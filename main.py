"""
FastAPI Backend for Ollama Chatbot
Connects frontend to local Ollama instance running Qwen2.5:3b-instruct
Includes guided patient intake flow with MySQL persistence (UUID7 user_id)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from typing import List, Optional, Dict, Any
import logging
import uuid
from uuid_extensions import uuid7str
import mysql.connector
from mysql.connector import Error as MySQLError
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_database_and_tables()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Chatbot API",
    description="Backend API for patient intake chatbot using Ollama",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = 'qwen3:4b'
OLLAMA_URL = 'http://localhost:11434'

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "12345678",
    "database": "healthcare_chatbot",
}

# ─── Ordered intake questions (field, Vietnamese question) ────────────────────
INTAKE_QUESTIONS: List[Dict[str, str]] = [
    {"field": "name",                  "table": "users",              "question": "Xin chào! Tôi là trợ lý y tế AI. Để hỗ trợ bạn tốt hơn, tôi sẽ hỏi một vài câu hỏi.\n\nHọ và tên đầy đủ của bạn là gì?"},
    {"field": "age",                   "table": "users",              "question": "Bạn bao nhiêu tuổi?"},
    {"field": "gender",                "table": "users",              "question": "Giới tính của bạn là gì? (Nam / Nữ / Khác)"},
    {"field": "job",                   "table": "users",              "question": "Nghề nghiệp hiện tại của bạn là gì?"},
    {"field": "address",               "table": "users",              "question": "Địa chỉ nơi ở của bạn?"},
    {"field": "reason",                "table": "li_do_kham_benh",    "question": "Lý do chính khiến bạn đến khám hôm nay là gì?"},
    {"field": "start_time",            "table": "li_do_kham_benh",    "question": "Triệu chứng bắt đầu từ khi nào? (Ví dụ: 3 ngày trước, từ hôm qua...)"},
    {"field": "tinh_chat_trieu_chung", "table": "li_do_kham_benh",    "question": "Hãy mô tả tính chất triệu chứng (mức độ đau, vị trí, tần suất xuất hiện...)?"},
    {"field": "tien_su_benh",          "table": "li_do_kham_benh",    "question": "Bạn có tiền sử bệnh gì không? (Ví dụ: tiểu đường, huyết áp, tim mạch...)"},
    {"field": "tien_su_thuoc",         "table": "li_do_kham_benh",    "question": "Bạn có đang dùng thuốc gì không? Nếu có, hãy liệt kê tên thuốc."},
    {"field": "tien_su_gia_dinh",      "table": "li_do_kham_benh",    "question": "Trong gia đình bạn có ai mắc bệnh di truyền hoặc bệnh mãn tính không?"},
    {"field": "thoi_quen_sinh_hoat",   "table": "li_do_kham_benh",    "question": "Hãy mô tả thói quen sinh hoạt hàng ngày (ăn uống, tập thể dục, hút thuốc, uống rượu...)"},
]

# ─── In-memory session store ──────────────────────────────────────────────────
# sessions[session_id] = { "answers": {field: value}, "step": int }
sessions: Dict[str, Dict[str, Any]] = {}

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    model: str

class HealthResponse(BaseModel):
    status: str
    ollama: str
    model: str

class IntakeStartResponse(BaseModel):
    session_id: str
    question: str
    field: str
    step: int
    total: int

class IntakeAnswerRequest(BaseModel):
    session_id: str
    answer: str

class IntakeAnswerResponse(BaseModel):
    done: bool
    question: Optional[str] = None
    field: Optional[str] = None
    step: Optional[int] = None
    total: Optional[int] = None
    user_id: Optional[str] = None
    message: Optional[str] = None

# ─── Database helpers ─────────────────────────────────────────────────────────
def get_db_connection():
    """Return a new MySQL connection."""
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn


def ensure_database_and_tables():
    """Create the database and tables if they do not exist."""
    try:
        # Connect without specifying a database first
        cfg_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
        conn = mysql.connector.connect(**cfg_no_db)
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cursor.execute(f"USE `{DB_CONFIG['database']}`;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                age INT,
                gender VARCHAR(20),
                job VARCHAR(100),
                address TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS li_do_kham_benh (
                record_id INT PRIMARY KEY AUTO_INCREMENT,
                user_id VARCHAR(36),
                reason TEXT NOT NULL,
                start_time TEXT,
                tinh_chat_trieu_chung TEXT,
                tien_su_benh TEXT,
                tien_su_thuoc TEXT,
                tien_su_gia_dinh TEXT,
                thoi_quen_sinh_hoat TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        conn.commit()
        logger.info("✅ Database and tables verified/created.")
    except MySQLError as e:
        logger.error(f"❌ Failed to create tables: {e}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def upsert_patient(answers: Dict[str, str]) -> str:
    """Insert users + li_do_kham_benh rows and return user_id (UUID7)."""
    user_id = uuid7str()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Insert user row
        cursor.execute(
            """INSERT INTO users (user_id, name, age, gender, job, address)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE name=VALUES(name), age=VALUES(age),
               gender=VALUES(gender), job=VALUES(job), address=VALUES(address)
            """,
            (
                user_id,
                answers.get("name", ""),
                int(answers["age"]) if answers.get("age", "").isdigit() else None,
                answers.get("gender", ""),
                answers.get("job", ""),
                answers.get("address", ""),
            )
        )

        # Insert medical visit row
        cursor.execute(
            """INSERT INTO li_do_kham_benh
               (user_id, reason, start_time, tinh_chat_trieu_chung,
                tien_su_benh, tien_su_thuoc, tien_su_gia_dinh, thoi_quen_sinh_hoat)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                answers.get("reason", ""),
                answers.get("start_time", ""),
                answers.get("tinh_chat_trieu_chung", ""),
                answers.get("tien_su_benh", ""),
                answers.get("tien_su_thuoc", ""),
                answers.get("tien_su_gia_dinh", ""),
                answers.get("thoi_quen_sinh_hoat", ""),
            )
        )

        conn.commit()
        logger.info(f"✅ Patient data saved. user_id={user_id}")
        return user_id
    except MySQLError as e:
        conn.rollback()
        logger.error(f"❌ DB insert failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()




# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Healthcare Chatbot API is running",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "intake_start": "/intake/start (POST)",
            "intake_answer": "/intake/answer (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models = response.json()
                model_names = [m["name"] for m in models.get("models", [])]
                model_available = any(MODEL_NAME in name for name in model_names)
                return HealthResponse(
                    status="healthy",
                    ollama="connected",
                    model="available" if model_available else "not found"
                )
            else:
                raise HTTPException(status_code=503, detail="Ollama returned an error")
    except httpx.TimeoutException:
        raise HTTPException(status_code=503, detail="Ollama connection timeout.")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama on port 11434")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/intake/start", response_model=IntakeStartResponse)
async def intake_start():
    """Start a new patient intake session and return the first question."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"answers": {}, "step": 0}

    first_q = INTAKE_QUESTIONS[0]
    return IntakeStartResponse(
        session_id=session_id,
        question=first_q["question"],
        field=first_q["field"],
        step=1,
        total=len(INTAKE_QUESTIONS),
    )


@app.post("/intake/answer", response_model=IntakeAnswerResponse)
async def intake_answer(request: IntakeAnswerRequest):
    """Receive an answer, save it, then return the next question or finish."""
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new intake.")

    step = session["step"]
    current_q = INTAKE_QUESTIONS[step]
    session["answers"][current_q["field"]] = request.answer.strip()
    session["step"] += 1

    next_step = session["step"]

    # More questions remain
    if next_step < len(INTAKE_QUESTIONS):
        next_q = INTAKE_QUESTIONS[next_step]
        return IntakeAnswerResponse(
            done=False,
            question=next_q["question"],
            field=next_q["field"],
            step=next_step + 1,
            total=len(INTAKE_QUESTIONS),
        )

    # All questions answered → save to DB
    try:
        user_id = upsert_patient(session["answers"])
        # Clean up session
        del sessions[request.session_id]
        return IntakeAnswerResponse(
            done=True,
            user_id=user_id,
            message=(
                "✅ Cảm ơn bạn đã cung cấp thông tin! Hồ sơ của bạn đã được lưu thành công. "
                "Bạn có thể tiếp tục trò chuyện với tôi về tình trạng sức khỏe của mình."
            ),
        )
    except Exception as e:
        logger.error(f"Failed to save patient: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể lưu dữ liệu: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Free-form chat endpoint (used after intake is complete)."""
    logger.info(f"Received message: {request.message[:50]}...")

    try:
        messages = []
        # System prompt for post-intake medical chat
        messages.append({
            "role": "system",
            "content": (
                "Bạn là trợ lý y tế AI thông minh, nhiệt tình và đáng tin cậy. "
                "Hãy trả lời các câu hỏi về sức khỏe một cách chính xác, rõ ràng và bằng tiếng Việt. "
                "Luôn nhắc người dùng gặp bác sĩ khi cần thiết."
            )
        })
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.message})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9},
                }
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Ollama API error: {response.text}")

            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            if not assistant_message:
                raise HTTPException(status_code=500, detail="Empty response from Ollama")

            return ChatResponse(response=assistant_message, model=MODEL_NAME)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Ollama timed out.")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama on port 11434")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch models")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Ollama: {str(e)}")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting Healthcare Chatbot Backend")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"MySQL: {DB_CONFIG['host']} / {DB_CONFIG['database']}")
    print(f"Server: http://localhost:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")