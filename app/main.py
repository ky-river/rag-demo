from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import closing
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Iterator, Literal, TypedDict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openai import APITimeoutError, OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.auth import AuthError, AuthStore, User
from app.evaluation import run_retrieval_evaluation
from app.rate_limit import SlidingWindowRateLimiter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
SESSION_DB = ROOT / "sessions.sqlite3"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env", override=True)


def positive_float_from_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def non_negative_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value >= 0 else default


MODEL_TIMEOUT_SECONDS = positive_float_from_env("MODEL_TIMEOUT_SECONDS", 60.0)
MODEL_MAX_RETRIES = non_negative_int_from_env("MODEL_MAX_RETRIES", 1)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
DEFAULT_COLLECTION_NAME = "rag_demo"
BUILD_COLLECTION_PREFIX = "rag_demo_build_"
COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")
ACTIVE_COLLECTION_FILE = DB_DIR / "active_collection.txt"
WORKSPACES_DATA_DIR = DATA_DIR / "users"
WORKSPACES_INDEX_DIR = DB_DIR / "users"
INDEX_REBUILD_LOCK = threading.Lock()
UPLOAD_LOCK = asyncio.Lock()
auth_store = AuthStore(SESSION_DB)
bearer_scheme = HTTPBearer(auto_error=False)


def workspace_data_dir(owner_id: str | None) -> Path:
    return WORKSPACES_DATA_DIR / owner_id if owner_id else DATA_DIR


def workspace_collection_prefix(owner_id: str | None) -> str:
    if not owner_id:
        return DEFAULT_COLLECTION_NAME
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"rag_demo_{digest}"


def active_collection_file(owner_id: str | None) -> Path:
    if not owner_id:
        return ACTIVE_COLLECTION_FILE
    return WORKSPACES_INDEX_DIR / owner_id / "active_collection.txt"


def read_active_collection_name(owner_id: str | None) -> str:
    prefix = workspace_collection_prefix(owner_id)
    try:
        name = active_collection_file(owner_id).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return prefix
    if (
        COLLECTION_NAME_PATTERN.fullmatch(name)
        and (
            name == prefix
            or name.startswith(f"{prefix}_build_")
            or (
                owner_id is None
                and (
                    name == DEFAULT_COLLECTION_NAME
                    or name.startswith(BUILD_COLLECTION_PREFIX)
                )
            )
        )
    ):
        return name
    return prefix


def persist_active_collection_name(owner_id: str | None, name: str) -> None:
    target = active_collection_file(owner_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(
        f".{target.name}.{uuid4().hex}.tmp"
    )
    temp_path.write_text(name, encoding="utf-8")
    os.replace(temp_path, target)


class BailianEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=MODEL_TIMEOUT_SECONDS,
            max_retries=MODEL_MAX_RETRIES,
        )

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        text = text.strip()
        if not text:
            raise ValueError("文本不能为空")

        response = self.client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return response.data[0].embedding


class QuestionRequest(BaseModel):
    question: str


class AuthRequest(BaseModel):
    email: str
    password: str


class StudyPlanRequest(BaseModel):
    gaps: list[str]


class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None


class ClearSessionRequest(BaseModel):
    session_id: str


class RetrievalEvaluationRequest(BaseModel):
    suite: Literal["smoke", "extended"] = "smoke"


class AgentGraphState(TypedDict, total=False):
    owner_id: str | None
    message: str
    intent: str
    tool: str
    tool_arguments: dict[str, str]
    routing: str
    history: list[dict[str, str]]
    answer: str
    sources: list[dict]


@dataclass
class WorkspaceIndex:
    collection_name: str
    vectorstore: Chroma
    bm25_documents: list[Document]
    bm25_index: BM25Okapi | None


embeddings = BailianEmbeddings()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=MODEL_MAX_RETRIES,
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_FILENAME_LENGTH = 120
PDF_SIGNATURE_READ_BYTES = 1024
AUTH_RATE_LIMIT_PER_MINUTE = non_negative_int_from_env(
    "AUTH_RATE_LIMIT_PER_MINUTE",
    10,
)
MODEL_RATE_LIMIT_PER_MINUTE = non_negative_int_from_env(
    "MODEL_RATE_LIMIT_PER_MINUTE",
    30,
)
WRITE_RATE_LIMIT_PER_MINUTE = non_negative_int_from_env(
    "WRITE_RATE_LIMIT_PER_MINUTE",
    10,
)
RATE_LIMIT_WINDOW_SECONDS = 60
RAG_TOP_K = 4
RAG_CANDIDATE_K = 12
RAG_RELEVANCE_THRESHOLD = 0.30
RRF_K = 10
RRF_VECTOR_WEIGHT = 1.0
RRF_BM25_WEIGHT = 3.0
RERANK_ENABLED = os.getenv("ENABLE_RERANK", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "BAAI/bge-reranker-v2-m3-int8",
)
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "4"))
APP_VERSION = os.getenv("APP_VERSION", "1.5")
RERANK_CACHE_DIR = Path(
    os.getenv("RERANK_CACHE_DIR", str(ROOT / ".cache" / "fastembed"))
)
if not RERANK_CACHE_DIR.is_absolute():
    RERANK_CACHE_DIR = ROOT / RERANK_CACHE_DIR
CHINESE_STOP_BIGRAMS = {
    "这个", "项目", "什么", "哪些", "怎么", "如何", "是否",
    "可以", "多少", "一个", "的是", "当前", "系统",
}
logger = logging.getLogger("rag_demo")
workspace_indexes: dict[str | None, WorkspaceIndex] = {}
rate_limiter = SlidingWindowRateLimiter()
MODEL_RATE_LIMIT_PATHS = {
    "/agent",
    "/agent-graph",
    "/agent-tools",
    "/ask",
    "/eval/retrieval",
    "/match",
    "/search",
    "/study-plan",
}
AUTH_RATE_LIMIT_PATHS = {"/auth/login", "/auth/register"}
reranker = None
reranker_error: str | None = None


class JsonLogFormatter(logging.Formatter):
    standard_fields = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.standard_fields and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)


def current_request_id() -> str:
    return request_id_context.get()


def create_chat_completion(operation: str, **kwargs):
    model = kwargs.get("model", "unknown")
    started_at = time.perf_counter()
    logger.info(
        "model_request_started",
        extra={
            "event": "model_request_started",
            "operation": operation,
            "model": model,
            "request_id": current_request_id(),
        },
    )
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        logger.error(
            "model_request_failed",
            extra={
                "event": "model_request_failed",
                "operation": operation,
                "model": model,
                "request_id": current_request_id(),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        raise

    logger.info(
        "model_request_completed",
        extra={
            "event": "model_request_completed",
            "operation": operation,
            "model": model,
            "request_id": current_request_id(),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )
    return response


def init_session_db() -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT '', "
            "messages_json TEXT NOT NULL, "
            "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE conversations "
                "ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user "
            "ON conversations(user_id)"
        )


def load_history(session_id: str, user_id: str) -> list[dict[str, str]]:
    with closing(sqlite3.connect(SESSION_DB)) as connection:
        row = connection.execute(
            "SELECT messages_json FROM conversations "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if not row:
        return []
    try:
        history = json.loads(row[0])
        return history if isinstance(history, list) else []
    except json.JSONDecodeError:
        return []


def save_history(
    session_id: str,
    history: list[dict[str, str]],
    user_id: str,
) -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        existing = connection.execute(
            "SELECT user_id FROM conversations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing and existing[0] != user_id:
            raise PermissionError("无权使用该会话")
        connection.execute(
            "INSERT INTO conversations(session_id, user_id, messages_json) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "messages_json = excluded.messages_json, updated_at = CURRENT_TIMESTAMP",
            (
                session_id,
                user_id,
                json.dumps(history[-10:], ensure_ascii=False),
            ),
        )


def clear_history(session_id: str, user_id: str) -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        connection.execute(
            "DELETE FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )


def bearer_token_from_request(request: Request) -> str | None:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def rate_limit_policy(method: str, path: str) -> tuple[str, int] | None:
    normalized_path = path.rstrip("/") or "/"
    if normalized_path in AUTH_RATE_LIMIT_PATHS:
        return "auth", AUTH_RATE_LIMIT_PER_MINUTE
    if normalized_path in MODEL_RATE_LIMIT_PATHS:
        return "model", MODEL_RATE_LIMIT_PER_MINUTE
    if (
        normalized_path == "/upload"
        or (method == "DELETE" and normalized_path.startswith("/documents/"))
    ):
        return "write", WRITE_RATE_LIMIT_PER_MINUTE
    return None


def require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    user = getattr(request.state, "authenticated_user", None)
    if user is not None:
        return user
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = auth_store.authenticate(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="登录状态已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


configure_logging()
init_session_db()

app = FastAPI(title="RAG Knowledge Base API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def auth_payload(user: User, token: str) -> dict:
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email},
    }


@app.post("/auth/register")
def register(request: AuthRequest):
    user, token = auth_store.register(request.email, request.password)
    return auth_payload(user, token)


@app.post("/auth/login")
def login(request: AuthRequest):
    user, token = auth_store.login(request.email, request.password)
    return auth_payload(user, token)


@app.get("/auth/me")
def auth_me(user: User = Depends(require_user)):
    return {"user": {"id": user.id, "email": user.email}}


@app.post("/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is not None:
        auth_store.revoke(credentials.credentials)
    return {"message": "已退出登录"}


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_request_id
        if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
        else str(uuid4())
    )
    request.state.request_id = request_id
    token = request_id_context.set(request_id)
    started_at = time.perf_counter()
    rate_limit_result = None
    try:
        policy = rate_limit_policy(request.method, request.url.path)
        if policy is not None:
            category, limit = policy
            subject = request.client.host if request.client else "unknown"
            if category != "auth":
                bearer_token = bearer_token_from_request(request)
                user = (
                    auth_store.authenticate(bearer_token)
                    if bearer_token
                    else None
                )
                if user is not None:
                    request.state.authenticated_user = user
                    subject = user.id

            rate_limit_result = rate_limiter.check(
                key=f"{category}:{subject}",
                limit=limit,
                window_seconds=RATE_LIMIT_WINDOW_SECONDS,
            )
            if not rate_limit_result.allowed:
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "请求过于频繁，请稍后重试",
                        "retry_after_seconds": rate_limit_result.retry_after_seconds,
                        "request_id": request_id,
                    },
                    headers={
                        "Retry-After": str(
                            rate_limit_result.retry_after_seconds
                        ),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            else:
                response = await call_next(request)
                response.headers["X-RateLimit-Remaining"] = str(
                    rate_limit_result.remaining
                )
        else:
            response = await call_next(request)
    except Exception as exc:
        logger.error(
            "request_failed",
            extra={
                "event": "request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
    finally:
        request_id_context.reset(token)


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "request_id": getattr(request.state, "request_id", current_request_id()),
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "request_id": getattr(request.state, "request_id", current_request_id()),
        },
    )
    for key, value in exc.headers.items():
        response.headers[key] = value
    return response


@app.exception_handler(APITimeoutError)
async def model_timeout_handler(request: Request, exc: APITimeoutError):
    request_id = getattr(request.state, "request_id", current_request_id())
    logger.warning(
        "model_request_timeout",
        extra={
            "event": "model_request_timeout",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=504,
        content={
            "error": "模型请求超时，请稍后重试",
            "request_id": request_id,
        },
    )


def tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_.+-]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        bigrams = [run[index:index + 2] for index in range(len(run) - 1)]
        tokens.extend(
            token for token in bigrams
            if token not in CHINESE_STOP_BIGRAMS
        )
    return tokens


def load_documents_from_store(store: Chroma) -> list[Document]:
    stored = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            stored.get("documents", []),
            stored.get("metadatas", []),
        )
        if content and content.strip()
    ]


def build_bm25_index(documents: list[Document]) -> BM25Okapi | None:
    tokenized_documents = [
        tokenize_for_bm25(document.page_content)
        for document in documents
    ]
    return BM25Okapi(tokenized_documents) if tokenized_documents else None


def indexed_pdf_files(owner_id: str | None) -> list[Path]:
    data_dir = workspace_data_dir(owner_id)
    data_dir.mkdir(parents=True, exist_ok=True)
    return [
        path
        for path in sorted(data_dir.iterdir())
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() == ".pdf"
        )
    ]


def cleanup_stale_build_collections(
    owner_id: str | None,
    workspace: WorkspaceIndex,
) -> None:
    build_prefix = f"{workspace_collection_prefix(owner_id)}_build_"
    try:
        collections = workspace.vectorstore._client.list_collections()
    except Exception as exc:
        logger.warning(
            "读取索引集合列表失败：%s",
            exc,
            extra={"event": "collection_list_failed"},
        )
        return

    for collection in collections:
        name = getattr(collection, "name", "")
        if not name.startswith(build_prefix):
            continue
        if name == workspace.collection_name:
            continue
        try:
            workspace.vectorstore._client.delete_collection(name)
            logger.info(
                "stale_index_collection_removed",
                extra={
                    "event": "stale_index_collection_removed",
                    "collection": name,
                },
            )
        except Exception as exc:
            logger.warning(
                "清理残留索引集合失败：%s",
                exc,
                extra={
                    "event": "stale_index_cleanup_failed",
                    "collection": name,
                },
            )


def get_workspace_index_locked(owner_id: str | None) -> WorkspaceIndex:
    workspace = workspace_indexes.get(owner_id)
    if workspace is not None:
        return workspace

    collection_name = read_active_collection_name(owner_id)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    documents = load_documents_from_store(vectorstore)
    workspace = WorkspaceIndex(
        collection_name=collection_name,
        vectorstore=vectorstore,
        bm25_documents=documents,
        bm25_index=build_bm25_index(documents),
    )
    cleanup_stale_build_collections(owner_id, workspace)
    workspace_indexes[owner_id] = workspace
    return workspace


def build_index(owner_id: str | None) -> int:
    """Build a staging index and switch to it only after validation succeeds."""
    with INDEX_REBUILD_LOCK:
        old_workspace = get_workspace_index_locked(owner_id)
        documents = []
        for pdf_path in indexed_pdf_files(owner_id):
            pdf_documents = PyPDFLoader(str(pdf_path)).load()
            for document in pdf_documents:
                document.metadata["filename"] = pdf_path.name
                if owner_id:
                    document.metadata["owner_id"] = owner_id
            documents.extend(pdf_documents)

        chunks = splitter.split_documents(documents)
        chunks = [
            chunk for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]
        if not chunks:
            raise ValueError("没有可用于建立索引的文本片段")

        old_collection_name = old_workspace.collection_name
        old_vectorstore = old_workspace.vectorstore
        collection_prefix = workspace_collection_prefix(owner_id)
        new_collection_name = (
            f"{collection_prefix}_build_{int(time.time())}_{uuid4().hex[:8]}"
        )
        staging_vectorstore = None

        try:
            staging_vectorstore = Chroma(
                collection_name=new_collection_name,
                persist_directory=str(DB_DIR),
                embedding_function=embeddings,
            )
            staging_vectorstore.add_documents(chunks)
            staged_count = staging_vectorstore._collection.count()
            if staged_count != len(chunks):
                raise RuntimeError(
                    f"索引校验失败：预期 {len(chunks)} 个片段，实际 {staged_count} 个"
                )
            new_bm25_index = build_bm25_index(chunks)
            persist_active_collection_name(owner_id, new_collection_name)
        except Exception:
            if staging_vectorstore is not None:
                try:
                    staging_vectorstore.delete_collection()
                except Exception as cleanup_error:
                    logger.warning(
                        "清理失败索引集合时出错：%s",
                        cleanup_error,
                        extra={
                            "event": "index_cleanup_failed",
                            "collection": new_collection_name,
                        },
                    )
            raise

        workspace_indexes[owner_id] = WorkspaceIndex(
            collection_name=new_collection_name,
            vectorstore=staging_vectorstore,
            bm25_documents=chunks,
            bm25_index=new_bm25_index,
        )

        if old_collection_name != new_collection_name:
            try:
                old_vectorstore.delete_collection()
            except Exception as cleanup_error:
                logger.warning(
                    "删除旧索引集合时出错：%s",
                    cleanup_error,
                    extra={
                        "event": "old_index_cleanup_failed",
                        "collection": old_collection_name,
                    },
                )

        logger.info(
            "index_rebuild_completed",
            extra={
                "event": "index_rebuild_completed",
                "owner_id": owner_id,
                "collection": new_collection_name,
                "previous_collection": old_collection_name,
                "chunk_count": len(chunks),
            },
        )
        return len(chunks)


def get_reranker():
    global reranker, reranker_error

    if not RERANK_ENABLED:
        return None
    if reranker is not None:
        return reranker
    if reranker_error is not None:
        return None

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        reranker = TextCrossEncoder(
            model_name=RERANK_MODEL,
            cache_dir=str(RERANK_CACHE_DIR),
        )
    except Exception as exc:
        reranker_error = str(exc)
        logger.warning("Cross-Encoder 初始化失败，回退到 RRF：%s", exc)
    return reranker


def rerank_items(question: str, items: list[dict], reranker_override=None) -> list[dict]:
    if len(items) <= 1:
        return items
    model = reranker_override or get_reranker()
    if model is None:
        return items

    candidate_count = min(max(RAG_TOP_K, RERANK_CANDIDATES), len(items))
    candidates = items[:candidate_count]
    documents = [
        item["document"].page_content
        for item in candidates
    ]

    try:
        scores = list(model.rerank(question, documents, batch_size=8))
    except Exception as exc:
        logger.warning("Cross-Encoder 推理失败，回退到 RRF：%s", exc)
        return items

    enriched = []
    for pre_rerank_rank, (item, score) in enumerate(
        zip(candidates, scores),
        start=1,
    ):
        ranked_item = dict(item)
        ranked_item["rerank_score"] = float(score)
        ranked_item["pre_rerank_rank"] = pre_rerank_rank
        enriched.append(ranked_item)

    enriched.sort(key=lambda item: item["rerank_score"], reverse=True)
    return enriched + items[candidate_count:]


def reranking_status() -> dict:
    return {
        "enabled": RERANK_ENABLED,
        "active": reranker is not None,
        "model": RERANK_MODEL if RERANK_ENABLED else None,
        "candidate_count": RERANK_CANDIDATES,
        "error": reranker_error,
    }


def health_status(owner_id: str | None) -> dict:
    with INDEX_REBUILD_LOCK:
        workspace = get_workspace_index_locked(owner_id)
    chunk_count = len(workspace.bm25_documents)
    document_count = len(
        {
            document.metadata.get("filename")
            for document in workspace.bm25_documents
            if document.metadata.get("filename")
        }
    )
    collection_error = None
    try:
        collection_count = workspace.vectorstore._collection.count()
    except Exception as exc:
        collection_count = None
        collection_error = str(exc)

    index_ready = (
        collection_error is None
        and collection_count == chunk_count
        and (chunk_count == 0 or workspace.bm25_index is not None)
    )
    if collection_error:
        status = "error"
    elif index_ready:
        status = "ok"
    else:
        status = "degraded"

    return {
        "status": status,
        "version": APP_VERSION,
        "index_ready": index_ready,
        "collection_name": workspace.collection_name,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "collection_count": collection_count,
        "collection_error": collection_error,
        "model": {
            "timeout_seconds": MODEL_TIMEOUT_SECONDS,
            "max_retries": MODEL_MAX_RETRIES,
        },
        "reranking": reranking_status(),
    }


def build_document_inventory(
    owner_id: str | None,
    workspace: WorkspaceIndex,
) -> list[dict[str, Any]]:
    chunk_counts = Counter(
        str(document.metadata.get("filename"))
        for document in workspace.bm25_documents
        if document.metadata.get("filename")
    )
    inventory = []
    for pdf_path in indexed_pdf_files(owner_id):
        stat = pdf_path.stat()
        inventory.append(
            {
                "filename": pdf_path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "chunk_count": chunk_counts.get(pdf_path.name, 0),
            }
        )
    return inventory


@app.get("/")
def home():
    return {"message": "RAG API is running", "version": APP_VERSION}


@app.get("/health")
def health():
    result = health_status(None)
    return JSONResponse(
        status_code=200 if result["index_ready"] else 503,
        content=result,
    )


@app.get("/workspace/health")
def workspace_health(user: User = Depends(require_user)):
    result = health_status(user.id)
    return JSONResponse(
        status_code=200 if result["index_ready"] else 503,
        content=result,
    )


@app.get("/documents")
async def list_documents(user: User = Depends(require_user)):
    async with UPLOAD_LOCK:
        with INDEX_REBUILD_LOCK:
            workspace = get_workspace_index_locked(user.id)
            documents = build_document_inventory(user.id, workspace)
            collection = workspace.collection_name
            chunk_count = len(workspace.bm25_documents)
    return {
        "documents": documents,
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "collection": collection,
    }


def document_key(document: Document) -> tuple:
    return (
        document.metadata.get("filename", ""),
        document.metadata.get("page", 0),
        document.page_content,
    )


def _retrieve_knowledge_locked(question: str, owner_id: str | None):
    workspace = get_workspace_index_locked(owner_id)
    raw_vector_results = workspace.vectorstore.similarity_search_with_score(
        question,
        k=RAG_CANDIDATE_K,
    )
    vector_results = [
        (
            document,
            max(
                0.0,
                min(1.0, 1.0 - float(distance) / math.sqrt(2)),
            ),
        )
        for document, distance in raw_vector_results
    ]

    vector_by_key = {}
    for rank, (document, score) in enumerate(vector_results, 1):
        if score >= RAG_RELEVANCE_THRESHOLD:
            vector_by_key[document_key(document)] = (
                document,
                float(score),
                rank,
            )

    lexical_scores = {}
    if workspace.bm25_index is not None:
        query_tokens = tokenize_for_bm25(question)
        if query_tokens:
            scores = workspace.bm25_index.get_scores(query_tokens)
            lexical_order = sorted(
                range(len(workspace.bm25_documents)),
                key=lambda index: scores[index],
                reverse=True,
            )
            for rank, index in enumerate(lexical_order, 1):
                score = float(scores[index])
                if score <= 0:
                    break
                document = workspace.bm25_documents[index]
                lexical_scores[document_key(document)] = (document, score, rank)

    candidate_keys = set(vector_by_key) | set(lexical_scores)
    fused = []
    for key in candidate_keys:
        vector_entry = vector_by_key.get(key)
        lexical_entry = lexical_scores.get(key)
        document = (vector_entry or lexical_entry)[0]
        hybrid_score = 0.0
        if vector_entry:
            hybrid_score += (
                RRF_VECTOR_WEIGHT / (RRF_K + vector_entry[2])
            )
        if lexical_entry:
            hybrid_score += (
                RRF_BM25_WEIGHT / (RRF_K + lexical_entry[2])
            )

        vector_score = vector_entry[1] if vector_entry else None
        lexical_score = lexical_entry[1] if lexical_entry else 0.0
        methods = []
        if vector_entry:
            methods.append("vector")
        if lexical_entry:
            methods.append("bm25")
        fused.append(
            {
                "document": document,
                "vector_score": vector_score,
                "lexical_score": lexical_score,
                "hybrid_score": hybrid_score,
                "retrieval_method": "+".join(methods),
            }
        )

    fused.sort(
        key=lambda item: (
            item["hybrid_score"],
            item["vector_score"] or 0,
            item["lexical_score"],
        ),
        reverse=True,
    )
    reranked = rerank_items(question, fused)
    docs = reranked[:RAG_TOP_K]
    sources = [
        {
            "filename": item["document"].metadata.get("filename", "未知文件"),
            "page": item["document"].metadata.get("page", 0) + 1,
            "content": item["document"].page_content,
            "relevance_score": (
                round(item["vector_score"], 4)
                if item["vector_score"] is not None
                else None
            ),
            "lexical_score": round(item["lexical_score"], 4),
            "hybrid_score": round(item["hybrid_score"], 6),
            "retrieval_method": item["retrieval_method"],
            "rerank_score": (
                round(item["rerank_score"], 6)
                if item.get("rerank_score") is not None
                else None
            ),
            "pre_rerank_rank": item.get("pre_rerank_rank"),
        }
        for item in docs
    ]
    return docs, sources


def retrieve_knowledge(question: str, owner_id: str | None):
    with INDEX_REBUILD_LOCK:
        return _retrieve_knowledge_locked(question, owner_id)


def format_retrieval_context(item: dict) -> str:
    document = item["document"]
    vector_score = (
        f"{item['vector_score']:.3f}"
        if item["vector_score"] is not None
        else "无"
    )
    return (
        f"来源：{document.metadata.get('filename', '未知文件')}，"
        f"第 {document.metadata.get('page', 0) + 1} 页，"
        f"向量分数 {vector_score}，"
        f"BM25 分数 {item['lexical_score']:.3f}，"
        f"融合分数 {item['hybrid_score']:.4f}\n"
        f"{document.page_content}"
    )


def answer_from_knowledge_base(
    question: str,
    history=None,
    owner_id: str | None = None,
):
    docs, sources = retrieve_knowledge(question, owner_id)
    if not docs:
        return {
            "answer": "当前资料中没有找到答案",
            "sources": [],
        }

    context = "\n\n".join(format_retrieval_context(item) for item in docs)
    prompt = f"""
请根据以下知识库内容回答问题。
如果内容中没有答案，请回答“当前资料中没有找到答案”。
不要编造信息。

知识库内容：
{context}

用户问题：
{question}
"""
    messages = [
        {"role": "system", "content": "你是一个严谨的知识库问答助手。"},
    ]
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})
    response = create_chat_completion(
        "knowledge_answer",
        model="qwen-plus",
        messages=messages,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }


def detect_intent(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ["匹配", "简历", "岗位", "jd"]):
        return "resume_match"
    if any(word in text for word in [
        "学习计划", "怎么学", "如何学习", "学习路线", "技能缺口", "学习 langgraph",
        "学习python", "学习 python", "学习rag", "学习 rag", "学习agent", "学习 agent",
    ]):
        return "study_plan"
    return "knowledge_qa"


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "检索当前知识库并回答项目文档、技术栈、系统架构、RAG 流程和 FAQ 等问题。"
                "当用户说“这个项目”“当前系统”或“我的文档”时，默认指当前知识库，"
                "必须调用本工具，不要先询问项目名称。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_study_plan",
            "description": "根据用户想学习的技能或主题制定学习计划。",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_match_guidance",
            "description": "处理简历和岗位匹配请求，并提示用户上传两个 PDF。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]

TOOL_TO_INTENT = {
    "knowledge_search": "knowledge_qa",
    "create_study_plan": "study_plan",
    "resume_match_guidance": "resume_match",
}
INTENT_TO_TOOL = {intent: tool for tool, intent in TOOL_TO_INTENT.items()}
AGENT_ROUTING_PROMPT = (
    "你是智能求职 Agent。每次只选择最合适的一个工具。"
    "项目技术栈、项目架构、RAG 流程、知识库或已上传文档相关问题必须调用 "
    "knowledge_search；用户说“这个项目”时默认指当前知识库，不要反问项目名称。"
    "学习路线和技能学习问题调用 create_study_plan；"
    "简历与岗位匹配问题调用 resume_match_guidance。"
    "不要自己猜测知识库内容。"
)


def normalize_tool_arguments(tool: str, arguments: dict, message: str) -> dict[str, str]:
    arguments = arguments if isinstance(arguments, dict) else {}
    if tool == "knowledge_search":
        return {"question": arguments.get("question") or message}
    if tool == "create_study_plan":
        return {"topic": arguments.get("topic") or message}
    return {}


def fallback_tool_choice(message: str) -> tuple[str, dict[str, str], str]:
    tool = INTENT_TO_TOOL[detect_intent(message)]
    return tool, normalize_tool_arguments(tool, {}, message), "fallback"


def select_agent_tool(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, dict[str, str], str]:
    response = create_chat_completion(
        "agent_routing",
        model="qwen-plus",
        messages=[
            {"role": "system", "content": AGENT_ROUTING_PROMPT},
            *history,
            {"role": "user", "content": message},
        ],
        tools=AGENT_TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        return fallback_tool_choice(message)

    tool_call = assistant_message.tool_calls[0]
    tool_name = tool_call.function.name
    if tool_name not in TOOL_TO_INTENT:
        return fallback_tool_choice(message)

    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}

    return (
        tool_name,
        normalize_tool_arguments(tool_name, arguments, message),
        "model",
    )


def run_study_plan_text(topic: str, history=None) -> str:
    response = create_chat_completion(
        "study_plan_text",
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个务实的技术学习规划助手。"},
            *(history or []),
            {"role": "user", "content": (
                "请根据用户需求制定一个简洁的 4 周学习计划。"
                "用中文回答，包含每周目标、任务和交付物。用户需求：" + topic
            )},
        ],
    )
    return response.choices[0].message.content or ""


def execute_agent_tool(
    name: str,
    arguments: dict,
    history=None,
    owner_id: str | None = None,
) -> dict:
    if name == "knowledge_search":
        return answer_from_knowledge_base(
            arguments["question"],
            history,
            owner_id=owner_id,
        )
    if name == "create_study_plan":
        return {"answer": run_study_plan_text(arguments["topic"], history), "sources": []}
    if name == "resume_match_guidance":
        return {
            "answer": "简历与岗位匹配需要两个 PDF 文件，请使用网页中的简历和岗位 JD 上传区域。",
            "sources": [],
        }
    raise ValueError(f"未知工具：{name}")


def graph_route(state: AgentGraphState) -> AgentGraphState:
    tool, arguments, routing = select_agent_tool(
        state["message"],
        state.get("history", []),
    )
    return {
        "intent": TOOL_TO_INTENT[tool],
        "tool": tool,
        "tool_arguments": arguments,
        "routing": routing,
    }


def graph_knowledge(state: AgentGraphState) -> AgentGraphState:
    result = execute_agent_tool(
        "knowledge_search",
        state.get("tool_arguments", {"question": state["message"]}),
        state.get("history", []),
        state.get("owner_id"),
    )
    return {"answer": result["answer"], "sources": result["sources"]}


def graph_study_plan(state: AgentGraphState) -> AgentGraphState:
    result = execute_agent_tool(
        "create_study_plan",
        state.get("tool_arguments", {"topic": state["message"]}),
        state.get("history", []),
        state.get("owner_id"),
    )
    return {"answer": result["answer"], "sources": result["sources"]}


def graph_resume_match(state: AgentGraphState) -> AgentGraphState:
    result = execute_agent_tool(
        "resume_match_guidance",
        state.get("tool_arguments", {}),
        state.get("history", []),
        state.get("owner_id"),
    )
    return {"answer": result["answer"], "sources": result["sources"]}


def graph_next_node(state: AgentGraphState) -> Literal["knowledge", "study", "resume"]:
    return {
        "knowledge_search": "knowledge",
        "create_study_plan": "study",
        "resume_match_guidance": "resume",
    }[state["tool"]]


agent_graph_builder = StateGraph(AgentGraphState)
agent_graph_builder.add_node("route", graph_route)
agent_graph_builder.add_node("knowledge", graph_knowledge)
agent_graph_builder.add_node("study", graph_study_plan)
agent_graph_builder.add_node("resume", graph_resume_match)
agent_graph_builder.add_edge(START, "route")
agent_graph_builder.add_conditional_edges(
    "route",
    graph_next_node,
    {"knowledge": "knowledge", "study": "study", "resume": "resume"},
)
agent_graph_builder.add_edge("knowledge", END)
agent_graph_builder.add_edge("study", END)
agent_graph_builder.add_edge("resume", END)
agent_graph = agent_graph_builder.compile()


def run_agent_graph(request: AgentRequest, owner_id: str):
    message = request.message.strip()
    if not message:
        return {"error": "请输入任务"}

    session_id = request.session_id or str(uuid4())
    history = load_history(session_id, owner_id)[-10:]
    graph_history = history.copy()
    result = agent_graph.invoke({
        "owner_id": owner_id,
        "message": message,
        "history": graph_history,
    })
    answer = result.get("answer", "")
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    try:
        save_history(session_id, history, owner_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "intent": result.get("intent"),
        "tool": result.get("tool"),
        "routing": result.get("routing"),
        "answer": answer,
        "sources": result.get("sources", []),
    }


@app.post("/agent-graph")
def agent_graph_endpoint(
    request: AgentRequest,
    user: User = Depends(require_user),
):
    return run_agent_graph(request, user.id)


@app.post("/agent-tools", deprecated=True)
def agent_tools(
    request: AgentRequest,
    user: User = Depends(require_user),
):
    """Compatibility alias for clients using the previous endpoint."""
    return run_agent_graph(request, user.id)


@app.post("/agent", deprecated=True)
def agent(
    request: AgentRequest,
    user: User = Depends(require_user),
):
    message = request.message.strip()
    if not message:
        return {"error": "请输入任务"}

    session_id = request.session_id or str(uuid4())
    history = load_history(session_id, user.id)[-10:]
    intent = detect_intent(message)
    if intent == "resume_match":
        reply = "简历与岗位匹配需要两个 PDF 文件，请使用下方的简历和岗位 JD 上传区域。"
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ])
        save_history(session_id, history, user.id)
        return {
            "session_id": session_id,
            "intent": intent,
            "message": reply,
        }

    if intent == "study_plan":
        messages = [
            {"role": "system", "content": "你是一个务实的技术学习规划助手。"},
            *history,
            {"role": "user", "content": (
                "请根据用户需求制定一个简洁的 4 周学习计划。"
                "用中文回答，包含每周目标、任务和交付物。用户需求：" + message
            )},
        ]
        response = create_chat_completion(
            "legacy_agent_study_plan",
            model="qwen-plus",
            messages=messages,
        )
        answer = response.choices[0].message.content
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        save_history(session_id, history, user.id)
        return {"session_id": session_id, "intent": intent, "answer": answer, "sources": []}

    result = answer_from_knowledge_base(message, history, owner_id=user.id)
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]},
    ])
    save_history(session_id, history, user.id)
    return {"session_id": session_id, "intent": intent, **result}


@app.post("/agent/clear")
def clear_agent_session(
    request: ClearSessionRequest,
    user: User = Depends(require_user),
):
    clear_history(request.session_id, user.id)
    return {"message": "会话已清空", "session_id": request.session_id}


def validate_pdf_filename(filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("只支持 PDF 文件")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"PDF 文件名不能超过 {MAX_FILENAME_LENGTH} 个字符")
    if any(ord(character) < 32 for character in filename):
        raise ValueError("PDF 文件名包含非法字符")


def has_pdf_signature(path: Path) -> bool:
    with path.open("rb") as buffer:
        header = buffer.read(PDF_SIGNATURE_READ_BYTES)
    return b"%PDF-" in header


def read_uploaded_pdf(upload: UploadFile, temp_dir: Path) -> str:
    filename = Path(upload.filename or "").name
    try:
        validate_pdf_filename(filename)
    except ValueError as exc:
        raise ValueError("简历和岗位 JD 都必须是 PDF 文件") from exc

    pdf_path = temp_dir / filename
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    if not has_pdf_signature(pdf_path):
        raise ValueError("文件内容不是有效的 PDF")

    documents = PyPDFLoader(str(pdf_path)).load()
    return "\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content and document.page_content.strip()
    )


@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    user: User = Depends(require_user),
):
    with tempfile.TemporaryDirectory(prefix="rag_match_") as temp_path:
        temp_dir = Path(temp_path)
        try:
            resume_text = read_uploaded_pdf(resume, temp_dir)
            job_text = read_uploaded_pdf(job_description, temp_dir)
        except ValueError as exc:
            return {"error": str(exc)}

    if not resume_text or not job_text:
        return {"error": "简历或岗位 JD 没有读取到有效文本"}

    prompt = f"""
你是一名专业的技术招聘分析助手。请比较下面的候选人简历和岗位 JD，严格只根据提供的内容分析，不要编造经历。

只输出合法 JSON，不要 Markdown 代码块，格式必须是：
{{"match_score": 0, "matched_skills": [], "skill_gaps": [], "resume_suggestions": [], "interview_questions": []}}
match_score 是 0 到 100 的整数；其他字段都是中文字符串数组。

候选人简历：
{resume_text[:30000]}

岗位 JD：
{job_text[:30000]}
"""

    response = create_chat_completion(
        "resume_match",
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是严谨的招聘分析助手。"},
            {"role": "user", "content": prompt},
        ],
    )

    raw_analysis = response.choices[0].message.content or "{}"
    try:
        analysis = json.loads(raw_analysis)
    except json.JSONDecodeError:
        analysis = {
            "match_score": None,
            "matched_skills": [],
            "skill_gaps": [],
            "resume_suggestions": [],
            "interview_questions": [],
            "raw_analysis": raw_analysis,
        }

    return {
        "resume_filename": Path(resume.filename or "").name,
        "job_description_filename": Path(job_description.filename or "").name,
        "analysis": analysis,
    }


@app.post("/study-plan")
def study_plan(
    request: StudyPlanRequest,
    user: User = Depends(require_user),
):
    gaps = [gap.strip() for gap in request.gaps if gap.strip()]
    if not gaps:
        return {"error": "没有可用于制定计划的技能缺口"}

    response = create_chat_completion(
        "study_plan_json",
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个务实的技术学习规划助手。"},
            {"role": "user", "content": (
                "请根据以下技能缺口制定一个 4 周学习计划。只输出合法 JSON，不要 Markdown。"
                "格式：{\"goal\":\"\",\"weeks\":[{\"week\":1,\"focus\":\"\",\"tasks\":[] ,\"deliverable\":\"\"}]}。"
                f"技能缺口：{json.dumps(gaps, ensure_ascii=False)}"
            )},
        ],
    )
    raw_plan = response.choices[0].message.content or "{}"
    try:
        plan = json.loads(raw_plan)
    except json.JSONDecodeError:
        plan = {"raw_plan": raw_plan}
    return {"plan": plan}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user: User = Depends(require_user),
):
    filename = Path(file.filename or "").name
    try:
        validate_pdf_filename(filename)
    except ValueError as exc:
        return {"error": str(exc)}

    data_dir = workspace_data_dir(user.id)
    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / filename
    upload_token = uuid4().hex
    temp_pdf_path = data_dir / f".upload_{upload_token}.pdf"
    backup_pdf_path = data_dir / f".backup_{upload_token}.pdf"
    total_bytes = 0

    async with UPLOAD_LOCK:
        try:
            with temp_pdf_path.open("wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        temp_pdf_path.unlink(missing_ok=True)
                        return {"error": "PDF 文件不能超过 10MB"}
                    buffer.write(chunk)

            if total_bytes == 0:
                temp_pdf_path.unlink(missing_ok=True)
                return {"error": "不能上传空文件"}
            if not has_pdf_signature(temp_pdf_path):
                temp_pdf_path.unlink(missing_ok=True)
                return {"error": "文件内容不是有效的 PDF"}

            documents = PyPDFLoader(str(temp_pdf_path)).load()
            has_text = any(
                document.page_content and document.page_content.strip()
                for document in documents
            )
            if not has_text:
                temp_pdf_path.unlink(missing_ok=True)
                return {"error": "PDF 中没有可读取的文字内容"}
        except Exception:
            temp_pdf_path.unlink(missing_ok=True)
            return {"error": "PDF 文件损坏或无法解析"}

        had_previous_file = pdf_path.exists()
        try:
            if had_previous_file:
                shutil.copy2(pdf_path, backup_pdf_path)
            os.replace(temp_pdf_path, pdf_path)
        except Exception:
            temp_pdf_path.unlink(missing_ok=True)
            if backup_pdf_path.exists():
                os.replace(backup_pdf_path, pdf_path)
            return {"error": "PDF 文件写入磁盘失败"}

        try:
            chunk_count = await run_in_threadpool(build_index, user.id)
        except Exception as exc:
            pdf_path.unlink(missing_ok=True)
            if had_previous_file and backup_pdf_path.exists():
                os.replace(backup_pdf_path, pdf_path)
            return {"error": f"PDF 入库失败，已保留原索引：{exc}"}

        backup_pdf_path.unlink(missing_ok=True)
        with INDEX_REBUILD_LOCK:
            collection = get_workspace_index_locked(user.id).collection_name
        return {
            "message": f"PDF 上传并入库成功，共新增 {chunk_count} 个文本片段",
            "filename": filename,
            "chunks": chunk_count,
            "collection": collection,
        }


@app.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    user: User = Depends(require_user),
):
    safe_filename = Path(filename).name
    if (
        safe_filename != filename
        or not safe_filename
        or len(safe_filename) > MAX_FILENAME_LENGTH
        or not safe_filename.lower().endswith(".pdf")
    ):
        return {"error": "文档名称不合法"}

    data_dir = workspace_data_dir(user.id)
    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / safe_filename
    delete_token = uuid4().hex
    backup_pdf_path = data_dir / f".deleted_{delete_token}.pdf"

    async with UPLOAD_LOCK:
        if not pdf_path.is_file():
            return {"error": "文档不存在或已经被删除"}

        try:
            os.replace(pdf_path, backup_pdf_path)
        except OSError:
            return {"error": "无法移动待删除的 PDF 文件"}

        try:
            chunk_count = await run_in_threadpool(build_index, user.id)
        except Exception as exc:
            try:
                os.replace(backup_pdf_path, pdf_path)
            except OSError as restore_error:
                logger.error(
                    "恢复删除失败文档时出错：%s",
                    restore_error,
                    extra={
                        "event": "document_delete_restore_failed",
                        "filename": safe_filename,
                    },
                )
            with INDEX_REBUILD_LOCK:
                collection = get_workspace_index_locked(user.id).collection_name
            return {
                "error": f"删除后索引重建失败，已保留原索引：{exc}",
                "collection": collection,
            }

        try:
            backup_pdf_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "清理已删除文档备份失败：%s",
                cleanup_error,
                extra={
                    "event": "document_delete_cleanup_failed",
                    "filename": safe_filename,
                },
            )

        with INDEX_REBUILD_LOCK:
            collection = get_workspace_index_locked(user.id).collection_name
        return {
            "message": f"文档已删除并重建索引，当前共 {chunk_count} 个文本片段",
            "filename": safe_filename,
            "chunks": chunk_count,
            "collection": collection,
        }


@app.post("/ask")
def ask(
    request: QuestionRequest,
    user: User = Depends(require_user),
):
    question = request.question.strip()
    if not question:
        return {"error": "问题不能为空"}
    return {
        "question": question,
        **answer_from_knowledge_base(question, owner_id=user.id),
    }


@app.post("/search")
def search(
    request: QuestionRequest,
    user: User = Depends(require_user),
):
    question = request.question.strip()
    if not question:
        return {"error": "问题不能为空"}

    started_at = time.perf_counter()
    _, sources = retrieve_knowledge(question, user.id)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    reranking = reranking_status()
    return {
        "question": question,
        "retrieval_mode": (
            "vector+bm25_rrf+cross_encoder"
            if reranking["active"]
            else "vector+bm25_rrf"
        ),
        "top_k": RAG_TOP_K,
        "candidate_k": RAG_CANDIDATE_K,
        "relevance_threshold": RAG_RELEVANCE_THRESHOLD,
        "reranking": reranking,
        "latency_ms": latency_ms,
        "sources": sources,
    }


@app.post("/eval/retrieval")
def evaluate_retrieval(
    request: RetrievalEvaluationRequest,
    user: User = Depends(require_user),
):
    try:
        return run_retrieval_evaluation(
            request.suite,
            lambda question: retrieve_knowledge(question, user.id)[1],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
