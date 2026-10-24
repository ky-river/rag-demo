from __future__ import annotations

import os
import json
import logging
import math
import re
import sqlite3
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from typing import Literal, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
SESSION_DB = ROOT / "sessions.sqlite3"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env", override=True)


class BailianEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
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


class StudyPlanRequest(BaseModel):
    gaps: list[str]


class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None


class ClearSessionRequest(BaseModel):
    session_id: str


class AgentGraphState(TypedDict, total=False):
    message: str
    intent: str
    tool: str
    tool_arguments: dict[str, str]
    routing: str
    history: list[dict[str, str]]
    answer: str
    sources: list[dict]


embeddings = BailianEmbeddings()
vectorstore = Chroma(
    collection_name="rag_demo",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)
bm25_index: BM25Okapi | None = None
bm25_documents: list[Document] = []
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
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
APP_VERSION = os.getenv("APP_VERSION", "1.3")
RERANK_CACHE_DIR = Path(
    os.getenv("RERANK_CACHE_DIR", str(ROOT / ".cache" / "fastembed"))
)
if not RERANK_CACHE_DIR.is_absolute():
    RERANK_CACHE_DIR = ROOT / RERANK_CACHE_DIR
CHINESE_STOP_BIGRAMS = {
    "这个", "项目", "什么", "哪些", "怎么", "如何", "是否",
    "可以", "多少", "一个", "的是", "当前", "系统",
}
logger = logging.getLogger(__name__)
reranker = None
reranker_error: str | None = None


def init_session_db() -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "session_id TEXT PRIMARY KEY, messages_json TEXT NOT NULL, "
            "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )


def load_history(session_id: str) -> list[dict[str, str]]:
    with closing(sqlite3.connect(SESSION_DB)) as connection:
        row = connection.execute(
            "SELECT messages_json FROM conversations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return []
    try:
        history = json.loads(row[0])
        return history if isinstance(history, list) else []
    except json.JSONDecodeError:
        return []


def save_history(session_id: str, history: list[dict[str, str]]) -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        connection.execute(
            "INSERT INTO conversations(session_id, messages_json) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "messages_json = excluded.messages_json, updated_at = CURRENT_TIMESTAMP",
            (session_id, json.dumps(history[-10:], ensure_ascii=False)),
        )


def clear_history(session_id: str) -> None:
    with closing(sqlite3.connect(SESSION_DB)) as connection, connection:
        connection.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))


init_session_db()

app = FastAPI(title="RAG Knowledge Base API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def load_documents_from_store() -> list[Document]:
    stored = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            stored.get("documents", []),
            stored.get("metadatas", []),
        )
        if content and content.strip()
    ]


def refresh_bm25_index(documents: list[Document] | None = None) -> None:
    global bm25_documents, bm25_index

    bm25_documents = documents if documents is not None else load_documents_from_store()
    tokenized_documents = [
        tokenize_for_bm25(document.page_content)
        for document in bm25_documents
    ]
    bm25_index = BM25Okapi(tokenized_documents) if tokenized_documents else None


def build_index() -> int:
    """从 data 中全部 PDF 重建索引，避免同名文件重复累积。"""
    global vectorstore

    documents = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        pdf_documents = PyPDFLoader(str(pdf_path)).load()
        for document in pdf_documents:
            document.metadata["filename"] = pdf_path.name
        documents.extend(pdf_documents)

    chunks = splitter.split_documents(documents)
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

    # 清空 Chroma 集合，不直接删除正在使用的 SQLite 文件。
    vectorstore.delete_collection()
    vectorstore = Chroma(
        collection_name="rag_demo",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    if chunks:
        vectorstore.add_documents(chunks)
    refresh_bm25_index(chunks)
    return len(chunks)


refresh_bm25_index()


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


def health_status() -> dict:
    chunk_count = len(bm25_documents)
    document_count = len(
        {
            document.metadata.get("filename")
            for document in bm25_documents
            if document.metadata.get("filename")
        }
    )
    collection_error = None
    try:
        collection_count = vectorstore._collection.count()
    except Exception as exc:
        collection_count = None
        collection_error = str(exc)

    index_ready = (
        collection_error is None
        and chunk_count > 0
        and bm25_index is not None
        and collection_count == chunk_count
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
        "document_count": document_count,
        "chunk_count": chunk_count,
        "collection_count": collection_count,
        "collection_error": collection_error,
        "reranking": reranking_status(),
    }


@app.get("/")
def home():
    return {"message": "RAG API is running", "version": APP_VERSION}


@app.get("/health")
def health():
    result = health_status()
    return JSONResponse(
        status_code=200 if result["index_ready"] else 503,
        content=result,
    )


def document_key(document: Document) -> tuple:
    return (
        document.metadata.get("filename", ""),
        document.metadata.get("page", 0),
        document.page_content,
    )


def retrieve_knowledge(question: str):
    raw_vector_results = vectorstore.similarity_search_with_score(
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
    if bm25_index is not None:
        query_tokens = tokenize_for_bm25(question)
        if query_tokens:
            scores = bm25_index.get_scores(query_tokens)
            lexical_order = sorted(
                range(len(bm25_documents)),
                key=lambda index: scores[index],
                reverse=True,
            )
            for rank, index in enumerate(lexical_order, 1):
                score = float(scores[index])
                if score <= 0:
                    break
                document = bm25_documents[index]
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


def answer_from_knowledge_base(question: str, history=None):
    docs, sources = retrieve_knowledge(question)
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
    response = client.chat.completions.create(
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
    response = client.chat.completions.create(
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
    response = client.chat.completions.create(
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


def execute_agent_tool(name: str, arguments: dict, history=None) -> dict:
    if name == "knowledge_search":
        return answer_from_knowledge_base(arguments["question"], history)
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
    )
    return {"answer": result["answer"], "sources": result["sources"]}


def graph_study_plan(state: AgentGraphState) -> AgentGraphState:
    result = execute_agent_tool(
        "create_study_plan",
        state.get("tool_arguments", {"topic": state["message"]}),
        state.get("history", []),
    )
    return {"answer": result["answer"], "sources": result["sources"]}


def graph_resume_match(state: AgentGraphState) -> AgentGraphState:
    result = execute_agent_tool(
        "resume_match_guidance",
        state.get("tool_arguments", {}),
        state.get("history", []),
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


def run_agent_graph(request: AgentRequest):
    message = request.message.strip()
    if not message:
        return {"error": "请输入任务"}

    session_id = request.session_id or str(uuid4())
    history = load_history(session_id)[-10:]
    result = agent_graph.invoke({
        "message": message,
        "history": history,
    })
    answer = result.get("answer", "")
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    save_history(session_id, history)
    return {
        "session_id": session_id,
        "intent": result.get("intent"),
        "tool": result.get("tool"),
        "routing": result.get("routing"),
        "answer": answer,
        "sources": result.get("sources", []),
    }


@app.post("/agent-graph")
def agent_graph_endpoint(request: AgentRequest):
    return run_agent_graph(request)


@app.post("/agent-tools", deprecated=True)
def agent_tools(request: AgentRequest):
    """Compatibility alias for clients using the previous endpoint."""
    return run_agent_graph(request)


@app.post("/agent", deprecated=True)
def agent(request: AgentRequest):
    message = request.message.strip()
    if not message:
        return {"error": "请输入任务"}

    session_id = request.session_id or str(uuid4())
    history = load_history(session_id)[-10:]
    intent = detect_intent(message)
    if intent == "resume_match":
        reply = "简历与岗位匹配需要两个 PDF 文件，请使用下方的简历和岗位 JD 上传区域。"
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ])
        save_history(session_id, history)
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
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
        )
        answer = response.choices[0].message.content
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        save_history(session_id, history)
        return {"session_id": session_id, "intent": intent, "answer": answer, "sources": []}

    result = answer_from_knowledge_base(message, history)
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]},
    ])
    save_history(session_id, history)
    return {"session_id": session_id, "intent": intent, **result}


@app.post("/agent/clear")
def clear_agent_session(request: ClearSessionRequest):
    clear_history(request.session_id)
    return {"message": "会话已清空", "session_id": request.session_id}


def read_uploaded_pdf(upload: UploadFile, temp_dir: Path) -> str:
    filename = Path(upload.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("简历和岗位 JD 都必须是 PDF 文件")

    pdf_path = temp_dir / filename
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

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

    response = client.chat.completions.create(
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
def study_plan(request: StudyPlanRequest):
    gaps = [gap.strip() for gap in request.gaps if gap.strip()]
    if not gaps:
        return {"error": "没有可用于制定计划的技能缺口"}

    response = client.chat.completions.create(
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
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        return {"error": "只支持 PDF 文件"}

    pdf_path = DATA_DIR / filename
    total_bytes = 0
    try:
        with pdf_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    pdf_path.unlink(missing_ok=True)
                    return {"error": "PDF 文件不能超过 10MB"}
                buffer.write(chunk)

        if total_bytes == 0:
            pdf_path.unlink(missing_ok=True)
            return {"error": "不能上传空文件"}

        # 提前解析一次，给损坏 PDF 或纯图片 PDF 返回清晰提示。
        documents = PyPDFLoader(str(pdf_path)).load()
        has_text = any(
            document.page_content and document.page_content.strip()
            for document in documents
        )
        if not has_text:
            pdf_path.unlink(missing_ok=True)
            return {"error": "PDF 中没有可读取的文字内容"}
    except Exception:
        pdf_path.unlink(missing_ok=True)
        return {"error": "PDF 文件损坏或无法解析"}

    try:
        chunk_count = build_index()
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        return {"error": f"PDF 入库失败：{exc}"}

    return {
        "message": f"PDF 上传并入库成功，共新增 {chunk_count} 个文本片段",
        "filename": filename,
        "chunks": chunk_count,
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return {"error": "问题不能为空"}
    return {"question": question, **answer_from_knowledge_base(question)}


@app.post("/search")
def search(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return {"error": "问题不能为空"}

    started_at = time.perf_counter()
    _, sources = retrieve_knowledge(question)
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
