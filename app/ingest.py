from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
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
            raise ValueError("发现空文本")
        response = self.client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return response.data[0].embedding

load_dotenv(ROOT / ".env", override=True)

pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("data 文件夹里没有 PDF 文件")

documents = []

for pdf_file in pdf_files:
    print(f"正在读取: {pdf_file.name}")
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)
chunks = [
    chunk for chunk in chunks
    if chunk.page_content and chunk.page_content.strip()
]
print([repr(chunk.page_content[:50]) for chunk in chunks])
print(f"原始页面: {len(documents)}")
print(f"切分片段: {len(chunks)}")

if DB_DIR.exists():
    shutil.rmtree(DB_DIR)
    print("已清空旧向量库")

embeddings = BailianEmbeddings()

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
    collection_name="rag_demo",
)

print(f"向量库已保存到: {DB_DIR}")