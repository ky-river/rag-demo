import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "chroma_db"

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
        response = self.client.embeddings.create(
            model="text-embedding-v3",
            input=text.strip(),
        )
        return response.data[0].embedding


embeddings = BailianEmbeddings()

vectorstore = Chroma(
    collection_name="rag_demo",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

question = input("请输入问题：")

docs = vectorstore.similarity_search(question, k=4)

context = "\n\n".join(
    f"来源：第 {doc.metadata.get('page', 0) + 1} 页\n{doc.page_content}"
    for doc in docs
)

prompt = f"""
请根据下面的资料回答问题。
如果资料中没有答案，请明确说“当前资料中没有找到答案”。
不要编造资料。

资料：
{context}

问题：
{question}
"""

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {
            "role": "system",
            "content": "你是一个严谨的企业知识库问答助手。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

print("\n回答：")
print(response.choices[0].message.content)
