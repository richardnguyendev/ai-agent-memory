# app/agent.py
import os
from dotenv import load_dotenv
from .memory import query_memory, add_memory, is_duplicate
from .db import messages_col
import time

load_dotenv()

# Try using LangChain Ollama wrapper; if not available, fallback to simple transform call
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except Exception as e:
    print("Lỗi import Ollama:", e)
    OLLAMA_AVAILABLE = False
print("OLLAMA_AVAILABLE =", OLLAMA_AVAILABLE)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")

if OLLAMA_AVAILABLE:
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
else:
    llm = None  # fallback below

def build_prompt(question, retrieved_docs):
    header = "Bạn là trợ lý lập trình chuyên nghiệp. Dưới đây là các đoạn liên quan từ lịch sử người dùng:\n\n"
    context = ""
    for i, d in enumerate(retrieved_docs):
        context += f"[Context {i+1}]\n{d['document']}\n\n"
    prompt = f"{header}{context} Hỏi: {question}\n\nTrả lời ngắn gọn, rõ ràng, kèm ví dụ nếu cần."
    return prompt

def call_llm(prompt):
    print("Gọi call_llm, OLLAMA_AVAILABLE =", OLLAMA_AVAILABLE)
    if OLLAMA_AVAILABLE:
        try:
            result = llm(prompt)
            print("Kết quả từ Ollama:", result)
            return result
        except Exception as e:
            print("Lỗi khi gọi Ollama:", e)
            return "Lỗi khi gọi Ollama: " + str(e)
        # LangChain Ollama returns text on __call__
        # return llm(prompt)
    else:
        # simple fallback: echo prompt (for dev) — replace with another LLM or OpenAI if you want
        return "LLM offline (Ollama) chưa được cài/khởi chạy. Thay vào đó bạn cần bật Ollama hoặc cấu hình LLM khác."

def handle_question(user_id, question):
    # 1) semantic retrieval
    retrieved = query_memory(question, k=5)

    # 2) build prompt with retrieved context
    prompt = build_prompt(question, retrieved)

    # 3) call LLM
    answer = call_llm(prompt)

    # 4) store QA in MongoDB (raw)
    messages_col.insert_one({
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "timestamp": time.time(),
        "retrieved_count": len(retrieved)
    })

    # 5) dedupe then add memory (store Q+A as a single doc)
    qa_text = f"Q: {question}\nA: {answer}"
    dup, dup_id, sim = is_duplicate(qa_text, top_k=3)
    if not dup:
        add_memory(qa_text, metadata={"user_id": user_id, "source": "chat"})
    else:
        # optional: update metadata to record frequency
        messages_col.update_one({"_id": messages_col.find_one({"_id": messages_col.find_one({"user_id": user_id})})}, {"$set": {"duplicate_of": dup_id, "similarity": sim}})
    return answer
