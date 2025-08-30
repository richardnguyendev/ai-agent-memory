# app/main.py
from fastapi import FastAPI
from .models import AskRequest
from .agent import handle_question

app = FastAPI(title="AI Agent Memory")

@app.post("/ask")
async def ask(req: AskRequest):
    answer = handle_question(req.user_id, req.question)
    return {"answer": answer}
