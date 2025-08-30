# app/db.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "aiagent")
COL_MESSAGES = os.getenv("MESSAGES_COLLECTION", "messages")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
messages_col = db[COL_MESSAGES]
