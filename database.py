import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(
    mongo_uri,
    tls=True,
    tlsCAFile=certifi.where()
)

db = client["test"]
def fetch_data(enrollmentNo):
    collection = db["students"]   
    record = collection.find_one({"enrollmentNo": enrollmentNo})
    return record


