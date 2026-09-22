from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
MODEL_NAME="deepseek:deepseek-v4-flash"
CHROMA_DIR = "./chroma_db"
DATA_DIR = "./data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3