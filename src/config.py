from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
MODEL_NAME="deepseek:deepseek-v4-flash"
DB_URL=os.getenv("DB_URL")

#__file__ 是这个文件本身的路径
# config.py 在 src/ 下，parent.parent 就是项目根目录
#运行时推导的路径,不是硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
DATA_DIR = str(PROJECT_ROOT / "data")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 8