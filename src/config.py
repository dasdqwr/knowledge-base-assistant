from pathlib import Path
import os

# ========== 必须放在最前面 ==========
# 把控制台切到 UTF-8。config 是所有模块的公共依赖，也是每个入口最早被导入的
# src 模块，所以把引导挂在这里，一处生效、全项目覆盖（含 scripts/）。
# 详见 src/utf8.py 里的说明：GBK 控制台下 print emoji 会抛异常并弄死服务。
from src.utf8 import enable_utf8_console

enable_utf8_console()

from dotenv import load_dotenv

load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
MODEL_NAME="deepseek:deepseek-v4-flash"
DB_URL=os.getenv("DB_URL")

# 嵌入模型名。放在这里是为了让"实际用的模型"和"索引签名里记的模型"共用同一个真相，
# 避免改了 model.py 却忘了改索引新鲜度判断，从而静默复用错误的向量库。
EMBEDDING_MODEL = "BAAI/bge-m3"

#__file__ 是这个文件本身的路径
# config.py 在 src/ 下，parent.parent 就是项目根目录
#运行时推导的路径,不是硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
DATA_DIR = str(PROJECT_ROOT / "data")

# 索引清单：它的存在 = "向量库完整且与当前 data/ 一致"。
# 详见 src/index_meta.py 里的"清单写入时机"说明。
MANIFEST_PATH = str(PROJECT_ROOT / "index_manifest.json")

# 支持索引的文档后缀。集中放这里，避免 loader / document / cli / streamlit 各写一份而漂移
# （曾经 document.py 认 .markdown 但 cli 不认，导致"列表里有、上传却被拒"）。
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".docx")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 8

# 单路召回的候选数（BM25 与向量各取这么多，再融合）
RETRIEVAL_K = 5
# 混合检索权重 [BM25, 向量]。BM25 权重高，因为本项目要解决的是
# "精确 API 名搜不到"这个问题（见 README 技术点 1）。
ENSEMBLE_WEIGHTS = [0.8, 0.2]
# 进入 Rerank 的候选数量
RERANK_CANDIDATES = 10