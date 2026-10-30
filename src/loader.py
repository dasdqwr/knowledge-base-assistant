from pathlib import Path

# 注意：src.config 必须在任何 huggingface / torch 相关导入之前。
# config 会 load_dotenv()，而 HF_HOME / HF_ENDPOINT 这类变量只有在下游库
# 首次 import 之前生效 —— huggingface_hub 在导入时就把缓存路径算成常量了，
# 之后再设置 os.environ 不会改变它，模型会照样下到旧位置（C 盘）。
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    FLUSH_AFTER_CODE,
    MAX_CODE_BLOCKS_PER_CHUNK,
    SUPPORTED_SUFFIXES,
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.md_chunker import chunk_markdown


def load_documents():
    """加载 data/ 目录下所有文档"""
    docs = []
    for file in Path(DATA_DIR).iterdir():
        if file.is_dir():
            continue
        # 跳过点文件（.gitkeep 之类），否则会白跑一次加载并打印"加载失败"
        if file.name.startswith("."):
            continue
        # 后缀统一小写比较，否则 .PDF 会被静默跳过
        suffix = file.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        try:
            if suffix == '.pdf':
                # 转成字符串传给加载器
                docs.extend(PyPDFLoader(str(file)).load())
                print(f"✅ 加载 PDF: {file.name}")
            elif suffix in ('.txt', '.md', '.markdown'):
                docs.extend(TextLoader(str(file), encoding="utf-8").load())
                print(f"✅ 加载文本: {file.name}")
            elif suffix == ".docx":
                docs.extend(Docx2txtLoader(str(file)).load())
                print(f"✅ 加载 Word: {file.name}")
        except Exception as e:
            print(f"❌ 加载失败 {file.name}: {e}")
    return docs

# ========== 切分器 ==========

def _get_recursive_splitter():
    """通用递归切分器（支持中文）"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    )


def _split_markdown(doc, fallback_splitter=None):
    """Markdown 专用切分：代码块原子化，保证围栏永不被切断。

    实现在 src/md_chunker.py。要点：
      · 围栏代码块整体不可切，加入后立即封口（每块最多 N 个代码块）
      · 只有正文段落才会交给递归切分器，而正文里没有围栏，所以安全
      · 未闭合围栏按代码整段保留，不丢内容

    fallback_splitter 参数保留以兼容旧调用方，实际由 chunk_markdown 内部
    用同样的分隔符构造正文切分器。
    """
    pieces = chunk_markdown(
        doc.page_content,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        max_code_blocks_per_chunk=MAX_CODE_BLOCKS_PER_CHUNK,
        flush_after_code=FLUSH_AFTER_CODE,
        prose_splitter=fallback_splitter or _get_recursive_splitter(),
    )
    return [
        Document(page_content=text, metadata={**doc.metadata, **meta})
        for text, meta in pieces
    ]


def _split_code(doc, language):
    """代码专用切分：按语法结构切"""
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = code_splitter.split_documents([doc])
    return chunks

# 后缀 → 编程语言映射
CODE_LANGUAGES = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".html": Language.HTML,
}

def split_documents(docs):
    """切分文档"""
    if not docs:
        return []
    fallback = _get_recursive_splitter()
    all_chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        suffix = Path(source).suffix.lower()

        if suffix in (".md", ".markdown"):
            chunks = _split_markdown(doc, fallback)
        elif suffix in CODE_LANGUAGES:
            chunks = _split_code(doc, CODE_LANGUAGES[suffix])
        else:
            chunks = fallback.split_documents([doc])

        all_chunks.extend(chunks)
    print(f"📄 切分为 {len(all_chunks)} 个块")
    return all_chunks

