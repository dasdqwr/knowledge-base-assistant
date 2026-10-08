from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_SUFFIXES


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


def _get_markdown_splitter():
    """Markdown 标题切分器"""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,   # 保留标题在内容里
    )

def _split_markdown(doc, fallback_splitter):
    """Markdown 专用切分：先按标题切，再对过大的块递归切"""
    md_splitter = _get_markdown_splitter()
    md_chunks = md_splitter.split_text(doc.page_content)

    result = []
    for chunk in md_chunks:
        chunk.metadata.update(doc.metadata)   # 保留 source
        #Markdown 按标题切完后，某节可能还是太长，这时用递归切分器再切
        if len(chunk.page_content) > CHUNK_SIZE:
            result.extend(fallback_splitter.split_documents([chunk]))
        else:
            result.append(chunk)
    return result


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

