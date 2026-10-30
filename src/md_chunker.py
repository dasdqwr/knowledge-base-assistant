"""Markdown 切分器：保证代码围栏永不被切断。

为什么需要自己写
----------------
LangChain 现有的两个 markdown 切分器都不够用（都实测过）：

  MarkdownHeaderTextSplitter
      围栏感知是安全的，但不对超长块做处理；一旦把超长块交给
      RecursiveCharacterTextSplitter（它不认识围栏），代码块就从中间被劈开。
      实测 227 块里 41 块（18.1%）围栏不配对。

  ExperimentalMarkdownSyntaxTextSplitter
      思路很好（代码块单独成块 + "Code" 元数据），但实测有三个问题：
        · 围栏正则锚定行首（^```），缩进的 ``` 识别不了 → 仍有 10 个坏块
        · 未闭合围栏时 _resolve_code_chunk 返回空串，静默丢掉文件剩余内容
        · 没有任何尺寸聚合 → 539 块里有 175 个碎片块
      所以借它的思路，不用它的实现。

本模块的做法
------------
把文档看作有序的「原子」序列，按规则装箱：

    header  标题行（h1-h3）。若当前块已有正文/代码则封口；
            否则继续累积 —— 连续标题不会各自生成碎片块。
    code    围栏代码块（围栏行 + 内容 + 闭合围栏），整体不可切。
            加入后立即封口，保证每块最多 MAX_CODE_BLOCKS_PER_CHUNK 个代码块。
    prose   正文段落（按空行切分）。只有正文段落才允许交给递归切分器。
    hr      水平分割线（---/***/___）。作为块边界，本身不进入块。

关键性质（由构造保证，不靠调参）
--------------------------------
    1. 切分器从不「看见」围栏内部的文本 → 输出块的围栏必然成对
    2. 未闭合围栏 → 剩余内容整体当作代码块，绝不丢内容
    3. 正文段落才进入递归切分器，而正文里没有围栏，所以安全

围栏识别比 CommonMark 更宽松
---------------------------
除了标准的 0-3 空格缩进，还容忍列表标记（`- ```java`）和更深的缩进。
理由：宁可把缩进代码块误判成围栏（后果是块变大），也不能漏判围栏
（后果是块被切坏）。文档是给人读的笔记，宽松识别更稳。
"""
from __future__ import annotations

import re
from typing import Iterator

# ---------------------------------------------------------------------------
# 行级识别
# ---------------------------------------------------------------------------
# 围栏：可选缩进 + 可选列表标记 + 3 个以上 ` 或 ~ + 可选 info string
FENCE_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:(?:[-*+]|\d+[.)])\s+)?"
    r"(?P<fence>`{3,}|~{3,})"
    r"(?P<info>[^`~]*)$"
)

# 标题：1-6 级，要求 # 后面有空格（避免把 #话题 当成标题）
HEADER_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.*)$")

# 水平分割线：整行只有同一种 -、* 或 _，至少 3 个
HR_RE = re.compile(r"^[ \t]{0,3}(?P<ch>[-*_])[ \t]*(?:(?P=ch)[ \t]*){2,}$")

# 空行
BLANK_RE = re.compile(r"^[ \t]*$")

# 默认进入 headers_to_split_on 的级别
DEFAULT_HEADER_LEVELS = (1, 2, 3)


def match_fence(line: str) -> re.Match | None:
    return FENCE_LINE_RE.match(line)


def is_hr(line: str) -> bool:
    return bool(HR_RE.match(line))


def is_fence_line(line: str) -> bool:
    """供测试与外部校验使用：这一行是不是围栏行。"""
    return bool(FENCE_LINE_RE.match(line))


# ---------------------------------------------------------------------------
# 原子扫描
# ---------------------------------------------------------------------------
# 原子类型
HEADER = "header"
CODE = "code"
PROSE = "prose"
HR = "hr"


def scan_atoms(
    lines: list[str],
    header_levels: tuple[int, ...] = DEFAULT_HEADER_LEVELS,
) -> Iterator[tuple]:
    """把行序列扫描成原子序列。

    产出：
        (HEADER, level, text)
        (CODE, block_text, language, closed)
        (PROSE, paragraph_text)
        (HR,)
    """
    buf: list[str] = []

    def flush_prose():
        if not buf:
            return
        raw = "\n".join(buf)
        buf.clear()
        # 按空行切段；段内保留原有换行（不破坏列表结构）
        for para in re.split(r"\n[ \t]*\n", raw):
            text = para.strip("\n")
            if text.strip():
                yield (PROSE, text)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = match_fence(line)
        if m:
            for atom in flush_prose():
                yield atom
            fence_char = m.group("fence")[0]
            fence_len = len(m.group("fence"))
            language = m.group("info").strip()
            start = i
            i += 1
            closed = False
            while i < n:
                m2 = match_fence(lines[i])
                if (
                    m2
                    and m2.group("fence")[0] == fence_char
                    and len(m2.group("fence")) >= fence_len
                ):
                    i += 1
                    closed = True
                    break
                i += 1
            yield (CODE, "\n".join(lines[start:i]), language, closed)
            continue

        hm = HEADER_RE.match(line)
        if hm and len(hm.group("hashes")) in header_levels:
            for atom in flush_prose():
                yield atom
            yield (HEADER, len(hm.group("hashes")), hm.group("text").strip())
            i += 1
            continue

        if HR_RE.match(line):
            for atom in flush_prose():
                yield atom
            yield (HR,)
            i += 1
            continue

        buf.append(line)
        i += 1

    for atom in flush_prose():
        yield atom


# ---------------------------------------------------------------------------
# 装箱
# ---------------------------------------------------------------------------
def _default_prose_splitter(chunk_size: int, overlap: int):
    """正文专用切分器。正文里没有围栏，所以用朴素分隔符是安全的。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    )


def _header_meta(path: dict[int, str]) -> dict:
    return {f"h{level}": text for level, text in sorted(path.items())}


def chunk_markdown(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_code_blocks_per_chunk: int = 1,
    flush_after_code: bool = False,
    header_levels: tuple[int, ...] = DEFAULT_HEADER_LEVELS,
    prose_splitter=None,
) -> list[tuple[str, dict]]:
    """切分 markdown。返回 [(块内容, 元数据)]。

    元数据键：h1/h2/h3（标题路径）、Code（代码块语言，仅代码块）、
    unclosed_code（True 表示该块内含未闭合围栏，用于排障）。

    flush_after_code
        False（默认，懒封口）：代码块之后允许紧跟的正文并入同一块。
            代码块和它下面那句解释通常是一体的，实测这样能少 60 个块、
            并消掉大部分"过小碎片块"。
        True：代码块后立即封口，块更纯但更容易产生碎片。
        两种模式下每块最多仍只有 max_code_blocks_per_chunk 个代码块 ——
        因为下一个代码块到达时会在开头触发 flush。
    """
    if prose_splitter is None:
        prose_splitter = _default_prose_splitter(chunk_size, overlap)

    lines = text.splitlines()
    atoms = scan_atoms(lines, header_levels)

    out: list[tuple[str, dict]] = []

    cur_parts: list[str] = []
    cur_len = 0
    cur_has_content = False      # 已有正文或代码
    cur_code_count = 0
    cur_meta: dict = {}

    path: dict[int, str] = {}    # 当前标题路径

    def reset():
        nonlocal cur_parts, cur_len, cur_has_content, cur_code_count, cur_meta
        cur_parts = []
        cur_len = 0
        cur_has_content = False
        cur_code_count = 0
        cur_meta = {}

    def start_if_empty():
        nonlocal cur_meta
        if not cur_parts:
            cur_meta = _header_meta(path)

    def emit(content: str, meta: dict | None = None):
        body = content.strip("\n")
        if body.strip():
            out.append((body, dict(meta if meta is not None else cur_meta)))

    def flush():
        if cur_parts:
            emit("\n".join(cur_parts), cur_meta)
        reset()

    def push(piece: str, *, content: bool = True):
        nonlocal cur_len, cur_has_content
        if cur_parts:
            cur_len += 1
        cur_parts.append(piece)
        cur_len += len(piece)
        if content:
            cur_has_content = True

    for atom in atoms:
        kind = atom[0]

        if kind == HEADER:
            _, level, htext = atom
            if cur_has_content:
                flush()
            # 低级别标题清空更深的级别（与 Markdown 语义一致）
            path = {k: v for k, v in path.items() if k < level}
            path[level] = htext
            start_if_empty()
            cur_meta = _header_meta(path)
            push("#" * level + " " + htext, content=False)
            continue

        if kind == HR:
            flush()
            continue

        if kind == CODE:
            _, block, language, closed = atom
            if cur_code_count >= max_code_blocks_per_chunk:
                flush()

            if len(block) > chunk_size:
                # 大代码块不可切。
                # 若当前块还没有正文（只有标题，或完全是空的），就让标题跟着它走 ——
                # 绝不为了压尺寸而单独吐出一个"只有标题"的碎片块
                # （标题本身只有几十字符，合并带来的超长可以忽略）。
                if not cur_has_content:
                    start_if_empty()
                    push(block)
                    cur_code_count += 1
                    if language:
                        cur_meta["Code"] = language
                    if not closed:
                        cur_meta["unclosed_code"] = True
                    flush()
                    continue
                # 当前块已有正文：先把它封口（不能把超大代码塞进去），
                # 再让代码块单独成块
                flush()
                start_if_empty()
                meta = dict(cur_meta)
                if language:
                    meta["Code"] = language
                if not closed:
                    meta["unclosed_code"] = True
                emit(block, meta)
                reset()
                continue

            if cur_parts and cur_len + 1 + len(block) > chunk_size:
                flush()
            start_if_empty()
            push(block)
            cur_code_count += 1
            if language:
                cur_meta["Code"] = language
            if not closed:
                cur_meta["unclosed_code"] = True
            # 默认懒封口：不在这里 flush，让紧跟的正文并入同一块。
            # 下一个代码块到达时，开头的 cur_code_count 检查会先 flush。
            if flush_after_code and cur_code_count >= max_code_blocks_per_chunk:
                flush()
            continue

        # PROSE
        para = atom[1]
        if len(para) > chunk_size:
            # 拆长段落时不要把已累积的标题孤立成单独一块：
            # 只含标题时，把标题接到第一片前面。
            pending = ""
            if cur_parts and not cur_has_content:
                pending = "\n".join(cur_parts) + "\n"
                meta_for_pieces = dict(cur_meta)
                reset()
            else:
                flush()
                meta_for_pieces = _header_meta(path)
            pieces = prose_splitter.split_text(para)
            if not pieces:
                continue
            for idx, piece in enumerate(pieces):
                body = piece if idx else (pending + piece)
                emit(body, meta_for_pieces)
            continue

        if cur_parts and cur_len + 1 + len(para) > chunk_size:
            flush()
        start_if_empty()
        push(para)

    flush()

    # 兜底：极端情况下（例如全是分割线）保证至少返回原文
    if not out and text.strip():
        out.append((text.strip("\n"), {}))

    return out
