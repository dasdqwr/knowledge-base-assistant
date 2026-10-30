"""src/md_chunker.py 的单元测试。

覆盖两类：
  A. 边界情况（8 项）—— 每个都是实测中真实出现过的形态
  B. 不变量 —— 对真实 data/ 断言"围栏成对、内容不丢"
"""
from __future__ import annotations

import re

import pytest

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.md_chunker import chunk_markdown, is_fence_line


def chunk(text: str, **kw) -> list[tuple[str, dict]]:
    params = dict(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    params.update(kw)
    return chunk_markdown(text, **params)


def fence_counts(text: str) -> int:
    return sum(1 for line in text.splitlines() if is_fence_line(line))


def assert_balanced(chunks):
    for i, (text, meta) in enumerate(chunks):
        assert fence_counts(text) % 2 == 0, (
            f"第 {i} 块围栏不配对（{fence_counts(text)} 个围栏行），"
            f"meta={meta}, 内容={text[:120]!r}"
        )


def nonws(text: str) -> str:
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# A. 边界情况
# ---------------------------------------------------------------------------
def test_bare_fence():
    """裸围栏（langchain.md / python.md 的真实形态）。"""
    text = "## 标题\n\n正文。\n\n```\nprint(1)\n```\n\n后续正文。\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    assert any("print(1)" in t for t, _ in chunks)


def test_fence_with_language():
    """带语言标识的围栏，语言要进 Code 元数据。"""
    text = "## 标题\n\n```java\nint a = 1;\n```\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    metas = [m for _, m in chunks if m.get("Code")]
    assert metas and metas[0]["Code"] == "java"


def test_list_indented_fence():
    """列表项里的围栏：`- ```java`。

    这是 ExperimentalMarkdownSyntaxTextSplitter 漏掉的形态，
    它的正则 ^``` 要求行首无缩进。
    """
    text = "- 示例：\n  - ```java\n    int a = 1;\n    ```\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    assert any("int a = 1;" in t for t, _ in chunks)


def test_space_indented_fence():
    """3 空格缩进的围栏（CommonMark 合法），实测在你的文件里出现过。"""
    text = "正文。\n\n   ```python\n   x = 1\n   ```\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    assert any("x = 1" in t for t, _ in chunks)


def test_tilde_fence():
    """波浪线围栏。"""
    text = "正文。\n\n~~~\necho hi\n~~~\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    assert any("echo hi" in t for t, _ in chunks)


def test_unclosed_fence_keeps_content():
    """未闭合围栏：剩余内容整体保留，不丢，且打标记。

    ExperimentalMarkdownSyntaxTextSplitter 在这里会返回空串，静默丢掉
    文件剩余所有内容。

    注意：这里**不**断言围栏成对 —— 源文档本身就少一个闭合围栏，
    我们选择"不伪造内容"（不自动补一个 ``` ），而是原样保留并打上
    unclosed_code 标记，让评估报告能看见"源文件有问题"。
    """
    text = "开头正文。\n\n```\nprint('unclosed')\n仍然在代码块里\n最后一行\n"
    chunks = chunk(text)
    joined = "\n".join(t for t, _ in chunks)
    assert "print('unclosed')" in joined
    assert "最后一行" in joined, "未闭合围栏后面的内容被丢掉了"
    assert any(m.get("unclosed_code") for _, m in chunks), "应标记 unclosed_code"
    # 恰好一个块围栏为奇数 —— 就是那个未闭合块，其它块必须成对
    odd = [t for t, _ in chunks if fence_counts(t) % 2 == 1]
    assert len(odd) == 1, f"应只有一个未闭合块，实际 {len(odd)} 个"


def test_hash_inside_code_is_not_header():
    """代码块里的 # 注释不能被当成标题（B4 原型踩过的坑）。"""
    text = (
        "## 真标题\n\n"
        "```python\n"
        "# 启用 LangSmith 监控\n"
        "import os\n"
        "## 这行是注释，不是标题\n"
        "```\n\n"
        "正文。\n"
    )
    chunks = chunk(text)
    assert_balanced(chunks)
    for text_content, meta in chunks:
        if "import os" in text_content or "启用 LangSmith" in text_content:
            assert meta.get("h2") == "真标题", f"元数据被代码里的 # 污染: {meta}"


def test_blank_line_inside_code_is_not_paragraph_boundary():
    """代码块内的空行不能把代码切成两段。"""
    text = "```python\ndef f():\n    a = 1\n\n    return a\n```\n"
    chunks = chunk(text)
    assert_balanced(chunks)
    joined = "\n".join(t for t, _ in chunks)
    assert "def f():" in joined and "return a" in joined
    # 同一块内应同时包含首尾
    assert any("def f():" in t and "return a" in t for t, _ in chunks)


def test_oversized_code_block_stays_whole():
    """超过 chunk_size 的代码块允许超长，但围栏必须成对。"""
    body = "\n".join(f"line_{i} = {i}" for i in range(400))   # 远超 800
    text = "## 大代码\n\n```python\n" + body + "\n```\n"
    chunks = chunk(text, chunk_size=200)
    assert_balanced(chunks)
    huge = [(t, m) for t, m in chunks if len(t) > 200]
    assert huge, "应该产生超长块"
    assert all(fence_counts(t) == 2 for t, _ in huge), "超长代码块应恰好一对围栏"


# ---------------------------------------------------------------------------
# B. 结构行为
# ---------------------------------------------------------------------------
def test_header_metadata_path():
    """标题路径元数据 h1/h2/h3 正确，且低级别标题清空更深级别。"""
    text = (
        "# 第一章\n\n"
        "## 1.1 小节\n\n正文一。\n\n"
        "## 1.2 小节\n\n### 1.2.1 子节\n\n正文二。\n\n"
        "# 第二章\n\n正文三。\n"
    )
    chunks = chunk(text)
    assert_balanced(chunks)
    # 正文二所在块应带 h1=第一章 h2=1.2 小节 h3=1.2.1 子节
    hit = [m for t, m in chunks if "正文二" in t]
    assert hit, "找不到正文二所在块"
    assert hit[0].get("h1") == "第一章"
    assert hit[0].get("h2") == "1.2 小节"
    assert hit[0].get("h3") == "1.2.1 子节"
    # 第二章之后的块不应残留 h2/h3
    hit2 = [m for t, m in chunks if "正文三" in t]
    assert hit2 and hit2[0].get("h1") == "第二章"
    assert "h2" not in hit2[0] and "h3" not in hit2[0]


def test_consecutive_headers_not_tiny():
    """连续标题不应各自生成"只有标题"的碎片块。"""
    text = "# 章\n\n## 节 A\n\n### 子节\n\n正文内容在这里。\n"
    chunks = chunk(text)
    tiny = [t for t, _ in chunks if len(t.strip()) < 20]
    assert not tiny, f"产生了碎片块: {tiny!r}"


def test_horizontal_rule_is_boundary():
    """水平分割线作为块边界，本身不进入块内容。"""
    text = "第一段。\n\n---\n\n第二段。\n"
    chunks = chunk(text)
    for t, _ in chunks:
        assert "---" not in t.strip(), f"分割线混进了块: {t!r}"


def test_max_code_blocks_per_chunk():
    """max_code_blocks_per_chunk=1 时每块最多一个代码块。"""
    text = (
        "## 标题\n\n```\na\n```\n\n```\nb\n```\n\n```\nc\n```\n"
    )
    chunks = chunk(text, max_code_blocks_per_chunk=1)
    for t, _ in chunks:
        assert fence_counts(t) <= 2, f"一块里出现了多个代码块: {t!r}"
    assert_balanced(chunks)


# ---------------------------------------------------------------------------
# C. 对真实数据的不变量
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", ["langchain.md", "python.md"])
def test_real_data_fences_balanced(filename):
    """真实文档切分后，每一块的围栏都必须成对。"""
    from pathlib import Path

    from src.config import DATA_DIR

    path = Path(DATA_DIR) / filename
    if not path.is_file():
        pytest.skip(f"{filename} 不存在")

    text = path.read_text(encoding="utf-8")
    chunks = chunk(text)
    assert chunks
    assert_balanced(chunks)


@pytest.mark.parametrize("filename", ["langchain.md", "python.md"])
def test_real_data_content_conserved(filename):
    """真实文档切分后，非空白字符不应明显丢失。

    允许输出多于输入（overlap 会重复内容），但不允许显著减少。
    """
    from pathlib import Path

    from src.config import DATA_DIR

    path = Path(DATA_DIR) / filename
    if not path.is_file():
        pytest.skip(f"{filename} 不存在")

    text = path.read_text(encoding="utf-8")
    out = "\n".join(t for t, _ in chunk(text))
    src_len = len(nonws(text))
    out_len = len(nonws(out))
    # 分割线、多余空行会被丢弃，给 1% 容差
    assert out_len >= src_len * 0.99, (
        f"{filename} 内容丢失：原文 {src_len} 非空白字符 → 输出 {out_len}"
    )
