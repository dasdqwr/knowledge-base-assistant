"""
控制台编码引导（唯一出口）。

只做一件事：把 sys.stdout / sys.stderr 挂到 UTF-8。

为什么需要它
------------
Windows 中文环境的控制台代码页是 cp936(GBK)，于是：

    sys.stdout.encoding = 'gbk'
    sys.stdout.errors   = 'strict'      ← 无法编码就抛异常
    sys.stderr.encoding = 'gbk'
    sys.stderr.errors   = 'backslashreplace'  ← 转义成 \\u2705，不抛

GBK 没有 emoji（U+1F680 这类字符）。项目里到处是 print("🚀 ...")，
一旦写向 stdout 就抛 UnicodeEncodeError；而这个异常会穿透 FastAPI 的
lifespan，让整个服务启动失败 —— 一条日志语句把进程弄死。

注意 GBK 对汉字是完全够用的，出问题的只有 emoji，所以这个坑很晚才暴露。

怎么保证"够早"
--------------
本模块是 bootstrap 单例。所有入口（run.py / main.py / cli.py /
scripts/*.py）都通过 `src.config` 间接触发它，而 config 必须在任何
会 print 的模块之前被导入：

    cli.py      → src.config 是第一个 src 导入
    main.py     → src.agent → src.cache → src.config
    run.py      → app → src.document / src.cache → src.config
    scripts/*   → src.loader → src.config

所以能力挂在 config 上，而不是散落在各个入口。新增入口只要导入
src.config（几乎必然）就自动获得保护。
"""

import sys


def enable_utf8_console() -> bool:
    """把标准输出/错误流切到 UTF-8。返回是否实际做了改动。

    幂等：重复调用无害。没有真实终端时（pythonw、stdout 被重定向为 None、
    被嵌入到别的宿主里）会静默跳过，而不是抛异常。
    """
    changed = False

    # (流, 错误处理器)
    #   stdout 原本是 strict，所以要显式给一个兜底策略，
    #          避免"没终端/字体不支持"时又是一次硬崩溃。
    #   stderr 原本是 backslashreplace，保持原语义，顺手固化下来。
    for stream, errors in (
        (sys.stdout, "replace"),
        (sys.stderr, "backslashreplace"),
    ):
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # 不是 TextIOWrapper（例如已被换成 StringIO）

        try:
            if getattr(stream, "encoding", "").lower().replace("-", "") == "utf8":
                continue  # 已经是 UTF-8（PYTHONIOENCODING / -X utf8），不用动
            reconfigure(encoding="utf-8", errors=errors)
            changed = True
        except Exception:
            # 保险：任何意外都不能让引导本身变成新的崩溃点
            pass

    return changed
