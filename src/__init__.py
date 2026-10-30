"""知识库助手核心包。

这里只做一件事：**保证 src.config 在任何第三方库之前被导入**。

原因：config 会 load_dotenv()，而下面这些变量只有在对应的库首次 import
**之前**生效，之后再设置 os.environ 不会改变已算好的常量：

    HF_HOME           —— huggingface_hub 导入时固定缓存路径
    HF_ENDPOINT       —— 同上，固定镜像地址
    HF_HUB_OFFLINE    —— 离线模式（本地模型已缓存时能省掉几十秒的联网重试）
    HF_HUB_DISABLE_XET —— 关掉会 401 的 Xet 后端

实测后果：如果 langchain/transformers 先被导入，镜像超时重试会让一次
"复用索引"的启动从 1.1 秒变成 126 秒。

因为包的 __init__ 一定先于任何 src.* 子模块执行，放在这里能覆盖所有入口
（run.py / app.py / cli.py / scripts/ / tests），不用在每个模块里记得调顺序。
"""
import src.config  # noqa: F401  （副作用导入：加载 .env + 初始化 UTF-8 控制台）
