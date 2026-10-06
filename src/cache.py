"""
Redis 缓存封装（同步客户端 + 线程池包装）

为什么不用 redis.asyncio：
    Windows + SelectorEventLoop 下，redis.asyncio 首次连接容易超时。
    同步客户端稳定可靠，用 asyncio.to_thread 丢到线程池即可不阻塞事件循环。
"""
import hashlib
import redis
import asyncio
from src.config import REDIS_PASSWORD, REDIS_HOST, REDIS_PORT


# 用同步客户端（你验证过能秒通）
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
)


def make_key(prefix: str, text: str) -> str:
    """
    把任意文本哈希成固定长度的 key。

    为什么要哈希：
        - 问题可能很长，直接当 key 太长
        - 问题可能含特殊字符（换行、空格、中文标点），不适合做 key
        - md5 后固定 32 位十六进制，稳定且安全

    参数：
        prefix：key 前缀，用于分类，如 "retrieval"、"answer"
        text：原始文本，如用户问题

    返回：
        形如 "retrieval:a1b2c3d4..." 的字符串
    """
    h = hashlib.md5(text.encode()).hexdigest()
    return f"{prefix}:{h}"

async def get_cached(key: str) -> str | None:
    """
    读缓存。

    参数：
        key：make_key() 生成的 key

    返回：
        命中：缓存的字符串
        未命中：None
        出错：None（并打印警告）
    """
    try:
        return await asyncio.to_thread(redis_client.get, key)
    except Exception as e:
        print(f"⚠️ Redis 读取失败: {e}")
        return None

async def set_cached(key: str, value: str, ttl: int = 3600) -> bool:
    """
    写缓存。

    参数：
        key：make_key() 生成的 key
        value：要缓存的字符串
        ttl：过期时间（秒），默认 3600（1 小时）

    返回：
        成功：True
        失败：False（并打印警告）

    说明：
        用 setex（set with expire），保证缓存会自动过期，
        避免文档更新后旧缓存永久残留。
    """
    try:
        await asyncio.to_thread(redis_client.setex, key, ttl, value)
        return True
    except Exception as e:
        print(f"⚠️ Redis 写入失败: {e}")
        return False

async def clear_pattern(pattern: str) -> int:
    """
    按前缀清空缓存，返回删除的 key 数量。

    参数：
        pattern：匹配模式，如 "retrieval:*"

    使用场景：
        文档更新、向量库重建后，旧的检索缓存全部失效，
        需要清空 "retrieval:*"，避免返回过时数据。

    说明：
        用 scan_iter 而不是 keys：
            - keys 会阻塞 Redis，数据多时很危险
            - scan_iter 是游标式遍历，不阻塞
    """
    def _clear():
        count = 0
        for key in redis_client.scan_iter(pattern):
            redis_client.delete(key)
            count += 1
        return count

    try:
        return await asyncio.to_thread(_clear)
    except Exception as e:
        print(f"⚠️ Redis 清理失败: {e}")
        return 0

async def close_redis() -> None:
    """
    关闭 Redis 连接。

    使用场景：
        FastAPI 的 lifespan 关闭阶段调用，
        优雅释放连接池资源。
    """
    try:
        await asyncio.to_thread(redis_client.close)
    except Exception:
        # 关闭失败也无所谓，进程马上就退出了
        pass