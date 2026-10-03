import asyncio
import redis.asyncio as redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

async def main():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=5,   # 加超时，避免一直卡
    )
    await r.set("test_async", "hello")
    result = await r.get("test_async")
    print(f"结果: {result}")
    await r.aclose()


asyncio.run(main())