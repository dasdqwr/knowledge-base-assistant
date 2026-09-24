import psycopg
from src.config import DB_URL

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        print("连接成功:", cur.fetchone()[0])