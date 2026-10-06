"""
PostgreSQL 连通性预检。

为什么不能直接 psycopg.connect：
    如果 5432 端口上有端口转发代理（例如 VMware 的 vmnat.exe 在转发到虚拟机），
    它能完成 TCP 握手，却不转发任何数据，psycopg 会一直静默挂起。
    所以这里先自己做一次 PostgreSQL 协议握手，用超时把"假连通"识别出来。

用法：
    python scripts/check_db.py
"""
import socket
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import DB_URL


def parse_target(url: str):
    u = urlsplit(url)
    return u.hostname or "127.0.0.1", u.port or 5432


def pg_handshake(host: str, port: int, timeout: float = 5.0) -> str:
    """发一个 PostgreSQL SSLRequest，看对端是否按协议回包。

    返回 'ok' / 'ssl' / 'not_pg' / 'timeout' / 'refused' / 具体错误
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
    except ConnectionRefusedError:
        return "refused"
    except socket.timeout:
        return "timeout"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)

    try:
        # SSLRequest：长度 8 + 魔数 80877103
        s.sendall(b"\x00\x00\x00\x08\x04\xd2\x16\x2f")
        data = s.recv(64)
        if data == b"S":
            return "ssl"
        if data == b"N":
            return "ok"
        if not data:
            return "not_pg"
        return "unexpected: " + repr(data[:32])
    except socket.timeout:
        return "timeout"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)
    finally:
        s.close()


def main() -> int:
    if not DB_URL:
        print("❌ DB_URL 未配置，检查 .env")
        return 1

    host, port = parse_target(DB_URL)
    print(f"目标: {host}:{port}")

    verdict = pg_handshake(host, port)
    print(f"PostgreSQL 协议握手: {verdict}")

    if verdict in ("ok", "ssl"):
        import psycopg

        conninfo = DB_URL.replace("postgresql+psycopg", "postgresql")
        try:
            with psycopg.connect(conninfo, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    print("✅ 连接成功:", cur.fetchone()[0][:80])
                    cur.execute("SELECT current_database(), current_user;")
                    print("✅ 库/用户:", cur.fetchone())
            return 0
        except Exception as e:
            print(f"❌ 协议层通了但登录失败: {type(e).__name__}: {e}")
            return 1

    if verdict == "timeout":
        print("❌ 端口接受 TCP 连接，但对端不回应 PostgreSQL 协议。")
        print("   常见原因：该端口是虚拟机端口转发（VMware vmnat.exe / VirtualBox），")
        print("   而虚拟机没开、或虚拟机里的 PostgreSQL 没启动。")
        return 1

    if verdict == "not_pg":
        print("❌ 对端立刻关闭了连接：5432 上没有 PostgreSQL 在服务。")
        return 1

    if verdict == "refused":
        print("❌ 连接被拒绝：没有进程监听该端口，PostgreSQL 未启动。")
        return 1

    print("❌ 意外的响应，端口上可能不是 PostgreSQL。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
