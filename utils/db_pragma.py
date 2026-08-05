"""SQLite 引擎 PRAGMA 配置。

集中管理 SQLite 连接级 PRAGMA，缓解多账号并发读写同一数据库文件时的争用：
- journal_mode=WAL：读写不互锁，并发写不再立即 SQLITE_BUSY
- synchronous=NORMAL：WAL 下安全的性能折中
- busy_timeout=30000：遇锁时等待最多 30s 再报错，而非立即失败
"""
from sqlalchemy import event
from sqlalchemy.engine import Engine


def setup_sqlite_pragmas(engine: Engine) -> None:
    """为 SQLite 引擎注册连接级 PRAGMA。

    每次 DBAPI 连接建立时执行一次。对非 SQLite 引擎无副作用（仅在 connect
    事件中执行 PRAGMA 语句，其他驱动会忽略或报错——本项目的引擎均为 SQLite）。
    """
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
