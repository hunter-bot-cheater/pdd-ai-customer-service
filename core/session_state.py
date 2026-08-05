"""
会话状态管理器

单例模式，用字典存储会话与过期时间戳，
用于记录"已转人工"状态，防止转人工后 AI 继续抢答。

session_key 格式：{shop_id}:{from_uid}
默认有效期 4 小时（14400 秒）。
"""
import time
import threading
from typing import Optional


class SessionState:
    """会话状态管理器（线程安全单例）"""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._handoffs = {}
                    obj._data_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    def mark_handoff(self, session_key: str, ttl_seconds: int = 14400) -> None:
        """标记会话为已转人工，默认有效 4 小时"""
        with self._data_lock:
            self._handoffs[session_key] = time.time() + ttl_seconds

    def is_handoff(self, session_key: str) -> bool:
        """判断会话是否处于转人工有效期内，过期自动清除"""
        with self._data_lock:
            expiry = self._handoffs.get(session_key)
            if expiry is None:
                return False
            if time.time() > expiry:
                del self._handoffs[session_key]
                return False
            return True

    def clear_handoff(self, session_key: str) -> None:
        """手动清除转人工标记"""
        with self._data_lock:
            self._handoffs.pop(session_key, None)


# 全局单例实例（与 SessionState() 返回同一对象）
session_state = SessionState()
