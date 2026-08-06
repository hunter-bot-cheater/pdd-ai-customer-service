"""
转人工后新消息通知跟踪器

规则：会话转人工后，买家后续发送的新消息仍需要通知企业微信人工客服，
但同一会话 5 分钟内最多通知一次，避免消息轰炸刷屏。

冷却时长可通过配置 notification.handoff_cooldown_seconds 调整（默认 300 秒）。
"""
import threading
import time
from typing import Dict

from config import get_config


class NotifyTracker:
    """转人工通知冷却跟踪器（线程安全）"""

    def __init__(self, cooldown_seconds: int = None):
        # 读取配置，默认 300 秒；测试中可显式传 0 关闭冷却
        if cooldown_seconds is None:
            cooldown_seconds = int(get_config("notification.handoff_cooldown_seconds", 300))
        self._cooldown = cooldown_seconds
        self._last_notify: Dict[str, float] = {}
        self._lock = threading.Lock()

    def should_notify(self, session_key: str) -> bool:
        """判断该会话当前是否允许再发一次通知（冷却期内返回 False）"""
        with self._lock:
            last = self._last_notify.get(session_key)
            if last is None:
                return True
            return (time.time() - last) >= self._cooldown

    def update_notify(self, session_key: str) -> None:
        """记录该会话最近一次通知时间"""
        with self._lock:
            self._last_notify[session_key] = time.time()

    def clear(self, session_key: str) -> None:
        """清除某会话的通知记录（用于测试）"""
        with self._lock:
            self._last_notify.pop(session_key, None)


# 全局单例（move_conversation 与 ai_handler 共用同一冷却表）
notify_tracker = NotifyTracker()
