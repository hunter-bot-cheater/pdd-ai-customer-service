"""
同一买家回复间隔跟踪器

规则：同一买家（from_uid）连续两次回复之间的间隔至少 4~6 秒，
若上一次回复距今不足 4 秒，则在发送前等待补齐，
避免机器人秒回导致平台风控或真人感缺失。

间隔可通过配置 ai_reply.uid_min_interval 调整（默认 4 秒）。
"""
import threading
import time
from typing import Dict

from config import get_config


class UIDSendTracker:
    """同一买家回复间隔跟踪器（线程安全）"""

    def __init__(self, min_interval: float = None):
        # 读取配置，默认 4 秒；测试中可显式传 0 关闭等待
        if min_interval is None:
            min_interval = float(get_config("ai_reply.uid_min_interval", 4.0))
        self._min_interval = min_interval
        self._last_send: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait_before_send(self, uid: str) -> float:
        """返回距离上一次发送还需等待的秒数（无需等待则返回 0）"""
        with self._lock:
            last = self._last_send.get(uid)
            if last is None:
                return 0.0
            elapsed = time.time() - last
            pad = self._min_interval - elapsed
            return max(0.0, pad)

    def record_send(self, uid: str) -> None:
        """记录一次发送时间"""
        with self._lock:
            self._last_send[uid] = time.time()

    def clear(self, uid: str) -> None:
        """清除某买家的发送记录（用于测试/会话重置）"""
        with self._lock:
            self._last_send.pop(uid, None)
