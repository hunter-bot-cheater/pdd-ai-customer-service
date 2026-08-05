"""
处理器基类和通用工具
"""
import json
from typing import Dict, Any, Optional
from utils.logger_loguru import get_logger
from bridge.context import Context, _context_value
from ..core.handlers import MessageHandler



class BaseHandler(MessageHandler):
    """处理器基类，提供通用功能"""

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or self.__class__.__name__

    async def log_message(self, context: Context, action: str, extra_info: str = ""):
        """统一的日志记录（不记录完整内容以保护隐私）"""
        message_id = _context_value(context, "msg_id", "unknown")
        content_length = len(str(context.content)) if context.content else 0
        extra_length = len(str(extra_info)) if extra_info else 0
        self.logger.info(
            f"{self.name} {action} - message_id={message_id} "
            f"content_length={content_length} extra_length={extra_length}"
        )

    def _get_user_info(self, context: Context) -> str:
        """提取用户信息"""
        try:
            if hasattr(context, 'kwargs') and context.kwargs:
                message_id = _context_value(context, "msg_id")
                if message_id:
                    return f"message:{message_id}"
            return "用户:unknown"
        except Exception:
            return "用户:unknown"
