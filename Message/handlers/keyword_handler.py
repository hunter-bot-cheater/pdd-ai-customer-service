"""
关键词检测处理器 - 检测转人工关键词并触发转人工流程
"""
import asyncio
from typing import Dict, Any
from bridge.context import Context, ContextType
from .base import BaseHandler
from database.db_manager import db_manager
from utils.logger_loguru import get_logger
from Agent.CustomerAgent.tools.move_conversation import transfer_conversation, TransferConversationParams

class KeywordDetectionHandler(BaseHandler):
    """关键词检测处理器 - 检测转人工关键词并触发转人工流程"""

    # 售后关键词：命中后强制转人工，避免 AI 回复敏感问题
    AFTER_SALE_KEYWORDS = [
        "退货", "退款", "售后", "质量问题", "破损", "漏发", "少发",
        "不满意", "投诉", "赔偿", "换货", "维修", "差评", "给差评",
        "假货", "质量差",
    ]

    def __init__(self):
        super().__init__("KeywordDetectionHandler")
        self.logger = get_logger("KeywordDetectionHandler")
        self.keywords = self._load_keywords()

        # 记录加载的关键词数量
        self.logger.info(f"关键词检测处理器初始化完成，加载了 {len(self.keywords)} 个关键词，{len(self.AFTER_SALE_KEYWORDS)} 个售后关键词")

    def _load_keywords(self):
        """从数据库加载关键词"""
        try:
            keywords_data = db_manager.get_all_keywords()
            keywords = {item['keyword'].lower() for item in keywords_data if item.get('keyword')}
            self.logger.debug(f"从数据库加载关键词: {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"加载关键词失败: {e}")
            # 如果加载失败，使用默认关键词
            default_keywords = {
                "转人工", "人工客服", "真人", "客服", "人工", "工单", "好评",
                "取消订单", "改地址", "转售后客服", "转售后", "返现", "过敏",
                "退款", "没有效果", "骗人", "投诉", "纠纷", "开发票", "开票",
                "烂", "取消", "备注"
            }
            self.logger.warning(f"使用默认关键词: {default_keywords}")
            return default_keywords

    def can_handle(self, context: Context) -> bool:
        """检查消息是否包含关键词"""
        # 只处理文本类型的消息
        if context.type != ContextType.TEXT:
            return False

        # 检查消息内容是否存在且为字符串
        if not context.content or not isinstance(context.content, str):
            return False

        # 将消息内容转换为小写进行检测
        content_lower = context.content.lower()

        # 检查是否包含任何转人工关键词
        for keyword in self.keywords:
            if keyword in content_lower:
                self.logger.debug(f"检测到转人工关键词: '{keyword}' 在消息: '{context.content}'")
                return True

        # 检查是否包含任何售后关键词
        for keyword in self.AFTER_SALE_KEYWORDS:
            if keyword in content_lower:
                self.logger.debug(f"检测到售后关键词: '{keyword}' 在消息: '{context.content}'")
                return True

        return False

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """转接到人工客服"""
        try:
            shop_id = metadata.get('shop_id')
            user_id = metadata.get('user_id')
            from_uid = metadata.get('from_uid')
            shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', None) or ""

            if not all([shop_id, user_id, from_uid]):
                return False

            content = context.content or ""
            # 售后关键词：强制转人工，即使失败也拦截，避免 AI 回复敏感问题
            is_after_sale = any(kw in content for kw in self.AFTER_SALE_KEYWORDS)
            reason = "售后关键词触发转人工" if is_after_sale else "用户主动转人工"

            params = TransferConversationParams(
                shop_id=str(shop_id),
                user_id=str(user_id),
                recipient_uid=str(from_uid),
                shop_name=str(shop_name),
            )

            try:
                result = await asyncio.to_thread(transfer_conversation, params, reason, True, content)
            except Exception as e:
                self.logger.error(f"调用转人工工具异常: {e}")
                return True if is_after_sale else False

            if "会话转接成功" in result:
                self.logger.info(f"会话已成功转接人工: {result}")
                return True

            self.logger.error(f"会话转接失败: {result}")
            # 规则 6: 转人工后静默处理，不向用户发送预设回复话术
            return True if is_after_sale else False

        except Exception as e:
            self.logger.error(f"客服转接处理失败: {e}")
            return False
            
    def reload_keywords(self) -> None:
        """重新加载关键词（用于管理员更新关键词后刷新）"""
        old_count = len(self.keywords)
        self.keywords = self._load_keywords()
        new_count = len(self.keywords)
        self.logger.info(f"关键词重新加载完成: {old_count} -> {new_count}")

    def get_keyword_count(self) -> int:
        """获取当前关键词数量"""
        return len(self.keywords)

    def get_keywords(self) -> set:
        """获取当前关键词列表"""
        return self.keywords.copy()