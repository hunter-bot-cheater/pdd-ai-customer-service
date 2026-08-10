"""
关键词检测处理器 - 检测转人工关键词并触发转人工流程

改造后语义（方案一 + category）：
- transfer 类关键词（必转硬短路）：命中立即转人工，不受意图判断影响，保留"用户说转人工必转"。
- after_sale 类关键词（售后软兜底）：不再直接转人工，仅作为意图分类的提示，
  由 Message.handlers.ai_handler 中的意图分类器基于语义决定是否转人工。
  这样"退货流程是什么"等纯咨询可放行给 AI，而"立刻给我退款"等操作诉求仍转人工。
"""
import asyncio
from datetime import datetime, time
from typing import Dict, Any, Set, Tuple
from bridge.context import Context, ContextType
from .base import BaseHandler
from database.db_manager import (
    db_manager,
    DEFAULT_TRANSFER_KEYWORDS,
    DEFAULT_AFTER_SALE_KEYWORDS,
)
from utils.logger_loguru import get_logger
from Agent.CustomerAgent.tools.move_conversation import transfer_conversation, TransferConversationParams

# 模块级售后软兜底词集合，供 ai_handler 的意图分类器用作提示，无需实例化 handler。
_AFTERSALE_KEYWORDS: Set[str] = set()


class KeywordDetectionHandler(BaseHandler):
    """关键词检测处理器 - 仅对 transfer 类关键词硬短路转人工"""

    # 保留为种子常量，供迁移播种与测试断言使用；不再用于硬短路转人工。
    AFTER_SALE_KEYWORDS = list(DEFAULT_AFTER_SALE_KEYWORDS)

    def __init__(self, business_hours=None):
        super().__init__("KeywordDetectionHandler")
        self.logger = get_logger("KeywordDetectionHandler")
        self.business_hours = business_hours or {"start": "08:00", "end": "23:00"}
        self.transfer_keywords, self.after_sale_keywords = self._load_keywords()

        # 记录加载的关键词数量
        self.logger.info(
            f"关键词检测处理器初始化完成，加载了 {len(self.transfer_keywords)} 个必转词，"
            f"{len(self.after_sale_keywords)} 个售后软兜底词"
        )

    def _load_keywords(self) -> Tuple[Set[str], Set[str]]:
        """从数据库按类别加载关键词，并刷新模块级售后词集合。"""
        global _AFTERSALE_KEYWORDS
        try:
            keywords_data = db_manager.get_all_keywords()
            transfer: Set[str] = set()
            after: Set[str] = set()
            for item in keywords_data:
                kw = (item.get('keyword') or '').lower()
                if not kw:
                    continue
                if (item.get('category') or 'transfer') == 'after_sale':
                    after.add(kw)
                else:
                    transfer.add(kw)
            _AFTERSALE_KEYWORDS = after
            self.logger.debug(f"loaded transfer={len(transfer)} after_sale={len(after)}")
            return transfer, after
        except Exception as e:
            self.logger.error(
                f"keyword load failed: error_type={type(e).__name__}"
            )
            # 如果加载失败，使用默认关键词（两类都兜底）
            transfer = {k.lower() for k in DEFAULT_TRANSFER_KEYWORDS}
            after = {k.lower() for k in DEFAULT_AFTER_SALE_KEYWORDS}
            _AFTERSALE_KEYWORDS = after
            self.logger.warning(
                f"using default keywords: transfer={len(transfer)} after_sale={len(after)}"
            )
            return transfer, after

    def can_handle(self, context: Context) -> bool:
        """仅当命中 transfer 类（必转）关键词时拦截并转人工。

        after_sale 类关键词不再在此拦截，放行给 ai_handler 做意图判断。
        """
        # 只处理文本类型的消息
        if not self._within_business_hours():
            return False
        if context.type != ContextType.TEXT:
            return False

        # 检查消息内容是否存在且为字符串
        if not context.content or not isinstance(context.content, str):
            return False

        # 将消息内容转换为小写进行检测
        content_lower = context.content.lower()

        # 仅 transfer 类关键词触发硬短路转人工
        for keyword in self.transfer_keywords:
            if keyword in content_lower:
                self.logger.debug(f"检测到必转关键词: '{keyword}' 在消息: '{context.content}'")
                return True

        return False

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """转接到人工客服（仅由 transfer 类关键词触发，保留子账号静默规则）"""
        try:
            shop_id = metadata.get('shop_id')
            user_id = metadata.get('user_id')
            from_uid = metadata.get('from_uid')
            shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', None) or ""

            if not all([shop_id, user_id, from_uid]):
                return False

            content = context.content or ""
            reason = "用户主动转人工"

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
                return True

            if "会话转接成功" in result:
                self.logger.info(f"会话已成功转接人工: {result}")
                return True

            self.logger.error(f"会话转接失败: {result}")
            # 规则 6: 转人工后静默处理，不向用户发送预设回复话术
            return True

        except Exception as e:
            self.logger.error(
                f"客服转接处理失败: error_type={type(e).__name__}"
            )
            return False
            
    def reload_keywords(self) -> None:
        """重新加载关键词（用于管理员更新关键词后刷新）"""
        old_transfer = len(self.transfer_keywords)
        old_after = len(self.after_sale_keywords)
        self.transfer_keywords, self.after_sale_keywords = self._load_keywords()
        self.logger.info(
            f"关键词重新加载完成: transfer {old_transfer} -> {len(self.transfer_keywords)}, "
            f"after_sale {old_after} -> {len(self.after_sale_keywords)}"
        )

    def get_keyword_count(self) -> int:
        """获取当前关键词数量（两类合计）"""
        return len(self.transfer_keywords) + len(self.after_sale_keywords)

    def get_keywords(self) -> set:
        """获取当前关键词列表（两类合计）"""
        return self.transfer_keywords | self.after_sale_keywords

    def _within_business_hours(self) -> bool:
        """Return whether manual-service routing is currently enabled.

        配置无效（缺失或格式错误）时保守返回 True（允许转人工），
        避免配置问题导致功能被静默禁用。
        """
        bh = self.business_hours
        if not isinstance(bh, dict):
            self.logger.error(
                f"营业时间配置无效（{bh!r}），应为包含 start/end 的配置对象，"
                f"如 start=\"08:00\", end=\"23:00\"。已默认允许转人工。"
            )
            return True
        try:
            start = time.fromisoformat(str(bh.get("start", "08:00")))
            end = time.fromisoformat(str(bh.get("end", "23:00")))
            current = datetime.now().time()
            if start <= end:
                return start <= current <= end
            return current >= start or current <= end
        except (TypeError, ValueError):
            self.logger.error(
                f"营业时间配置无效（start={bh.get('start')!r}, end={bh.get('end')!r}），"
                f"格式应为 HH:MM，如 \"08:00\"/\"23:00\"。已默认允许转人工。"
            )
            return True


def reload_after_sale_global() -> None:
    """刷新模块级售后词集合（UI 或运行时增删后调用，无需实例化 handler）。"""
    try:
        keywords_data = db_manager.get_all_keywords()
        after = {
            (item.get('keyword') or '').lower()
            for item in keywords_data
            if (item.get('category') or 'transfer') == 'after_sale' and item.get('keyword')
        }
        global _AFTERSALE_KEYWORDS
        _AFTERSALE_KEYWORDS = after
    except Exception as e:  # pragma: no cover
        get_logger("KeywordDetectionHandler").warning(f"刷新售后词集合失败: {e}")


def match_after_sale_keyword(text: str) -> bool:
    """判断文本是否命中售后软兜底词（供意图分类器用作提示）。"""
    if not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in _AFTERSALE_KEYWORDS)
