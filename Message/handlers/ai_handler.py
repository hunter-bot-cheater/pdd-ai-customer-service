"""
AI回复处理器
专注的AI处理，移除复杂预处理和发送逻辑
"""
import random
import asyncio
from typing import Dict, Any, Optional
from bridge.context import Context, ContextType
from .base import BaseHandler
from .preprocessor import MessagePreprocessor
from Agent.bot import Bot
from core.session_state import SessionState
from Agent.CustomerAgent.tools.move_conversation import transfer_conversation, TransferConversationParams


class AIReplyHandler(BaseHandler):
    """专注的AI回复处理器"""

    def __init__(self, bot: Bot = None, auto_reply_types: set = None):
        super().__init__("AIReplyHandler")
        # 从 DI 容器获取 CustomerAgent 单例（如果未传入）
        if bot is None:
            from core.di_container import container
            from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
            bot = container.get(CustomerAgent)
        self.bot = bot
        self.preprocessor = MessagePreprocessor()
        self.auto_reply_types = auto_reply_types or {
            ContextType.TEXT,
            ContextType.GOODS_INQUIRY,
            ContextType.GOODS_SPEC,
            ContextType.ORDER_INFO,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION
        }

    def can_handle(self, context: Context) -> bool:
        """检查是否可以处理该消息"""
        # 支持多种消息类型
        return context.type in self.auto_reply_types

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """处理AI回复（支持分句发送）"""
        try:
            # ===== 0. 转人工状态检测：有效期内禁止 AI 抢答 =====
            shop_id = metadata.get('shop_id')
            from_uid = metadata.get('from_uid')
            if shop_id and from_uid:
                session_key = f"{shop_id}:{from_uid}"
                if SessionState().is_handoff(session_key):
                    self.logger.info(f"会话已在转人工状态，重新触发转人工流程: session_key={session_key}")
                    await self._retrigger_handoff(context, metadata)
                    return True  # 跳过 AI 回复

            # 1. 预处理消息
            processed_content = self.preprocessor.process(context.content, context.type)

            # 2. 调用AI生成回复
            reply = await self._get_ai_reply(processed_content, context)
            if not reply:
                self.logger.warning("AI回复生成失败，使用备用回复")
                return await self._handle_fallback(context, metadata)

            # ========== 3. 分句发送 ==========
            # 按句子分隔符拆分：句号、问号、感叹号、分号、换行等
            import re
            # 使用正则按标点拆分，保留分隔符
            sentences = re.split(r'(?<=[。！？；\n])', reply)
            # 过滤空字符串
            sentences = [s.strip() for s in sentences if s.strip()]

            # 如果句子数量太少（1-2句），直接整条发送
            if len(sentences) <= 2:
                success = await self._send_reply(context, reply, metadata)
                if success:
                    await self.log_message(context, "AI回复发送成功", f"回复: {reply[:50]}...")
                else:
                    self.logger.warning("AI回复发送失败")
                    return await self._handle_fallback(context, metadata)
                return True

            # 句子较多时，逐条发送
            self.logger.info(f"分句发送：共 {len(sentences)} 句")
            for i, sentence in enumerate(sentences):
                # 如果句子太长，可以不再拆分，直接发送
                success = await self._send_reply(context, sentence, metadata)
                if success:
                    self.logger.debug(f"分句 {i + 1}/{len(sentences)} 发送成功: {sentence[:30]}...")
                else:
                    self.logger.warning(f"分句 {i + 1} 发送失败: {sentence}")
                    # 发送失败时，可以选择继续还是中断
                    # 建议继续，避免丢失后续内容
                    # break  # 如果希望中断，取消注释

                # 每条之间间隔 1.5~2.5 秒（模拟打字停顿）
                if i < len(sentences) - 1:
                    await asyncio.sleep(random.uniform(1.5, 2.5))

            return True

        except Exception as e:
            self.logger.error(f"AI回复处理失败: {e}")
            return await self._handle_fallback(context, metadata)

    async def _retrigger_handoff(self, context: Context, metadata: Dict[str, Any]) -> None:
        """会话处于转人工有效期内：再次触发转人工并通知人工客服"""
        try:
            shop_id = metadata.get('shop_id')
            user_id = metadata.get('user_id')
            from_uid = metadata.get('from_uid')
            shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', '') or ""

            params = TransferConversationParams(
                shop_id=str(shop_id),
                user_id=str(user_id),
                recipient_uid=str(from_uid),
                shop_name=str(shop_name),
            )
            result = await asyncio.to_thread(
                transfer_conversation,
                params,
                "有效期内再次发消息",
                True,
                context.content or "",
            )
            self.logger.info(f"重新触发转人工结果: {result}")
        except Exception as e:
            self.logger.error(f"重新触发转人工失败: {e}")

    async def _get_ai_reply(self, query: str, context: Context) -> Optional[str]:
        """获取AI回复"""
        if not self.bot:
            return None

        try:
            # 优先使用异步接口，其次回退到同步接口
            if hasattr(self.bot, 'async_reply'):
                res = await self.bot.async_reply(query, context)
                return getattr(res, 'content', str(res))
            elif hasattr(self.bot, 'reply'):
                res = self.bot.reply(query, context)
                return getattr(res, 'content', str(res))
            else:
                self.logger.warning("Bot不支持reply或async_reply方法")
                return None

        except Exception as e:
            self.logger.error(f"AI Bot调用失败: {e}")
            return None

    async def _send_reply(self, context: Context, reply: str, metadata: Dict[str, Any]) -> bool:
        """发送回复"""
        try:
            # 从metadata中提取必要信息
            shop_id = metadata.get('shop_id')
            user_id = metadata.get('user_id')
            from_uid = metadata.get('from_uid')

            if not all([shop_id, user_id, from_uid]):
                self.logger.warning(f"缺少发送信息: shop_id={shop_id}, user_id={user_id}, from_uid={from_uid}")
                return False

            # 通过发送器抽象发送（同步 HTTP + DB，放工作线程避免阻塞事件循环）
            from bridge.sender import get_sender
            sender = get_sender(context.channel_type)
            if not sender:
                self.logger.warning(f"无可用发送器: channel_type={context.channel_type}")
                return False
            result = await asyncio.to_thread(sender.send_text, shop_id, user_id, from_uid, reply)
            if isinstance(result, dict) and result.get("success"):
                return True
            return False

        except Exception as e:
            self.logger.error(f"发送回复失败: {e}")
            return False

    async def _handle_fallback(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """备用回复处理"""
        try:
            # 简单的自动回复
            reply_text = "亲，感谢您的咨询！客服正在为您处理，请稍等片刻。"

            # 记录备用回复
            self.logger.info("使用备用回复")

            # 尝试发送备用回复
            success = await self._send_reply(context, reply_text, metadata)
            if not success:
                # 如果发送失败，记录日志并返回False让下游有机会处理
                await self.log_message(context, "备用回复发送失败", f"内容: {reply_text}")
                return False

            await self.log_message(context, "备用回复发送成功", f"内容: {reply_text}")
            return True

        except Exception as e:
            self.logger.error(f"备用回复处理失败: {e}")
            return True  # 即使失败也返回True，避免重复处理
