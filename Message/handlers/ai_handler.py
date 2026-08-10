"""
AI回复处理器
专注的AI处理，移除复杂预处理和发送逻辑
"""
import random
import asyncio
import datetime
from typing import Dict, Any, Optional, List
from bridge.context import Context, ContextType
from .base import BaseHandler
from .preprocessor import MessagePreprocessor
from Agent.bot import Bot
from config import get_config
from core.session_state import SessionState
from core.notify_tracker import notify_tracker
from core.uid_send_tracker import UIDSendTracker


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

        # ===== 真人模拟时序参数（规则 1/3/4/2）=====
        # 规则 1: 已读 6~8 秒，打字 4~6 秒，合计 10~14 秒
        self.read_seconds_min = float(get_config("ai_reply.read_seconds_min", 6))
        self.read_seconds_max = float(get_config("ai_reply.read_seconds_max", 8))
        self.typing_seconds_min = float(get_config("ai_reply.typing_seconds_min", 4))
        self.typing_seconds_max = float(get_config("ai_reply.typing_seconds_max", 6))
        # 规则 4: 拆分后的多条消息间隔 3~6 秒
        self.split_interval_min = float(get_config("ai_reply.split_interval_min", 3))
        self.split_interval_max = float(get_config("ai_reply.split_interval_max", 6))
        # 规则 3: 单条消息最大字数
        self.max_message_len = int(get_config("ai_reply.max_message_len", 25))
        # 规则 2: 同一买家连续回复间隔跟踪器
        self.uid_tracker = UIDSendTracker()

    def can_handle(self, context: Context) -> bool:
        """检查是否可以处理该消息"""
        # 支持多种消息类型
        return context.type in self.auto_reply_types

    def _is_outside_business_hours(self) -> bool:
        """判断当前时间是否处于营业时间之外（非营业时间返回 True）

        从 config.json 的 business_hours 读取 start/end（默认 08:00-23:00），
        支持 start > end 的跨零点营业区间（如 23:00-08:00）。
        配置解析失败时保守地按营业时间内处理。
        """
        try:
            business_hours = get_config("business_hours", {}) or {}
            start_str = str(business_hours.get("start", "08:00"))
            end_str = str(business_hours.get("end", "23:00"))
            start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.datetime.strptime(end_str, "%H:%M").time()
            now = datetime.datetime.now().time()

            if start_time <= end_time:
                return not (start_time <= now <= end_time)
            # 跨零点营业（如 23:00-08:00）：营业区间为 [start, 24:00) ∪ [00:00, end]
            return not (now >= start_time or now <= end_time)
        except Exception as e:
            self.logger.warning(f"营业时间解析失败，按营业时间内处理: {e}")
            return False

    async def _notify_after_hours(self, context: Context, metadata: Dict[str, Any], from_uid: Optional[str]) -> None:
        """非营业时间通知人工客服（发送失败仅记录警告日志，不影响主流程）"""
        try:
            from Message.handlers.notify import async_send_wechat_notification, build_handoff_message
            shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', '') or ""
            message = build_handoff_message(
                shop_name=shop_name,
                buyer_uid=from_uid or "",
                reason="非营业时间自动转人工",
                last_message=context.content or "",
            )
            ok = await async_send_wechat_notification(message)
            if not ok:
                self.logger.warning(f"非营业时间通知发送失败: session={from_uid}")
        except Exception as e:
            self.logger.warning(f"非营业时间通知发送异常: {e}")

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """处理AI回复（支持分条短消息发送 + 真人时序模拟）"""
        try:
            # ===== 0. 营业时间检查：非营业时间静默转人工，不执行 AI 回复 =====
            if self._is_outside_business_hours():
                shop_id = metadata.get('shop_id')
                from_uid = metadata.get('from_uid')
                session_key = f"{shop_id}:{from_uid}"
                if shop_id and from_uid:
                    SessionState().mark_handoff(session_key)
                self.logger.info(f"非营业时间，会话 {session_key} 静默标记为转人工，不进行 AI 回复")
                await self._notify_after_hours(context, metadata, from_uid)
                return True  # 拦截后续处理

            # ===== 0. 转人工状态检测：规则 8，转人工后 AI 完全忽略该会话后续消息 =====
            shop_id = metadata.get('shop_id')
            from_uid = metadata.get('from_uid')
            if shop_id and from_uid:
                session_key = f"{shop_id}:{from_uid}"
                if SessionState().is_handoff(session_key):
                    self.logger.info(f"会话已转人工，AI 忽略该会话后续消息: session_key={session_key}")
                    # 规则 9: 转人工后新消息仍通知人工客服（带冷却防刷屏）
                    await self._notify_handoff_new_message(context, metadata, session_key)
                    return True  # 直接返回，不等待、不回复

            # 1. 预处理消息
            processed_content = self.preprocessor.process(context.content, context.type)

            # 1.5 意图识别路由：基于语义判断是否需要转人工（替代旧版售后关键词硬匹配）
            # 售后咨询（consult）放行给 AI 自主回答；操作/投诉/负面情绪转人工+通知。
            if await self._maybe_transfer_by_intent(context, metadata, processed_content):
                return True

            # 2. 调用AI生成回复
            reply = await self._get_ai_reply(processed_content, context)
            if not reply:
                self.logger.warning("AI回复生成失败，使用备用回复")
                return await self._handle_fallback(context, metadata)

            # 3. 规则 3: 拆分为不超过 max_message_len 字的短消息
            messages = self._split_reply(reply)

            # 4. 规则 1+2: 模拟已读+打字延迟（合计 10~14 秒），并补齐同一买家回复间隔
            await self._simulate_human_delay(from_uid)

            # 5. 逐条发送，遵守分条间隔与业务码校验（规则 4、5）
            self.logger.info(f"分条发送：共 {len(messages)} 条")
            sent_count = 0
            for i, msg in enumerate(messages):
                # 需求二：清洗后为空的分条跳过，不发送（与业务码失败区分，避免误停后续分条）
                if not self._clean_text(msg).strip():
                    self.logger.warning(f"分条 {i + 1} 清洗后为空，跳过该条: {msg!r}")
                    continue
                success = await self._send_reply(context, msg, metadata)
                if success:
                    sent_count += 1
                    self.logger.debug(f"分条 {i + 1}/{len(messages)} 发送成功: {msg[:30]}...")
                else:
                    # 规则 5: 发送结果业务码非 0（或请求失败）→ 记录日志并停止后续发送
                    self.logger.error(f"分条 {i + 1} 发送失败，停止后续发送: {msg[:30]}...")
                    break

                # 规则 4: 拆分后的多条消息间隔 3~6 秒
                if i < len(messages) - 1:
                    await asyncio.sleep(random.uniform(self.split_interval_min, self.split_interval_max))

            if sent_count == 0:
                self.logger.warning("AI回复全部发送失败，使用备用回复")
                return await self._handle_fallback(context, metadata)

            await self.log_message(context, "AI回复发送成功", f"回复: {reply[:50]}...")
            return True

        except Exception as e:
            self.logger.error(
                f"AI回复处理失败: error_type={type(e).__name__}"
            )
            return await self._handle_fallback(context, metadata)

    async def _notify_handoff_new_message(self, context: Context, metadata: Dict[str, Any], session_key: str) -> None:
        """规则 9: 转人工后买家新消息仍通知人工客服（同会话 5 分钟冷却防刷屏）"""
        try:
            if not notify_tracker.should_notify(session_key):
                self.logger.debug(f"会话通知处于冷却期内，跳过通知: session_key={session_key}")
                return

            # 延迟导入，避免与 Message 包产生循环依赖
            from Message.handlers.notify import build_handoff_message, send_wechat_notification_sync
            shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', '') or ""
            message = build_handoff_message(
                shop_name=shop_name,
                buyer_uid=metadata.get('from_uid') or "",
                reason="转人工后买家新消息",
                last_message=context.content or "",
            )
            await asyncio.to_thread(send_wechat_notification_sync, message)
            notify_tracker.update_notify(session_key)
            self.logger.info(f"已通知人工客服买家新消息: session_key={session_key}")
        except Exception as e:
            self.logger.error(f"发送转人工后新消息通知失败: {e}")

    def _split_reply(self, reply: str) -> List[str]:
        """规则 3: 将回复拆分为不超过 max_message_len 字的短消息

        优先在句号/问号/感叹号等句末标点处断句，其次在逗号/顿号处细分，
        单段仍超长时按字符硬切，保证每条消息字数不超限。

        URL 保护：先用占位符（__URL_n__）替换文本中的 URL，拆分完成后还原，
        确保 URL 完整无损、不被截断。含 URL 的消息允许略超字数上限
        （URL 无法拆分）。
        """
        if not reply:
            return []
        if self.max_message_len <= 0:
            return [reply]

        import re
        # 1. URL → 占位符，避免拆分过程中截断 URL
        masked, url_map = self._mask_urls(reply)

        # 2. 按句末标点拆分，保留分隔符
        sentences = re.split(r'(?<=[。！？；\n])', masked)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        for sent in sentences:
            if len(sent) <= self.max_message_len:
                chunks.append(sent)
                continue

            # 句子超长：按逗号/顿号再细分
            parts = re.split(r'(?<=[，、])', sent)
            parts = [p.strip() for p in parts if p.strip()]

            current = ""
            for part in parts:
                if len(part) > self.max_message_len:
                    # 仍超长：按字符硬切（URL 占位符整体不可拆分）
                    for sub in self._split_overlong(part, url_map):
                        if current and len(current) + len(sub) <= self.max_message_len:
                            current += sub
                        else:
                            if current:
                                chunks.append(current)
                            current = sub
                elif len(current) + len(part) <= self.max_message_len:
                    current += part
                else:
                    chunks.append(current)
                    current = part
            if current:
                chunks.append(current)

        # 3. 还原占位符为原始 URL
        return [self._restore_urls(c, url_map) for c in chunks]

    def _mask_urls(self, text: str):
        """将文本中的 URL 替换为占位符，返回 (掩码文本, {占位符: 原始URL})

        URL 末尾的中英文句号、逗号、问号等标点不属于 URL，剥离出掩码范围，
        避免 URL 被当作整块不可分割的内容而吞掉后续标点/文字。
        """
        if not text:
            return text, {}
        import re
        url_map: Dict[str, str] = {}
        parts = []
        last = 0
        i = 0
        # URL 字符集：除空白外，也在中英文句末/逗号标点处终止，避免吞掉后续中文内容
        for m in re.finditer(r'https?://[^\s，。！？；、]+', text):
            raw = m.group(0)
            url = raw.rstrip('，。！？；、,.!?;:()（）…')
            if not url:
                continue
            start, end = m.start(), m.start() + len(url)
            parts.append(text[last:start])
            placeholder = f'__URL_{i}__'
            url_map[placeholder] = url
            parts.append(placeholder)
            last = end
            i += 1
        parts.append(text[last:])
        return ''.join(parts), url_map

    def _restore_urls(self, text: str, url_map: Dict[str, str]) -> str:
        """将占位符还原为原始 URL"""
        if not text or not url_map:
            return text
        for placeholder, url in url_map.items():
            text = text.replace(placeholder, url)
        return text

    def _split_overlong(self, text: str, url_map: Dict[str, str]) -> List[str]:
        """将超长文本按字符硬切为不超过 max_message_len 的分片

        URL 占位符视为原子 token，绝不截断；含 URL 的分片允许略超上限。
        """
        if not text:
            return []
        import re
        tokens = [t for t in re.split(r'(__URL_\d+__)', text) if t]
        chunks: List[str] = []
        current = ""
        for tok in tokens:
            if tok in url_map:
                # URL 占位符整体不可拆分；塞不下时单独成条（允许略超）
                if current and len(current) + len(tok) <= self.max_message_len:
                    current += tok
                else:
                    if current:
                        chunks.append(current)
                    current = tok
                continue
            # 普通文本：逐字符填充
            while tok:
                room = self.max_message_len - len(current)
                if room <= 0:
                    chunks.append(current)
                    current = ""
                    continue
                take = min(len(tok), room)
                current += tok[:take]
                tok = tok[take:]
        if current:
            chunks.append(current)
        return chunks

    async def _simulate_human_delay(self, uid: Optional[str]) -> None:
        """模拟真人已读+打字延迟（规则 1），并补齐同一买家回复间隔（规则 2）"""
        # 规则 1: 已读 6~8 秒 + 打字 4~6 秒 = 总计 10~14 秒
        read_sec = random.uniform(self.read_seconds_min, self.read_seconds_max)
        typing_sec = random.uniform(self.typing_seconds_min, self.typing_seconds_max)
        delay = read_sec + typing_sec

        # 规则 2: 同一买家连续回复间隔不足 4 秒时补齐
        pad = self.uid_tracker.wait_before_send(uid)
        if pad > delay:
            self.logger.debug(f"同一买家回复间隔不足，补齐等待: pad={pad:.1f}s")
            delay = pad

        self.logger.debug(f"模拟已读+打字延迟: {delay:.1f}s (读{read_sec:.1f}s + 打{typing_sec:.1f}s)")
        await asyncio.sleep(delay)

    async def _maybe_transfer_by_intent(
        self, context: Context, metadata: Dict[str, Any], processed_content: str
    ) -> bool:
        """意图识别路由：若 LLM 判定为操作/投诉/负面情绪且置信度达标，则转人工。

        失败/超时时保守返回 False（不转人工），避免把敏感诉求误交给 AI 回复，
        也避免把正常咨询误转。子账号静默标记与通知规则由 transfer_conversation 处理。
        """
        try:
            from Message.handlers import intent_classifier as ic_module
            from Message.handlers.keyword_handler import match_after_sale_keyword

            classifier = ic_module.get_intent_classifier()
            if classifier is None or not classifier.enabled:
                return False

            after_sale_hint = match_after_sale_keyword(processed_content)
            result = await classifier.classify(processed_content, after_sale_hint=after_sale_hint)
        except Exception as e:
            self.logger.warning(f"意图分类异常，保守不转人工: {e}")
            return False

        intent = (result or {}).get("intent")
        confidence = float((result or {}).get("confidence", 0.0) or 0.0)
        self.logger.info(f"意图分类结果: intent={intent}, confidence={confidence}")

        if ic_module.IntentClassifier.should_transfer(intent, confidence, classifier.threshold):
            return await self._transfer_by_intent(context, metadata, processed_content)
        return False

    async def _transfer_by_intent(self, context: Context, metadata: Dict[str, Any], last_message: str) -> bool:
        """执行意图触发的转人工（语义层），保留营业时间/子账号静默规则。"""
        shop_id = metadata.get('shop_id')
        user_id = metadata.get('user_id')
        from_uid = metadata.get('from_uid')
        shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', None) or ""
        if not all([shop_id, user_id, from_uid]):
            return False

        from Agent.CustomerAgent.tools.move_conversation import (
            transfer_conversation,
            TransferConversationParams,
        )
        params = TransferConversationParams(
            shop_id=str(shop_id),
            user_id=str(user_id),
            recipient_uid=str(from_uid),
            shop_name=str(shop_name),
        )
        try:
            result = await asyncio.to_thread(
                transfer_conversation, params, "AI意图识别触发转人工", True, last_message
            )
        except Exception as e:
            self.logger.error(f"意图转人工调用异常: {e}")
            return True

        if "会话转接成功" in result:
            self.logger.info(f"意图触发转接人工成功: {result}")
            return True

        self.logger.error(f"意图转接失败: {result}")
        # 规则 6：转人工失败仍静默拦截，避免 AI 误答敏感诉求
        return True

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
            self.logger.error(
                f"AI Bot调用失败: error_type={type(e).__name__}"
            )
            return None

    def _clean_text(self, text: str) -> str:
        """需求二：移除所有中英文句号、逗号、问号、分号，使回复更简洁自然"""
        if not text:
            return text
        import re
        return re.sub(r'[，,。.；;？?]', '', text)

    async def _send_reply(self, context: Context, reply: str, metadata: Dict[str, Any]) -> bool:
        """发送回复，并校验发送结果业务码（规则 5）"""
        # 需求二：发送前清洗标点（句号、逗号、问号、分号），清洗后为空则取消发送
        reply = self._clean_text(reply)
        if not reply.strip():
            self.logger.warning(f"清洗后回复为空，取消发送: reply={reply!r}")
            return False
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

            # 规则 5: 校验发送结果与业务码，失败记录日志并返回 False（停止后续发送）
            ok = self._check_send_result(result)
            if ok:
                # 规则 2: 记录本次发送时间，用于同一买家回复间隔控制
                self.uid_tracker.record_send(from_uid)
            return ok

        except Exception as e:
            self.logger.error(
                f"发送回复失败: error_type={type(e).__name__}"
            )
            return False

    def _check_send_result(self, result) -> bool:
        """规则 5: 校验发送结果，业务码非 0 视为失败"""
        if not result:
            return False
        if isinstance(result, str):
            # send_text 在业务错误（如 error_code=10002）时返回错误文案字符串
            self.logger.error(f"发送失败（业务错误）: {result}")
            return False
        if result.get("success"):
            error_code = result.get("result", {}).get("error_code", 0)
            if error_code:
                self.logger.error(f"发送业务码非 0: error_code={error_code}, result={result}")
                return False
            return True
        self.logger.error(f"发送请求失败: {result}")
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
            self.logger.error(
                f"备用回复处理失败: error_type={type(e).__name__}"
            )
            return True  # 即使失败也返回True，避免重复处理
