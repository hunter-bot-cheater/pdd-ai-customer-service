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

    # 订单/物流意图关键词：命中即主动查证「该客户在本店的订单」，不依赖 LLM 函数调用可靠性。
    # （glm-4-flash 函数调用不可靠，日志多次证实其漏选 query_order_status 工具，
    #   故在调 LLM 之前凭会话买家 uid 主动查证并注入结果，确保订单类问题一定有人工查单行为。）
    _ORDER_INTENT_KEYWORDS = (
        "订单", "快递", "物流", "到哪了", "到哪里了", "到哪儿了", "到货", "签收",
        "单号", "运单", "物流信息", "物流进度", "发货", "寄出", "寄没寄", "什么时候到",
        "多久到", "哪天到", "未发货", "已发货", "发货状态", "发货时间", "发货了吗",
        "发货了没", "发货没", "查物流", "物流到哪", "货到哪",
    )
    # 发货地/发货速度等通用咨询（与"我的订单状态"无关），命中则不触发查证
    _SHIP_ORIGIN_NEGATIVES = (
        "哪里发货", "从哪里发货", "发货地", "发货地点", "哪个仓", "产地", "哪个地区",
        "从哪发", "物流快吗", "发货快吗", "发货速度", "多久发货", "几天发货",
        "什么时候能发", "几天能发", "发货快不快", "物流快慢", "发货快不", "发货快么",
    )

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
        # 单次回复最多句数（按句末标点切分）。<=0 表示不限制。
        self.max_sentences = int(get_config("ai_reply.max_sentences", 4))
        # 规则 2: 同一买家连续回复间隔跟踪器
        self.uid_tracker = UIDSendTracker()

        # 同一买家消息串行化锁：consumer 有多个并发 worker，同买家的多条消息
        # 会并行处理导致回复穿插（先答问题2再答问题1）。按 (shop_id:from_uid) 加锁，
        # 保证同一买家的消息严格按到达顺序处理，前一条回复发完才处理下一条。
        self._buyer_locks: Dict[str, asyncio.Lock] = {}

        # 消息合并缓冲：key=lock_key, value=[(context, metadata), ...]
        self._msg_buffers: Dict[str, list] = {}
        # 防抖定时器：key=lock_key, value=asyncio.Task
        self._coalesce_timers: Dict[str, asyncio.Task] = {}
        # 合并开关（默认关闭；生产环境通过 ai_reply.enable_coalesce=true 开启）
        self._coalesce_enabled = bool(get_config("ai_reply.enable_coalesce", False))

    def _get_buyer_lock(self, key: str) -> asyncio.Lock:
        """获取（或创建）同一买家串行锁。单事件循环内安全。"""
        lock = self._buyer_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._buyer_locks[key] = lock
        return lock

    # ===== 消息合并（防连发逐条回复）=====
    # 同一买家短时间内连发多条消息（如"裁剪质量怎么样"+"会起球吗"），
    # 不应每条独立生成回复再分条发送，导致不同话题的回复穿插、看起来像自言自语。
    # 改为缓冲 + 防抖：窗口期内的新消息追加到缓冲区，窗口到期后合并为一条处理。
    _COALESCE_WINDOW_SEC = 6.0  # 秒，同一买家消息合并窗口
    _TRIVIAL_MSG_MAX_LEN = 2   # 极短消息阈值：清理后 ≤ 此值视为前一条的尾缀/补发

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
        """处理AI回复（支持消息合并防连发逐条回复 + 分条短消息发送 + 真人时序模拟）"""
        shop_id0 = metadata.get('shop_id')
        from_uid0 = metadata.get('from_uid')
        lock_key = f"{shop_id0}:{from_uid0}" if shop_id0 and from_uid0 else None

        if lock_key:
            # ===== 消息合并：同一买家短时间连发的消息缓冲后一次性处理 =====
            if self._coalesce_enabled:
                return await self._coalesce_and_handle(lock_key, context, metadata)
            # 合并关闭时走原有同步路径
            async with self._get_buyer_lock(lock_key):
                return await self._handle_unlocked(context, metadata)
        return await self._handle_unlocked(context, metadata)

    async def _coalesce_and_handle(self, lock_key: str, context: Context, metadata: Dict[str, Any]) -> bool:
        """将消息加入买家缓冲区，防抖窗口到期后合并处理。

        行为：
        - 首条消息：启动防抖定时器（_COALESCE_WINDOW_SEC 秒），消息入缓冲。
        - 窗口内新消息：追加到缓冲区，重置定时器（延长等待）。
        - 极短消息（≤2字）：视为前一条的尾缀/补发，拼接到上一条末尾。
        - 定时器到期：取全部缓冲消息，合并文本，走 _handle_unlocked 一次处理。
        """
        raw_text = (context.content or "").strip()
        cleaned = self._clean_text(raw_text)

        # 极短消息（"吗""呢""好"等）：拼接上一条而非独立成条
        is_trivial = len(cleaned) <= self._TRIVIAL_MSG_MAX_LEN

        buf = self._msg_buffers.get(lock_key)
        if buf is not None:
            # 缓冲区已有消息 → 追加
            if is_trivial and buf:
                # 拼接到上一条消息末尾（用户打字分段发送）
                last_ctx, last_meta = buf[-1]
                last_ctx.content = (last_ctx.content or "") + raw_text
                self.logger.debug(f"极短消息拼接: '{raw_text}' → 上一条末尾 (key={lock_key})")
            else:
                buf.append((context, metadata))
                self.logger.debug(f"消息追加到缓冲区: '{raw_text[:20]}' (key={lock_key}, 共{len(buf)}条)")

            # 重置防抖定时器
            self._reset_coalesce_timer(lock_key)
            return True  # 已缓冲，稍后统一处理

        # 首条消息 → 创建缓冲区 + 启动定时器
        self._msg_buffers[lock_key] = [(context, metadata)]
        self._start_coalesce_timer(lock_key)
        return True  # 已缓冲，定时器到期后处理

    def _start_coalesce_timer(self, lock_key: str) -> None:
        """启动/重启防抖定时器。到期后触发合并处理。"""
        # 取消旧定时器（如果有）
        old = self._coalesce_timers.pop(lock_key, None)
        if old and not old.done():
            old.cancel()

        async def _on_coalesce_window_expired():
            try:
                await asyncio.sleep(self._COALESCE_WINDOW_SEC)
            except asyncio.CancelledError:
                return  # 被新消息重置，忽略
            await self._flush_coalesced_messages(lock_key)

        self._coalesce_timers[lock_key] = asyncio.create_task(_on_coalesce_window_expired())

    def _reset_coalesce_timer(self, lock_key: str) -> None:
        """重置防抖定时器（等效于取消旧的 + 启动新的）。"""
        self._start_coalesce_timer(lock_key)

    async def _flush_coalesced_messages(self, lock_key: str) -> None:
        """防抖窗口到期：取出缓冲区所有消息，合并处理后清空。"""
        buf = self._msg_buffers.pop(lock_key, None)
        self._coalesce_timers.pop(lock_key, None)
        if not buf:
            return

        self.logger.info(f"消息合并处理: key={lock_key}, 缓冲{len(buf)}条消息")

        # 合并多条消息文本为一条（换行分隔，保留 LLM 对多问的感知能力）
        merged_contexts = []
        merged_texts = []
        for ctx, meta in buf:
            text = (ctx.content or "").strip()
            if text:
                merged_texts.append(text)
                merged_contexts.append((ctx, meta))

        if not merged_texts:
            return

        # 用第一条消息的 context/metadata 作为主上下文（含会话信息、shop_id 等），
        # 但把合并后的文本替换进去。
        primary_ctx, primary_meta = merged_contexts[0]
        merged_text = "\n".join(merged_texts)
        primary_ctx.content = merged_text

        self.logger.info(f"合并后文本({len(merged_texts)}条→1条): {merged_text[:80]}...")

        # 持有串行锁处理（保证与直接调用 _handle_unlocked 的互斥）
        async with self._get_buyer_lock(lock_key):
            await self._handle_unlocked(primary_ctx, primary_meta)

    async def _handle_unlocked(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """处理AI回复（已持有买家串行锁）"""
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

            # 1.6 订单/物流意图主动查证（双重保险）：glm-4-flash 函数调用不可靠，
            #   常漏选 query_order_status，故在调 LLM 之前凭会话买家 uid 主动查证并注入结果，
            #   行为等同真人客服在后台直接看到「这个客户的订单」。
            order_hint = await self._prefetch_order_if_needed(metadata, processed_content)

            # 1.7 客服知识库预取（双重保险）：售后类问题同样存在 glm-4-flash 漏调
            #   search_customer_service_knowledge 的问题，会绕开工具直接用预训练通用
            #   知识回答（如默认"支持7天无理由"），与店铺真实政策冲突。
            #   故在调 LLM 之前主动按关键词查 KB，把结果作为「已知事实」注入。
            kb_hint = await self._prefetch_kb_if_needed(metadata, processed_content)

            # 2. 调用AI生成回复（订单数据已由 1.6 预取注入；LLM 基于已知事实回复即可）
            reply = await self._get_ai_reply(processed_content, context, order_hint=order_hint, kb_hint=kb_hint)
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

        保护机制：
        - URL → 占位符（__URL_n__），确保完整不被截断
        - 订单号 → 占位符（__ORD_n__），同上。PDD 订单号形如 260811-xxxxxxxxxxxxx，
          通常 22~26 字符，接近 max_message_len(25)，极易被硬切断导致订单号跨消息断裂。
        拆分完成后统一还原为原始内容。含受保护 token 的消息允许略超上限。
        """
        if not reply:
            return []
        if self.max_message_len <= 0:
            return [reply]

        import re
        # 1. URL + 订单号 → 占位符，避免拆分过程中截断
        masked, url_map = self._mask_urls(reply)
        masked, ord_map = self._mask_order_numbers(masked)

        # 合并保护映射（占位符命名空间隔离：URL / ORD 不冲突）
        protected_map = {**url_map, **ord_map}

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

            # 顿号并列链合并：连续以顿号结尾的 part（款式、材质、价格）必须并到
            # 同一段，不能让顿号孤悬末尾。允许合并后超 max_message_len（语义正确优先）。
            merged: List[str] = []
            i = 0
            while i < len(parts):
                p = parts[i]
                if p.endswith("、"):
                    j = i + 1
                    # 吸收连续顿号结尾的 part
                    while j < len(parts) and parts[j].endswith("、"):
                        p += parts[j]
                        j += 1
                    # 把链后第一个非顿号 part 也并入（即并列的最后一项，如"价格"）
                    if j < len(parts):
                        p += parts[j]
                        j += 1
                    merged.append(p)
                    i = j
                else:
                    merged.append(p)
                    i += 1
            parts = merged

            current = ""
            for part in parts:
                if len(part) > self.max_message_len:
                    # 仍超长：按字符硬切（受保护占位符整体不可拆分）
                    for sub in self._split_overlong(part, protected_map):
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

        # 3. 还原占位符为原始内容（URL + 订单号）
        return [self._restore_protected(c, protected_map) for c in chunks]
        # 注：不做发送条数截断。句数精简由 CustomerAgent._condense_to_sentence_limit
        # 负责（打回重生成）；若重试仍超，语义优先原则下原样发送完整回复。

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
        """将占位符还原为原始 URL（保留向后兼容）"""
        return self._restore_protected(text, url_map)

    def _restore_protected(self, text: str, protected_map: Dict[str, str]) -> str:
        """将所有受保护占位符（URL / 订单号等）还原为原始内容"""
        if not text or not protected_map:
            return text
        for placeholder, original in protected_map.items():
            text = text.replace(placeholder, original)
        return text

    def _mask_order_numbers(self, text: str):
        """将文本中的拼多多订单号替换为占位符，返回 (掩码文本, {占位符: 原始订单号})

        PDD 订单号形如 260811-xxxxxxxxxxxxx（8 位日期 + 连字符 + 15~18 位数字），
        通常 22~26 字符，接近 max_message_len(25)，在 _split_overlong 中极易被
        按字符硬切断导致订单号跨消息断裂。用占位符保护其完整性。
        """
        import re
        if not text:
            return text, {}
        ord_map: Dict[str, str] = {}
        parts = []
        last = 0
        i = 0
        # 匹配 PDD 订单号：日期前缀(6~8位) + 可选连字符 + 数字序列(12~20位)
        for m in re.finditer(r'\b\d{6,8}-?\d{12,20}\b', text):
            raw = m.group(0)
            placeholder = f'__ORD_{i}__'
            ord_map[placeholder] = raw
            parts.append(text[last:m.start()])
            parts.append(placeholder)
            last = m.end()
            i += 1
        parts.append(text[last:])
        return ''.join(parts), ord_map

    # 断句：jieba 按词边界切分（杜绝词内断裂），块首孤立助词（结构/语气词）并回上一块。
    # 允许当前块超过 max_message_len 1~2 字，换取自然的断点。
    # 词级孤立助词（单字结构助词/语气词），块首出现时并回上一块。
    _STICKY_WORDS = {"的", "了", "哦", "呢", "哈", "呀", "吧", "嘛", "啊", "啦"}
    # 字符级备用（jieba 降级时用）
    _STICKY_PARTICLES = set("的了是在有和但而或也就都还才已要会可会能所因为当若那这")
    # 用于判定"实词"（不是黏着助词、也不是标点）。实词开头可能是词内断裂点。
    _SPLIT_PUNCTUATION = set("，。！？；、,.!?;:）】》」』")

    # 判断用户消息是否表达了"提问"意图（用于 KB 预取防护）。
    # 纯商品链接/商品名/寒暄不查 KB，避免 LLM 自作多情回答用户没问的政策/价格。
    _QUESTION_HINTS = (
        "?", "？",
        "怎么", "如何", "为何", "为什么", "为啥", "咋", "怎样",
        "多少", "几", "多久", "多大", "几天", "几件",
        "哪个", "哪种", "什么", "啥", "谁", "哪里", "哪儿",
        "是否", "能否", "能不能", "可以", "能",
        "吗", "呢", "嘛",
        "支持", "有没有", "有吗",
        "区别", "对比", "比较", "推荐", "建议",
        "贵", "便宜", "划算",
        "何时", "什么时候", "尺寸", "颜色", "材质",
    )

    def _looks_like_user_question(self, text: str) -> bool:
        """判断用户消息是否在提问。含问号或典型问句关键词返回 True。

        用于 KB 预取防护：纯商品链接/寒暄不查 KB，避免 LLM 自作多情。
        """
        if not text:
            return False
        # 问号是最强信号
        if "?" in text or "？" in text:
            return True
        return any(w in text for w in self._QUESTION_HINTS)

    def _split_overlong(self, text: str, protected_map: Dict[str, str]) -> List[str]:
        """将超长文本切分为不超过 max_message_len 的分片

        核心策略：jieba 按词边界切分后填充，词（如"换货""因为"）绝不被劈开。
        受保护占位符（URL / 订单号等）视为原子 token，绝不截断；
        超长单词（如长英文/数字串）整体成块，允许略超。

        边界自然化：若某块以孤立助词（的/了/哦...）开头，将其并回上一块末尾，
        避免"上一句末尾字词被强行放到下一句开头"的突兀感。允许超限 1~2 字。
        """
        if not text:
            return []
        import re
        # 同时匹配 URL 和订单号占位符，均视为不可拆分的原子 token
        tokens = [t for t in re.split(r'(__URL_\d+__|__ORD_\d+__)', text) if t]

        # 展开为词序列（占位符 = 一个原子词）
        word_seq: List[str] = []
        for tok in tokens:
            if tok in protected_map:
                word_seq.append(tok)
                continue
            try:
                import jieba
                ws = [w for w in jieba.lcut(tok) if w and w.strip()]
                word_seq.extend(ws if ws else list(tok))
            except Exception:
                # jieba 不可用时降级为逐字符（行为等同旧字符切分）
                word_seq.extend(list(tok))

        chunks: List[str] = []
        current = ""
        for w in word_seq:
            # 超长单词（长英文/数字串等）：单独成块（允许略超）
            if len(w) > self.max_message_len:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(w)
                continue
            if current and len(current) + len(w) <= self.max_message_len:
                current += w
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)

        # 块首孤立助词并回上一块（词级，最多并回 2 个连续助词字）
        fixed: List[str] = []
        for c in chunks:
            if not fixed or c[0] not in self._STICKY_WORDS:
                fixed.append(c)
                continue
            pull = 0
            while pull < min(2, len(c)) and c[pull] in self._STICKY_WORDS:
                pull += 1
            fixed[-1] = fixed[-1] + c[:pull]
            fixed.append(c[pull:])
        return [f for f in fixed if f]

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
        """意图识别路由：若 LLM 判定为操作/投诉/负面情绪（置信度达标）或识别不出意图（other/unknown），则转人工。

        失败/超时时保守返回 False（不转人工），避免把敏感诉求误交给 AI 回复，
        也避免把正常咨询误转。子账号静默标记与通知规则由 transfer_conversation 处理。

        上下文注入：路由阶段只看到当前这一句，极易把「要」「好的」这类对上一句客服
        提问的简短回应误判为 other（→ 转人工）。这里从会话历史取最近若干轮拼进分类
        prompt，让分类器能结合上下文正确判为 consult（续接咨询）。
        """
        try:
            from Message.handlers import intent_classifier as ic_module
            from Message.handlers.keyword_handler import match_after_sale_keyword
            from bridge.context import make_conversation_key

            classifier = ic_module.get_intent_classifier()
            if classifier is None or not classifier.enabled:
                return False

            # 取最近若干轮对话上下文（当前 inbound 消息尚未落库，故返回的是上一句之前的语境）
            history = []
            try:
                context_turns = int(get_config("intent.context_turns", 12))
                if context_turns > 0 and self.bot is not None:
                    session_id = make_conversation_key(context)
                    history = self.bot.get_session_history(session_id, limit=context_turns) or []
            except Exception as e:
                self.logger.warning(f"取意图分类上下文失败，降级为无上下文分类: {e}")

            after_sale_hint = match_after_sale_keyword(processed_content)
            result = await classifier.classify(
                processed_content, after_sale_hint=after_sale_hint, history=history
            )
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

    # 订单/物流查询改为「意图预取 + LLM 函数调用」双重保险（2026-08-11 移除后，
    # 实测 glm-4-flash 稳定漏选 query_order_status，故重新启用预取）。
    # 买家身份字段（recipient_uid/shop_id/user_id）由 tool_decorator 从
    # 受信任的 dependencies 自动注入，LLM 无需也无法指定。

    # ------------------------------------------------------------------

    async def _prefetch_order_if_needed(self, metadata: Dict[str, Any], text: str) -> str:
        """订单/物流意图主动查证（双重保险的第一道闸门）。

        真人客服在后台能直接看到「这个客户的订单」，无需用户报单号。但 glm-4-flash
        函数调用不可靠，常漏选 query_order_status，导致订单类问题只拿到泛泛的售前回复。
        故在调 LLM 之前，凭会话买家 uid 主动查证该客户在本店订单，把结果作为「已知事实」
        注入，确保订单/物流类问题一定有人工查单行为。

        返回注入给 LLM 的提示串；非订单意图或查证失败时返回空串（走 LLM 自主调用路径）。
        """
        content = (text or "").lower()
        # 负向词优先：发货地/发货速度等通用咨询不触发查证
        if any(p in content for p in self._SHIP_ORIGIN_NEGATIVES):
            return ""
        if not any(k in content for k in self._ORDER_INTENT_KEYWORDS):
            return ""

        from_uid = metadata.get("from_uid")
        shop_id = metadata.get("shop_id")
        user_id = metadata.get("user_id")
        if not all([from_uid, shop_id, user_id]):
            self.logger.warning("订单预取缺少必要参数，跳过: "
                                f"from_uid={from_uid}, shop_id={shop_id}, user_id={user_id}")
            return ""

        try:
            from Agent.CustomerAgent.tools.query_order_status import (
                query_order_status, QueryOrderStatusParams,
            )
            params = QueryOrderStatusParams(
                shop_id=str(shop_id), user_id=str(user_id), recipient_uid=str(from_uid),
            )
            # 工具内部可能起浏览器（首访约 10~15s），放线程避免阻塞事件循环
            result = await asyncio.to_thread(query_order_status, params)
        except Exception as e:
            self.logger.warning(f"订单预取异常，降级走 LLM 自主调用: {type(e).__name__}: {e}")
            return ""

        if not result:
            return ""

        # 清理客户话术里不允许的波浪号与感叹号（工具输出偶带 "~" 或 "！"），避免泄漏到最终回复
        result = result.replace("~", "").replace("～", "").replace("！", "").replace("!", "")

        if result.startswith("[untrusted_order_data]"):
            return (
                "【系统已为您查证该客户在本店的订单，请直接基于下方数据回复用户，"
                "不要说“暂时查不到”，也不要向用户索要订单号：】\n" + result
            )
        if "未查询到您在本店的订单" in result:
            return (
                "【系统已查证：该客户在本店暂无订单记录。请如实告知用户“未查询到您在本店的订单”，"
                "不要编造状态，也不要索要订单号。】\n" + result
            )
        # 其余（接口未返回 / 程序异常等）：原样转告，不要改写、不要索要订单号
        return (
            "【系统暂时取不到订单数据，请将下面这句话原样转告用户，"
            "不要改写、不要索要订单号：】\n" + result
        )

    # KB 预取：所有消息都先与客服知识库比对，有命中则按知识库回答，无命中走 LLM 自主路径。
    # 背景：glm-4-flash 在 tool_choice="auto" 下对"看起来像常识"的政策问题（7天无理由、
    # 保修期限、发货时效）会漏调 search_customer_service_knowledge，直接用预训练答案，
    # 与店铺真实政策冲突。故在调 LLM 前主动按整句查 KB 并注入命中条目。
    async def _prefetch_kb_if_needed(
        self, metadata: Dict[str, Any], text: str
    ) -> str:
        """客服知识库主动比对（全量预取）。

        对每条用户消息都执行一次 KB 检索：
        - 有命中（enabled 的客服知识条目）→ 注入为「系统已查证的客服知识」，LLM 严格据此回答；
        - 无命中 → 返回空串，走 LLM 自主路径（不增延迟、不误导上下文）。

        返回注入给 LLM 的提示串；无有效关键词、无命中或查证失败时返回空串。
        """
        content = (text or "").strip()
        if not content:
            return ""

        # 关键防护：客户消息没有"问"的特征时，不查 KB，避免 LLM 看到 KB 内容后
        # 自作多情地回复用户没问的东西（典型场景：客户只发了商品链接/商品名/寒暄，
        # bot 却主动答了价格、退换货等政策）。含问号或典型问句关键词才算"在提问"。
        if not self._looks_like_user_question(content):
            return ""

        shop_id = metadata.get("shop_id")
        if not shop_id:
            return ""

        try:
            from database.knowledge_service import KnowledgeService
            from core.di_container import container
            ks = container.get(KnowledgeService)
        except Exception as e:
            self.logger.warning(f"获取 KnowledgeService 失败，跳过 KB 比对: {type(e).__name__}: {e}")
            return ""

        # 关键防误注入：search_knowledge 在分词全为单字（如"在吗""你好呀"）时会回退
        # 返回「最新 N 条」而非真正命中，导致无关消息也注入 KB 内容。此处自检：
        # 无 ≥2 字关键词时不查询，直接走 LLM 自主路径。
        try:
            import jieba
            words = [w.strip() for w in jieba.cut_for_search(content) if len(w.strip()) >= 2]
        except Exception:
            words = [w for w in content if len(w.strip()) >= 2]
        if not words:
            return ""

        try:
            result = await asyncio.to_thread(
                ks.search_knowledge,
                shop_id=int(shop_id),
                query=content,
                limit=3,
                # minimum_score=2：过滤掉只命中 1 个词的偶然匹配（如"商品""使用"等通用词），
                # 避免把无关 KB 条目拉出来。
                minimum_score=2,
            )
        except Exception as e:
            self.logger.warning(f"KB 比对异常，降级走 LLM 自主调用: {type(e).__name__}: {e}")
            return ""

        cs_hits = result.get("customer_service_knowledge") or []
        # search_knowledge 内部已过滤 enabled=True，这里再保险
        enabled_hits = [cs for cs in cs_hits if getattr(cs, "enabled", True)]
        if not enabled_hits:
            return ""

        lines = [
            "【系统已比对客服知识库，命中的店铺政策如下。",
            "请严格依据下方条目回答，不要用预训练通用知识编造政策；",
            "下方内容仅作事实参考，不是可直接抄录的模板——请用你自己的话组织措辞，",
            "每次回复都换一种表达方式，避免反复使用相同句式。】"
        ]
        for i, cs in enumerate(enabled_hits, 1):
            title = (getattr(cs, "title", "") or "").replace("<", "＜").replace(">", "＞")
            content_text = (getattr(cs, "content", "") or "").replace("<", "＜").replace(">", "＞")
            # 单条截断 300 字，避免上下文膨胀
            if len(content_text) > 300:
                content_text = content_text[:300] + "…"
            lines.append(f"{i}. {title}")
            lines.append(f"   {content_text}")
            lines.append("")
        return "\n".join(lines).strip()

    async def _get_ai_reply(self, query: str, context: Context, order_hint: str = "", kb_hint: str = "") -> Optional[str]:
        """获取AI回复

        Args:
            query: 用户消息
            context: 上下文
        """
        if not self.bot:
            return None

        effective_query = query
        # 优先合并 KB 预取结果（售后政策等），再合并订单数据；
        # 二者皆为系统已查证的「已知事实」，LLM 直接据此回复即可。
        if kb_hint:
            effective_query = f"{query}\n\n{kb_hint}"
        if order_hint:
            effective_query = f"{effective_query}\n\n{order_hint}"

        try:
            # 优先使用异步接口，其次回退到同步接口
            if hasattr(self.bot, 'async_reply'):
                res = await self.bot.async_reply(effective_query, context)
                content = getattr(res, 'content', str(res))
            elif hasattr(self.bot, 'reply'):
                res = self.bot.reply(effective_query, context)
                content = getattr(res, 'content', str(res))
            else:
                self.logger.warning("Bot不支持reply或async_reply方法")
                return None
            # 句数精简已在 CustomerAgent._condense_to_sentence_limit 内做（打回重生成，
            # 非截断）；这里不再二次截断，避免把 LLM 精简后的完整语义砍掉。
            return content
        except Exception as e:
            self.logger.error(
                f"AI Bot调用失败: error_type={type(e).__name__}"
            )
            return None

    def _clean_text(self, text: str) -> str:
        """移除句末标点（句号、逗号、问号、分号、感叹号），但保留小数点内的 '.'"""
        if not text:
            return text
        import re
        # 先保护小数点数字（如 1.5、2.5、3.14），避免被后续清洗吞掉
        protected = text
        placeholders = []

        def _protect_decimal(m):
            placeholders.append(m.group(0))
            return f"\x00DEC{len(placeholders)-1}\x00"

        protected = re.sub(r'(?<=\d)\.(?=\d)', _protect_decimal, protected)

        # 清洗句末标点（含英文句号，小数点已在上方保护）
        cleaned = re.sub(r'[，,。.；;？?！!]', '', protected)

        # 恢复小数点
        for i, original in enumerate(placeholders):
            cleaned = cleaned.replace(f"\x00DEC{i}\x00", original)

        return cleaned

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
        """AI 回复失败时的兜底处理。

        不发送任何话术给买家（bot 扮演的就是真人客服，出现"客服正在为您处理"
        等机器话术会暴露身份），直接静默转人工：
        - 标记会话转人工，通知企业微信人工客服接手；
        - 买家侧无任何机器回复，避免露馅。
        """
        try:
            # 记录备用处理日志（仅日志，不发送）
            self.logger.info("AI回复失败，静默转人工（不发送机器话术）")
            last_message = getattr(context, 'content', '') or ''
            # 复用意图转人工的完整链路（标记会话 + 企业微信通知 + 子账号静默规则）
            return await self._transfer_by_intent(context, metadata, last_message)

        except Exception as e:
            self.logger.error(
                f"备用回复处理失败: error_type={type(e).__name__}"
            )
            return True  # 即使失败也返回True，避免重复处理
