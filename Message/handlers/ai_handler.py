"""
AI回复处理器
专注的AI处理，移除复杂预处理和发送逻辑
"""
import random
import asyncio
import datetime
import time
from collections import deque
from typing import Dict, Any, Optional, List
from bridge.context import Context, ContextType
from .base import BaseHandler
from .preprocessor import MessagePreprocessor
from Agent.bot import Bot
from config import get_config
from core.session_state import SessionState
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

    # 用户要求：无法回答一律直接转人工，禁止说"我不太清楚"等推脱话。
    # 兜底拦截：LLM 未调用 transfer_conversation 却输出下列推脱话术时，
    # 强制静默转人工，不把推脱话发给买家。模式特意保守，仅匹配明确"答不上来"的措辞，
    # 避免误伤正常回复（如"不太清楚您想要哪个颜色"是反问澄清，不含"我/呢"等标记，不拦截）。
    _CANNOT_ANSWER_PATTERNS = (
        "我不太清楚", "我不大清楚", "我不太懂", "我不大懂",
        "不太清楚呢", "不清楚呢", "不大清楚呢",
        "回答不了", "没法回答", "不能回答", "帮不了您", "帮不到您",
        "没法给您准确", "无法给您准确", "给您准确答复", "没法给您确切",
        "这块我没法", "这个我回答不来", "我回答不上来",
        "我确实不知道", "我真的不知道", "不太了解这方面", "这块我不太了解",
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
        # 合并窗口（秒）：同一买家该时间内的连发消息会被合并。可配置，默认 4 秒。
        self._coalesce_window = float(get_config("ai_reply.coalesce_window_sec", 4.0))
        # 合并窗口内"已读"计时起点：收到首条即开始，使已读延迟与窗口并行消耗，
        # 避免「6秒窗口 + 已读6~8秒 + 打字4~6秒」叠加导致冷启动过慢。
        self._coalesce_read_starts: Dict[str, float] = {}
        # 合并开关（默认关闭；生产环境通过 ai_reply.enable_coalesce=true 开启）
        self._coalesce_enabled = bool(get_config("ai_reply.enable_coalesce", False))

        # 会话内已发消息缓存：用于发送前/发送后「防重复改写」。
        # 拼多多平台对短时间内发给同一买家的「相同消息」会返回 40013 拦截
        # （error=请勿重复发送相同消息）。买家重复提问时 bot 会生成雷同答案被拦，
        # 故缓存近期已发文本，命中重复则换措辞重写，既回复客户又绕开平台风控。
        self._recent_sent: Dict[str, deque] = {}      # session_key -> deque[(ts, text)]
        self._recent_sent_ttl = float(get_config("ai_reply.repeat_cache_ttl_sec", 300))  # 5 分钟
        self._recent_sent_max = int(get_config("ai_reply.repeat_cache_max", 12))         # 每条会话保留条数
        # 40013 重复拦截后改写重试上限（避免无限改写死循环）
        self._repeat_rewrite_max = int(get_config("ai_reply.repeat_rewrite_max", 1))

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
        # 收到首条即开始「已读」计时（语义上此刻视为已读，后续窗口期并行消耗已读延迟）
        self._coalesce_read_starts[lock_key] = time.monotonic()
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
                await asyncio.sleep(self._coalesce_window)
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

        # 计算合并窗口期间已消耗的「已读」时间，传给后续延迟模拟予以扣除，
        # 避免已读延迟与窗口等待叠加导致总延迟过长（收到首条即开始已读计时）。
        read_start = self._coalesce_read_starts.pop(lock_key, None)
        elapsed_read = (time.monotonic() - read_start) if read_start else 0.0

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
        primary_meta["_coalesce_read_elapsed"] = elapsed_read
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
                    self.logger.info(
                        f"会话已转人工，AI 忽略该会话后续消息（静默，不回复不再通知）: session_key={session_key}"
                    )
                    # 转人工仅首次通知；后续买家新消息静默忽略，不重复发企业微信通知
                    return True  # 直接返回，不等待、不回复

            # 1. 预处理消息
            processed_content = self.preprocessor.process(context.content, context.type)

            # 1.4 成衣面料用量咨询直接转人工：做衣服/做短袖/给小孩做衣服要多少米布等
            # 这类问题需要版型、款式、体型数据，AI 估算极易误导，命中即转人工。
            if self._is_garment_fabric_usage_query(processed_content):
                self.logger.info(f"命中成衣用量咨询，直接转人工: {processed_content[:60]}")
                return await self._transfer_by_intent(
                    context, metadata, processed_content or "",
                    reason="成衣用量咨询→转人工",
                )

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

            # 1.8 规格感知米数计算：买家问"X米能不能买/怎么拍"时，按商品规格程序化
            #   计算拍法（或不可裁结论），硬性注入 prompt，LLM 只复述不自行计算。
            meter_hint = await self._prefetch_meter_hint(metadata, processed_content, context)

            # 2. 调用AI生成回复（订单数据已由 1.6 预取注入；LLM 基于已知事实回复即可）
            reply = await self._get_ai_reply(
                processed_content, context,
                order_hint=order_hint, kb_hint=kb_hint, meter_hint=meter_hint,
            )
            if not reply:
                self.logger.warning("AI回复生成失败，使用备用回复")
                return await self._handle_fallback(context, metadata)

            # 2.5 兜底拦截：用户要求"无法回答直接转人工、禁止说'我不太清楚'"。
            #     若 LLM 未真正转人工却输出推脱话术，强制静默转人工，绝不能把推脱话发给买家。
            if self._is_cannot_answer(reply):
                session_key = f"{metadata.get('shop_id')}:{metadata.get('from_uid')}"
                if SessionState().is_handoff(session_key):
                    self.logger.warning("LLM 输出推脱话术，但会话已转人工，直接拦截不发送")
                    return True
                self.logger.warning("LLM 输出推脱话术（疑似无法回答），强制静默转人工")
                return await self._transfer_by_intent(
                    context, metadata, processed_content or "",
                    reason="AI无法回答→转人工",
                )

            # 2.6 尺寸政策硬拦截：本店不接受特殊尺寸/长度备注。LLM（尤其 glm-4-flash）在
            #   KB 未命中时会编造"有特殊长度需求可在订单备注说明"类变通话术（实例：
            #   "大概有多长"缺问句特征 → KB 未注入 → 模型自由发挥）。命中即带纠正指令
            #   重答一次；仍违规则替换为安全模板，绝不把违规承诺发给买家。
            if self._violates_size_policy(reply) or self._is_wrong_size_refusal(processed_content, reply):
                if self._is_wrong_size_refusal(processed_content, reply):
                    self.logger.warning(f"AI 回复错误拒绝整米长度，纠正后重答: {reply[:80]!r}")
                else:
                    self.logger.warning(f"AI 回复疑似违反尺寸政策，纠正后重答: {reply[:80]!r}")
                correction = (
                    "【纠正指令】你上一版回复违反了店铺尺寸政策。店铺规则：不接受任何特殊尺寸/长度备注，"
                    "规格有就能买，规格库里无法组合出来的长度就不能买。判断标准看当前咨询的商品："
                    "花型名称含「贡缎」且幅宽1.5米的商品支持0.5米递增（1.5米、2.5米、3.5米等规格里有的都能买）；"
                    "普通花型只售整米，但整米可以通过多拍组合实现（如4米拍2个2米、5米拍2米+3米、6米拍2个3米），"
                    "不要对整米需求说'不支持''面料长度固定'。请重新回答用户问题，严禁提及订单备注、定制、"
                    "按需裁剪或任何变通说法，也严禁错误拒绝可实现的整米长度，就事论事。"
                )
                reply2 = await self._get_ai_reply(
                    processed_content, context,
                    order_hint=order_hint,
                    kb_hint=(kb_hint + "\n\n" + correction) if kb_hint else correction,
                )
                if reply2 and not self._violates_size_policy(reply2) and not self._is_wrong_size_refusal(processed_content, reply2) and not self._is_cannot_answer(reply2):
                    reply = reply2
                else:
                    reply = (
                        "亲，这款面料按商品规格售卖，整米可以拍多个组合（如4米拍2个2米），"
                        "小数米或特殊长度不支持备注呢。"
                    )
                    self.logger.warning("纠正重答仍违反尺寸政策/错误拒售或为空，使用安全模板回复")

            # 3. 规则 3: 拆分为不超过 max_message_len 字的短消息
            messages = self._split_reply(reply)

            # 4. 规则 1+2: 模拟已读+打字延迟（合计 10~14 秒），并补齐同一买家回复间隔
            #    合并场景下扣除窗口期已消耗的已读时间（_coalesce_read_elapsed），避免双重计时。
            already_read = float(metadata.pop("_coalesce_read_elapsed", 0.0) or 0.0)
            await self._simulate_human_delay(from_uid, already_elapsed_read=already_read)

            # 5. 发送前：检测与近期已发消息重复（平台 40013 防刷），命中则先改写，减少失败往返
            session_key = f"{metadata.get('shop_id')}:{metadata.get('from_uid')}"
            messages = await self._prededupe(messages, session_key, context)

            # 6. 逐条发送，遵守分条间隔与业务码校验（规则 4、5）
            #    平台 40013「请勿重复发送相同消息」拦截时，换措辞改写重试（最多 _repeat_rewrite_max 次），
            #    不静默放弃、也不直接转人工——买家需要被回应，改写后仍失败才转人工兜底。
            self.logger.info(f"分条发送：共 {len(messages)} 条")
            sent_count = 0
            i = 0
            n = len(messages)
            rewrite_attempts = 0
            while i < n:
                msg = messages[i]
                # 发送前再次确认：人工客服在生成/延迟/分条间隔期间介入，立即放弃本次回复，
                # 避免"程序已准备发送、人已先回"的竞态下仍把 AI 回复发出去。
                if SessionState().is_handoff(session_key):
                    self.logger.info(
                        f"发送前检测到人工客服介入，放弃本次待发送回复"
                        f"（共 {n} 条，已发 {sent_count}）: session_key={session_key}"
                    )
                    break
                # 需求二：清洗后为空的分条跳过，不发送（与业务码失败区分，避免误停后续分条）
                if not self._clean_text(msg).strip():
                    self.logger.warning(f"分条 {i + 1} 清洗后为空，跳过该条: {msg!r}")
                    i += 1
                    continue
                success, err_code, err_msg = await self._send_reply(context, msg, metadata)
                if success:
                    sent_count += 1
                    self._record_sent(session_key, msg)
                    self.logger.debug(f"分条 {i + 1}/{n} 发送成功: {msg[:30]}...")
                    i += 1
                    # 规则 4: 拆分后的多条消息间隔 3~6 秒
                    if i < n:
                        await asyncio.sleep(random.uniform(self.split_interval_min, self.split_interval_max))
                    continue
                # 发送失败
                is_repeat_block = (err_code == 40013 and (not err_msg or "重复" in str(err_msg)))
                if is_repeat_block and rewrite_attempts < self._repeat_rewrite_max:
                    rewrite_attempts += 1
                    self.logger.warning(
                        f"分条 {i + 1} 被平台防重复拦截（40013），改写措辞重试（第 {rewrite_attempts} 次）"
                    )
                    remaining = messages[i:]
                    rewritten = await self._rewrite_messages(remaining, context)
                    if rewritten and rewritten != remaining:
                        # 用改写后的内容替换剩余分条，从当前位置重新发送（不前进 i）
                        messages = messages[:i] + rewritten
                        n = len(messages)
                        continue
                    self.logger.error("改写失败或内容未变，停止发送后续分条")
                    break
                self.logger.error(f"分条 {i + 1} 发送失败（error_code={err_code}），停止后续发送: {msg[:30]}...")
                break

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
        "多少", "几", "多久", "多大", "多长", "多宽", "几米", "米数", "几天", "几件",
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

    async def _simulate_human_delay(self, uid: Optional[str], already_elapsed_read: float = 0.0) -> None:
        """模拟真人已读+打字延迟（规则 1），并补齐同一买家回复间隔（规则 2）

        already_elapsed_read: 合并场景下窗口等待期间已消耗的已读秒数，予以扣除，
        使「已读延迟」与「合并窗口」并行而非叠加。
        """
        # 规则 1: 已读 6~8 秒 + 打字 4~6 秒 = 总计 10~14 秒
        read_sec = random.uniform(self.read_seconds_min, self.read_seconds_max)
        # 扣除窗口期已消耗的已读时间（如窗口 4 秒，则已读只剩 2~4 秒）
        read_sec = max(0.0, read_sec - already_elapsed_read)
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
            # 2026-09-05 修复：买家发"谢谢""好的""👌"等纯寒暄被 LLM 误判为 other → 强制改 consult。
            # 避免每句礼貌回应都被踢给真人（按 should_transfer 规则 other+置信度≥阈值会触发转人工）。
            result = self._force_consult_if_pure_thanks(processed_content, result)
        except Exception as e:
            self.logger.warning(f"意图分类异常，保守不转人工: {e}")
            return False

        intent = (result or {}).get("intent")
        confidence = float((result or {}).get("confidence", 0.0) or 0.0)
        self.logger.info(f"意图分类结果: intent={intent}, confidence={confidence}")

        if ic_module.IntentClassifier.should_transfer(intent, confidence, classifier.threshold):
            return await self._transfer_by_intent(
                context, metadata, processed_content, intent=intent
            )
        return False

    async def _transfer_by_intent(
        self,
        context: Context,
        metadata: Dict[str, Any],
        last_message: str,
        reason: str = "AI意图识别触发转人工",
        intent: str = "",
    ) -> bool:
        """执行意图触发的转人工（语义层），保留营业时间/子账号静默规则。

        Args:
            reason: 转人工原因，进入企业微信通知文案（如"AI无法回答→转人工"）。
            intent: 意图分类结果（operation/complaint/negative_emotion/other/unknown 等），
                用于选择转人工前的安抚话术类型。

        2026-09-05 新增：判定转人工的瞬间，先给买家发一句"核实/处理类"拖延话术
        （像真人客服在亲自帮忙查），避免人工客服未及时接入导致"3 分钟回复率"下降。
        绝不用"正在为您转接人工客服"类话术（会暴露 AI）。已转人工/非营业时间不发送。
        """
        shop_id = metadata.get('shop_id')
        user_id = metadata.get('user_id')
        from_uid = metadata.get('from_uid')
        shop_name = metadata.get('shop_name') or getattr(context.kwargs, 'shop_name', None) or ""
        if not all([shop_id, user_id, from_uid]):
            return False

        # 转人工前检查：若会话已处于转人工状态，不再重复标记/通知，避免真人介入后
        # 竞态路径（如发送前检测到 handoff 进入 fallback）再次发送企业微信通知。
        session_key = f"{shop_id}:{from_uid}"
        if SessionState().is_handoff(session_key):
            self.logger.info(
                f"会话已处于转人工状态，跳过重复转接/通知: "
                f"session_key={session_key}, reason={reason}"
            )
            return True

        # 2026-09-05：转人工瞬间先发一句"核实/处理类"安抚话术（像真人），
        # 保住三分钟回复率。失败不阻断转人工主流程。
        try:
            from Message.handlers.handoff_soother import (
                classify_category,
                pick_soother,
            )

            category = classify_category(intent=intent, reason=reason)
            soother = pick_soother(category, session_key=session_key)
            await self._send_reply(context, soother, metadata)
            self.logger.info(f"已发送转人工安抚话术: category={category}, text={soother!r}")
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"转人工安抚话术发送失败（不阻断转接）: {type(e).__name__}: {e}")

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
                transfer_conversation, params, reason, True, last_message
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
            "若条目与系统尺寸政策冲突（如出现「备注米数」「特殊长度可备注」「按需裁剪」等表述），"
            "一律视为条目笔误，以系统提示中的尺寸政策为准；",
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

    # ===== 规格感知米数计算（2026-09-05）=====
    # 从产品知识 specifications 解析商品基础规格，程序化判断买家目标米数能否组合，
    # 有解注入推荐拍法、无解注入"无法裁剪+最近可卖规格"，LLM 只复述不自行计算。

    _CN_METER_MAP = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                     "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

    @classmethod
    def _extract_target_meter(cls, text: str) -> Optional[float]:
        """从买家消息提取目标米数（排除"宽1.5米/幅宽1.5米"这类幅宽表述）。"""
        import re

        if not text:
            return None
        tokens: List[float] = []
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*米", text):
            prefix = text[max(0, m.start() - 2):m.start()]
            if prefix.endswith("宽") or prefix.endswith("幅宽"):
                continue
            try:
                tokens.append(float(m.group(1)))
            except ValueError:
                continue
        for m in re.finditer(r"([一二两三四五六七八九十])\s*米", text):
            prefix = text[max(0, m.start() - 2):m.start()]
            if prefix.endswith("宽") or prefix.endswith("幅宽"):
                continue
            tokens.append(float(cls._CN_METER_MAP[m.group(1)]))
        return tokens[-1] if tokens else None

    @staticmethod
    def _should_advise_meter(text: str, meter: Optional[float]) -> bool:
        """消息像在问"买/拍/裁 X 米"才触发规格计算（避免"幅宽1.5米"纯陈述误触发）。"""
        import re

        if meter is None:
            return False
        return bool(re.search(r"买|拍|裁|卖|要|来|能|可以|支持|够|需要|做", text or ""))

    async def _prefetch_meter_hint(self, metadata: Dict[str, Any], text: str, context: Context) -> str:
        """米数购买/裁剪意图 → 按商品规格程序化计算拍法（或不可裁结论），注入 LLM。"""
        from Message.handlers.meter_advisor import (
            DEFAULT_SPECS,
            build_meter_hint,
            parse_meter_values,
        )

        meter = self._extract_target_meter(text)
        if not self._should_advise_meter(text, meter):
            return ""

        shop_id = metadata.get("shop_id")
        product_name, specs = "", None

        def _load():
            from bridge.context import make_conversation_key
            from core.di_container import container
            from database.knowledge_service import KnowledgeService

            # 1) 会话历史最近商品卡片
            goods_id = None
            try:
                session_id = make_conversation_key(context)
                history = self.bot.get_session_history(session_id, limit=30) or []
                for msg in reversed(history):
                    m = re.search(r"商品ID[：:]\s*(\d+)", str((msg or {}).get("content", "")))
                    if m:
                        goods_id = int(m.group(1))
                        break
            except Exception as e:
                self.logger.debug(f"读会话历史商品卡片失败: {e}")

            ks = container.get(KnowledgeService)
            sid = int(shop_id) if str(shop_id).isdigit() else shop_id
            pk = ks.get_product_by_goods_id(sid, goods_id) if goods_id else None

            # 2) 商品名 2-gram 模糊匹配
            if pk is None:
                best, best_score = None, 0
                for cand in (ks.list_products_by_shop(sid) or []):
                    name = cand.goods_name or ""
                    score = sum(1 for i in range(len(name) - 1) if name[i:i + 2] in text)
                    if score > best_score:
                        best, best_score = cand, score
                pk = best
            return pk

        try:
            pk = await asyncio.to_thread(_load)
        except Exception as e:
            self.logger.warning(f"商品规格查询失败，按普通花型默认规格计算: {e}")
            pk = None

        if pk is not None:
            product_name = pk.goods_name or ""
            specs = parse_meter_values(getattr(pk, "specifications", None))
        if not specs:
            specs = DEFAULT_SPECS  # 无商品上下文 → 普通花型默认 1/2/3

        hint = build_meter_hint(float(meter), specs, product_name)
        self.logger.info(
            f"规格米数计算: meter={meter}, product={product_name or '(默认普通花型)'}, "
            f"hint={hint[:80]}..."
        )
        return hint

    async def _get_ai_reply(self, query: str, context: Context, order_hint: str = "", kb_hint: str = "", meter_hint: str = "") -> Optional[str]:
        """获取AI回复

        Args:
            query: 用户消息
            context: 上下文
        """
        if not self.bot:
            return None

        effective_query = query
        # 优先合并 KB 预取结果（售后政策等），再合并订单数据与规格米数计算；
        # 三者皆为系统已查证的「已知事实」，LLM 直接据此回复即可。
        if kb_hint:
            effective_query = f"{query}\n\n{kb_hint}"
        if meter_hint:
            effective_query = f"{effective_query}\n\n{meter_hint}"
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

    def _is_cannot_answer(self, reply: str) -> bool:
        """判断 LLM 回复是否为『答不上来』的推脱话术（需强制转人工）。"""
        if not reply:
            return False
        # 去掉空格与波浪号噪声，降低规避匹配的可能
        r = reply.replace(" ", "").replace("~", "").replace("～", "")
        return any(pat in r for pat in self._CANNOT_ANSWER_PATTERNS)

    # 纯感谢/确认短语白名单：买家说"谢谢""好的""👌"等纯礼貌回应时不应触发转人工。
    # 背景：2026-09-05 复盘发现 LLM 把"谢谢"判为 other 且置信度 0.8，按 should_transfer
    # 规则被转人工，导致正常对话被踢给真人客服。修复：识别纯感谢/确认短语并强制改 intent=consult。
    _PURE_THANKS_ACK_PHRASES = (
        "谢谢", "感谢", "谢啦", "谢了", "多谢", "tks", "thx", "3q", "thanks", "thank", "thankyou",
        "好的", "好", "行", "可以", "没问题", "可以的", "好呀", "好的呢", "好的呀", "好嘞",
        "嗯", "嗯嗯", "哦", "哦哦", "嗯哼", "嗯嗯嗯",
        "ok", "okay", "okk", "👌", "👍", "🙏",
        "收到", "了解", "明白了", "知道了", "懂了", "了解了解", "好的了解",
        "再见", "拜拜", "bye", "👋", "下次见",
        "辛苦了", "麻烦你了", "麻烦您了",
    )
    # 含售后词不算纯感谢（例："好的我退款""谢谢但要退货"必须按原始意图处理）
    _PURE_THANKS_BLOCKERS = (
        "退", "投诉", "差评", "骗", "假货", "质量差", "赔", "赔偿", "不满意", "过敏",
        "破损", "漏发", "少发", "换货", "维修", "气", "烂", "骗人", "纠纷",
    )

    @classmethod
    def _is_pure_thanks_or_ack(cls, text: str) -> bool:
        """消息是否只含感谢/确认/礼貌等纯寒暄内容（不含售后词）。

        用于意图分类后兜底：买家说"谢谢""好的""👌"等纯礼貌回应被 LLM 误判为 other
        时强制改 consult，避免每句寒暄都踢给真人。
        """
        if not text:
            return False
        import re

        t = text.strip().lower()
        # 含售后词则不算纯感谢
        if any(k in t for k in cls._PURE_THANKS_BLOCKERS):
            return False
        # 去掉常见标点/表情/空白，仅留汉字与字母数字
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", t)
        if not cleaned:
            # 纯表情/标点 → 视为纯寒暄（用户发个👍也应继续对话）
            return True
        # 全部由白名单短语拼接而成（允许重复 + 句末常见语气词 亲/哈/呀/呢/呗/啦/哦/了）
        ack_re = re.compile(
            r"^(?:"
            r"谢谢|感谢|谢啦|谢了|多谢|tks|thx|3q|thanks|thank|thankyou|"
            r"好的?|行|可以|没问题|好呀|好的呢|好的呀|好嘞|"
            r"嗯+|嗯哼|哦+|ok|okay|okk|"
            r"收到|了解|明白了?|知道了?|懂了|了解了解|好的了解|"
            r"再见|拜拜|bye|下次见|"
            r"辛苦了|麻烦你|麻烦您"
            r")+(?:亲|哈|呀|呢|呗|啦|哦|了)*$"
        )
        return bool(ack_re.match(cleaned))

    @classmethod
    def _force_consult_if_pure_thanks(
        cls, text: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """纯感谢/确认短语被 LLM 误判为 other → 强制改 consult，避免无谓转人工。"""
        if not result:
            return result or {"intent": "unknown", "confidence": 0.0}
        intent = result.get("intent")
        if intent == "other" and cls._is_pure_thanks_or_ack(text):
            return {"intent": "consult", "confidence": 0.95}
        return result

    # 成衣面料用量咨询：买家让估算做/买某件衣服需要多少米布。这类问题依赖版型/
    # 款式/体型，超出店铺客服能力，AI 不能估算，必须直接转人工（2026-09-05 复盘）。
    #
    # 命中规则：消息含「服装款式词」AND 含「面料/尺寸用量询问词」 → 转人工；
    # 或含「给小孩/孩子/宝宝 + 做/买 + 衣服」组合 → 转人工。
    # 用量词不要求伴随"做/缝制"等动作词（"推荐买多少米布""需要多少布"也算），
    # 因为本质上都是让 AI 推荐面料用量，超出客服能力。

    _GARMENT_KEYWORDS = (
        # 上装
        "衣服", "短袖", "长袖", "衬衫", "寸衫", "衬衣", "polo", "t恤", "卫衣",
        "外套", "大衣", "风衣", "西装", "套装", "马甲", "背心", "打底衫",
        "棉衣", "棉袄", "羽绒服", "毛衣", "针织衫", "夹克", "短外套",
        "上衣", "罩衫", "中衣", "吊带",
        # 下装
        "裤子", "长裤", "短裤", "打底裤", "背带",
        # 裙装
        "裙子", "连衣裙", "半裙", "长裙", "短裙",
        # 其他成衣
        "旗袍", "家居服", "睡衣", "内衣", "校服", "工作服", "围裙",
        "童装", "孕妇装", "宝宝装",
        # 泛称/口语
        "小孩衣服", "孩子衣服", "宝宝衣服", "衣裳",
    )
    _USAGE_KEYWORDS = (
        # 多少米/多少布
        "多少米", "几米", "多少布", "几米布", "多少布料",
        "多少尺寸", "多大尺寸", "尺寸多少",
        # 用量推荐
        "用多少", "买多少", "需要多少", "要多少", "需要用多少",
        "推荐多少", "推荐几",
        # 是否够（X 套/件 衣服够吗 / 9 米够做几套）
        "够吗", "够不够", "够用", "够做", "够裁", "够缝",
        "够不够用", "够用吗", "够做吗",
        # 怎么算
        "怎么算", "怎么买", "怎么量",
    )

    # 尺寸政策违禁词：分句内出现且无否定词，即视为向买家暗示可变通（2026-09-04 复盘）
    _SIZE_POLICY_FORBIDDEN = ("备注", "定制", "按需裁剪", "特殊长度", "特殊尺寸", "任选")
    _SIZE_POLICY_NEGATION = ("不", "没有", "无法", "别", "禁止", "仅", "只", "勿", "无须", "无需")

    # 错误拒售话术：买家询问整米长度时，模型不能说出这些推脱/拒绝表述（2026-09-05 复盘）
    _WRONG_REFUSAL_PATTERNS = (
        "面料长度固定",
        "不能修改",
        "不支持按您要求",
        "不支持这个长度",
    )

    def _violates_size_policy(self, reply: str) -> bool:
        """检测回复是否暗示可通过备注/定制获得规格外的尺寸或长度。

        按标点分句逐句判断：分句含违禁词且无否定词 → 违规。
        「不接受备注」「不卖特殊尺寸」等否定表述不算违规。
        背景：glm-4-flash 在 KB 未命中时会编造"有特殊长度需求可在订单备注说明"
        类话术（店铺明确不接受备注特殊米数），必须在发送前拦截。
        """
        if not reply:
            return False
        import re
        for seg in re.split(r"[。！？!?\n，,；;：:]", reply):
            s = seg.strip()
            if not s:
                continue
            if any(w in s for w in self._SIZE_POLICY_FORBIDDEN) and not any(
                n in s for n in self._SIZE_POLICY_NEGATION
            ):
                return True
        return False

    def _is_wrong_size_refusal(self, user_text: str, reply: str) -> bool:
        """检测模型是否错误拒绝可实现的整米长度。

        买家询问整米长度（如4米、5米、6米）时，正确答复应是"拍X个Y米组合"；
        若模型回复"面料长度固定""不能修改""不支持按您要求...裁剪"等推脱话术，
        视为错误拒售，需触发纠正。
        """
        if not user_text or not reply:
            return False
        import re

        if not re.search(r"(?:[1-9]\d*|[一二两三四五六七八九十]+)\s*米", user_text):
            return False
        r = reply.replace(" ", "").replace("~", "").replace("～", "")
        return any(pat in r for pat in self._WRONG_REFUSAL_PATTERNS)

    def _is_garment_fabric_usage_query(self, text: str) -> bool:
        """检测买家是否在问"做/买某件衣服需要多少米布"或推荐面料用量。

        此类问题需要结合版型、款式、个人胖瘦，店铺客服无法仅凭文字估算，
        必须直接转人工，避免 AI 瞎估导致买家误解（如"做一套衣服拍三米"）。

        命中规则（任一）：
        1. 消息含服装款式词 AND 含面料/尺寸用量询问词（2026-09-05 复盘扩量）：
           例如"做长袖要多少米布""短袖需要多少布""推荐几米布""1.5米能做几件衣服"。
        2. 含"给小孩/孩子/宝宝 + 做/买 + 衣服"组合（例如"给小孩做衣服"）。
        3. 数量/件套词 + 服装款式词（2026-09-05 复盘扩量）：不限定对象——
           大人/老人/小孩/不提对象都算。例："做3套衣服够吗""买2件大人的衣服"
           "做1套老人的衣服""做3件衬衫"——只要批量成衣问询，AI 无法估用量，转人工。
        """
        if not text:
            return False
        import re

        t = text.lower()
        has_garment = any(k in t for k in self._GARMENT_KEYWORDS)
        has_usage = any(k in t for k in self._USAGE_KEYWORDS)
        # 兜底 A："给小孩/孩子/宝宝 做/买 衣服" 不强求含用量词
        has_for_child = (
            any(k in t for k in ("给小孩", "给孩子", "给宝宝", "给闺女", "给儿子"))
            and any(k in t for k in ("做", "买", "衣服"))
        )
        # 兜底 B：数量/件套词 + 服装款式词（不限人群，如"做3套衣服""2件大人的衣服"）
        has_qty_garment = (
            re.search(r"(?:做|买|要|裁)?\s*\d+\s*[套件条个]", t) is not None
            and has_garment
        )
        return (has_garment and has_usage) or has_for_child or has_qty_garment

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

            # 规则 5: 校验发送结果与业务码
            ok, error_code, error_msg = self._check_send_result(result)
            if ok:
                # 规则 2: 记录本次发送时间，用于同一买家回复间隔控制
                self.uid_tracker.record_send(from_uid)
            return (ok, error_code, error_msg)

        except Exception as e:
            self.logger.error(
                f"发送回复失败: error_type={type(e).__name__}"
            )
            return False

    def _check_send_result(self, result):
        """规则 5: 校验发送结果，业务码非 0 视为失败。

        返回 (ok, error_code, error_msg)：
        - ok: 是否发送成功
        - error_code: 平台业务码（成功为 0；无结果时为 -1）
        - error_msg: 平台返回的错误描述（成功为 None）
        """
        if not result:
            return (False, -1, "empty_result")
        if isinstance(result, str):
            # send_text 在业务错误（如 error_code=10002）时返回错误文案字符串
            self.logger.error(f"发送失败（业务错误）: {result}")
            return (False, -1, result)
        if result.get("success"):
            inner = result.get("result", {}) or {}
            error_code = inner.get("error_code", 0)
            error_msg = inner.get("error")
            if error_code:
                self.logger.error(f"发送业务码非 0: error_code={error_code}, error={error_msg}, result={result}")
                return (False, error_code, error_msg)
            return (True, 0, None)
        self.logger.error(f"发送请求失败: {result}")
        return (False, -1, "request_failed")

    # ===== 平台防重复拦截（40013）应对 =====
    # 拼多多对「同一客服账号短时间内发给同一买家的相同消息」返回 40013
    # （error=请勿重复发送相同消息）。买家重复提问时 bot 会生成雷同答案被拦。
    # 处理原则：买家需要被回应，不能静默放弃、也不能无意义转人工占用人力。
    # 做法：检测重复 → 换措辞重写（保持事实不变）→ 重发；改写后仍失败才转人工兜底。

    def _record_sent(self, session_key: str, text: str) -> None:
        """记录本会话已成功发送的文本，供后续重复检测"""
        cleaned = self._clean_text(text).strip()
        if not cleaned:
            return
        buf = self._recent_sent.get(session_key)
        if buf is None:
            buf = deque(maxlen=self._recent_sent_max)
            self._recent_sent[session_key] = buf
        buf.append((time.time(), cleaned))

    def _is_recent_duplicate(self, session_key: str, text: str) -> bool:
        """判断 text 是否与本会话近期已发消息字面重复（平台 40013 的判定核心）"""
        cleaned = self._clean_text(text).strip()
        if not cleaned:
            return False
        buf = self._recent_sent.get(session_key)
        if not buf:
            return False
        now = time.time()
        for ts, prev in buf:
            if now - ts > self._recent_sent_ttl:
                continue
            if prev == cleaned:
                return True
        return False

    async def _rewrite_messages(self, messages: List[str], context: Context) -> List[str]:
        """把待发送消息用不同措辞重写（保持关键事实不变），用于绕过平台重复拦截。

        复用 bot.async_reply 的底层 LLM 通道，prompt 要求只换说法、不改事实、不调工具。
        """
        joined = "\n".join(m for m in messages if m.strip())
        if not joined.strip():
            return messages
        prompt = (
            "请把下面这几句客服回复用不同的措辞、不同的语序重新表达一遍，"
            "必须保持原有意思和所有关键事实（如尺寸、价格、政策）完全不变，"
            "只是换种自然的说法，不要新增或删减任何信息，不要调用任何工具。"
            "逐句输出，每句一行。\n\n"
            f"{joined}"
        )
        try:
            if not (self.bot and hasattr(self.bot, "async_reply")):
                return messages
            res = await self.bot.async_reply(prompt, context)
            content = getattr(res, "content", str(res)) or ""
            rewritten = [m.strip() for m in content.split("\n") if m.strip()]
            if not rewritten:
                return messages
            out = []
            for r in rewritten:
                out.extend(self._split_reply(r))
            return out if out else messages
        except Exception as e:
            self.logger.error(f"防重复改写失败: {e}")
            return messages

    async def _prededupe(self, messages: List[str], session_key: str, context: Context) -> List[str]:
        """发送前：若任一待发消息与近期已发重复，整批改写后返回（减少一次失败往返）"""
        if not any(self._is_recent_duplicate(session_key, m) for m in messages):
            return messages
        self.logger.warning(f"检测到待发消息与近期已发重复，发送前改写: session={session_key}")
        return await self._rewrite_messages(messages, context)

    async def _handle_fallback(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """AI 回复失败时的兜底处理。

        2026-09-05 起：转人工瞬间由 _transfer_by_intent 自动补发一条"核实/处理类"
        拖延话术（像真人客服在帮忙查），再静默转人工标记会话、通知企业微信人工客服
        接手。买家不会看到"系统错误/请稍后再试"等露馅话术，也不会长时间无人应答。
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
