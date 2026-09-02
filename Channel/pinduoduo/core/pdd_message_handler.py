"""Pinduoduo message conversion and account-scoped dispatch."""

from __future__ import annotations

import asyncio
import json

from bridge.context import ChannelType, Context, ContextType
from Channel.pinduoduo.pdd_message import PDDChatMessage
from config import get_config
from core.session_state import SessionState
from database import db_manager
from utils.config_updater import update_config_with_uid
from utils.logger_loguru import get_logger


class MessageHandlerMixin:
    async def _setup_message_consumer(
        self,
        queue_name: str,
        shop_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Create one consumer and one Agent for this PDDChannel/account."""
        from Message import handler_chain
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        from core.di_container import container, configure_standard_services

        try:
            existing_consumer = self.consumer_manager.get_consumer(queue_name)
            if existing_consumer is not None:
                await self.consumer_manager.stop_consumer(queue_name, remove=True)
                self.queue_manager.remove_queue(queue_name)

            consumer = self.consumer_manager.create_consumer(
                queue_name,
                max_concurrent=10,
            )
            if self._account_agent is None:
                if not container.is_registered(CustomerAgent):
                    configure_standard_services()
                self._account_agent = container.get(CustomerAgent)

            handlers = handler_chain(
                use_ai=True,
                business_hours=self.business_hours,
                bot=self._account_agent,
            )
            for handler in handlers:
                consumer.add_handler(handler)

            await self.consumer_manager.start_consumer(queue_name)
            self.logger.debug(f"message consumer started: {queue_name}")
        except Exception as exc:
            self.logger.error(
                f"message consumer setup failed: error_type={type(exc).__name__}"
            )
            raise

    async def _process_websocket_message(
        self,
        message: str,
        shop_id: str,
        user_id: str,
        username: str,
        queue_name: str,
    ) -> None:
        try:
            if not message or not message.strip():
                return

            message_data = json.loads(message)
            msg_type = message_data.get("message", {}).get("type", "unknown")
            self.logger.debug(
                f"received message: type={msg_type}, shop_id={shop_id}"
            )

            pdd_message = PDDChatMessage(message_data)
            context = await asyncio.to_thread(
                self._convert_to_context, pdd_message, shop_id, user_id, username
            )
            if not context:
                return

            if self._should_process_immediately(context):
                await self._handle_immediate_message(context, shop_id, user_id)
            elif self._should_queue_message(context):
                msg_id = await self.queue_manager.get_or_create_queue(queue_name).put(
                    context
                )
                self.logger.debug(
                    f"message queued: {queue_name}, ID: {msg_id}, type: {context.type}"
                )
        except json.JSONDecodeError:
            self.logger.error("invalid websocket JSON message")
        except Exception as exc:
            self.logger.error(
                f"websocket message handling failed: error_type={type(exc).__name__}"
            )

    def _should_process_immediately(self, context: Context) -> bool:
        return context.type in {
            ContextType.SYSTEM_STATUS,
            ContextType.AUTH,
            ContextType.WITHDRAW,
            ContextType.SYSTEM_HINT,
            ContextType.MALL_CS,
            ContextType.TRANSFER,
        }

    def _should_queue_message(self, context: Context) -> bool:
        return context.type in {
            ContextType.TEXT,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION,
            ContextType.GOODS_INQUIRY,
            ContextType.ORDER_INFO,
            ContextType.GOODS_CARD,
            ContextType.GOODS_SPEC,
        }

    async def _handle_immediate_message(
        self,
        context: Context,
        shop_id: str,
        user_id: str,
    ) -> None:
        """立即处理系统消息（认证/撤回/转接等，不进入 AI 队列）"""
        kwargs = context.kwargs
        username = getattr(kwargs, "username", None)
        recipient_uid = getattr(kwargs, "from_uid", None)
        if isinstance(kwargs, dict):
            username = username or kwargs.get("username")
            recipient_uid = recipient_uid or kwargs.get("from_uid")
        username = username or ""
        recipient_uid = recipient_uid or ""
        try:
            # 认证消息：捕获完整 UID 写入配置，无需创建 SendMessage
            if context.type == ContextType.AUTH:
                self._capture_auth_uid(context)
                return

            from Channel.pinduoduo.utils.API.send_message import SendMessage
            send_message = SendMessage(shop_id, user_id)
            if context.type == ContextType.WITHDRAW:
                self.logger.info(f"收到撤回消息: {context.content}")
                send_message.send_text(recipient_uid, "[玫瑰]")

            elif context.type == ContextType.SYSTEM_STATUS:
                self.logger.debug(f"系统状态消息: {context.content}")

            elif context.type == ContextType.SYSTEM_HINT:
                self.logger.info(f"系统提示: {context.content}")

            elif context.type == ContextType.MALL_CS:
                # 诊断：把客服侧的 from_uid / 昵称 / is_aut 一并打出，便于区分真人子账号与机器人
                _fk = context.kwargs
                _fuid = getattr(_fk, "from_uid", None) or ""
                _fnick = getattr(_fk, "nickname", None) or ""
                _faut = getattr(_fk, "is_aut", None)
                self.logger.info(
                    f"收到客服消息: from_uid={_fuid} nickname={_fnick} is_aut={_faut} content={context.content}"
                )
                # 人工客服（非本程序账号）主动回复买家 → 静默转人工：
                # 标记该买家会话在 config.handoff.valid_hours 时长内 AI 完全静默、
                # 不回复、也不发任何企业微信通知；过期后自动恢复自动回复。
                self._handle_human_agent_reply(context, shop_id, user_id)

            elif context.type == ContextType.SYSTEM_BIZ:
                self.logger.info(f"系统业务消息: {context.content}")

            elif context.type == ContextType.MALL_SYSTEM_MSG:
                self.logger.info(f"商城系统消息: {context.content}")

            elif context.type == ContextType.TRANSFER:
                self.logger.info(f"转接消息: {context.content}")
                send_message.send_text(recipient_uid, "[玫瑰]")

        except Exception as e:
            self.logger.error(f"立即处理消息失败: {e}")

    def _handle_human_agent_reply(
        self, context: Context, shop_id: str, user_id: str
    ) -> None:
        """检测人工客服（非本程序账号）主动回复买家，静默标记该会话转人工。

        平台会把人工客服发出的消息以 MALL_CS 角色推回程序；本程序自己通过
        SendMessage 发出的回复也可能以相同角色回灌（from_uid 为本账号
        cs_{shop_id}_{user_id}）。必须过滤掉本账号自身的消息，否则 AI 会被自己的
        回复误判为"人工介入"而停手。

        命中后调用 SessionState.mark_handoff：在 config.handoff.valid_hours 时长内，
        该买家会话 AI 完全静默、不回复，也不发任何企业微信通知；到期后自动恢复。
        """
        kwargs = context.kwargs
        from_uid = getattr(kwargs, "from_uid", None) or ""
        to_uid = getattr(kwargs, "to_uid", None) or ""
        is_aut = getattr(kwargs, "is_aut", None)
        cs_uid = getattr(kwargs, "cs_uid", None) or ""
        template_name = getattr(kwargs, "template_name", None) or ""

        # 过滤本账号（主账号 / 子账号）自身发出的消息，避免误标转人工
        own_uids = {
            str(u)
            for u in (get_config("transfer.sub_account_uids", []) or [])
        }
        own_uids |= {
            str(u)
            for u in (get_config("transfer.main_account_user_ids", []) or [])
        }
        own_uids.add(f"cs_{shop_id}_{user_id}")
        self.logger.debug(
            f"人工介入判定: from_uid={from_uid} to_uid={to_uid} is_aut={is_aut} "
            f"cs_uid={cs_uid} template_name={template_name} own_uids={sorted(own_uids)}"
        )
        if from_uid and from_uid in own_uids:
            self.logger.debug(
                f"收到本账号发出的客服消息（from_uid={from_uid}），忽略，不标记转人工"
            )
            return

        # 区分「真人客服子账号」与「店铺机器人(店小蜜/平台自动回复)」：
        # 依据真实报文（2026-09-02 抓取两条 mall_cs 消息对比）：
        #   - 真人子账号发言：from.cs_uid 存在（子账号唯一标识，如 YAEOB4MY...），
        #     且 message 无 template_name。
        #   - 店铺机器人发言：from 无 cs_uid（只有 csid/mall_id/role/uid），
        #     且 message.template_name == "mall_robot_text_msg"（含 robot 字样）。
        # 结论：有 cs_uid => 真人，应标记转人工让 AI 静默；无 cs_uid 或
        # 模板名为 robot => 机器人，AI 继续自动回复，不静默。
        is_robot = (not cs_uid) or ("robot" in template_name.lower())
        if is_robot:
            self.logger.info(
                f"收到店铺机器人/店小蜜消息（cs_uid={cs_uid!r} template_name={template_name!r}），"
                f"不标记转人工，AI 继续自动回复"
            )
            return

        if not to_uid:
            self.logger.debug("无法识别买家 UID，跳过人工介入静默标记")
            return

        session_key = f"{shop_id}:{to_uid}"
        try:
            SessionState().mark_handoff(session_key)
            self.logger.info(
                f"检测到人工客服主动回复，会话静默转人工（不再回复且不通知）: "
                f"session_key={session_key}"
            )
        except Exception as e:
            self.logger.error(f"人工介入静默标记失败（不影响主流程）: {e}")

    def _capture_auth_uid(self, context: Context) -> None:
        """认证成功后捕获完整 UID 并写入 config.json

        需求：运行时从 WebSocket 认证响应中获取完整 UID
        （子账号如 'cs_661962391_189109418'，主账号为纯数字），
        自动写入 transfer.main_account_user_ids 与 transfer.sub_account_uids，
        供主/子账号判定与转人工流程使用。
        """
        # _convert_to_context 已将认证 dict 序列化为 JSON 字符串
        try:
            auth_info = json.loads(context.content or "")
        except (json.JSONDecodeError, TypeError):
            self.logger.warning("认证消息格式异常，跳过 UID 捕获")
            return
        if not isinstance(auth_info, dict):
            self.logger.warning("认证消息格式异常，跳过 UID 捕获")
            return

        result = auth_info.get('result')
        if result != 'ok':
            self.logger.warning(f"{context.kwargs.username}认证失败")
            return
        self.logger.info(f"{context.kwargs.username}认证成功")

        uid = auth_info.get('uid')
        if not uid:
            self.logger.warning("认证响应缺少 uid，跳过 UID 捕获")
            return
        update_config_with_uid(str(uid))

    def _convert_to_context(self, pdd_message: PDDChatMessage, shop_id: str, user_id: str, username: str) -> Context:
        """将拼多多消息转换为Context格式"""
        shop_info = db_manager.get_shop(self.channel_name, shop_id) or {}
        shop_name = shop_info.get("shop_name", "")
        content = pdd_message.content
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        return Context.create_pinduoduo_context(
            content=content,
            msg_id=str(pdd_message.msg_id) if pdd_message.msg_id is not None else "",
            from_user=str(pdd_message.from_user or ""),
            from_uid=str(pdd_message.from_uid or ""),
            to_user=str(pdd_message.to_user or ""),
            to_uid=str(pdd_message.to_uid or ""),
            nickname=str(pdd_message.nickname or ""),
            is_aut=pdd_message.is_aut,
            cs_uid=pdd_message.cs_uid,
            template_name=pdd_message.template_name,
            timestamp=pdd_message.timestamp,
            user_msg_type=pdd_message.user_msg_type,
            shop_id=str(shop_id),
            user_id=str(user_id),
            username=str(username),
            shop_name=str(shop_name),
            raw_data=pdd_message.raw_data,
            channel_type=ChannelType.PINDUODUO,
        )


__all__ = ["MessageHandlerMixin"]
