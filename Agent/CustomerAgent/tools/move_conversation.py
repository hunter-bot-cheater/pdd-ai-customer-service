"""
会话转接工具

将当前会话转接给人工客服。
"""
from typing import Optional, Union
from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from bridge.sender import get_sender
from config import get_config
from core.session_state import SessionState
from utils.logger_loguru import get_logger

logger = get_logger("TransferConversationTool")


class TransferConversationParams(BaseModel):
    """会话转接参数"""
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")
    recipient_uid: Optional[Union[str, int]] = Field(default=None, description="接收转接的用户UID")
    shop_name: Optional[str] = Field(default=None, description="店铺名称")


def _is_main_account(shop_id, user_id) -> bool:
    """判断当前账号是否为主账号

    规则 7：子账号不调用转人工 API，仅静默标记会话并通知人工客服；
    主账号则尝试真实转移会话（转移失败也保持静默，不回复用户预设话术）。

    判定依据（运行时认证时自动写入 config.json）：
    1. transfer.sub_account_uids：子账号完整 UID 列表
       （cs_{shop_id}_{user_id}）——命中即子账号；
    2. 兼容旧配置 transfer.main_account_user_ids：user_id 或完整子账号
       UID 命中其中的子账号项，也判定为子账号；
    3. 未配置任何列表时，默认全部账号按主账号处理。
    """
    try:
        user_id_str = str(user_id)
        sub_uid = f"cs_{shop_id}_{user_id}"

        # 1. 子账号列表优先判定
        sub_uids = get_config("transfer.sub_account_uids", []) or []
        sub_set = {str(u) for u in sub_uids}
        if sub_uid in sub_set or user_id_str in sub_set:
            return False

        # 2. 兼容旧配置：main_account_user_ids 中的完整子账号 UID
        main_uids = get_config("transfer.main_account_user_ids", []) or []
        if main_uids:
            main_set = {str(u) for u in main_uids}
            if user_id_str in main_set:
                return True
            if sub_uid in main_set:
                return False
            # 未知账号：保守地按主账号处理，保证转人工流程可用
            return True
        return True
    except Exception as e:
        # 判断失败时保守地按主账号处理，保证转人工流程可用
        logger.error(f"判断主账号失败，默认按主账号处理: {e}")
        return True


def _mark_handoff(params: TransferConversationParams) -> None:
    """转接成功后标记会话为已转人工（失败仅记录日志，不影响主流程）"""
    try:
        session_key = f"{params.shop_id}:{str(params.recipient_uid)}"
        SessionState().mark_handoff(session_key)
        logger.info(f"已标记会话转人工状态: session_key={session_key}")
    except Exception as e:
        logger.error(f"标记转人工状态失败（不影响转接主流程）: {e}")


def _notify_handoff(params: TransferConversationParams, reason: str, last_message: str) -> None:
    """发送企业微信转人工通知（失败仅记录日志，不影响主流程）"""
    try:
        # 延迟导入，避免与 Message 包（handlers -> ai_handler）产生循环依赖
        from Message.handlers.notify import build_handoff_message, send_wechat_notification_sync
        message = build_handoff_message(
            shop_name=params.shop_name or "",
            buyer_uid=str(params.recipient_uid) if params.recipient_uid else "",
            reason=reason,
            last_message=last_message,
        )
        send_wechat_notification_sync(message)
    except Exception as e:
        logger.error(f"发送企业微信通知失败（不影响转接主流程）: {e}")


@agent_tool(
    name="transfer_conversation",
    description="将当前会话转接给人工客服。",
    param_model=TransferConversationParams,
    side_effect=True,
)
def transfer_conversation(
    params: TransferConversationParams,
    reason: str = "用户主动转人工",
    send_notification: bool = True,
    last_message: str = "",
) -> str:
    """
    将当前会话转接给人工客服。

    Args:
        params: 转接参数
        reason: 转人工原因（用于通知文案）
        send_notification: 是否发送企业微信通知
        last_message: 触发转人工的用户消息
    """
    try:
        if not all([params.shop_id, params.user_id, params.recipient_uid]):
            return "转接失败：缺少必要的会话信息"

        # 规则 7：子账号不调用转人工 API，仅静默标记会话 + 通知人工客服
        if not _is_main_account(params.shop_id, params.user_id):
            logger.info(
                f"子账号会话转人工（静默标记，不调用API）: "
                f"shop_id={params.shop_id}, user_id={params.user_id}, recipient_uid={params.recipient_uid}"
            )
            _mark_handoff(params)
            if send_notification:
                _notify_handoff(params, reason, last_message)
            return "会话转接成功"

        sender = get_sender()
        cs_list = sender.get_cs_list(str(params.shop_id), str(params.user_id))
        my_cs_uid = f"cs_{params.shop_id}_{params.user_id}"
        if cs_list and isinstance(cs_list, dict):
            # 过滤掉自己，不转接给自己
            available_cs_uids = [uid for uid in cs_list.keys() if uid != my_cs_uid]

            if available_cs_uids:
                # 选择第一个可用的客服
                cs_uid = available_cs_uids[0]
                # 转移会话
                transfer_result = sender.transfer_to_cs(str(params.shop_id), str(params.user_id), str(params.recipient_uid), cs_uid)

                if transfer_result and transfer_result.get('success'):
                    logger.info(f"会话转接成功: recipient_uid={params.recipient_uid}, to_cs_uid={cs_uid}")
                    # 转接成功后标记会话，防止 AI 抢答
                    _mark_handoff(params)
                    if send_notification:
                        _notify_handoff(params, reason, last_message)
                    return "会话转接成功"
                else:
                    logger.warning(f"会话转接失败: transfer_result={transfer_result}")
                    return "会话转接失败"
            else:
                logger.warning("会话转接失败: 当前无可用的人工客服")
                return "当前无可用的人工客服"
        logger.warning("会话转接失败：无法获取客服列表")
        return "会话转接失败：无法获取客服列表"

    except Exception as e:
        logger.error(f"转接过程中发生错误: {type(e).__name__}")
        return "转接失败，请稍后重试"
