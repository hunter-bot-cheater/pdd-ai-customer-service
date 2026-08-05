"""
企业微信群机器人通知

转人工时通过企业微信群机器人 Webhook 发送通知，
提醒人工客服及时处理。通知失败仅记录日志，不影响主流程。
"""
import asyncio
from datetime import datetime
from typing import Optional

import requests

from config import get_config
from utils.logger_loguru import get_logger

logger = get_logger("WechatNotify")


def _get_webhook() -> str:
    """读取企业微信群机器人 Webhook 配置"""
    return get_config("notification.wechat_webhook", "") or ""


def build_handoff_message(
    shop_name: str = "",
    buyer_uid: str = "",
    reason: str = "用户主动转人工",
    last_message: str = "",
    timestamp: Optional[datetime] = None,
) -> str:
    """构建转人工通知消息文本"""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "【Agent-Customer 转人工通知】",
        f"店铺：{shop_name or '未知'}",
        f"买家ID：{buyer_uid or '未知'}",
        f"触发原因：{reason}",
        f"用户消息：{last_message or '（无文本内容）'}",
        f"时间：{ts}",
        "请及时处理！",
    ]
    return "\n".join(lines)


def _post_wechat_sync(message: str) -> bool:
    """同步发送企业微信文本消息（在调用线程内执行，含网络超时）"""
    webhook = _get_webhook()
    if not webhook:
        logger.warning("未配置企业微信 Webhook (notification.wechat_webhook)，跳过通知")
        return False

    payload = {"msgtype": "text", "text": {"content": message}}
    try:
        resp = requests.post(webhook, json=payload, timeout=5)
        result = resp.json()
        if resp.status_code == 200 and result.get("errcode") == 0:
            logger.debug(f"企业微信通知发送成功: {result}")
            return True
        logger.error(f"企业微信通知发送失败: status={resp.status_code}, resp={result}")
        return False
    except Exception as e:
        logger.error(f"企业微信通知发送异常: {e}")
        return False


async def async_send_wechat_notification(message: str) -> bool:
    """异步发送企业微信通知，内部放入工作线程避免阻塞事件循环"""
    return await asyncio.to_thread(_post_wechat_sync, message)


def send_wechat_notification_sync(message: str) -> bool:
    """同步发送企业微信通知（供同步工具内部调用）"""
    return _post_wechat_sync(message)
