"""
拼多多消息处理类
"""
from bridge.context import  ContextType
from Message.message import ChatMessage
from enum import IntEnum
import json
from typing import Any, Dict, Optional
from utils.logger_loguru import get_logger

class PDDMsgType(IntEnum):
    """拼多多消息类型枚举"""
    TEXT = 0
    IMAGE = 1
    VIDEO = 14
    WITHDRAW = 1002
    EMOTION = 5
    GOODS_SPEC = 64
    TRANSFER = 24


class PDDSubType(IntEnum):
    """拼多多消息子类型枚举"""
    ORDER_INFO = 1
    GOODS_INQUIRY = 0


def _safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """安全获取嵌套字典值，避免链式get()时中间值为None导致AttributeError"""
    result = data
    for key in keys:
        if not isinstance(result, dict):
            return default
        result = result.get(key)
        if result is None:
            return default
    return result


class BaseMessageHandler:
    def __init__(self, msg):
        self.msg = msg
        self.data = msg.get("message",{})
    def get_basic_info(self):
        """获取基础信息"""
        return {
            "msg_id": self.data.get("msg_id"),
            "nickname": self.data.get("nickname"),
            "from_role": self.data.get("from",{}).get("role"),
            "from_uid": self.data.get("from",{}).get("uid"),
            "to_role": self.data.get("to",{}).get("role"),
            "to_uid": self.data.get("to",{}).get("uid"),
            "timestamp": self.data.get("time"),
            "is_aut": self.data.get("is_aut"),
            # 区分真人客服子账号与店铺机器人（店小蜜）的关键字段：
            # 真人子账号发言时 from.cs_uid 存在（子账号唯一标识），机器人消息无此字段。
            "cs_uid": self.data.get("from",{}).get("cs_uid"),
            # 机器人模板消息标识，如 mall_robot_text_msg；真人普通消息无此字段。
            "template_name": self.data.get("template_name"),
        }

        
class MessageTypeHandler:
    """消息类型处理类"""

    @staticmethod
    def _get_content(msg_data: Dict[str, Any], context_type: ContextType, path: tuple) -> tuple:
        """通用内容提取"""
        return context_type, _safe_get(msg_data, *path)

    @staticmethod
    def handle_text(msg_data):
        """处理文本消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.TEXT, ("message", "content"))

    @staticmethod
    def handle_image(msg_data):
        """处理图片消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.IMAGE, ("message", "content"))

    @staticmethod
    def handle_video(msg_data):
        """处理视频消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.VIDEO, ("message", "content"))

    @staticmethod
    def handle_emotion(msg_data):
        """处理表情消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.EMOTION, ("info", "description"))

    @staticmethod
    def handle_withdraw(msg_data):
        """处理撤回消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.WITHDRAW, ("info", "withdraw_hint"))

    @staticmethod
    def handle_goods_inquiry(msg_data):
        """处理商品咨询消息"""
        goods_info = {
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "goods_price": _safe_get(msg_data, "message", "info", "goodsPrice"),
            "goods_thumb_url": _safe_get(msg_data, "message", "info", "goodsThumbUrl"),
            "link_url": _safe_get(msg_data, "message", "info", "linkUrl"),
        }
        return ContextType.GOODS_INQUIRY,goods_info

    @staticmethod
    def handle_goods_spec(msg_data):
        """咨询商品规格"""
        goods_info = {
            "goods_id": _safe_get(msg_data, "message", "info", "data", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "data", "goodsName"),
            "goods_price": _safe_get(msg_data, "message", "info", "data", "goodsPrice"),
            "goods_spec": _safe_get(msg_data, "message", "info", "data", "spec"),
        }
        return ContextType.GOODS_SPEC,goods_info

    @staticmethod
    def handle_order_info(msg_data):
        """处理订单信息消息"""
        order_info = {
            "order_id": _safe_get(msg_data, "message", "info", "orderSequenceNo"),
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "afterSalesStatus": _safe_get(msg_data, "message", "info", "afterSalesStatus"),
            "afterSalesType": _safe_get(msg_data, "message", "info", "afterSalesType"),
            "spec": _safe_get(msg_data, "message", "info", "spec"),
        }
        return ContextType.ORDER_INFO,order_info

    @staticmethod
    def handle_mall_system_msg(msg_data):
        """处理商城消息"""
        system_msg = {
            "user_id": _safe_get(msg_data, "message", "data", "user_id"),
        }
        return ContextType.MALL_SYSTEM_MSG,system_msg


    @staticmethod
    def handle_auth(msg_data):
        """处理认证消息"""
        auth_info = {
            "uid": _safe_get(msg_data, "uid"),
            "result": _safe_get(msg_data, "auth", "result"),
            "status": _safe_get(msg_data, "status"),
        }
        return ContextType.AUTH,auth_info

    @staticmethod
    def handle_transfer(msg_data):
        """处理转接消息"""
        transfer_info = {
            "from_uid": _safe_get(msg_data, "message", "from", "uid"),
            "to_uid": _safe_get(msg_data, "message", "to", "uid")
        }
        return ContextType.TRANSFER,transfer_info
class PDDChatMessage(ChatMessage):
    """拼多多消息实现类"""
    
    def __init__(self, msg):
        super().__init__(msg)
        self.logger = get_logger("PDDChatMessage")
        self.msg = msg
        self.base_handler = BaseMessageHandler(msg)
        #获取基本信息
        basic_info = self.base_handler.get_basic_info()
        self.msg_id = basic_info.get("msg_id")
        self.nickname = basic_info.get("nickname")
        self.from_user = basic_info.get("from_role")
        self.from_uid = basic_info.get("from_uid")
        self.to_user = basic_info.get("to_role")
        self.to_uid = basic_info.get("to_uid")
        self.is_aut = basic_info.get("is_aut")
        # 区分真人子账号与机器人
        self.cs_uid = basic_info.get("cs_uid")
        self.template_name = basic_info.get("template_name")

        # 检查是否非用户消息
        if self.from_user == "mall_cs":
            self.user_msg_type = ContextType.MALL_CS
            self.content = self.msg.get("message",{}).get("content")
            # === 诊断：打印客服侧原始报文，确认 from.uid / nickname / is_aut，
            #     用于区分真人客服子账号与店铺机器人(店小蜜)。纯日志，不改业务逻辑。 ===
            self.logger.info(
                "=== 客服侧(MALL_CS)原始消息 JSON(用于区分人/机器人) ===\n"
                + json.dumps(self.msg, ensure_ascii=False, indent=2)
            )
            return
        # 处理消息
        self._process_message()
        
    def _process_message(self):
        # ===== 新增：打印原始消息用于调试 =====
        import json
        # 强制输出，使用 INFO 级别确保能看到
        self.logger.info(f"=== 原始消息完整 JSON ===\n{json.dumps(self.msg, ensure_ascii=False, indent=2)}")
        """处理消息"""
        self.msg_type = (
                self.msg.get("response") or
                self.msg.get("event") or
                self.msg.get("cmd") or
                self.msg.get("type")
        )
        if self.msg_type == "push":
            message_obj = self.msg.get("message", {})
            user_msg_type = self.msg.get("message", {}).get("type")
            if user_msg_type == PDDMsgType.TEXT:
                sub_type=self.msg.get("message",{}).get("sub_type")
                if sub_type == PDDSubType.ORDER_INFO:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_order_info(self.msg)
                elif sub_type == PDDSubType.GOODS_INQUIRY:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_goods_inquiry(self.msg)
                else:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_text(self.msg)
            elif user_msg_type == PDDMsgType.IMAGE:
                self.user_msg_type,self.content = MessageTypeHandler.handle_image(self.msg)
            elif user_msg_type == PDDMsgType.VIDEO:
                self.user_msg_type,self.content = MessageTypeHandler.handle_video(self.msg)
            elif user_msg_type == PDDMsgType.WITHDRAW:
                self.user_msg_type,self.content = MessageTypeHandler.handle_withdraw(self.msg)
            elif user_msg_type == PDDMsgType.EMOTION:
                self.user_msg_type,self.content = MessageTypeHandler.handle_emotion(self.msg)
            elif user_msg_type == PDDMsgType.GOODS_SPEC:
                self.user_msg_type,self.content = MessageTypeHandler.handle_goods_spec(self.msg)
            elif user_msg_type == PDDMsgType.TRANSFER:
                self.user_msg_type,self.content = MessageTypeHandler.handle_transfer(self.msg)
            else:
                self.user_msg_type = ContextType.SYSTEM_STATUS
                self.content = f"不支持的消息类型: {user_msg_type}, 完整消息: {self.msg}"
                self.logger.warning(self.content)
        elif self.msg_type == "auth":
            self.user_msg_type,self.content = MessageTypeHandler.handle_auth(self.msg)
        elif self.msg_type == "mall_system_msg":
            self.user_msg_type,self.content = MessageTypeHandler.handle_mall_system_msg(self.msg)
        else:
            self.user_msg_type = ContextType.SYSTEM_STATUS
            self.content = f"不支持的消息类型: {self.msg_type}"
