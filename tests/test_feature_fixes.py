"""
四个核心功能修复的验证测试

覆盖：
1. 转人工后 AI 继续抢答 → 会话状态管理（4小时有效期），有效期内再次触发转人工
2. 售后关键词自动转人工 → 售后词库命中后强制转人工并拦截
3. 库存查询调用工具 → 系统提示词强制调用 get_shop_products + 库存输出
4. 转人工通知到企业微信群 → Webhook 通知格式与发送

运行方式（在项目根目录）：
    .venv/Scripts/python.exe -m unittest tests.test_feature_fixes -v
"""
import asyncio
import json
import threading
import time
import unittest
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

warnings.filterwarnings("ignore")

# ============================================================================
# 初始化：按 app.py 顺序注册标准服务（工具模块导入依赖 DI 容器）
# ============================================================================
from config import config as _app_config
from core.di_container import configure_standard_services
configure_standard_services(_app_config)

from core.session_state import SessionState, session_state
from Agent.CustomerAgent.tools import move_conversation as mc_module
from Agent.CustomerAgent.tools.move_conversation import (
    transfer_conversation,
    TransferConversationParams,
)
from Message.handlers.ai_handler import AIReplyHandler
from Message.handlers import keyword_handler as kh_module
from Message.handlers.keyword_handler import KeywordDetectionHandler
from Message.handlers import notify as notify_module
from Message.handlers.notify import (
    build_handoff_message,
    async_send_wechat_notification,
)
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.tools.get_product_list import _format_products_output
from Agent.CustomerAgent.custom.tool_decorator import TOOL_REGISTRY

from bridge.context import Context, ContextType, ChannelType
from bridge.reply import Reply, ReplyType


# ============================================================================
# 测试替身
# ============================================================================

class FakeSender:
    """模拟 PinduoduoSender，记录调用不发起真实网络请求"""

    def __init__(self, cs_list=None, success=True):
        self.cs_list = cs_list or {
            "cs_shop1_user1": {"username": "客服1"},
            "cs_shop1_user2": {"username": "客服2"},
        }
        self.success = success
        self.calls = {"transfer_to_cs": [], "send_text": []}

    def get_cs_list(self, shop_id, user_id):
        return self.cs_list

    def transfer_to_cs(self, shop_id, user_id, recipient_uid, cs_uid):
        self.calls["transfer_to_cs"].append((shop_id, user_id, recipient_uid, cs_uid))
        return {"success": self.success}

    def send_text(self, shop_id, user_id, recipient_uid, text):
        self.calls["send_text"].append((shop_id, user_id, recipient_uid, text))
        return {"success": True}


class MockBot:
    """记录是否被调用的模拟 Bot"""

    def __init__(self):
        self.calls = []

    async def async_reply(self, query, context=None):
        self.calls.append((query, context))
        return Reply(ReplyType.TEXT, "好的，亲")


class CaptureWebhook:
    """记录企业微信通知内容的替身（替换真实发送）"""

    def __init__(self):
        self.messages = []

    def send(self, message: str) -> bool:
        self.messages.append(message)
        return True


class _WebhookHTTPServer(BaseHTTPRequestHandler):
    """本地 Webhook 服务器，用于验证真实 POST 请求格式"""

    last_payload = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _WebhookHTTPServer.last_payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"errcode": 0, "errmsg": "ok"}).encode())

    def log_message(self, *args):
        pass


def make_context(content, shop_id="shop1", user_id="user1", from_uid="buyer1", shop_name="峰哥编织"):
    return Context.create_pinduoduo_context(
        content=content,
        from_uid=from_uid,
        user_id=user_id,
        shop_id=shop_id,
        shop_name=shop_name,
        user_msg_type=ContextType.TEXT,
        channel_type=ChannelType.PINDUODUO,
    )


def make_metadata(shop_id="shop1", user_id="user1", from_uid="buyer1", shop_name="峰哥编织"):
    return {
        "shop_id": shop_id,
        "user_id": user_id,
        "from_uid": from_uid,
        "shop_name": shop_name,
        "user_key": "pinduoduo_buyer1",
    }


# ============================================================================
# 测试 1：会话状态管理器
# ============================================================================

class TestSessionState(unittest.TestCase):
    def setUp(self):
        self.s = SessionState()
        self.s.clear_handoff("1:buyer1")

    def test_singleton(self):
        self.assertIs(SessionState(), SessionState())
        self.assertIs(SessionState(), session_state)

    def test_mark_and_check(self):
        self.s.mark_handoff("1:buyer1")
        self.assertTrue(self.s.is_handoff("1:buyer1"))

    def test_unknown_session_not_handoff(self):
        self.assertFalse(self.s.is_handoff("999:unknown"))

    def test_clear_handoff(self):
        self.s.mark_handoff("1:buyer1")
        self.s.clear_handoff("1:buyer1")
        self.assertFalse(self.s.is_handoff("1:buyer1"))

    def test_expiry(self):
        self.s.mark_handoff("1:buyer1", ttl_seconds=0.1)
        self.assertTrue(self.s.is_handoff("1:buyer1"))
        time.sleep(0.2)
        self.assertFalse(self.s.is_handoff("1:buyer1"))

    def test_thread_safety(self):
        errors = []

        def worker(i):
            try:
                for _ in range(50):
                    key = f"t:buyer{i % 5}"
                    self.s.mark_handoff(key, ttl_seconds=1)
                    self.s.is_handoff(key)
                    if i % 2 == 0:
                        self.s.clear_handoff(key)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


# ============================================================================
# 测试 2：转人工工具标记会话 + 发送通知
# ============================================================================

class TestTransferConversation(unittest.TestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        self.capture = CaptureWebhook()
        self.sender = FakeSender()
        self._sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._sender_patch.start()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()

    def tearDown(self):
        self._sender_patch.stop()
        self._notify_patch.stop()

    def test_success_marks_session_and_notifies(self):
        params = TransferConversationParams(
            shop_id="shop1", user_id="user1", recipient_uid="buyer1", shop_name="峰哥编织",
        )
        result = transfer_conversation(params)
        self.assertEqual(result, "会话转接成功")
        # 会话被标记
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 已发送通知
        self.assertEqual(len(self.capture.messages), 1)
        msg = self.capture.messages[0]
        self.assertIn("峰哥编织", msg)
        self.assertIn("buyer1", msg)
        self.assertIn("用户主动转人工", msg)
        # 转接调用发生
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)

    def test_failure_does_not_mark_session(self):
        sender = FakeSender(success=False)
        mc_module.get_sender = lambda: sender  # noqa: E731
        params = TransferConversationParams(
            shop_id="shop1", user_id="user1", recipient_uid="buyer1",
        )
        result = transfer_conversation(params)
        self.assertEqual(result, "会话转接失败")
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))

    def test_no_available_cs(self):
        sender = FakeSender(cs_list={"cs_shop1_user1": {"username": "客服1"}})
        mc_module.get_sender = lambda: sender  # noqa: E731
        params = TransferConversationParams(
            shop_id="shop1", user_id="user1", recipient_uid="buyer1",
        )
        result = transfer_conversation(params)
        self.assertEqual(result, "当前无可用的人工客服")
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))

    def test_missing_params(self):
        params = TransferConversationParams(shop_id="shop1")
        result = transfer_conversation(params)
        self.assertIn("缺少必要的会话信息", result)


# ============================================================================
# 测试 3：AI 处理器 —— 转人工有效期内不抢答，并重新触发转人工
# ============================================================================

class TestAIHandlerHandoff(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        self.bot = MockBot()
        self.handler = AIReplyHandler(bot=self.bot)
        self.sender = FakeSender()
        self._sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._sender_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()

    def tearDown(self):
        self._sender_patch.stop()
        self._notify_patch.stop()

    async def test_handoff_active_skips_ai_and_retriggers(self):
        SessionState().mark_handoff("shop1:buyer1")
        context = make_context("在吗，请问发货了吗")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # AI 未被调用（不抢答）
        self.assertEqual(self.bot.calls, [])
        # 再次触发转人工
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        # 再次通知人工，原因标注"有效期内再次发消息"
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("有效期内再次发消息", self.capture.messages[0])
        self.assertIn("在吗，请问发货了吗", self.capture.messages[0])

    async def test_handoff_expired_ai_replies_normally(self):
        # 已过期（标记 0.1 秒）
        SessionState().mark_handoff("shop1:buyer1", ttl_seconds=0.1)
        time.sleep(0.2)
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("你好")
            ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # AI 正常回复
        self.assertEqual(len(self.bot.calls), 1)
        # 未重复转人工
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])

    async def test_no_handoff_ai_replies_normally(self):
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("你好")
            ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertEqual(len(self.bot.calls), 1)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])


# ============================================================================
# 测试 4：售后关键词触发转人工
# ============================================================================

class TestKeywordHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        self.sender = FakeSender()
        self._mc_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._mc_patch.start()
        self._kh_patch = mock.patch.object(kh_module, "get_sender", return_value=self.sender)
        self._kh_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()
        self.handler = KeywordDetectionHandler()

    def tearDown(self):
        self._mc_patch.stop()
        self._kh_patch.stop()
        self._notify_patch.stop()

    def test_after_sale_keywords_defined(self):
        for kw in ["退货", "退款", "售后", "质量问题", "破损", "漏发", "少发",
                   "不满意", "投诉", "赔偿", "换货", "维修", "差评", "给差评",
                   "假货", "质量差"]:
            self.assertIn(kw, KeywordDetectionHandler.AFTER_SALE_KEYWORDS)

    def test_can_handle_after_sale(self):
        context = make_context("我要退款")
        self.assertTrue(self.handler.can_handle(context))

    def test_can_handle_regular_keyword(self):
        context = make_context("转人工")
        self.assertTrue(self.handler.can_handle(context))

    def test_cannot_handle_normal_message(self):
        context = make_context("这件衣服多大码")
        self.assertFalse(self.handler.can_handle(context))

    async def test_after_sale_triggers_transfer_and_blocks(self):
        context = make_context("我要退款，商品有质量问题")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # 转人工被触发
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        # 会话被标记
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 通知原因
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("售后关键词触发转人工", self.capture.messages[0])
        self.assertIn("我要退款，商品有质量问题", self.capture.messages[0])

    async def test_after_sale_failure_still_blocks(self):
        # 无可用客服 → 转人工失败，但售后场景仍返回 True（拦截，避免 AI 回复敏感问题）
        sender = FakeSender(cs_list={"cs_shop1_user1": {"username": "客服1"}})
        mc_module.get_sender = lambda: sender  # noqa: E731
        kh_module.get_sender = lambda channel_type: sender
        context = make_context("我要投诉")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))
        # 无可用客服时向用户说明（保持原有体验）
        self.assertIn("当前没有其他客服在线", sender.calls["send_text"][0][3])


# ============================================================================
# 测试 5：库存查询调用工具（系统提示词 + 库存输出）
# ============================================================================

class TestInventory(unittest.TestCase):
    def test_prompt_has_mandatory_inventory_instruction(self):
        builder = MessageBuilder(instructions=["test"])
        prompt = builder.system_prompt
        self.assertIn("get_shop_products", prompt)
        self.assertIn("库存", prompt)
        self.assertIn("有货吗", prompt)
        # 强制规则 + 兜底文案 + 严禁编造
        self.assertIn("必须调用 get_shop_products", prompt)
        self.assertIn("亲，当前库存信息暂无法实时查询，建议您在商品详情页查看实时库存。", prompt)
        self.assertIn("严禁编造", prompt)

    def test_get_shop_products_tool_registered(self):
        self.assertIn("get_shop_products", TOOL_REGISTRY)

    def test_product_list_output_includes_stock(self):
        products = [{
            "goods_id": 1001,
            "goods_name": "测试商品",
            "price": "10.00",
            "quantity": 42,
        }]
        out = _format_products_output(products, total=1, page=1)
        self.assertIn("库存: 42 件", out)
        self.assertIn("测试商品", out)

    def test_product_list_output_without_stock(self):
        products = [{"goods_id": 1001, "goods_name": "测试商品", "price": "10.00"}]
        out = _format_products_output(products, total=1, page=1)
        self.assertNotIn("库存:", out)


# ============================================================================
# 测试 6：企业微信群机器人通知
# ============================================================================

class TestWechatNotify(unittest.TestCase):
    def test_build_handoff_message_format(self):
        msg = build_handoff_message(
            shop_name="峰哥编织",
            buyer_uid="buyer1",
            reason="售后关键词触发转人工",
            last_message="我要退款",
        )
        self.assertIn("【Agent-Customer 转人工通知】", msg)
        self.assertIn("店铺：峰哥编织", msg)
        self.assertIn("买家ID：buyer1", msg)
        self.assertIn("触发原因：售后关键词触发转人工", msg)
        self.assertIn("用户消息：我要退款", msg)
        self.assertIn("请及时处理！", msg)

    def test_async_send_wechat_notification_posts_to_webhook(self):
        server = HTTPServer(("127.0.0.1", 0), _WebhookHTTPServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/cgi-bin/webhook/send"
        try:
            with mock.patch.object(notify_module, "_get_webhook", return_value=url):
                result = asyncio.run(async_send_wechat_notification("hello wechat test"))
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(result)
        payload = _WebhookHTTPServer.last_payload
        self.assertIsNotNone(payload)
        self.assertEqual(payload["msgtype"], "text")
        self.assertEqual(payload["text"]["content"], "hello wechat test")

    def test_no_webhook_returns_false_without_error(self):
        with mock.patch.object(notify_module, "_get_webhook", return_value=""):
            result = asyncio.run(async_send_wechat_notification("x"))
        self.assertFalse(result)

    def test_webhook_returns_error_code_is_false(self):
        class _ErrorHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"errcode": 93000, "errmsg": "invalid key"}).encode())
            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _ErrorHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/send"
        try:
            with mock.patch.object(notify_module, "_get_webhook", return_value=url):
                result = asyncio.run(async_send_wechat_notification("x"))
        finally:
            server.shutdown()
            server.server_close()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
