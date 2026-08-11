"""
核心功能修复 + 真人模拟规则的验证测试

覆盖：
A. 四个核心功能修复
1. 转人工后 AI 继续抢答 → 会话状态管理（4小时有效期）
2. 售后关键词自动转人工 → 售后词库命中后强制转人工并拦截
3. 库存查询调用工具 → 系统提示词强制调用 get_shop_products + 库存输出
4. 转人工通知到企业微信群 → Webhook 通知格式与发送

B. 九条真人模拟硬性规则
1. 模拟已读+打字：总延迟 10~14 秒（已读 6~8 秒 + 打字 4~6 秒）
2. 同一买家连续回复间隔 4~6 秒（不足补齐）
3. 单条消息最多 25 字，超长拆分
4. 拆分后的多条消息间隔 3~6 秒
5. 发送结果业务码校验，非 0 停止后续发送
6. 转人工后静默处理，不发送预设回复话术
7. 子账号转人工：不调用转人工 API，仅标记 + 通知
8. 转人工后 AI 完全忽略该会话后续消息（直接返回 True）
9. 转人工后新消息仍通知人工客服（5 分钟冷却防刷屏）

运行方式（在项目根目录）：
    .venv/Scripts/python.exe -m unittest tests.test_feature_fixes -v
"""
import asyncio
import datetime as _real_dt
import json
import shutil
import tempfile
import threading
import time
import unittest
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

warnings.filterwarnings("ignore")

# ============================================================================
# 初始化：按 app.py 顺序注册标准服务（工具模块导入依赖 DI 容器）
# ============================================================================
from config import config as _app_config
from config import get_config
from core.di_container import configure_standard_services
configure_standard_services(_app_config)

from core.session_state import SessionState, session_state
from core.uid_send_tracker import UIDSendTracker
from core.notify_tracker import NotifyTracker, notify_tracker
import importlib
from database.db_manager import DatabaseManager
# database/__init__.py 将包属性 db_manager 重绑定为 DI 代理，
# 因此必须通过 importlib 取真实子模块才能 mock 其 get_db_manager。
database_db_manager_module = importlib.import_module("database.db_manager")
from Agent.CustomerAgent.tools import move_conversation as mc_module
from Agent.CustomerAgent.tools.move_conversation import (
    transfer_conversation,
    TransferConversationParams,
)
from Message.handlers.ai_handler import AIReplyHandler
from Message.handlers import ai_handler as ai_module
from Message.handlers import keyword_handler as kh_module
from Message.handlers.keyword_handler import KeywordDetectionHandler
from Message.handlers import notify as notify_module
from Message.handlers.notify import (
    build_handoff_message,
    async_send_wechat_notification,
)
from utils.config_updater import update_config_with_uid
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.tools.get_product_list import _format_products_output
from Agent.CustomerAgent.custom.tool_decorator import TOOL_REGISTRY, execute_tool

from bridge.context import Context, ContextType, ChannelType

from Message.handlers import intent_classifier as _ic_module


class _ConsultIntentClassifier:
    """测试桩：意图恒为 consult（不触发转人工），用于隔离发送/回复/转人工静默等
    逻辑的单元测试，避免这些用例被真实意图路由（含 other/unknown→转人工）干扰。

    意图路由本身由 TestAIHandlerIntentRouting / TestIntentClassifier 单独覆盖。
    """

    enabled = True
    threshold = 0.6
    model_name = "stub"
    api_key = ""
    api_base = ""

    async def classify(self, text, after_sale_hint=False, history=None):
        return {"intent": "consult", "confidence": 0.99}

    @staticmethod
    def should_transfer(intent, confidence, threshold):
        return _ic_module.IntentClassifier.should_transfer(intent, confidence, threshold)


def _patch_intent_to_consult():
    """返回已启动的 patch，使 get_intent_classifier 返回 consult 桩。"""
    return mock.patch.object(
        _ic_module, "get_intent_classifier", return_value=_ConsultIntentClassifier()
    )
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


class BusinessErrorSender(FakeSender):
    """发送若干次成功后，返回业务码非 0 的结果（用于规则 5 测试）"""

    def __init__(self, fail_after=1):
        super().__init__()
        self.fail_after = fail_after
        self.count = 0

    def send_text(self, shop_id, user_id, recipient_uid, text):
        self.calls["send_text"].append((shop_id, user_id, recipient_uid, text))
        self.count += 1
        if self.count > self.fail_after:
            return {"success": True, "result": {"error_code": 10002}}
        return {"success": True}


class MockBot:
    """记录是否被调用的模拟 Bot，可配置返回的回复文本"""

    def __init__(self, reply_text="好的，亲"):
        self.calls = []
        self.reply_text = reply_text

    async def async_reply(self, query, context=None):
        self.calls.append((query, context))
        return Reply(ReplyType.TEXT, self.reply_text)


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
# 配置辅助：加速测试（关闭真人模拟时序等待），用完恢复默认
# ============================================================================

def _zero_ai_reply_delays():
    """将 AI 回复时序参数置 0，加速测试"""
    _app_config.set("ai_reply.read_seconds_min", 0, save=False)
    _app_config.set("ai_reply.read_seconds_max", 0, save=False)
    _app_config.set("ai_reply.typing_seconds_min", 0, save=False)
    _app_config.set("ai_reply.typing_seconds_max", 0, save=False)
    _app_config.set("ai_reply.split_interval_min", 0, save=False)
    _app_config.set("ai_reply.split_interval_max", 0, save=False)
    _app_config.set("ai_reply.uid_min_interval", 0, save=False)


def _restore_ai_reply_delays():
    """恢复 AI 回复时序参数为生产默认值"""
    _app_config.set("ai_reply.read_seconds_min", 6, save=False)
    _app_config.set("ai_reply.read_seconds_max", 8, save=False)
    _app_config.set("ai_reply.typing_seconds_min", 4, save=False)
    _app_config.set("ai_reply.typing_seconds_max", 6, save=False)
    _app_config.set("ai_reply.split_interval_min", 3, save=False)
    _app_config.set("ai_reply.split_interval_max", 6, save=False)
    _app_config.set("ai_reply.uid_min_interval", 4, save=False)


def _set_inside_business_hours():
    """将营业时间设为包含当前时刻的 ±1 小时窗口，避免 handle 测试受运行时刻影响"""
    _app_config.set("business_hours", _inside_business_hours_dict(), save=False)


def _inside_business_hours_dict():
    """返回包含当前时刻的 ±1 小时营业时间字典（供直接构造 handler 使用）"""
    now = _real_dt.datetime.now()
    start = (now - _real_dt.timedelta(hours=1)).strftime("%H:%M")
    end = (now + _real_dt.timedelta(hours=1)).strftime("%H:%M")
    return {"start": start, "end": end}


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

    def test_persist_handoff_survives_memory_clear(self):
        """转人工标记持久化到数据库：清空内存缓存（模拟进程重启）后仍能恢复"""
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_db = str(tmp_dir / "test_session_state.db")
        temp_manager = DatabaseManager(db_path=tmp_db)
        key = "persist:buyer1"
        try:
            with mock.patch.object(
                database_db_manager_module,
                "get_db_manager",
                return_value=temp_manager,
            ):
                s = self.s
                s._db = None  # 强制重新解析到被 mock 的 get_db_manager
                s.mark_handoff(key)
                # 清空内存缓存，模拟进程重启后内存丢失
                with s._data_lock:
                    s._handoffs.clear()
                self.assertTrue(s.is_handoff(key))
                s.clear_handoff(key)
                self.assertFalse(s.is_handoff(key))
        finally:
            # 恢复单例状态，避免污染后续测试
            with self.s._data_lock:
                self.s._db = None
                self.s._handoffs.pop(key, None)
            temp_manager.dispose()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_memory_cache_works_when_db_unavailable(self):
        """数据库不可用时降级为纯内存缓存，不抛异常"""
        key = "memonly:buyer1"
        try:
            with mock.patch.object(
                database_db_manager_module,
                "get_db_manager",
                side_effect=RuntimeError("db unavailable"),
            ):
                s = self.s
                s._db = None
                s.mark_handoff(key)  # 不应抛异常
                self.assertTrue(s.is_handoff(key))
                s.clear_handoff(key)
                self.assertFalse(s.is_handoff(key))
        finally:
            with self.s._data_lock:
                self.s._db = None
                self.s._handoffs.pop(key, None)


# ============================================================================
# 测试 2：转人工工具标记会话 + 发送通知（规则 6/7）
# ============================================================================

class TestTransferConversation(unittest.TestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        # 测试期间统一按"全部主账号"处理，不受真实 config 主账号列表影响
        self._orig_main = get_config("transfer.main_account_user_ids", [])
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        self.capture = CaptureWebhook()
        self.sender = FakeSender()
        self._sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._sender_patch.start()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()

    def tearDown(self):
        _app_config.set("transfer.main_account_user_ids", self._orig_main, save=False)
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

    def test_int_recipient_uid_coerced_to_str(self):
        """AI 可能以整数形式提取 recipient_uid，参数模型应接受并转字符串使用"""
        params = TransferConversationParams(
            shop_id="shop1", user_id="user1", recipient_uid=5927195871573, shop_name="峰哥编织",
        )
        result = transfer_conversation(params)
        self.assertIn("会话转接成功", result)
        # 会话标记使用字符串形式的 UID
        self.assertTrue(SessionState().is_handoff("shop1:5927195871573"))
        # 转人工 API 收到字符串形式的 recipient_uid
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        shop_id, user_id, recipient_uid, cs_uid = self.sender.calls["transfer_to_cs"][0]
        self.assertEqual(recipient_uid, "5927195871573")


# ============================================================================
# 测试 3：主/子账号转人工行为（规则 7）
# ============================================================================

class TestSubAccountTransfer(unittest.TestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        self.capture = CaptureWebhook()
        self.sender = FakeSender()
        self._sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._sender_patch.start()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()
        # 配置：user2 为子账号（完整 UID cs_shop1_user2），其余默认按主账号处理
        self._orig_main = get_config("transfer.main_account_user_ids", [])
        self._orig_sub = get_config("transfer.sub_account_uids", [])
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        _app_config.set("transfer.sub_account_uids", ["cs_shop1_user2"], save=False)

    def tearDown(self):
        _app_config.set("transfer.main_account_user_ids", self._orig_main, save=False)
        _app_config.set("transfer.sub_account_uids", self._orig_sub, save=False)
        self._sender_patch.stop()
        self._notify_patch.stop()

    def test_sub_account_marks_and_notifies_without_api(self):
        """规则 7: 子账号不调用转人工 API，仅标记会话 + 通知人工客服"""
        params = TransferConversationParams(
            shop_id="shop1", user_id="user2", recipient_uid="buyer1", shop_name="峰哥编织",
        )
        result = transfer_conversation(params)
        self.assertIn("会话转接成功", result)
        # 会话被标记（后续 AI 不再抢答）
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 子账号不调用转人工 API
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        # 但仍通知人工客服
        self.assertEqual(len(self.capture.messages), 1)

    def test_main_account_calls_api(self):
        """主账号尝试真实转接"""
        params = TransferConversationParams(
            shop_id="shop1", user_id="user1", recipient_uid="buyer1", shop_name="峰哥编织",
        )
        result = transfer_conversation(params)
        self.assertIn("会话转接成功", result)
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        self.assertEqual(len(self.capture.messages), 1)

    def test_unconfigured_all_treated_as_main(self):
        """未配置 main_account_user_ids 时，全部按主账号处理"""
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        _app_config.set("transfer.sub_account_uids", [], save=False)
        params = TransferConversationParams(
            shop_id="shop1", user_id="any_user", recipient_uid="buyer1",
        )
        result = transfer_conversation(params)
        self.assertIn("会话转接成功", result)
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)


# ============================================================================
# 测试 3.1：主/子账号判定（需求一：完整 UID 捕获后正确分类）
# ============================================================================

class TestIsMainAccount(unittest.TestCase):
    """写入完整 UID 后，_is_main_account 对主/子账号的判定"""

    def setUp(self):
        self._orig_main = get_config("transfer.main_account_user_ids", [])
        self._orig_sub = get_config("transfer.sub_account_uids", [])
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        _app_config.set("transfer.sub_account_uids", [], save=False)

    def tearDown(self):
        _app_config.set("transfer.main_account_user_ids", self._orig_main, save=False)
        _app_config.set("transfer.sub_account_uids", self._orig_sub, save=False)

    def test_sub_account_list_marks_sub(self):
        """子账号 UID 命中 sub_account_uids → 判定为子账号"""
        _app_config.set("transfer.sub_account_uids", ["cs_661962391_189109418"], save=False)
        self.assertFalse(mc_module._is_main_account("661962391", "189109418"))

    def test_other_user_still_main(self):
        _app_config.set("transfer.sub_account_uids", ["cs_661962391_189109418"], save=False)
        self.assertTrue(mc_module._is_main_account("661962391", "661962391"))

    def test_full_uid_in_main_list_detects_sub(self):
        """兼容旧配置：main_account_user_ids 存完整子账号 UID → 判定为子账号"""
        _app_config.set(
            "transfer.main_account_user_ids", ["cs_661962391_189109418", "661962391"], save=False
        )
        self.assertFalse(mc_module._is_main_account("661962391", "189109418"))
        self.assertTrue(mc_module._is_main_account("661962391", "661962391"))

    def test_bare_user_id_in_main_list_is_main(self):
        """兼容旧配置：裸 user_id 命中 main_account_user_ids → 主账号"""
        _app_config.set("transfer.main_account_user_ids", ["user1"], save=False)
        self.assertTrue(mc_module._is_main_account("shop1", "user1"))

    def test_empty_config_all_main(self):
        self.assertTrue(mc_module._is_main_account("shop1", "any_user"))


# ============================================================================
# 测试 3.2：运行时 WebSocket 认证捕获完整 UID（需求一）
# ============================================================================

class TestAuthUidCapture(unittest.TestCase):
    """认证成功后从 WebSocket 认证响应捕获完整 UID 并写入配置"""

    def setUp(self):
        from Channel.pinduoduo.core.pdd_message_handler import MessageHandlerMixin
        from utils.logger_loguru import get_logger
        self.handler = MessageHandlerMixin()
        self.handler.logger = get_logger("TestAuthUidCapture")

    def make_auth_context(self, content):
        return Context.create_pinduoduo_context(
            content=content,
            from_uid="",
            user_id="189109418",
            shop_id="661962391",
            username="cs_661962391_189109418",
            user_msg_type=ContextType.AUTH,
            channel_type=ChannelType.PINDUODUO,
        )

    def test_captures_full_sub_uid(self):
        # _convert_to_context 已将认证 dict 序列化为 JSON 字符串
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            self.handler._capture_auth_uid(
                self.make_auth_context(json.dumps({"uid": "cs_661962391_189109418", "result": "ok"}))
            )
        m.assert_called_once_with("cs_661962391_189109418")

    def test_captures_main_uid(self):
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            self.handler._capture_auth_uid(
                self.make_auth_context(json.dumps({"uid": "661962391", "result": "ok"}))
            )
        m.assert_called_once_with("661962391")

    def test_auth_failure_does_not_capture(self):
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            self.handler._capture_auth_uid(
                self.make_auth_context(json.dumps({"uid": "cs_661962391_189109418", "result": "fail"}))
            )
        m.assert_not_called()

    def test_missing_uid_does_not_capture(self):
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            self.handler._capture_auth_uid(self.make_auth_context(json.dumps({"result": "ok"})))
        m.assert_not_called()

    def test_invalid_json_does_not_capture(self):
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            self.handler._capture_auth_uid(self.make_auth_context("not-a-json"))
        m.assert_not_called()

    def test_handle_immediate_auth_writes_uid(self):
        """AUTH 消息走 _handle_immediate_message → 捕获 UID，无需 SendMessage"""
        with mock.patch("Channel.pinduoduo.core.pdd_message_handler.update_config_with_uid") as m:
            asyncio.run(self.handler._handle_immediate_message(
                self.make_auth_context(json.dumps({"uid": "cs_661962391_189109418", "result": "ok"})),
                "661962391",
                "189109418",
            ))
        m.assert_called_once_with("cs_661962391_189109418")


# ============================================================================
# 测试 4：AI 处理器 —— 转人工有效期内忽略消息（规则 8/9）
# ============================================================================

class TestAIHandlerHandoff(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        notify_tracker.clear("shop1:buyer1")
        # 加速测试：关闭真人模拟时序等待（规则 1/2/4）
        _zero_ai_reply_delays()
        _set_inside_business_hours()
        self.bot = MockBot()
        self.handler = AIReplyHandler(bot=self.bot)
        self.sender = FakeSender()
        self._sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._sender_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()
        # 隔离意图路由：本类只测"转人工后静默/忽略"，不涉及意图触发转人工
        self._intent_patch = _patch_intent_to_consult()
        self._intent_patch.start()

    def tearDown(self):
        self._sender_patch.stop()
        self._notify_patch.stop()
        self._intent_patch.stop()
        _restore_ai_reply_delays()

    async def test_handoff_active_skips_ai_and_notifies(self):
        """规则 8: 转人工后 AI 完全忽略该会话后续消息（不等待、不回复、不重触发）"""
        SessionState().mark_handoff("shop1:buyer1")
        context = make_context("在吗，请问发货了吗")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # AI 未被调用（不抢答）
        self.assertEqual(self.bot.calls, [])
        # 不再重复触发转人工 API
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        # 规则 9: 新消息仍通知人工客服，原因标注"转人工后买家新消息"
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("转人工后买家新消息", self.capture.messages[0])
        self.assertIn("在吗，请问发货了吗", self.capture.messages[0])

    async def test_handoff_new_message_notify_cooldown(self):
        """规则 9: 同会话冷却期内再次来消息，不重复通知（防刷屏）"""
        SessionState().mark_handoff("shop1:buyer1")
        context = make_context("在吗")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertEqual(len(self.capture.messages), 1)
        # 冷却期内（5 分钟内）第二条消息 → 不再通知
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertEqual(len(self.capture.messages), 1)

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
        # 未触发转人工
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])

    async def test_no_handoff_ai_replies_normally(self):
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("你好")
            ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertEqual(len(self.bot.calls), 1)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])


# ============================================================================
# 测试 5：售后关键词触发转人工（规则 6 静默处理）
# ============================================================================

class TestKeywordHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        # 测试期间统一按"全部主账号"处理，避免受真实 config 主账号列表影响
        self._orig_main = get_config("transfer.main_account_user_ids", [])
        self._orig_sub = get_config("transfer.sub_account_uids", [])
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        _app_config.set("transfer.sub_account_uids", [], save=False)
        self.sender = FakeSender()
        self._mc_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._mc_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()
        # 营业时间窗口设为包含当前时刻，避免 can_handle 的营业时间门控受运行时刻影响
        self.handler = KeywordDetectionHandler(business_hours=_inside_business_hours_dict())

    def tearDown(self):
        _app_config.set("transfer.main_account_user_ids", self._orig_main, save=False)
        _app_config.set("transfer.sub_account_uids", self._orig_sub, save=False)
        self._mc_patch.stop()
        self._notify_patch.stop()

    def test_after_sale_keywords_defined(self):
        for kw in ["退货", "退款", "售后", "质量问题", "破损", "漏发", "少发",
                   "不满意", "投诉", "赔偿", "换货", "维修", "差评", "给差评",
                   "假货", "质量差"]:
            self.assertIn(kw, KeywordDetectionHandler.AFTER_SALE_KEYWORDS)

    def test_can_handle_after_sale(self):
        """改造后：售后软兜底词不再硬短路转人工，交由意图分类判断。"""
        context = make_context("我要退款")
        self.assertFalse(self.handler.can_handle(context))

    def test_transfer_keyword_hard_trigger(self):
        """必转词仍硬短路转人工（保留'用户说转人工必转'）。"""
        context = make_context("请转人工帮我")
        self.assertTrue(self.handler.can_handle(context))

    def test_can_handle_regular_keyword(self):
        # 不依赖数据库中的关键词状态，注入受控关键词源验证"转人工"检测
        with mock.patch.object(kh_module.db_manager, "get_all_keywords", return_value=[{"keyword": "转人工"}]):
            handler = KeywordDetectionHandler(business_hours=_inside_business_hours_dict())
        context = make_context("转人工")
        self.assertTrue(handler.can_handle(context))

    def test_match_after_sale_keyword(self):
        """模块级售后词匹配函数：命中 after_sale 词返回 True，普通词 False。"""
        # 直接测试函数（不依赖实例），用注入集合验证逻辑
        from Message.handlers import keyword_handler as kh
        import Message.handlers.keyword_handler as kh_real
        # 临时替换模块级集合
        old = kh_real._AFTERSALE_KEYWORDS
        kh_real._AFTERSALE_KEYWORDS = {"退款", "退货"}
        try:
            self.assertTrue(kh.match_after_sale_keyword("我要退款"))
            self.assertFalse(kh.match_after_sale_keyword("这件衣服多大码"))
        finally:
            kh_real._AFTERSALE_KEYWORDS = old

    def test_cannot_handle_normal_message(self):
        context = make_context("这件衣服多大码")
        self.assertFalse(self.handler.can_handle(context))

    def test_within_business_hours_invalid_format_allows_manual(self):
        """配置 start/end 格式错误时保守返回 True（允许转人工），不禁用功能"""
        handler = KeywordDetectionHandler(business_hours={"start": "bad", "end": "23:00"})
        self.assertTrue(handler._within_business_hours())
        handler = KeywordDetectionHandler(business_hours={"start": "08:00", "end": None})
        self.assertTrue(handler._within_business_hours())

    def test_within_business_hours_invalid_type_allows_manual(self):
        """business_hours 非 dict（如字符串/缺失）时保守返回 True（允许转人工）"""
        handler = KeywordDetectionHandler(business_hours="bad-config")
        self.assertTrue(handler._within_business_hours())

    def test_within_business_hours_missing_fields_uses_defaults(self):
        """start/end 缺失时回退默认 08:00-23:00，返回正常布尔判断"""
        handler = KeywordDetectionHandler(business_hours={})
        result = handler._within_business_hours()
        self.assertIsInstance(result, bool)

    def test_within_business_hours_valid_config(self):
        """有效配置按营业时间正常判断"""
        handler = KeywordDetectionHandler(business_hours=_inside_business_hours_dict())
        result = handler._within_business_hours()
        self.assertTrue(result)

    async def test_after_sale_triggers_transfer_and_blocks(self):
        """必转词命中即转人工并拦截（硬短路路径，保留规则 6 静默）。"""
        context = make_context("请转人工")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # 转人工被触发
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        # 会话被标记
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 通知原因
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("用户主动转人工", self.capture.messages[0])
        self.assertIn("请转人工", self.capture.messages[0])

    async def test_after_sale_failure_still_blocks_silently(self):
        """规则 6: 转人工失败（无可用客服）时仍拦截，但不发预设回复话术"""
        sender = FakeSender(cs_list={"cs_shop1_user1": {"username": "客服1"}})
        mc_module.get_sender = lambda: sender  # noqa: E731
        context = make_context("请转人工")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))
        # 不向用户发送任何预设回复话术（静默处理）
        self.assertEqual(sender.calls["send_text"], [])

    async def test_sub_account_keyword_silent_mark_and_notify(self):
        """规则 7: 子账号触发必转词 → 不调用转人工 API，仅静默标记 + 通知"""
        _app_config.set("transfer.sub_account_uids", ["cs_shop1_user1"], save=False)
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        context = make_context("转人工", user_id="user1")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # 子账号不调用转人工 API
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        # 会话被静默标记
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 仍通知人工客服
        self.assertEqual(len(self.capture.messages), 1)


# ============================================================================
# 测试 6：真人模拟规则（规则 1/2/3/4/5）
# ============================================================================

class TestHumanReplyRules(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        _zero_ai_reply_delays()
        _set_inside_business_hours()
        self.bot = MockBot()
        self.handler = AIReplyHandler(bot=self.bot)
        self.sender = FakeSender()
        # 隔离意图路由：本类只测发送/拆分/清洗规则，避免意图触发转人工干扰断言
        self._intent_patch = _patch_intent_to_consult()
        self._intent_patch.start()

    def tearDown(self):
        self._intent_patch.stop()
        _restore_ai_reply_delays()

    def test_split_reply_respects_max_len(self):
        """规则 3: 超长回复被拆分为不超过 25 字的短消息，且内容完整保留"""
        handler = AIReplyHandler(bot=MockBot())
        long_reply = "亲，这款商品我们一般下单后48小时内发货哦，遇到大促可能会稍微延迟，但会尽快给您发出的。请放心购买哈。"
        chunks = handler._split_reply(long_reply)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertLessEqual(len(c), handler.max_message_len)
        # 拆分后内容应与原文一致（无空格的中文串）
        self.assertEqual("".join(chunks), long_reply)

    def test_split_reply_keeps_short_reply_whole(self):
        handler = AIReplyHandler(bot=MockBot())
        short = "亲，在的哦。"
        chunks = handler._split_reply(short)
        self.assertEqual(chunks, ["亲，在的哦。"])

    def test_split_reply_empty(self):
        handler = AIReplyHandler(bot=MockBot())
        self.assertEqual(handler._split_reply(""), [])

    def test_check_send_result_business_code(self):
        """规则 5: 业务码校验"""
        handler = AIReplyHandler(bot=MockBot())
        # 成功 + 业务码 0 → 通过
        self.assertTrue(handler._check_send_result({"success": True, "result": {"error_code": 0}}))
        # 无 result 字段，默认业务码 0 → 通过
        self.assertTrue(handler._check_send_result({"success": True}))
        # success=True 但业务码非 0 → 失败
        self.assertFalse(handler._check_send_result({"success": True, "result": {"error_code": 10002}}))
        # 返回错误文案字符串 → 失败
        self.assertFalse(handler._check_send_result("该消息发送失败，请稍后重试"))
        # 返回 None / success=False → 失败
        self.assertFalse(handler._check_send_result(None))
        self.assertFalse(handler._check_send_result({"success": False}))

    async def test_human_delay_is_10_to_14_seconds(self):
        """规则 1: 已读 6~8 秒 + 打字 4~6 秒 = 合计 10~14 秒"""
        # 显式设置为生产默认值
        _restore_ai_reply_delays()
        handler = AIReplyHandler(bot=MockBot())
        with mock.patch("Message.handlers.ai_handler.asyncio.sleep") as mock_sleep:
            await handler._simulate_human_delay("buyer1")
        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        self.assertGreaterEqual(delay, 10.0)
        self.assertLessEqual(delay, 14.0)

    async def test_business_code_stops_subsequent_sends(self):
        """规则 5: 发送业务码非 0 时记录日志并停止后续发送"""
        self.bot = MockBot(reply_text="这是一条很长的回复，用来拆分成多条消息。第二句的内容。第三句的内容。")
        handler = AIReplyHandler(bot=self.bot)
        sender = BusinessErrorSender(fail_after=1)
        with mock.patch("bridge.sender.get_sender", return_value=sender):
            context = make_context("你好")
            ok = await handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # 第一条成功，第二条业务码非 0 → 停止，不再发送第三条
        self.assertEqual(len(sender.calls["send_text"]), 2)

    async def test_split_messages_all_sent_successfully(self):
        """规则 3/4: 多条拆分消息全部发送成功（发送时已清洗标点）"""
        self.bot = MockBot(reply_text="第一条短句。第二条短句。第三条短句。")
        handler = AIReplyHandler(bot=self.bot)
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("你好")
            ok = await handler.handle(context, make_metadata())
        self.assertTrue(ok)
        texts = [call[3] for call in self.sender.calls["send_text"]]
        self.assertEqual(len(texts), 3)
        self.assertEqual(texts, ["第一条短句", "第二条短句", "第三条短句"])


# ============================================================================
# 测试 6.1：URL 保护 —— 拆分时 URL 完整无损、不被截断
# ============================================================================

class TestSplitUrlProtection(unittest.TestCase):
    """_split_reply 对 URL 的保护：占位符替换 → 拆分 → 还原"""

    def setUp(self):
        self.handler = AIReplyHandler(bot=MockBot())

    def test_url_single_message_preserved(self):
        """示例：含 URL 的整条消息不超过上限时，原样保留为一条"""
        reply = "亲，链接是 https://www.pinduoduo.com/search?keyword=保温杯，您看下。"
        chunks = self.handler._split_reply(reply)
        self.assertEqual(chunks, [reply])

    def test_url_intact_in_long_message(self):
        """URL 位于长消息中：完整出现在同一条，不被截断"""
        url = "https://mobile.yangkeduo.com/goods.html?goods_id=123456789012345"
        reply = f"亲这是商品链接{url}，这款保温杯很受欢迎销量很好质量也不错您可以放心购买。"
        chunks = self.handler._split_reply(reply)
        joined = "".join(chunks)
        self.assertIn(url, joined)
        self.assertEqual(sum(1 for c in chunks if url in c), 1)

    def test_multiple_urls_all_intact(self):
        url1 = "https://mobile.yangkeduo.com/goods.html?goods_id=111"
        url2 = "https://mobile.yangkeduo.com/goods.html?goods_id=222"
        reply = f"第一个链接{url1}，第二个链接{url2}，都可以看看哦。"
        chunks = self.handler._split_reply(reply)
        joined = "".join(chunks)
        self.assertIn(url1, joined)
        self.assertIn(url2, joined)
        self.assertEqual(sum(1 for c in chunks if url1 in c), 1)
        self.assertEqual(sum(1 for c in chunks if url2 in c), 1)

    def test_url_longer_than_max_stays_whole(self):
        """URL 单独超过字数上限：允许该条略超，但 URL 完整"""
        url = "https://www.pinduoduo.com/mobile/mall/goods_detail?goods_id=" + "9" * 40
        reply = f"链接{url}。"
        chunks = self.handler._split_reply(reply)
        joined = "".join(chunks)
        self.assertIn(url, joined)
        self.assertEqual(sum(1 for c in chunks if url in c), 1)
        # 含 URL 的消息允许超过 25 字
        self.assertTrue(any(len(c) > self.handler.max_message_len for c in chunks))

    def test_url_followed_by_long_text_not_glued(self):
        """URL 后的标点/长文本不被并入 URL，仍可正常拆分"""
        url = "https://x.com/abc"
        reply = (f"链接：{url}，后面这一段是很长的描述文字超过二十五个字了请查看详情，"
                 "另一句也超过二十五字了请仔细核对一下。")
        chunks = self.handler._split_reply(reply)
        joined = "".join(chunks)
        self.assertIn(url, joined)
        self.assertEqual(sum(1 for c in chunks if url in c), 1)
        # 长文本未被 URL 吞并，存在独立于 URL 的更多分片
        self.assertGreater(len(chunks), 1)

    def test_url_does_not_break_reconstruction_without_spaces(self):
        """无空格干扰时，拆分后拼接应与原文一致（URL 完整）"""
        url = "https://mobile.yangkeduo.com/goods.html?goods_id=123456"
        reply = f"亲这是链接{url}您看下这个商品。"
        chunks = self.handler._split_reply(reply)
        self.assertEqual("".join(chunks), reply)


# ============================================================================
# 测试：营业时间检查（非营业时间静默转人工 + 企业微信通知）
# ============================================================================

class _FixedDateTime(_real_dt.datetime):
    """固定 now() 的 datetime 子类，供营业时间判断单测使用"""
    fixed_now = None

    @classmethod
    def now(cls, tz=None):
        return cls.fixed_now


class _FakeDatetimeModule:
    """模拟 datetime 模块：datetime.now() 固定、strptime 继承自真实 datetime"""
    datetime = _FixedDateTime


class TestBusinessHours(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        notify_tracker.clear("shop1:buyer1")
        _zero_ai_reply_delays()
        self._orig_bh = get_config("business_hours", {})
        self.bot = MockBot()
        self.handler = AIReplyHandler(bot=self.bot)
        self.sender = FakeSender()
        # _send_reply 内部从 bridge.sender 取发送器
        self._bridge_sender_patch = mock.patch("bridge.sender.get_sender", return_value=self.sender)
        self._bridge_sender_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(
            notify_module, "_post_wechat_sync",
            side_effect=lambda msg: (self.capture.messages.append(msg), True)[1],
        )
        self._notify_patch.start()
        # 隔离意图路由：本类只测营业时间逻辑，避免意图触发转人工干扰正常回复断言
        self._intent_patch = _patch_intent_to_consult()
        self._intent_patch.start()

    def tearDown(self):
        _app_config.set("business_hours", self._orig_bh, save=False)
        self._bridge_sender_patch.stop()
        self._notify_patch.stop()
        self._intent_patch.stop()
        _restore_ai_reply_delays()

    def _set_business_window(self, start, end):
        _app_config.set("business_hours", {"start": start, "end": end}, save=False)

    def _time_str(self, delta_minutes):
        """返回 now + delta 分钟的 %H:%M 字符串（跨零点由 datetime 运算保证）"""
        return (_real_dt.datetime.now() + _real_dt.timedelta(minutes=delta_minutes)).strftime("%H:%M")

    async def test_outside_hours_marks_handoff_and_notifies_no_ai(self):
        """非营业时间：静默转人工、通知企业微信、不执行 AI 回复"""
        self._set_business_window(self._time_str(2), self._time_str(4))
        context = make_context("在吗，发货了吗")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # 不调用 AI
        self.assertEqual(self.bot.calls, [])
        # 不发送任何回复
        self.assertEqual(self.sender.calls["send_text"], [])
        # 静默标记转人工
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        # 通知企业微信，原因标注"非营业时间自动转人工"
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("非营业时间自动转人工", self.capture.messages[0])
        self.assertIn("峰哥编织", self.capture.messages[0])
        self.assertIn("buyer1", self.capture.messages[0])
        self.assertIn("在吗，发货了吗", self.capture.messages[0])

    async def test_inside_hours_replies_normally(self):
        """营业时间内：正常 AI 回复，不标记转人工、不通知"""
        self._set_business_window(self._time_str(-2), self._time_str(2))
        context = make_context("你好")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        # AI 正常回复
        self.assertEqual(len(self.bot.calls), 1)
        self.assertGreaterEqual(len(self.sender.calls["send_text"]), 1)
        # 未标记转人工、未通知
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))
        self.assertEqual(self.capture.messages, [])

    async def test_outside_hours_notify_failure_does_not_affect_flow(self):
        """非营业时间通知发送失败：仅记录警告，仍静默转人工并拦截"""
        self._notify_patch.stop()
        self._notify_patch = mock.patch.object(notify_module, "_post_wechat_sync", return_value=False)
        self._notify_patch.start()
        self._set_business_window(self._time_str(2), self._time_str(4))
        context = make_context("在吗")
        ok = await self.handler.handle(context, make_metadata())
        self.assertTrue(ok)
        self.assertEqual(self.bot.calls, [])
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))

    def test_is_outside_business_hours_config(self):
        """_is_outside_business_hours 对配置区间与跨零点区间的判断"""
        _FixedDateTime.fixed_now = _real_dt.datetime(2026, 8, 7, 12, 0, 0)
        with mock.patch.object(ai_module, "datetime", _FakeDatetimeModule):
            self._set_business_window("09:00", "18:00")
            self.assertFalse(self.handler._is_outside_business_hours())
            # 跨零点营业：18:00-09:00，12:00 在区间外
            self._set_business_window("18:00", "09:00")
            self.assertTrue(self.handler._is_outside_business_hours())
            # 全天营业：00:00-23:59
            self._set_business_window("00:00", "23:59")
            self.assertFalse(self.handler._is_outside_business_hours())

    def test_invalid_business_hours_defaults_to_inside(self):
        """配置解析失败时保守地按营业时间内处理"""
        with mock.patch.object(ai_module, "get_config", return_value={"start": "bad", "end": "23:00"}):
            self.assertFalse(self.handler._is_outside_business_hours())


# ============================================================================
# 测试 7：同一买家回复间隔跟踪器（规则 2）
# ============================================================================

class TestUIDSendTracker(unittest.TestCase):
    def test_pads_short_interval(self):
        tracker = UIDSendTracker(min_interval=4.0)
        tracker.record_send("buyer1")
        # 立即再发：需要补齐到至少 4 秒
        pad = tracker.wait_before_send("buyer1")
        self.assertGreater(pad, 3.5)
        self.assertLessEqual(pad, 4.0)

    def test_unknown_uid_no_wait(self):
        tracker = UIDSendTracker(min_interval=4.0)
        self.assertEqual(tracker.wait_before_send("buyer9"), 0.0)

    def test_after_wait_no_padding(self):
        tracker = UIDSendTracker(min_interval=4.0)
        tracker.record_send("buyer1")
        time.sleep(4.2)
        self.assertEqual(tracker.wait_before_send("buyer1"), 0.0)

    def test_clear(self):
        tracker = UIDSendTracker(min_interval=4.0)
        tracker.record_send("buyer1")
        tracker.clear("buyer1")
        self.assertEqual(tracker.wait_before_send("buyer1"), 0.0)


# ============================================================================
# 测试 8：转人工通知冷却跟踪器（规则 9）
# ============================================================================

class TestNotifyTracker(unittest.TestCase):
    def test_cooldown_blocks_repeat_notify(self):
        tracker = NotifyTracker(cooldown_seconds=300)
        self.assertTrue(tracker.should_notify("shop1:buyer1"))
        tracker.update_notify("shop1:buyer1")
        # 冷却期内不允许再次通知
        self.assertFalse(tracker.should_notify("shop1:buyer1"))

    def test_clear_resets_cooldown(self):
        tracker = NotifyTracker(cooldown_seconds=300)
        tracker.update_notify("shop1:buyer1")
        tracker.clear("shop1:buyer1")
        self.assertTrue(tracker.should_notify("shop1:buyer1"))

    def test_module_singleton_shared_between_modules(self):
        """规则 9: notify_tracker 是模块级单例，move_conversation 与 ai_handler 共用"""
        from Message.handlers.ai_handler import notify_tracker as ai_notify_tracker
        self.assertIsInstance(notify_tracker, NotifyTracker)
        self.assertIs(ai_notify_tracker, notify_tracker)


# ============================================================================
# 测试 9：库存查询调用工具（系统提示词 + 库存输出）
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
# 测试 10：企业微信群机器人通知
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


# ============================================================================
# 测试 11：自动写入 UID 到 config.json（需求一）
# ============================================================================

class TestConfigUpdater(unittest.TestCase):
    """需求一：账号认证成功后自动写入完整 UID 到 config.json"""

    def setUp(self):
        # 备份 config.json 原内容，测试结束后恢复
        config_path = Path("config.json")
        self._orig_content = config_path.read_text(encoding="utf-8") if config_path.exists() else None
        # 清空内存中的列表，构造"未配置"场景
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        _app_config.set("transfer.sub_account_uids", [], save=False)
        self._orig_llm = get_config("llm.model_name", "")

    def tearDown(self):
        # 恢复 config.json 原内容并重载缓存，避免污染真实配置
        if self._orig_content is not None:
            Path("config.json").write_text(self._orig_content, encoding="utf-8")
        _app_config.reload()

    def test_adds_uid(self):
        ok = update_config_with_uid("661962391")
        self.assertTrue(ok)
        self.assertIn("661962391", get_config("transfer.main_account_user_ids", []))

    def test_dedup(self):
        update_config_with_uid("661962391")
        update_config_with_uid("661962391")
        lst = get_config("transfer.main_account_user_ids", [])
        self.assertEqual(lst.count("661962391"), 1)

    def test_multiple_uids(self):
        update_config_with_uid("111")
        update_config_with_uid("222")
        lst = get_config("transfer.main_account_user_ids", [])
        self.assertEqual(sorted(lst), ["111", "222"])

    def test_empty_uid_rejected(self):
        self.assertFalse(update_config_with_uid(""))
        self.assertFalse(update_config_with_uid(None))

    def test_preserves_other_config(self):
        update_config_with_uid("661962391")
        # 其他配置字段未被覆盖或删除
        self.assertEqual(get_config("llm.model_name", ""), self._orig_llm)
        self.assertIsNotNone(get_config("notification.wechat_webhook", None))

    def test_sub_account_uid_written_to_both_lists(self):
        """完整子账号 UID（cs_ 开头）同时写入 main 与 sub 列表"""
        ok = update_config_with_uid("cs_661962391_189109418")
        self.assertTrue(ok)
        self.assertIn("cs_661962391_189109418", get_config("transfer.main_account_user_ids", []))
        self.assertIn("cs_661962391_189109418", get_config("transfer.sub_account_uids", []))

    def test_main_uid_not_in_sub_list(self):
        """主账号 UID（纯数字）只写 main 列表，不写 sub 列表"""
        update_config_with_uid("661962391")
        self.assertIn("661962391", get_config("transfer.main_account_user_ids", []))
        self.assertNotIn("661962391", get_config("transfer.sub_account_uids", []))

    def test_sub_list_dedup(self):
        update_config_with_uid("cs_661962391_189109418")
        update_config_with_uid("cs_661962391_189109418")
        subs = get_config("transfer.sub_account_uids", [])
        self.assertEqual(subs.count("cs_661962391_189109418"), 1)


# ============================================================================
# 测试 12：回复标点清洗（需求二）
# ============================================================================

class TestCleanText(unittest.TestCase):
    """需求二：_clean_text 移除中英文句号、逗号、问号"""

    def setUp(self):
        self.handler = AIReplyHandler(bot=MockBot())

    def test_removes_chinese_punctuation(self):
        self.assertEqual(
            self.handler._clean_text("亲，在的哦。您有什么需要帮忙？"),
            "亲在的哦您有什么需要帮忙",
        )

    def test_removes_english_punctuation(self):
        self.assertEqual(
            self.handler._clean_text("Hello, world. How are you?"),
            "Hello world How are you",
        )

    def test_empty_and_punct_only(self):
        self.assertEqual(self.handler._clean_text(""), "")
        self.assertEqual(self.handler._clean_text("。，？,."), "")

    def test_no_punctuation_unchanged(self):
        self.assertEqual(self.handler._clean_text("亲在的哦"), "亲在的哦")

    def test_removes_semicolon(self):
        self.assertEqual(
            self.handler._clean_text("亲，已为您处理；请问还有其他需要？"),
            "亲已为您处理请问还有其他需要",
        )
        self.assertEqual(
            self.handler._clean_text("Done; please wait."),
            "Done please wait",
        )


class TestSendCleanAndSkip(unittest.IsolatedAsyncioTestCase):
    """需求二：发送前清洗；纯标点分条跳过；整条纯标点走备用回复（同样被清洗）"""

    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        _zero_ai_reply_delays()
        _set_inside_business_hours()
        self.sender = FakeSender()
        self.bot = MockBot()
        # 隔离意图路由：本类只测发送前清洗/跳过规则，避免意图触发转人工干扰断言
        self._intent_patch = _patch_intent_to_consult()
        self._intent_patch.start()

    def tearDown(self):
        self._intent_patch.stop()

    def tearDown(self):
        _restore_ai_reply_delays()

    async def test_send_removes_punctuation(self):
        self.bot = MockBot(reply_text="亲，这款商品48小时内发货哦。请您放心。")
        handler = AIReplyHandler(bot=self.bot)
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("发货时间？")
            ok = await handler.handle(context, make_metadata())
        self.assertTrue(ok)
        texts = [call[3] for call in self.sender.calls["send_text"]]
        self.assertTrue(texts)
        for t in texts:
            for p in "，,。.？?":
                self.assertNotIn(p, t)
        self.assertIn("亲这款商品48小时内发货哦", texts[0])
        self.assertIn("请您放心", texts[1])

    async def test_punct_only_chunk_skipped(self):
        # 回复含纯标点分条 → 该条跳过，其余正常发送
        self.bot = MockBot(reply_text="你好。？？？.。")
        handler = AIReplyHandler(bot=self.bot)
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("在吗")
            ok = await handler.handle(context, make_metadata())
        self.assertTrue(ok)
        texts = [call[3] for call in self.sender.calls["send_text"]]
        self.assertEqual(texts, ["你好"])

    async def test_pure_punctuation_reply_falls_back_cleaned(self):
        # 整条回复仅由标点组成 → 清洗后为空，走备用回复（备用回复同样被清洗）
        self.bot = MockBot(reply_text="？？？")
        handler = AIReplyHandler(bot=self.bot)
        with mock.patch("bridge.sender.get_sender", return_value=self.sender):
            context = make_context("在吗")
            ok = await handler.handle(context, make_metadata())
        self.assertTrue(ok)
        texts = [call[3] for call in self.sender.calls["send_text"]]
        self.assertTrue(texts)
        for t in texts:
            for p in "，,。.？?":
                self.assertNotIn(p, t)


# ============================================================================
# 测试 13：send_goods_link 兼容 AI 传入的整数 recipient_uid
# ============================================================================

class TestSendGoodsLinkIntUid(unittest.TestCase):
    """AI 以整数形式提取 recipient_uid 时，工具应正常执行并转为字符串"""

    def setUp(self):
        from Agent.CustomerAgent.tools import send_goods_link as sg_module
        self.sg_module = sg_module
        self.calls = []

        class _FakeSender:
            def __init__(self, calls):
                self.calls = calls

            def send_product_card(self, shop_id, user_id, recipient_uid, goods_id, biz_type=2):
                self.calls.append((shop_id, user_id, recipient_uid, goods_id, biz_type))
                return {"success": True}

        class _FakeProductManager:
            """隔离校验通过：返回与请求 goods_id 一致的商品详情"""

            def __init__(self, *args, **kwargs):
                pass

            def get_product_detail(self, goods_id):
                return {"success": True, "product_info": {"goods_id": goods_id}}

        self.sender = _FakeSender(self.calls)
        self._patch = mock.patch.object(sg_module, "get_sender", return_value=self.sender)
        self._patch.start()
        self._pm_patch = mock.patch.object(
            sg_module, "ProductManager", _FakeProductManager
        )
        self._pm_patch.start()

    def tearDown(self):
        self._patch.stop()
        self._pm_patch.stop()

    def test_params_model_accepts_int_uid(self):
        p = self.sg_module.SendGoodsLinkParams(
            recipient_uid=5927195871573, goods_id=100123, shop_id="1", user_id="2"
        )
        self.assertEqual(p.recipient_uid, 5927195871573)

    def test_execute_tool_with_int_uid(self):
        """身份字段来自受信任 dependencies（int 值），goods_id 为 LLM 业务参数"""
        result = execute_tool(
            "send_goods_link",
            json.dumps({"goods_id": 100123}),
            {
                "recipient_uid": 5927195871573,
                "shop_id": 661962391,
                "user_id": 189109418,
                "channel_type": "pinduoduo",
            },
        )
        self.assertIn("商品卡片发送成功", result)
        self.assertEqual(len(self.calls), 1)
        shop_id, user_id, recipient_uid, goods_id, biz_type = self.calls[0]
        self.assertEqual(recipient_uid, "5927195871573")
        self.assertEqual(shop_id, "661962391")
        self.assertEqual(user_id, "189109418")
        self.assertEqual(goods_id, 100123)

    def test_execute_tool_missing_params(self):
        """身份字段齐全但缺少业务参数 goods_id → 工具自身校验拦截"""
        result = execute_tool(
            "send_goods_link",
            json.dumps({}),
            {
                "recipient_uid": "buyer1",
                "shop_id": 661962391,
                "user_id": 189109418,
                "channel_type": "pinduoduo",
            },
        )
        self.assertIn("缺少必要", result)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ============================================================================
# 测试 14：意图分类器（轻量语义路由，替代售后关键词硬匹配）
# ============================================================================

from Message.handlers import intent_classifier as ic_module  # noqa: E402
from Message.handlers.intent_classifier import IntentClassifier  # noqa: E402


class TestIntentClassifier(unittest.IsolatedAsyncioTestCase):
    """意图分类器纯单元测试（不发起真实网络请求）"""

    def test_should_transfer_matrix(self):
        """操作/投诉/负面情绪且置信度达标 → 转；咨询/低置信 → 不转；
        other 高置信(≥阈值) → 转（LLM确认无法处理）；other 低置信(<阈值) → 不转（分类器不确定，让AI回）；
        unknown → 保守转（分类失败/超时）。"""
        IC = IntentClassifier
        self.assertTrue(IC.should_transfer("operation", 0.9, 0.6))
        self.assertTrue(IC.should_transfer("complaint", 0.6, 0.6))
        self.assertTrue(IC.should_transfer("negative_emotion", 0.7, 0.6))
        self.assertFalse(IC.should_transfer("consult", 0.99, 0.6))
        self.assertFalse(IC.should_transfer("operation", 0.3, 0.6))
        # other: 高置信才转，低置信不转
        self.assertTrue(IC.should_transfer("other", 0.9, 0.6))
        self.assertFalse(IC.should_transfer("other", 0.5, 0.6))   # 低置信 other → AI 先回
        self.assertFalse(IC.should_transfer("other", 0.3, 0.6))
        # unknown: 无论置信度都转（保守兜底）
        self.assertTrue(IC.should_transfer("unknown", 0.0, 0.6))

    def test_parse_valid(self):
        c = IntentClassifier({})
        resp = type("R", (), {"content": '前缀 {"intent":"operation","confidence":0.82} 后缀'})()
        self.assertEqual(c._parse(resp), {"intent": "operation", "confidence": 0.82})

    def test_parse_invalid(self):
        c = IntentClassifier({})
        self.assertEqual(
            c._parse(type("R", (), {"content": "无法解析"})()),
            {"intent": "unknown", "confidence": 0.0},
        )

    async def test_classify_disabled(self):
        c = IntentClassifier({"enabled": False})
        r = await c.classify("你好")
        self.assertEqual(r["intent"], "unknown")

    async def test_classify_empty(self):
        c = IntentClassifier({})
        r = await c.classify("")
        self.assertEqual(r["intent"], "unknown")

    async def test_classify_calls_llm_and_caches(self):
        c = IntentClassifier({})
        calls = []

        async def fake(text, hint, history=None):
            calls.append(text)
            return {"intent": "consult", "confidence": 0.9}

        with mock.patch.object(c, "_call_llm", fake):
            r1 = await c.classify("运费谁出")
            r2 = await c.classify("运费谁出")
        self.assertEqual(r1["intent"], "consult")
        self.assertEqual(len(calls), 1)  # 第二次命中缓存，不再调用 LLM

    def test_build_user_includes_history(self):
        """_build_user 在有上下文时拼接最近若干轮，并标注当前消息；无上下文时不出现上下文块。"""
        c = IntentClassifier({})
        history = [
            {"role": "user", "content": "有没有适合冬天的帽子"},
            {"role": "assistant", "content": "有的亲，这几款毛线帽都不错，需要推荐吗"},
            {"role": "system", "content": "（系统提示）"},
        ]
        user = c._build_user("要", False, history=history)
        self.assertIn("对话上下文", user)
        self.assertIn("买家：有没有适合冬天的帽子", user)
        self.assertIn("客服：有的亲，这几款毛线帽都不错，需要推荐吗", user)
        self.assertIn("当前消息：要", user)
        # system 角色也按角色名标注（不丢失）
        self.assertIn("系统：", user)
        # 无上下文：不应出现上下文块，仅当前消息
        user2 = c._build_user("要", False)
        self.assertNotIn("对话上下文", user2)
        self.assertIn("当前消息：要", user2)

    async def test_classify_passes_history_to_llm(self):
        """classify 把传入的 history 透传给 _call_llm / _build_user。"""
        c = IntentClassifier({})
        captured = {}

        async def fake(text, hint, history=None):
            captured["text"] = text
            captured["history"] = history
            return {"intent": "consult", "confidence": 0.9}

        hist = [{"role": "assistant", "content": "需要推荐吗"}]
        with mock.patch.object(c, "_call_llm", fake):
            await c.classify("要", after_sale_hint=False, history=hist)
        self.assertEqual(captured["text"], "要")
        self.assertEqual(captured["history"], hist)


# ============================================================================
# 测试 15：AI 处理器意图路由（基于语义转人工，保留既有规则）
# ============================================================================

class _StubClassifier:
    """测试替身：直接返回指定意图，跳过真实 LLM。"""

    def __init__(self, intent, confidence=0.9, threshold=0.6, enabled=True):
        self.intent = intent
        self.confidence = confidence
        self.threshold = threshold
        self.enabled = enabled

    async def classify(self, text, after_sale_hint=False, history=None):
        return {"intent": self.intent, "confidence": self.confidence}


class _ContextAwareStub:
    """测试替身：模拟真实分类器的「上下文判意图」行为，并记录收到的 history。

    行为：孤立的「要」→ 判 other（→ 转人工）；但当上下文里客服刚问过「需要推荐吗」
    时，「要」应判 consult（续接咨询，不转人工）。用于验证方案 A 的上下文注入修复。
    """

    def __init__(self):
        self.captured = None  # (text, history)
        self.enabled = True
        self.threshold = 0.6

    async def classify(self, text, after_sale_hint=False, history=None):
        self.captured = (text, history)
        if "要" in text and history:
            for h in history:
                if h.get("role") == "assistant" and "需要推荐吗" in (h.get("content") or ""):
                    return {"intent": "consult", "confidence": 0.9}
        return {"intent": "other", "confidence": 0.5}


class TestAIHandlerIntentRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        SessionState().clear_handoff("shop1:buyer1")
        notify_tracker.clear("shop1:buyer1")
        _zero_ai_reply_delays()
        _set_inside_business_hours()
        self.bot = MockBot()
        self.handler = AIReplyHandler(bot=self.bot)
        self.sender = FakeSender()
        # AI 回复路径在方法内动态 from bridge.sender import get_sender，需 patch 该模块
        self._sender_patch = mock.patch("bridge.sender.get_sender", return_value=self.sender)
        self._sender_patch.start()
        # 转人工路径使用 move_conversation 模块顶层导入的 get_sender，需单独 patch
        self._mc_sender_patch = mock.patch.object(mc_module, "get_sender", return_value=self.sender)
        self._mc_sender_patch.start()
        self.capture = CaptureWebhook()
        self._notify_patch = mock.patch.object(notify_module, "send_wechat_notification_sync", self.capture.send)
        self._notify_patch.start()
        self._orig_main = get_config("transfer.main_account_user_ids", [])
        self._orig_sub = get_config("transfer.sub_account_uids", [])

    def tearDown(self):
        self._sender_patch.stop()
        self._mc_sender_patch.stop()
        self._notify_patch.stop()
        _app_config.set("transfer.main_account_user_ids", self._orig_main, save=False)
        _app_config.set("transfer.sub_account_uids", self._orig_sub, save=False)
        _restore_ai_reply_delays()

    async def _route(self, message, intent, confidence=0.9):
        stub = _StubClassifier(intent, confidence=confidence)
        with mock.patch.object(ic_module, "get_intent_classifier", return_value=stub):
            return await self.handler.handle(make_context(message), make_metadata())

    async def test_consult_does_not_transfer(self):
        """售后咨询（consult）→ AI 自主回答，不转人工。"""
        ok = await self._route("退货流程是什么", "consult")
        self.assertTrue(ok)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        self.assertEqual(len(self.bot.calls), 1)  # AI 已回答

    async def test_operation_transfers(self):
        """操作诉求（operation）→ 转人工 + 企业微信通知。"""
        ok = await self._route("立刻给我退款", "operation")
        self.assertTrue(ok)
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("AI意图识别触发转人工", self.capture.messages[0])

    async def test_complaint_transfers(self):
        ok = await self._route("投诉你们", "complaint")
        self.assertTrue(ok)
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        self.assertEqual(len(self.capture.messages), 1)

    async def test_negative_emotion_transfers(self):
        """负面情绪（negative_emotion）→ 转人工（即便未明确要求操作）。"""
        ok = await self._route("你们太慢了，气死我了", "negative_emotion")
        self.assertTrue(ok)
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        self.assertEqual(len(self.capture.messages), 1)

    async def test_unknown_transfers(self):
        """分类未知（识别不出意图）→ 保守转人工。"""
        ok = await self._route("在吗", "unknown")
        self.assertTrue(ok)
        self.assertEqual(len(self.sender.calls["transfer_to_cs"]), 1)
        self.assertEqual(self.bot.calls, [])

    async def test_low_confidence_no_transfer(self):
        """置信度低于阈值 → 不转人工。"""
        ok = await self._route("我要退货", "operation", confidence=0.2)
        self.assertTrue(ok)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])

    async def test_disabled_no_transfer(self):
        """意图分类禁用 → 不转人工，走 AI 回复。"""
        stub = _StubClassifier("operation", enabled=False)
        with mock.patch.object(ic_module, "get_intent_classifier", return_value=stub):
            ok = await self.handler.handle(make_context("立刻给我退款"), make_metadata())
        self.assertTrue(ok)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])

    async def test_sub_account_intent_silent(self):
        """规则 7：子账号意图转人工 → 不调 API，仅静默标记 + 通知。"""
        _app_config.set("transfer.sub_account_uids", ["cs_shop1_user1"], save=False)
        _app_config.set("transfer.main_account_user_ids", [], save=False)
        ok = await self._route("我要退货", "operation")
        self.assertTrue(ok)
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        self.assertTrue(SessionState().is_handoff("shop1:buyer1"))
        self.assertEqual(len(self.capture.messages), 1)

    async def test_short_reply_in_context_does_not_transfer(self):
        """方案 A 修复验证：孤立看是 other 的「要」，在「需要推荐吗」上下文中应判 consult，不转人工。

        路由阶段取最近若干轮历史注入分类器；上下文里客服刚问「需要推荐吗」，
        买家回「要」应续接咨询（consult），而非被误判 other 转人工。
        """
        _app_config.set("intent.context_turns", 12, save=False)
        fake_history = [
            {"role": "user", "content": "有没有适合冬天的帽子"},
            {"role": "assistant", "content": "有的亲，这几款毛线帽都不错，需要推荐吗"},
        ]
        stub = _ContextAwareStub()
        # MockBot 无 get_session_history；临时挂上，让路由阶段取到上下文
        self.handler.bot.get_session_history = lambda sid, limit=None: fake_history
        with mock.patch.object(ic_module, "get_intent_classifier", return_value=stub):
            ok = await self.handler.handle(make_context("要"), make_metadata())
        self.assertTrue(ok)
        # 分类器确实拿到了上下文，且其中包含客服的「需要推荐吗」
        self.assertIsNotNone(stub.captured)
        _text, _history = stub.captured
        self.assertIn("要", _text)
        self.assertTrue(
            any(
                h.get("role") == "assistant" and "需要推荐吗" in (h.get("content") or "")
                for h in (_history or [])
            )
        )
        # 上下文判为 consult → 不转人工
        self.assertEqual(self.sender.calls["transfer_to_cs"], [])
        self.assertFalse(SessionState().is_handoff("shop1:buyer1"))


# ============================================================================
# 测试 16：关键词类别（category）持久化与迁移播种
# ============================================================================

class TestKeywordCategoryDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = DatabaseManager(db_path=str(Path(self.tmp) / "kw.db"))

    def tearDown(self):
        self.db.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_and_get_category(self):
        self.assertTrue(self.db.add_keyword("测试售后词", "after_sale"))
        row = self.db.get_keyword("测试售后词")
        self.assertEqual(row["category"], "after_sale")
        all_kw = self.db.get_all_keywords()
        self.assertTrue(
            any(k["keyword"] == "测试售后词" and k["category"] == "after_sale" for k in all_kw)
        )

    def test_default_category_transfer(self):
        self.assertTrue(self.db.add_keyword("必转测试词"))
        self.assertEqual(self.db.get_keyword("必转测试词")["category"], "transfer")

    def test_seed_after_sale_present(self):
        """迁移播种：after_sale 默认词存在且为 after_sale，必转词为 transfer。"""
        all_kw = {k["keyword"]: k["category"] for k in self.db.get_all_keywords()}
        self.assertEqual(all_kw.get("退款"), "after_sale")
        self.assertEqual(all_kw.get("转人工"), "transfer")
