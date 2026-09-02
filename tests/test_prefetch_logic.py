"""离线验证 ai_handler._prefetch_order_if_needed 的关键词门控与提示词格式化。

不启动浏览器、不打 PDD 接口：用 stub 替换 query_order_status 工具，仅验证：
- 订单/物流意图命中 → 按工具返回形态生成正确提示串
- 非订单意图 / 发货地负向词 → 返回空串（不触发查证）
- 缺参数 → 返回空串
- 波浪号清理
"""
import asyncio
import sys
import types
import logging
import os
import unittest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from Message.handlers.ai_handler import AIReplyHandler


class _FakeParams:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestPrefetchLogic(unittest.IsolatedAsyncioTestCase):
    """ai_handler._prefetch_order_if_needed 离线桩测试"""

    @classmethod
    def setUpClass(cls):
        cls._orig_qos_mod = sys.modules.get("Agent.CustomerAgent.tools.query_order_status")
        cls._fake_returns = {}

        def _fake_qos(params):
            uid = str(getattr(params, "recipient_uid", ""))
            return cls._fake_returns.get(uid, "")

        fake_mod = types.ModuleType("Agent.CustomerAgent.tools.query_order_status")
        fake_mod.query_order_status = _fake_qos
        fake_mod.QueryOrderStatusParams = _FakeParams
        sys.modules["Agent.CustomerAgent.tools.query_order_status"] = fake_mod

        # 构造实例但绕过 __init__（避免 DI 容器依赖）
        cls._handler = object.__new__(AIReplyHandler)
        cls._handler.logger = logging.getLogger("test_prefetch")

    @classmethod
    def tearDownClass(cls):
        if cls._orig_qos_mod is not None:
            sys.modules["Agent.CustomerAgent.tools.query_order_status"] = cls._orig_qos_mod
        else:
            sys.modules.pop("Agent.CustomerAgent.tools.query_order_status", None)

    async def _prefetch(self, meta, text):
        return await self._handler._prefetch_order_if_needed(meta, text)

    async def test_order_intent_with_data(self):
        self._fake_returns["5927195871573"] = (
            "[untrusted_order_data]\n您的订单（共 1 条）：\n"
            "订单号: 260811-330091805023251\n[/untrusted_order_data]"
        )
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "我的订单到哪里了")
        self.assertIn("untrusted_order_data", r)
        self.assertIn("直接基于下方数据", r)

    async def test_order_intent_no_order(self):
        self._fake_returns["5927195871573"] = "未查询到您在本店的订单记录哦。如果您刚下单，可能稍有延迟"
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "发货了吗")
        self.assertIn("未查询到您在本店的订单", r)
        self.assertIn("如实告知", r)

    async def test_api_error_transparent(self):
        self._fake_returns["5927195871573"] = "亲，我这边系统暂时没刷出您订单的最新状态呢。您可以在拼多多App里点"
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "我的快递什么时候到?")
        self.assertIn("原样转告", r)

    async def test_non_order_intent_empty(self):
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "有人吗")
        self.assertEqual(r, "")

    async def test_negative_shipping_location_empty(self):
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "你们是从哪里发货的")
        self.assertEqual(r, "")

    async def test_missing_params_empty(self):
        r = await self._prefetch({"from_uid": "x"}, "我的订单")
        self.assertEqual(r, "")

    async def test_wave_char_removed(self):
        self._fake_returns["5927195871573"] = "未查询到您在本店的订单记录哦～如果您刚下单"
        meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}
        r = await self._prefetch(meta, "我的订单到哪了")
        self.assertNotIn("~", r)
        self.assertNotIn("～", r)


if __name__ == "__main__":
    unittest.main()
