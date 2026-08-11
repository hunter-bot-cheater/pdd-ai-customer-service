"""离线验证 ai_handler._prefetch_order_if_needed 的关键词门控与提示词格式化。

不启动浏览器、不打 PDD 接口：用 stub 替换 query_order_status 工具，仅验证：
- 订单/物流意图命中 → 按工具返回形态生成正确提示串
- 非订单意图 / 发货地负向词 → 返回空串（不触发查证）
- 缺参数 → 返回空串
"""
import asyncio
import sys
import types
import logging
import os

PROJ = "D:/ai客服/Customer-Agent-main"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# 在方法运行时的懒加载 import 之前注入 stub 模块
_FAKE_RETURNS = {}


def _fake_qos(params):
    uid = str(getattr(params, "recipient_uid", ""))
    return _FAKE_RETURNS.get(uid, "")


class _FakeParams:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_fake_mod = types.ModuleType("Agent.CustomerAgent.tools.query_order_status")
_fake_mod.query_order_status = _fake_qos
_fake_mod.QueryOrderStatusParams = _FakeParams
sys.modules["Agent.CustomerAgent.tools.query_order_status"] = _fake_mod

from Message.handlers.ai_handler import AIReplyHandler  # noqa: E402

# 构造实例但绕过 __init__（避免 DI 容器依赖）
_h = object.__new__(AIReplyHandler)
_h.logger = logging.getLogger("test_prefetch")


async def run():
    meta = {"from_uid": "5927195871573", "shop_id": "661962391", "user_id": "189109418"}

    # 用例1：订单意图 + 有订单数据
    _FAKE_RETURNS["5927195871573"] = (
        "[untrusted_order_data]\n您的订单（共 1 条）：\n"
        "订单号: 260811-330091805023251\n[/untrusted_order_data]"
    )
    r1 = await _h._prefetch_order_if_needed(meta, "我的订单到哪里了")
    print("用例1 有数据:", "untrusted_order_data" in r1, "| 指令含'直接基于下方数据'?", "直接基于下方数据" in r1)

    # 用例2：订单意图 + 无订单
    _FAKE_RETURNS["5927195871573"] = "未查询到您在本店的订单记录哦。如果您刚下单，可能稍有延迟"
    r2 = await _h._prefetch_order_if_needed(meta, "发货了吗")
    print("用例2 无订单:", "未查询到您在本店的订单" in r2, "| 指令含'如实告知'?", "如实告知" in r2)

    # 用例3：接口异常（真人化兜底话术）→ 原样转告
    _FAKE_RETURNS["5927195871573"] = "亲，我这边系统暂时没刷出您订单的最新状态呢。您可以在拼多多App里点"
    r3 = await _h._prefetch_order_if_needed(meta, "我的快递什么时候到?")
    print("用例3 转告:", "原样转告" in r3)

    # 用例4：非订单消息 → 空串
    r4 = await _h._prefetch_order_if_needed(meta, "有人吗")
    print("用例4 非订单空串?", r4 == "")

    # 用例5：发货地负向词 → 空串
    r5 = await _h._prefetch_order_if_needed(meta, "你们是从哪里发货的")
    print("用例5 发货地负向空串?", r5 == "")

    # 用例6：缺参数 → 空串
    r6 = await _h._prefetch_order_if_needed({"from_uid": "x"}, "我的订单")
    print("用例6 缺参数空串?", r6 == "")

    # 用例7：波浪号清理（工具输出带 ~ 应被剥离）
    _FAKE_RETURNS["5927195871573"] = "未查询到您在本店的订单记录哦～如果您刚下单"
    r7 = await _h._prefetch_order_if_needed(meta, "我的订单到哪了")
    print("用例7 波浪号清理?", "~" not in r7 and "～" not in r7)

    ok = (
        ("untrusted_order_data" in r1 and "直接基于下方数据" in r1)
        and ("未查询到您在本店的订单" in r2 and "如实告知" in r2)
        and ("原样转告" in r3)
        and r4 == "" and r5 == "" and r6 == ""
        and ("~" not in r7 and "～" not in r7)
    )
    print("\n===== 总结 =====", "全部通过 OK" if ok else "存在失败!")


asyncio.run(run())
