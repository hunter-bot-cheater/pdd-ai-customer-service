"""退款/售后订单识别与格式化离线测试（无需浏览器）。

验证：userAllOrder 返回「未发货，退款成功」这类已退款订单时，bot 必须
识别出退款状态、不再把它报成「未发货请等待」，并在格式化输出里明确标注。
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Agent.CustomerAgent.tools.read_session_orders import (
    _classify_refund, normalize_customer_order,
)
from Agent.CustomerAgent.tools.query_order_status import _format_output


def test_classify_refund():
    assert _classify_refund("未发货，退款成功") == "已退款"
    assert _classify_refund("已退款") == "已退款"
    assert _classify_refund("退款中") == "退款中"
    assert _classify_refund("售后处理中") == "售后处理中"
    assert _classify_refund("待发货") == ""
    assert _classify_refund("已发货") == ""
    print("[ok] _classify_refund 分类正确")


def test_refunded_order_normalize():
    item = {
        "orderSn": "260811-330091805023251",
        "orderStatus": 1,
        "orderStatusStr": "未发货，退款成功",
        "shippingStatus": 0,
        "orderAmount": 2800,
        "uid": 4239748275,
        "orderGoodsList": {"goodsName": "保温杯", "spec": "红银", "goodsNumber": 1},
    }
    o = normalize_customer_order(item)
    assert o["refund_status_desc"] == "已退款", o
    assert o["needs_shipping"] is False, o
    assert "无需发货" in o["shipping_status_desc"], o
    print("[ok] 退款单归一化: refund=已退款, needs_shipping=False, 物流=无需发货")


def test_active_order_normalize():
    item = {
        "orderSn": "260812-ACTIVE",
        "orderStatus": 1,
        "orderStatusStr": "待发货",
        "shippingStatus": 0,
        "orderAmount": 5000,
        "uid": 4239748275,
    }
    o = normalize_customer_order(item)
    assert o["refund_status_desc"] == "", o
    assert o["needs_shipping"] is True, o
    assert o["shipping_status_desc"] == "未发货", o
    print("[ok] 在途单归一化: needs_shipping=True, 物流=未发货")


def test_format_distinguishes_refunded():
    refunded = normalize_customer_order({
        "orderSn": "260811-330091805023251", "orderStatus": 1,
        "orderStatusStr": "未发货，退款成功", "shippingStatus": 0,
        "orderAmount": 2800, "uid": 4239748275,
        "orderGoodsList": {"goodsName": "保温杯", "spec": "红银", "goodsNumber": 1},
    })
    active = normalize_customer_order({
        "orderSn": "260812-ACTIVE", "orderStatus": 1,
        "orderStatusStr": "待发货", "shippingStatus": 0,
        "orderAmount": 5000, "uid": 4239748275,
    })
    out = _format_output([refunded, active], is_filtered=True)
    assert "1 条已退款/已关闭、1 条在途" in out, out
    assert "退款/售后: 已退款（该订单已无需发货" in out, out
    assert "无需发货（已退款/已关闭）" in out, out
    # 在途单仍按未发货展示
    assert "物流状态: 未发货" in out, out
    print("[ok] _format_output 区分已退款与在途，并标注退款状态")


def test_format_no_wave_after_strip():
    # 回归：预取层会 strip 波浪号，这里确认格式化本身不主动注入波浪号
    refunded = normalize_customer_order({
        "orderSn": "260811-330091805023251", "orderStatus": 1,
        "orderStatusStr": "未发货，退款成功", "shippingStatus": 0,
        "orderAmount": 2800, "uid": 4239748275,
    })
    out = _format_output([refunded], is_filtered=True)
    assert "～" not in out and "~" not in out, out
    print("[ok] 格式化输出不含波浪号")


if __name__ == "__main__":
    test_classify_refund()
    test_refunded_order_normalize()
    test_active_order_normalize()
    test_format_distinguishes_refunded()
    test_format_no_wave_after_strip()
    print("\n全部退款识别测试通过 OK")
