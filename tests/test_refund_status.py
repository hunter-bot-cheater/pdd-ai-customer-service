"""退款/售后订单识别与格式化离线测试（无需浏览器）。

验证：userAllOrder 返回「未发货，退款成功」这类已退款订单时，bot 必须
识别出退款状态、不再把它报成「未发货请等待」；格式化输出层对已退款/已关闭
订单做过滤，不展示给客户。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Agent.CustomerAgent.tools.read_session_orders import (
    _classify_refund,
    normalize_customer_order,
)
from Agent.CustomerAgent.tools.query_order_status import _format_output


class TestRefundStatus(unittest.TestCase):
    def test_classify_refund(self):
        self.assertEqual(_classify_refund("未发货，退款成功"), "已退款")
        self.assertEqual(_classify_refund("已退款"), "已退款")
        self.assertEqual(_classify_refund("退款中"), "退款中")
        self.assertEqual(_classify_refund("售后处理中"), "售后处理中")
        self.assertEqual(_classify_refund("待发货"), "")
        self.assertEqual(_classify_refund("已发货"), "")

    def test_refunded_order_normalize(self):
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
        self.assertEqual(o["refund_status_desc"], "已退款")
        self.assertFalse(o["needs_shipping"])
        self.assertIn("无需发货", o["shipping_status_desc"])

    def test_active_order_normalize(self):
        item = {
            "orderSn": "260812-ACTIVE",
            "orderStatus": 1,
            "orderStatusStr": "待发货",
            "shippingStatus": 0,
            "orderAmount": 5000,
            "uid": 4239748275,
        }
        o = normalize_customer_order(item)
        self.assertEqual(o["refund_status_desc"], "")
        self.assertTrue(o["needs_shipping"])
        self.assertEqual(o["shipping_status_desc"], "未发货")

    def test_format_filters_refunded_orders(self):
        """已退款/已关闭订单不展示给客户，仅保留在途单"""
        refunded = normalize_customer_order({
            "orderSn": "260811-330091805023251",
            "orderStatus": 1,
            "orderStatusStr": "未发货，退款成功",
            "shippingStatus": 0,
            "orderAmount": 2800,
            "uid": 4239748275,
            "orderGoodsList": {"goodsName": "保温杯", "spec": "红银", "goodsNumber": 1},
        })
        active = normalize_customer_order({
            "orderSn": "260812-ACTIVE",
            "orderStatus": 1,
            "orderStatusStr": "待发货",
            "shippingStatus": 0,
            "orderAmount": 5000,
            "uid": 4239748275,
        })
        out = _format_output([refunded, active], is_filtered=True)
        self.assertNotIn("260811-330091805023251", out)
        self.assertIn("260812-ACTIVE", out)
        self.assertIn("物流状态: 未发货", out)

    def test_format_all_refunded_shows_empty(self):
        """全部订单均已退款/关闭 → 返回'暂无有效在途订单'"""
        refunded = normalize_customer_order({
            "orderSn": "260811-330091805023251",
            "orderStatus": 1,
            "orderStatusStr": "未发货，退款成功",
            "shippingStatus": 0,
            "orderAmount": 2800,
            "uid": 4239748275,
            "orderGoodsList": {"goodsName": "保温杯", "spec": "红银", "goodsNumber": 1},
        })
        out = _format_output([refunded], is_filtered=True)
        self.assertIn("暂无有效在途订单", out)
        self.assertNotIn("260811-330091805023251", out)

    def test_format_no_wave_after_strip(self):
        """回归：预取层会 strip 波浪号，格式化本身不主动注入波浪号"""
        refunded = normalize_customer_order({
            "orderSn": "260811-330091805023251",
            "orderStatus": 1,
            "orderStatusStr": "未发货，退款成功",
            "shippingStatus": 0,
            "orderAmount": 2800,
            "uid": 4239748275,
        })
        out = _format_output([refunded], is_filtered=True)
        self.assertNotIn("～", out)
        self.assertNotIn("~", out)


if __name__ == "__main__":
    unittest.main()
