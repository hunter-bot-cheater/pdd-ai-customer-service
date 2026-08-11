from ..base_request import BaseRequest
import time
from utils.logger_loguru import get_logger


class OrderQuery(BaseRequest):
    """
    拼多多订单查询API（mms.pinduoduo.com 商家后台内部接口）

    能力：
    - 按时间范围查询店铺近期订单列表
    - 返回订单号、商品名、发货状态、物流信息等

    认证：复用 BaseRequest 的 cookie 机制（含 anti-content）
    """

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id)
        if cookies:
            self.update_cookies(cookies)

    def get_recent_orders(self, days=7, page=1, page_size=20):
        """
        查询近期订单列表

        Args:
            days (int): 查询最近N天的订单，默认7天
            page (int): 页码，默认1
            page_size (int): 每页数量，默认20，最大50

        Returns:
            dict: {
                "success": bool,
                "total": int,
                "orders": [...],
                "error_msg": str or None
            }
        """
        import math

        now = int(time.time())
        end_ts = now
        start_ts = now - (days * 86400)

        url = "https://mms.pinduoduo.com/mangkhut/mms/recentOrderList"

        data = {
            "orderType": 1,
            "afterSaleType": 1,
            "remarkStatus": -1,
            "urgeShippingStatus": -1,
            "groupStartTime": start_ts,
            "groupEndTime": end_ts,
            "pageNumber": page,
            "pageSize": min(page_size, 50),
            "sortType": 10,
        }

        anti_content = (
            self.cookies.get("anti_content")
            or self.cookies.get("anti-content", "")
        )
        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "anti-content": anti_content,
            "content-type": "application/json;charset=UTF-8",
            "origin": "https://mms.pinduoduo.com",
            "referer": "https://mms.pinduoduo.com/chat-merchant/index.html",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            ),
        }

        result = self.post(url, json_data=data, headers=headers)

        if result and (result.get("success") is True or "result" in result):
            return self._parse_order_list(result)
        else:
            # 真实响应字段为蛇形 error_code/error_msg（已对真实账号验证）
            error_msg = (
                result.get("error_msg")
                or result.get("errorMsg")
                or result.get("error_code")
                or "订单查询接口无响应"
                if result
                else "订单查询接口无响应"
            )
            self.logger.error(f"订单列表查询失败: {error_msg}")
            return {"success": False, "error_msg": error_msg, "orders": [], "total": 0}

    def _parse_order_list(self, response_data):
        """解析 recentOrderList 响应"""
        try:
            result_data = response_data.get("result", {})
            total = result_data.get("totalItemNum", 0)
            raw_items = result_data.get("pageItems", [])

            orders = []
            for item in raw_items:
                order = {
                    "order_sn": item.get("orderSn", ""),
                    "order_sequence_no": item.get("orderSequenceNo", ""),
                    "goods_name": item.get("goodsName", ""),
                    "goods_id": item.get("goodsId", ""),
                    "spec": item.get("spec", ""),
                    "quantity": item.get("quantity", 0),
                    "order_status": item.get("orderStatus"),
                    "order_status_desc": _map_order_status(item.get("orderStatus")),
                    "shipping_status": item.get("shippingStatus"),
                    "shipping_status_desc": _map_shipping_status(
                        item.get("shippingStatus")
                    ),
                    "logistics_company": item.get("logisticsCompany", ""),
                    "logistics_sn": item.get("logisticsSn", ""),
                    "confirm_time": item.get("confirmTime", ""),
                    "pay_time": item.get("payTime", ""),
                    "after_sales_status": item.get("afterSalesStatus"),
                    "buyer_nick": item.get("buyerNick", "") or "",
                }
                orders.append(order)

            return {"success": True, "total": total, "orders": orders}

        except Exception as e:
            self.logger.error(f"解析订单列表失败: error_type={type(e).__name__}")
            return {"success": False, "error_msg": f"解析异常: {e}", "orders": [], "total": 0}


# ---- 状态映射（基于拼多多商家后台通用状态码）----

def _map_order_status(status_code):
    """订单成交状态码 → 中文描述"""
    _MAP = {
        0: "未支付",
        1: "已支付待成团",
        2: "已成交（已签收）",
        3: "已取消",
        5: "已结算",
        8: "非多多进宝商品",
    }
    return _MAP.get(status_code, f"未知状态({status_code})")


def _map_shipping_status(code):
    """发货状态码 → 中文描述"""
    _MAP = {
        0: "未发货",
        1: "已发货",
        2: "已揽收",
        3: "运输中",
        4: "派送中",
        5: "已签收",
        -1: "无需发货",
    }
    return _MAP.get(code, f"物流状态({code})")
