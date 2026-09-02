"""离线验证新的「按买家 uid 直接归因」路由逻辑（不依赖实时登录，避免触发 PDD 风控）。

用两个不同买家各自在本店的订单构造 fake 池（按 uid 返回其订单），验证：
- 买家A(1单) 查询 -> 只返回A的订单（不泄漏买家B的订单）
- 买家B(2单) 查询 -> 返回B的两单（都是本人，无跨买家泄漏）
- 买家无订单 -> 返回「未查询到您在本店的订单」
- 买家B + 给定 order_sn -> 在该买家订单内精确匹配那一单
- 买家B + 错误 order_sn -> 提示未找到并列出其本人订单
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import Agent.CustomerAgent.tools.query_order_status as qos
from Agent.CustomerAgent.tools.query_order_status import query_order_status, QueryOrderStatusParams

SHOP, USER = "661962391", "189109418"

# 买家A：1 单
ORDER_A = {
    "order_sn": "260811-330091805023251", "goods_name": "浙江绍兴特产保温杯,限量款",
    "order_status": 1, "order_status_desc": "未发货，退款成功", "shipping_status": 0,
    "shipping_status_desc": "未发货", "logistics_company": "", "logistics_sn": "",
    "pay_amount": 28.0, "buyer_id": "4239748275", "quantity": 1, "spec": "红银",
    "raw": {},
}
# 买家B：2 单
ORDER_B1 = {
    "order_sn": "260811-626796839282389", "goods_name": "保温杯", "order_status": 1,
    "order_status_desc": "待发货", "shipping_status": 0, "shipping_status_desc": "未发货",
    "logistics_company": "", "logistics_sn": "", "pay_amount": 28.0,
    "buyer_id": "555555555", "quantity": 1, "spec": "红银", "raw": {},
}
ORDER_B2 = {
    "order_sn": "260811-682759350642389", "goods_name": "杯子", "order_status": 2,
    "order_status_desc": "已发货", "shipping_status": 1, "shipping_status_desc": "已发货",
    "logistics_company": "中通快递", "logistics_sn": "ZT123456", "pay_amount": 50.0,
    "buyer_id": "555555555", "quantity": 1, "spec": "", "raw": {},
}

# 按 uid 返回该买家订单的假池
_FAKE = {
    "4239748275": [ORDER_A],
    "555555555": [ORDER_B1, ORDER_B2],
}


class _FakePool:
    def fetch_customer_orders(self, name, password="", uid="", headless=True):
        orders = _FAKE.get(str(uid), [])
        return {"success": True, "orders": orders, "total": len(orders), "returned": len(orders)}


qos.get_order_browser_pool = lambda: _FakePool()

print("===== 用例1：买家A(1单) -> 只返回A的订单（不泄漏B） =====")
r1 = query_order_status(QueryOrderStatusParams(shop_id=SHOP, user_id=USER, recipient_uid="4239748275"))
print(r1)
print("  含A订单?", "含 OK" if ORDER_A["order_sn"] in r1 else "缺失!")
print("  泄漏B订单?", "泄漏!" if (ORDER_B1["order_sn"] in r1 or ORDER_B2["order_sn"] in r1) else "未泄漏 OK")

print("\n===== 用例2：买家B(2单) -> 返回B两单（无跨买家泄漏） =====")
r2 = query_order_status(QueryOrderStatusParams(shop_id=SHOP, user_id=USER, recipient_uid="555555555"))
print(r2)
print("  含B两单?", "含 OK" if (ORDER_B1["order_sn"] in r2 and ORDER_B2["order_sn"] in r2) else "缺失!")
print("  泄漏A订单?", "泄漏!" if ORDER_A["order_sn"] in r2 else "未泄漏 OK")

print("\n===== 用例3：买家无订单 -> 提示未查询到 =====")
r3 = query_order_status(QueryOrderStatusParams(shop_id=SHOP, user_id=USER, recipient_uid="999000999"))
print(r3)
print("  是否含任何真实订单号?", "含(泄漏/错误)!" if any(o["order_sn"] in r3 for o in (ORDER_A, ORDER_B1, ORDER_B2)) else "不含 OK")

print("\n===== 用例4：买家B + 给定 order_sn -> 精确匹配那一单 =====")
r4 = query_order_status(QueryOrderStatusParams(shop_id=SHOP, user_id=USER, recipient_uid="555555555", order_sn="260811-682759350642389"))
print(r4)
print("  含目标单?", "含 OK" if ORDER_B2["order_sn"] in r4 else "缺失!")
print("  是否误含另一单?", "误含!" if ORDER_B1["order_sn"] in r4 else "未误含 OK")

print("\n===== 用例5：买家B + 错误 order_sn -> 提示未找到并列出其订单 =====")
r5 = query_order_status(QueryOrderStatusParams(shop_id=SHOP, user_id=USER, recipient_uid="555555555", order_sn="260811-000000000000000"))
print(r5)
print("  是否列出B本人订单?", "是 OK" if (ORDER_B1["order_sn"] in r5 and ORDER_B2["order_sn"] in r5) else "否!")
print("  是否泄漏A订单?", "泄漏!" if ORDER_A["order_sn"] in r5 else "未泄漏 OK")

ok = (
    (ORDER_A["order_sn"] in r1) and (ORDER_B1["order_sn"] not in r1 and ORDER_B2["order_sn"] not in r1)
    and (ORDER_B1["order_sn"] in r2 and ORDER_B2["order_sn"] in r2) and (ORDER_A["order_sn"] not in r2)
    and all(o["order_sn"] not in r3 for o in (ORDER_A, ORDER_B1, ORDER_B2))
    and (ORDER_B2["order_sn"] in r4) and (ORDER_B1["order_sn"] not in r4)
    and (ORDER_B1["order_sn"] in r5 and ORDER_B2["order_sn"] in r5) and (ORDER_A["order_sn"] not in r5)
)
print("\n===== 总结 =====", "全部通过 OK" if ok else "存在失败!")
