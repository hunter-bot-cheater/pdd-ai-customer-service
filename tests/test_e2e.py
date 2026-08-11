"""端到端：调用真实 query_order_status 工具（走真实读取器）查询买家 4239748275 的订单。"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Agent.CustomerAgent.tools.query_order_status as q


def main():
    params = q.QueryOrderStatusParams(
        shop_id="661962391",
        user_id="189109418",
        recipient_uid="4239748275",
        days=90,
    )
    print(">>> 调用 query_order_status 工具（真实读取器）...")
    out = q.query_order_status(params)
    print("=== 工具返回 ===")
    print(out)
    print("=== 校验 ===")
    assert "260811-330091805023251" in out, "未包含订单号！"
    assert "未发货" in out or "待发货" in out, "未包含发货状态！"
    print("✅ 端到端通过：工具正确返回了订单号与未发货/待发货状态")


if __name__ == "__main__":
    main()
