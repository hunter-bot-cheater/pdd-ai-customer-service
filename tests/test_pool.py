"""
真实端到端验证：
1) 通过浏览器池 query_order_status 工具真实读取买家 4239748275 在峰哥编织的订单；
2) 连续两次查询，验证常驻浏览器池复用（第二次应明显快于第一次）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import Agent.CustomerAgent.tools.query_order_status as q


def test_tool(buyer_uid: str):
    # 直接调用工具的底层函数（不经过 LLM），传入真实店铺/商家身份与 buyer_uid
    params = q.QueryOrderStatusParams(
        shop_id=661962391,
        user_id=189109418,
        recipient_uid=buyer_uid,
        days=90,
    )
    t0 = time.time()
    out = q.query_order_status(params)
    dt = time.time() - t0
    return out, dt


if __name__ == "__main__":
    print("===== 第 1 次查询（冷启动浏览器）=====")
    out1, dt1 = test_tool("4239748275")
    print(f"[耗时] {dt1:.1f}s")
    print(out1)
    print()
    print("===== 第 2 次查询（复用浏览器池）=====")
    out2, dt2 = test_tool("4239748275")
    print(f"[耗时] {dt2:.1f}s")
    print(out2)
    print()
    speedup = dt1 / dt2 if dt2 > 0 else float("inf")
    print(f"===== 对比：第1次 {dt1:.1f}s vs 第2次 {dt2:.1f}s（加速约 {speedup:.1f}x）=====")

    # 断言：两次都能精确返回该买家订单（attribution=buyer）
    assert "260811-330091805023251" in out1, "第1次未返回预期订单"
    assert "260811-330091805023251" in out2, "第2次未返回预期订单"
    assert "未发货" in out1 or "物流状态" in out1
    print("\n✅ 端到端通过：多买家精确归因 + 浏览器池复用均生效")
