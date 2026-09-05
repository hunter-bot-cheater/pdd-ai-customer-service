"""
整数米数组合计算工具（2026-09-05 复盘）。

普通花型店铺只售 1米/2米/3米 基础规格，买家要任意整数米数（X米，X≥4）
都可以用 1/2/3 多拍组合实现。本模块提供贪心算法计算最少件数组合方案，
供 message_builder 规则说明、ai_handler 兜底注入、单元测试复用。

算法（贪心：优先用 3 米，余数用 2 米，最后用 1 米凑，r=1 时拆 1 个 3 米）：
    y3, r = divmod(X, 3)
    r == 0 → y3 × 3米
    r == 2 → y3 × 3米 + 1 × 2米
    r == 1 且 y3 ≥ 2 → (y3-1) × 3米 + 2 × 2米
    r == 1 且 y3 == 1（X==4）→ 1 × 2米 + 2 × 1米
"""
from __future__ import annotations

from typing import List, Tuple


def compute_integer_meter_plan(meter: int) -> List[Tuple[int, int]]:
    """计算 X 米的最小件数组合方案。

    Args:
        meter: 整数米数（X ≥ 4）。

    Returns:
        [(规格米数, 件数), ...] 按规格从大到小排列，例如 7 → [(3, 1), (2, 2)]。
        若 meter 超出可组合范围（理论上 ≥4 都能组合），返回空列表。

    Raises:
        ValueError: meter 不是正整数。
    """
    if not isinstance(meter, int) or meter < 1:
        raise ValueError(f"meter 必须是正整数，得到 {meter!r}")

    if meter <= 3:
        return [(meter, 1)]

    y3, r = divmod(meter, 3)
    plan: List[Tuple[int, int]] = []

    if r == 0:
        plan.append((3, y3))
    elif r == 2:
        plan.append((3, y3))
        plan.append((2, 1))
    else:  # r == 1
        if y3 >= 2:
            plan.append((3, y3 - 1))
            plan.append((2, 2))
        else:
            # meter == 4: 1 × 2米 + 2 × 1米
            plan.append((2, 1))
            plan.append((1, meter - 2))

    return plan


def format_integer_meter_plan(plan: List[Tuple[int, int]]) -> str:
    """格式化组合方案为自然语言，如 [(3,1),(2,2)] → '1件3米 + 2件2米'。

    Args:
        plan: compute_integer_meter_plan 的输出。

    Returns:
        中文连接字符串，空 plan 返回 ""。
    """
    parts = [f"{n}件{spec}米" for spec, n in plan if n > 0]
    return " + ".join(parts)


def total_pieces(plan: List[Tuple[int, int]]) -> int:
    """组合方案的总件数。"""
    return sum(n for _, n in plan if n > 0)