"""
整数 / 0.5 米递增组合计算工具（2026-09-05 复盘）。

普通花型店铺只售 1米/2米/3米 基础规格，买家要任意整数米数（X米，X≥4）
都可以用 1/2/3 多拍组合实现。贡缎/1.5米宽幅商品支持 0.5米递增
（规格 1/1.5/2/2.5/3），需要用 DP 计算最少件数+最少规格种类的方案。

本模块提供两套算法：
  - compute_integer_meter_plan(X)：贪心 + 同规格 tie-breaker（件数 ≤ greedy 才用同规格）。
    例：4米→[(2,2)]（同规格 2件 2米，胜过 greedy [(3,1),(1,1)] 2件？同件数选同规格），
       8米→[(3,2),(2,1)]（greedy 3件混搭，胜过 [(2,4)] 4件同规格）。
  - compute_0p5_meter_plan(X)：DP（最少件数 + 最少规格种类）。
    例：5.5米→[(3,1),(2.5,1)]，6.5米→[(2.5,1),(2,2)]。
"""
from __future__ import annotations

from typing import List, Tuple


# =============================================================================
# 普通花型（整数米）组合计算
# =============================================================================

def _greedy_integer_plan(meter: int) -> List[Tuple[int, int]]:
    """贪心：尽量用 3 米，余数用 2 米凑，r=1 时拆 1 个 3 米变成 2米+1米（仅当 y3≥2 时拆）。"""
    y3, r = divmod(meter, 3)
    plan: dict = {}
    if y3 > 0:
        plan[3] = y3
    if r == 2:
        plan[2] = plan.get(2, 0) + 1
    elif r == 1:
        if y3 >= 2:
            # 拆 1 个 3 米：变成 2米+2米+1米
            plan[3] = y3 - 1
            plan[2] = plan.get(2, 0) + 2
        elif y3 == 1:
            # meter == 4: 1 件 3 米 + 1 件 1 米（2 件混搭）
            plan[3] = 1
            plan[1] = 1
        else:
            # meter == 1
            plan[1] = 1
    return [(s, n) for s, n in sorted(plan.items(), reverse=True) if n > 0]


def compute_integer_meter_plan(meter: int) -> List[Tuple[int, int]]:
    """计算 X 米的整数米组合方案（普通花型，规格 [1, 2, 3]）。

    策略：贪心最少件数 + 同规格优先（仅当同规格件数 ≤ greedy 件数才用同规格）。
    这样 4米→[(2,2)]（同规格胜出），8米→[(3,2),(2,1)]（greedy 3 件胜同规格 4 件）。

    Args:
        meter: 整数米数（X ≥ 1）。

    Returns:
        [(规格米数, 件数), ...] 按规格从大到小排列。

    Raises:
        ValueError: meter 不是正整数。
    """
    if not isinstance(meter, int) or meter < 1:
        raise ValueError(f"meter 必须是正整数，得到 {meter!r}")

    if meter <= 3:
        return [(meter, 1)]

    greedy = _greedy_integer_plan(meter)
    greedy_count = sum(n for _, n in greedy)

    # 同规格候选（最大规格优先，件数更少）
    for spec in [3, 2]:
        if meter % spec == 0:
            n = meter // spec
            if n <= greedy_count:
                return [(spec, n)]

    return greedy


# =============================================================================
# 贡缎 / 1.5 米宽幅（支持 0.5 米递增）组合计算
# =============================================================================

def compute_0p5_meter_plan(meter: float) -> List[Tuple[float, int]]:
    """计算 X 米的 0.5 米递增组合方案（贡缎/1.5米宽幅，规格 [1, 1.5, 2, 2.5, 3]）。

    算法：DP 凑出最少件数 + 最少规格种类。
    例：5.5米→[(3,1),(2.5,1)]，6.5米→[(2.5,1),(2,2)]（3件 2 种规格）。

    Args:
        meter: 目标米数（必须是 0.5 的倍数，如 5.5/6.5/7.5）。

    Returns:
        [(规格, 件数), ...] 按规格从大到小排列；无法组合时返回 None。

    Raises:
        ValueError: meter 不是正数或不是 0.5 的倍数。
    """
    if not isinstance(meter, (int, float)) or meter <= 0:
        raise ValueError(f"meter 必须是正数，得到 {meter!r}")

    # 把 X 转成 0.5 米为单位的整数 N = X * 2
    N = int(round(meter * 2))
    if abs(meter - N / 2) > 0.001:
        raise ValueError(f"meter 必须是 0.5 的倍数，得到 {meter!r}")

    specs_float = [3.0, 2.5, 2.0, 1.5, 1.0]
    specs_int = sorted([int(round(s * 2)) for s in specs_float], reverse=True)

    # DP：件数最少优先；件数相同时规格种类最少（同规格）优先。
    # 这样 4米→[(2,2)]（2件同规格胜 1件3米+1件1米）、7.5米→[(2.5,3)]（3件全同规格），
    # 而 4.5米→[(3,1),(1.5,1)] 或 [(2.5,1),(2,1)]（2件混搭，件数最少胜 3件1.5米）。
    # dp[i] = 最少件数；dp_kinds[i] = 同件数下的最少规格种类
    INF = float("inf")
    dp = [INF] * (N + 1)
    dp_kinds = [INF] * (N + 1)
    parent = [-1] * (N + 1)
    parent_spec = [0] * (N + 1)
    dp[0] = 0
    dp_kinds[0] = 0

    for i in range(N + 1):
        if dp[i] == INF:
            continue
        for spec in specs_int:
            nxt = i + spec
            if nxt > N:
                continue
            new_count = dp[i] + 1
            # 计算 new_kinds：i 状态的最后一个规格是什么？
            if i == 0:
                # 第一次拿规格，规格种类计 1
                new_kinds = 1
            else:
                last_spec = parent_spec[i]
                new_kinds = dp_kinds[i] + (0 if last_spec == spec else 1)
            # 更新条件：件数更少 或 (件数相等且规格种类更少)
            if dp[nxt] > new_count or (
                dp[nxt] == new_count and dp_kinds[nxt] > new_kinds
            ):
                dp[nxt] = new_count
                dp_kinds[nxt] = new_kinds
                parent[nxt] = i
                parent_spec[nxt] = spec

    if dp[N] == INF:
        return None

    # 回溯
    plan_int: dict = {}
    cur = N
    while cur > 0:
        prev = parent[cur]
        s = parent_spec[cur]
        plan_int[s] = plan_int.get(s, 0) + 1
        cur = prev
    return [(s / 2, n) for s, n in sorted(plan_int.items(), reverse=True)]


# =============================================================================
# 通用格式化
# =============================================================================

def format_meter_plan(plan: List[Tuple[float, int]]) -> str:
    """格式化任意组合方案为中文，如 [(3,1),(2.5,1)] → '1件3米 + 1件2.5米'。

    支持整数米和 0.5 米递增规格。
    """
    parts = []
    for spec, n in plan:
        if n <= 0:
            continue
        # 规格显示：整数显示 "1米"，小数显示 "1.5米"
        spec_str = f"{spec:g}米"  # g 去掉 .0
        parts.append(f"{n}件{spec_str}")
    return " + ".join(parts)


def total_pieces(plan: List[Tuple[float, int]]) -> int:
    """组合方案的总件数。"""
    return sum(n for _, n in plan if n > 0)


# 兼容旧调用（整数米专用）
def format_integer_meter_plan(plan: List[Tuple[int, int]]) -> str:
    """兼容旧 API：整数米组合方案格式化。"""
    return format_meter_plan([(float(s), n) for s, n in plan])