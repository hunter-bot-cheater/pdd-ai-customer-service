"""
规格感知米数顾问（2026-09-05）

从产品知识的 specifications 解析商品基础规格（米数），判断买家目标米数能否
用规格线性组合：
- 能组合 → 给出"几件几米"拍法（同规格覆盖优先，件数其次）
- 不能组合（如普通花型要 0.3 米）→ 给出"无法裁剪"结论 + 最接近的可卖规格

供 ai_handler 在调 LLM 前预计算并把结论硬性注入 prompt，LLM 只复述、不自行计算，
杜绝此前"推荐 1件5米"这类规格外组合。
"""
from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Sequence, Tuple

# 中文数字 → 阿拉伯（规格里常写"一米/二米/三米"）
_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# 阿拉伯米数，如 "1.5米" / "10米"
_ARA_METER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*米")
# 中文米数，如 "一米" / "两米" / "十米"
_CN_METER_RE = re.compile(r"([一二两三四五六七八九十])\s*米")

# 普通花型默认规格（无商品上下文时）
DEFAULT_SPECS: List[float] = [1.0, 2.0, 3.0]


def parse_meter_values(specs) -> List[float]:
    """从 product_knowledge.specifications 解析规格米数集合（升序去重）。

    specs 为 JSON 字符串或 list，每项形如：
      "颜色: 晴野碎芳 | 尺寸: 1.5米（多拍连裁）"
      "款式: 彩色枫叶 | 套餐: 一米"
    优先取「尺寸/套餐」字段，找不到则全文搜索米数。
    会剔除括号内含"宽/幅/门"的注释（如"一米（门幅1.43米）"里的 1.43 是幅宽
    而非可拍长度，不纳入规格集）。
    """
    if isinstance(specs, str):
        try:
            data = json.loads(specs)
        except Exception:
            return []
    else:
        data = specs or []

    vals = set()
    for item in data:
        if not isinstance(item, str):
            continue
        segs: List[str] = []
        for key in ("尺寸", "套餐"):
            m = re.search(re.escape(key) + r"\s*[:：]\s*([^|]+)", item)
            if m:
                segs.append(m.group(1))
        if not segs:
            segs = [item]
        for seg in segs:
            # 剔除括号内含 宽/幅/门 的注释片段（幅宽说明），避免把门幅当可拍规格
            seg_clean = re.sub(
                r"（[^（）]*(?:宽|幅|门)[^（）]*）|\([^()]*(?:宽|幅|门)[^()]*\)",
                "",
                seg,
            )
            m = _ARA_METER_RE.search(seg_clean)
            if m:
                vals.add(float(m.group(1)))
                continue
            m2 = _CN_METER_RE.search(seg_clean)
            if m2:
                vals.add(float(_CN_NUM[m2.group(1)]))
    return sorted(vals)


def _dp_tables(N: int, specs_int: Sequence[int]):
    """DP：dp[i]=最少件数；kinds[i]=同件数下最少规格种类。返回 (dp, parent, pspec)。"""
    INF = float("inf")
    dp = [INF] * (N + 1)
    kinds = [INF] * (N + 1)
    parent = [-1] * (N + 1)
    pspec = [0] * (N + 1)
    dp[0] = 0
    kinds[0] = 0
    for i in range(N + 1):
        if dp[i] == INF:
            continue
        for s in specs_int:
            nxt = i + s
            if nxt > N:
                continue
            nc = dp[i] + 1
            nk = 1 if i == 0 else kinds[i] + (0 if pspec[i] == s else 1)
            if dp[nxt] > nc or (dp[nxt] == nc and kinds[nxt] > nk):
                dp[nxt] = nc
                kinds[nxt] = nk
                parent[nxt] = i
                pspec[nxt] = s
    return dp, parent, pspec


def _backtrack(N: int, parent, pspec) -> List[Tuple[float, int]]:
    plan: dict = {}
    cur = N
    while cur > 0:
        s = pspec[cur]
        plan[s] = plan.get(s, 0) + 1
        cur = parent[cur]
    return sorted(((s / 10.0, n) for s, n in plan.items()), reverse=True)


def compute_plan_for_specs(
    meter: float, specs: Iterable[float]
) -> Optional[List[Tuple[float, int]]]:
    """判断 meter 能否由 specs 线性组合；能则返回 [(规格, 件数), ...]。

    规则：DP 求最少件数（同件数下最少规格种类）；若存在"同一规格多件覆盖"
    且其件数 ≤ DP 最少件数 + 1，则优先采用同规格方案
    （4.5米{1,1.5,2,2.5,3,10} → 3件1.5米；10.5米 → 7件1.5米件数过多改用 DP 混搭）。
    不可组合返回 None。
    """
    if meter is None or meter <= 0:
        return None
    specs_int = sorted({int(round(s * 10)) for s in specs if s and s > 0})
    if not specs_int:
        return None
    N = int(round(meter * 10))
    if N <= 0:
        return None

    dp, parent, pspec = _dp_tables(N, specs_int)
    if dp[N] == float("inf"):
        return None
    dp_min = dp[N]

    # 同规格覆盖优先（件数比最少件数多 1 件以内就采用）
    for s in sorted(specs_int, reverse=True):
        if N % s == 0:
            n = N // s
            if n <= dp_min + 1:
                return [(s / 10.0, n)]

    return _backtrack(N, parent, pspec)


def nearest_sellable(meter: float, specs: Iterable[float]) -> Optional[float]:
    """meter 不可组合时，返回规格可达集合中最接近 meter 的可卖米数。"""
    if meter is None or meter <= 0:
        return None
    specs_int = sorted({int(round(s * 10)) for s in specs if s and s > 0})
    if not specs_int:
        return None
    N = int(round(meter * 10))
    bound = N + max(specs_int)
    dp, _, _ = _dp_tables(bound, specs_int)
    best = None  # (差值, 米数x10)
    for x in range(1, bound + 1):
        if dp[x] < float("inf"):
            d = abs(x - N)
            if best is None or d < best[0] or (d == best[0] and x < best[1]):
                best = (d, x)
    return (best[1] / 10.0) if best else None


def format_spec_list(specs: Sequence[float]) -> str:
    parts = []
    for s in sorted(specs):
        parts.append(f"{s:g}米")
    return "/".join(parts)


def build_meter_hint(
    meter: float,
    specs: Sequence[float],
    product_name: str = "",
) -> str:
    """生成注入给 LLM 的系统结论（有解=拍法；无解=无法裁剪+最近可卖规格）。"""
    name = product_name or "当前商品"
    specs_str = format_spec_list(specs)
    plan = compute_plan_for_specs(meter, specs)
    if plan:
        parts = " + ".join(f"{n}件{s:g}米" for s, n in plan)
        total = sum(s * n for s, n in plan)
        return (
            f"【系统已按商品规格计算】买家想要 {meter:g} 米。商品「{name}」的规格为：{specs_str}。\n"
            f"推荐拍法：{parts}，共 {sum(n for _, n in plan)} 件连成 {total:g} 米一整段裁切。\n"
            f"请直接按此回复买家怎么拍，严禁自行另算米数，严禁推荐上述规格之外的米数。"
        )
    nearest = nearest_sellable(meter, specs)
    nearest_txt = f"{nearest:g} 米" if nearest else "规格内的米数"
    return (
        f"【系统已按商品规格计算】买家想要 {meter:g} 米，但商品「{name}」的规格为：{specs_str}，"
        f"无法组合出该长度，不能裁这个米数。\n"
        f"请回复无法裁剪/不支持这个长度，并推荐最接近的可卖规格 {nearest_txt}（按上述规格组合购买）。"
        f"严禁推荐规格之外的米数，严禁说可以定制或备注特殊长度。"
    )
