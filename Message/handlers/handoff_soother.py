"""
转人工安抚话术库（2026-09-05 新增）

需求：触发转人工的瞬间，由系统先给买家发一句"核实/处理类"拖延话术，
避免人工客服未能及时接入导致"3 分钟回复率"下降。

设计约束（用户审核确认）：
- **只发"核实/处理类"话术，绝不用"转接类"话术**（如"正在为您转接人工客服"），
  那会暴露背后是 AI 而非真人。
- 买家主动要求转人工（关键词硬短路）：**不发送**（买家已明确要人工，再说
  "我帮您核实"反而假）。
- 非营业时间：**不发送**（该路径走静默标记+通知，不经过话术发送）。
- 已处于转人工状态的会话后续消息：**不发送**（避免刷屏）。
- 同一场景有多个变体，随机挑选；记录最近一次用过的下标避免连续重复。

话术风格：口语化、像真人客服在亲自帮忙查/处理，句末语气词自然变化。
"""
from __future__ import annotations

import random
from typing import Dict, List

# =============================================================================
# 各场景话术池
# =============================================================================

# 售后操作：退货/退款/换货/取消订单/改地址等（情绪一般急切）
_AFTER_SALE_POOL: List[str] = [
    "亲，售后这边我来帮您核实处理，请稍等一下",
    "收到亲，马上帮您查看处理，稍等片刻哈",
    "亲，您别着急，这个我帮您核实一下，稍等片刻",
]

# 投诉/差评/索赔：先道歉稳住情绪
_COMPLAINT_POOL: List[str] = [
    "亲，非常抱歉给您带来不好的体验，我马上帮您核实处理，请稍等",
    "实在抱歉亲，您说的情况我一定帮您处理好，稍等片刻",
    "亲，真的不好意思，这个我马上帮您核实，请稍等一下",
]

# 负面情绪：愤怒/不满/催促（加急安抚）
_NEGATIVE_POOL: List[str] = [
    "亲，实在不好意思让您着急了，我马上帮您加急处理，请稍等",
    "亲您别急，我这就帮您核实，稍等片刻哈",
    "理解您的心情亲，正在加急帮您处理，请稍等一下",
]

# 意图不明（other/unknown）保守转人工：短句即可
_UNCERTAIN_POOL: List[str] = [
    "亲，麻烦稍等一下，我帮您核实下情况哈",
    "收到亲，稍等片刻，我确认好马上回复您",
    "亲，稍等一下哈，我帮您看看怎么处理",
]

# 成衣用量咨询（做衣服要多少米布）：显得专业、愿意帮
_GARMENT_POOL: List[str] = [
    "亲，做衣服的用量得结合款式和身高来算，我帮您问下师傅，稍等哈",
    "亲，这个用量我帮您核实一下再回复您，请稍等",
    "亲，稍等哈，我帮您确认一下做衣服的用料建议",
]

# AI 答不上来 / 知识库未覆盖 / 其他无法回答：通用核实话术
_UNANSWERABLE_POOL: List[str] = [
    "亲，您问的这个我帮您核实一下，请稍等",
    "亲，这块我马上帮您确认后回复，稍等一下哈",
    "亲，稍等一下，我帮您查清楚再回复您",
]

# 兜底（不应走到，防止遗漏）
_GENERIC_POOL: List[str] = [
    "亲，稍等一下，我帮您核实后回复您",
    "收到亲，稍等片刻，我确认好马上回复您",
    "亲，您稍等一下哈，我这就帮您看下",
]

# 场景分类 -> 话术池
_POOLS: Dict[str, List[str]] = {
    "after_sale": _AFTER_SALE_POOL,
    "complaint": _COMPLAINT_POOL,
    "negative": _NEGATIVE_POOL,
    "uncertain": _UNCERTAIN_POOL,
    "garment": _GARMENT_POOL,
    "unanswerable": _UNANSWERABLE_POOL,
    "generic": _GENERIC_POOL,
}

_LAST_USED: Dict[str, int] = {}  # session_key -> 最近一次使用的下标


def classify_category(intent: str = "", reason: str = "") -> str:
    """根据意图分类与转人工原因映射到话术池类别。

    Args:
        intent: 意图分类结果（operation/complaint/negative_emotion/other/unknown/...）。
        reason: 转人工原因（企业微信通知文案，如 'AI无法回答→转人工'）。
    """
    intent = (intent or "").strip().lower()
    reason = reason or ""

    if intent == "operation":
        return "after_sale"
    if intent == "complaint":
        return "complaint"
    if intent == "negative_emotion":
        return "negative"
    if intent in ("other", "unknown"):
        return "uncertain"
    if "成衣用量" in reason or "衣服" in reason or "布料" in reason:
        return "garment"
    if "AI无法回答" in reason or "知识库未覆盖" in reason or "无法回答" in reason:
        return "unanswerable"
    return "generic"


def pick_soother(category: str, session_key: str = "") -> str:
    """从指定场景话术池随机挑一条，尽量避免与上次连续重复。

    Args:
        category: classify_category 的输出（未知则落 generic）。
        session_key: 会话标识（shop_id:from_uid），用于避免同会话连续重复。
    """
    pool = _POOLS.get(category) or _GENERIC_POOL
    if len(pool) <= 1:
        return pool[0]

    last = _LAST_USED.get(session_key, -1)
    # 候选池：去掉最近一次用的那条，保证本轮变化
    candidates = [i for i in range(len(pool)) if i != last] or list(range(len(pool)))
    idx = random.choice(candidates)
    _LAST_USED[session_key] = idx
    return pool[idx]


# 供单元测试直接取池子
def get_pool(category: str) -> List[str]:
    return _POOLS.get(category) or _GENERIC_POOL