"""
订单号禁忌终态过滤器（纵深防御）

背景（二阶风险）：
    系统提示词层（message_builder 的"订单号使用规则"）只能约束 LLM 当下行为，
    无法约束知识库检索返回并被 LLM 复述进回复的内容。若运营在知识库录入
    "请提供订单号"类话术，终态回复可能重新出现索要订单号。

本模块提供与内容源无关的终态过滤：
    - 优先级高于一切内容源（含知识库工具返回文本）。
    - 仅当用户明确提及具体订单时才允许索要订单号。
    - 否则逐句（保留原标点结构）删除含"索要订单号"模式的子句；
      若删空则回退合规话术。

配置（config.json 的 order_number_guard 段，可覆盖；缺省时启用代码默认基线）：
    enabled                         是否启用过滤
    allow_when_user_references_order 用户提及具体订单时允许索要
    user_order_patterns             判定"用户提及具体订单"的正则
    solicit_patterns                判定"机器人索要订单号"的正则
    replacement                     删空时的合规回退话术
"""
import re

from config import get_config

# 安全基线默认值：config 缺省时仍生效（默认安全）。
# config_base 中的同名段为可覆盖配置与文档来源，二者应保持一致。
DEFAULT_ORDER_NUMBER_GUARD = {
    "enabled": True,
    "allow_when_user_references_order": True,
    "user_order_patterns": [
        r"订单号\s*[:：是]\s*\d+",
        r"我的订单",
        r"查询订单",
        r"查订单",
        r"订单状态",
        r"订单\s*\d+",
        r"帮我处理订单",
        r"处理订单",
    ],
    "solicit_patterns": [
        r"请提供(您|你)的订单号",
        r"请(您|你)提供订单号",
        r"(您|你)(的)?订单号(是|为)?(多少|什么)",
        r"把订单号发(给|我)",
        r"发(送)?(一下)?(您|你)的订单号",
        r"提供(一下)?订单号",
        r"订单号(发|给)(我|一下)",
        r"需(要|提供)订单号",
        r"告诉(我)?(您|你)(的)?订单号",
    ],
    "replacement": "亲，您可直接在订单页面自助操作哦。",
}

_PUNCT_CHARS = "。！？!?\n，,：:；;"


def get_guard_config() -> dict:
    """读取 config 中的 order_number_guard；缺失时回退到安全基线默认。"""
    cfg = get_config("order_number_guard")
    if not isinstance(cfg, dict):
        return dict(DEFAULT_ORDER_NUMBER_GUARD)
    merged = dict(DEFAULT_ORDER_NUMBER_GUARD)
    merged.update(cfg)
    # 列表型字段以 config 提供为准（整体替换，避免默认+配置拼接）
    for key in ("user_order_patterns", "solicit_patterns"):
        if key in cfg and isinstance(cfg[key], list):
            merged[key] = cfg[key]
    return merged


def _user_references_order(query: str, patterns) -> bool:
    if not query:
        return False
    return any(re.search(p, query) for p in patterns)


def enforce_order_number_guard(
    text: str,
    query: str = "",
    config: dict = None,
) -> str:
    """对最终回复文本执行订单号禁忌过滤。

    Args:
        text: 最终回复文本（已合并一切内容源，含知识库复述）。
        query: 用户本轮消息，用于判断"是否提及具体订单"。
        config: 可选，直接传入 guard 配置；缺省则从 config.json 读取。

    Returns:
        过滤后的文本。无违规或禁用时原样返回。
    """
    guard = config if isinstance(config, dict) else get_guard_config()
    if not guard.get("enabled", True):
        return text
    if not text:
        return text

    solicit = guard.get("solicit_patterns") or DEFAULT_ORDER_NUMBER_GUARD["solicit_patterns"]

    # 上下文豁免：用户明确提及具体订单时，允许索要订单号
    if guard.get("allow_when_user_references_order", True):
        user_patterns = (
            guard.get("user_order_patterns")
            or DEFAULT_ORDER_NUMBER_GUARD["user_order_patterns"]
        )
        if _user_references_order(query, user_patterns):
            return text

    # 按标点切分并保留分隔符，逐"子句"判定，仅丢弃违规子句，保留原标点结构
    parts = re.split(rf"([{_PUNCT_CHARS}]+)", text)
    kept = []
    violated = False
    for part in parts:
        if part == "":
            continue
        if re.fullmatch(rf"[{_PUNCT_CHARS}]+", part):
            kept.append(part)  # 纯分隔符始终保留
            continue
        if any(re.search(p, part) for p in solicit):
            violated = True
            continue
        kept.append(part)

    if not violated:
        return text

    result = "".join(kept).strip()
    if not result:
        result = guard.get("replacement") or DEFAULT_ORDER_NUMBER_GUARD["replacement"]
    return result
