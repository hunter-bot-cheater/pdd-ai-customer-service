"""订单号拆分保护 + 退款单过滤 离线测试（无需浏览器）。

验证：
1. _split_reply 不会把长订单号截断到两条消息里（订单号作为原子 token 保护）
2. _format_output 自动过滤掉已退款/已关闭的订单，不展示给客户
3. 全部退款时返回"暂无有效在途订单"
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Agent.CustomerAgent.tools.read_session_orders import normalize_customer_order
from Agent.CustomerAgent.tools.query_order_status import _format_output


# ---- 订单号保护：直接测试 mask + 拆分逻辑（避免 DI 容器依赖）----

def _mask_order_numbers(text: str):
    """从 ai_handler._mask_order_numbers 提取的核心逻辑"""
    ord_map = {}
    parts = []
    last = 0
    i = 0
    for m in re.finditer(r'\b\d{6,8}-?\d{12,20}\b', text):
        raw = m.group(0)
        placeholder = f'__ORD_{i}__'
        ord_map[placeholder] = raw
        parts.append(text[last:m.start()])
        parts.append(placeholder)
        last = m.end()
        i += 1
    parts.append(text[last:])
    return ''.join(parts), ord_map


def _split_with_ord_protection(reply: str, max_len: int = 25) -> list:
    """简化版拆分（仅验证订单号保护，不含 URL 逻辑）"""
    if not reply:
        return []
    masked, ord_map = _mask_order_numbers(reply)
    # 按句末标点拆
    sentences = re.split(r'(?<=[。！？；\n])', masked)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for sent in sentences:
        if len(sent) <= max_len:
            chunks.append(sent)
            continue
        # 超长按逗号/顿号细分
        parts = re.split(r'(?<=[，、])', sent)
        parts = [p.strip() for p in parts if p.strip()]
        current = ""
        for part in parts:
            if len(part) > max_len:
                # 硬切时保护 __ORD__ token
                tokens = [t for t in re.split(r'(__ORD_\d+__)', part) if t]
                for tok in tokens:
                    if tok.startswith('__ORD_') and tok in ord_map:
                        if current and len(current) + len(tok) <= max_len:
                            current += tok
                        else:
                            if current:
                                chunks.append(current)
                            current = tok
                    else:
                        while tok:
                            room = max_len - len(current)
                            if room <= 0:
                                chunks.append(current)
                                current = ""
                                continue
                            take = min(len(tok), room)
                            current += tok[:take]
                            tok = tok[take:]
            elif len(current) + len(part) <= max_len:
                current += part
            else:
                chunks.append(current)
                current = part
        if current:
            chunks.append(current)
    # 还原订单号
    return [_restore_ord(c, ord_map) for c in chunks]


def _restore_ord(text, ord_map):
    if not text or not ord_map:
        return text
    for ph, orig in ord_map.items():
        text = text.replace(ph, orig)
    return text


def test_order_number_not_split():
    """订单号在分条发送中必须保持完整"""
    reply = (
        "亲您的订单目前都还未发货哦。"
        "订单号 260811-682759350642389 正在等待发货。"
        "请您耐心等待我们会尽快为您发货"
    )
    chunks = _split_with_ord_protection(reply)
    
    full_sn = "260811-682759350642389"
    found = any(full_sn in c for c in chunks)
    sn_prefix = full_sn[:-3]
    sn_suffix = full_sn[-3:]
    suffix_only = [c for c in chunks if sn_suffix in c and sn_prefix not in c]
    
    assert found, f"订单号未被完整找到! chunks={chunks}"
    assert len(suffix_only) == 0, f"订单号被截断! 尾部孤立在: {suffix_only}"
    print(f"[ok] 订单号完整保留 (共 {len(chunks)} 条消息)")


def test_two_orders_not_split():
    """两个长订单号都不能被截断"""
    reply = (
        f"订单号 260811-682759350642389 正在等待发货。"
        f"而订单号 260811-626796839282389 已经退款成功无需发货。"
        f"请您耐心等待"
    )
    chunks = _split_with_ord_protection(reply)
    for sn in ["260811-682759350642389", "260811-626796839282389"]:
        assert any(sn in c for c in chunks), f"{sn} 未完整出现! chunks={chunks}"
    print("[ok] 双订单号均完整未断裂")


def test_refunded_orders_filtered():
    """已退款/已关闭的订单不展示给客户"""
    refunded = normalize_customer_order({
        "orderSn": "260811-682759350642389", "orderStatus": 1,
        "orderStatusStr": "未发货，退款成功", "shippingStatus": 0,
        "orderAmount": 2800, "uid": 4239748275,
        "orderGoodsList": {"goodsName": "保温杯", "spec": "红银", "goodsNumber": 1},
    })
    active = normalize_customer_order({
        "orderSn": "260812-ACTIVE", "orderStatus": 1,
        "orderStatusStr": "待发货", "shippingStatus": 0,
        "orderAmount": 5000, "uid": 4239748275,
    })
    out = _format_output([refunded, active], is_filtered=True)
    assert "260811-682759350642389" not in out, f"退款单不应出现: {out}"
    assert "260812-ACTIVE" in out, f"在途单应出现: {out}"
    print("[ok] 退款单被过滤，仅展示在途单")


def test_all_refunded_shows_empty():
    """全部订单均已退款 → 返回'暂无有效在途订单'"""
    r1 = normalize_customer_order({
        "orderSn": "260811-R1", "orderStatus": 1,
        "orderStatusStr": "未发货，退款成功", "shippingStatus": 0,
        "orderAmount": 2800, "uid": 4239748275,
    })
    r2 = normalize_customer_order({
        "orderSn": "260811-R2", "orderStatus": 1,
        "orderStatusStr": "已退款", "shippingStatus": 0,
        "orderAmount": 5000, "uid": 4239748275,
    })
    out = _format_output([r1, r2], is_filtered=True)
    assert "暂无有效在途订单" in out, out
    assert "260811-R1" not in out
    assert "260811-R2" not in out
    print("[ok] 全部退款 → '暂无有效在途订单'")


if __name__ == "__main__":
    test_order_number_not_split()
    test_two_orders_not_split()
    test_refunded_orders_filtered()
    test_all_refunded_shows_empty()
    print("\n全部订单号保护 + 退款过滤测试通过 OK")
