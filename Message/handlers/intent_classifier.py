"""
意图分类器（轻量语义路由）

替代旧版"售后关键词硬匹配"的转人工触发逻辑。基于现有 LLM（GLM-4-Flash）
对买家消息做意图分类，输出 {intent, confidence}，由 ai_handler 据此决定是否转人工：

- consult           ：纯咨询（退货流程是什么 / 运费谁出 / 保修多久）→ AI 自主回答
- operation         ：明确要求售后操作（我要退货 / 立刻退款）→ 转人工
- complaint         ：投诉 / 给差评 / 要求赔偿 → 转人工
- negative_emotion  ：强烈不满 / 愤怒 / 催促（太慢了 / 气死我了）→ 转人工
- other / unknown   ：其他 / 分类失败 → 保守不转人工

设计约束（来自改造需求）：
- 不引入外部 API，复用 config.json 中已有的 LLM 配置。
- 轻量：极小输出 token + 哈希缓存 + 超时回退；分类失败保守不转，避免误转。
- 可测试：LLM 调用封装在 _call_llm，测试可整体替换 get_intent_classifier 返回值。
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional

from utils.logger_loguru import get_logger
from config import get_config

logger = get_logger("IntentClassifier")

# 意图取值集合
INTENT_CONSULT = "consult"
INTENT_OPERATION = "operation"
INTENT_COMPLAINT = "complaint"
INTENT_NEGATIVE_EMOTION = "negative_emotion"
INTENT_OTHER = "other"
INTENT_UNKNOWN = "unknown"
_TRANSFER_INTENTS = {INTENT_OPERATION, INTENT_COMPLAINT, INTENT_NEGATIVE_EMOTION}

DEFAULT_PROMPT = (
    "你是一个电商客服系统的消息意图分类器。只输出一个 JSON 对象，不要输出任何其它内容。\n"
    "判断买家消息的意图，可选值：\n"
    '- "consult"：仅咨询政策/流程/信息，不需要立即操作（例：退货流程是什么、运费谁出、保修多久）\n'
    '- "operation"：明确要求执行售后操作（例：我要退货、立刻给我退款、把东西退掉）\n'
    '- "complaint"：投诉、要给差评、要求赔偿（例：投诉你们、你们是骗子、给差评）\n'
    '- "negative_emotion"：表达强烈不满/愤怒/着急/催促，即便未明确要求操作也需升级（例：太慢了、气死我了、催一下）\n'
    '- "other"：其它\n'
    "输出格式：{\"intent\": \"上述之一\", \"confidence\": 0.0到1.0之间的数字}\n"
    "注意：如果消息包含售后相关词（如退货/退款），但只是在询问流程或政策，应判为 consult；"
    "只有真正要求执行操作、投诉或宣泄强烈负面情绪时才判为非 consult。"
)


class IntentClassifier:
    """基于 LLM 的意图分类器（单例由 get_intent_classifier 管理）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model_name = cfg.get("model_name") or get_config("llm.model_name", "glm-4-flash")
        self.api_key = cfg.get("api_key") or get_config("llm.api_key", "")
        self.api_base = cfg.get("api_base") or get_config("llm.api_base", "")
        self.threshold = float(cfg.get("threshold", 0.6))
        self.cache_ttl = float(cfg.get("cache_ttl_seconds", cfg.get("cache_ttl", 600)))
        self.timeout = float(cfg.get("timeout_seconds", cfg.get("timeout", 0.8)))
        self.max_tokens = int(cfg.get("max_tokens", 32))
        self.prompt = cfg.get("prompt") or DEFAULT_PROMPT
        self._client = None
        self._cache: Dict[str, tuple] = {}  # key -> (expiry_ts, result)

    # ===== 客户端懒加载 =====
    def _ensure_client(self):
        if self._client is None:
            from Agent.CustomerAgent.custom.llm_client import LLMClient
            self._client = LLMClient(
                api_key=self.api_key,
                api_base=self.api_base,
                model_name=self.model_name,
                temperature=0.0,
            )
        return self._client

    # ===== 缓存 =====
    def _cache_key(self, text: str, after_sale_hint: bool) -> str:
        return hashlib.md5(f"{text}|{int(after_sale_hint)}".encode("utf-8")).hexdigest()

    def _get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        item = self._cache.get(key)
        if item and item[0] > time.time():
            return item[1]
        self._cache.pop(key, None)
        return None

    def _set_cache(self, key: str, result: Dict[str, Any]) -> None:
        self._cache[key] = (time.time() + self.cache_ttl, result)

    # ===== 对外接口 =====
    async def classify(self, text: str, after_sale_hint: bool = False) -> Dict[str, Any]:
        """分类单条消息。返回 {intent, confidence}。失败/禁用返回 unknown。

        超时/异常产生的保守 unknown 结果**不写入缓存**，避免一次瞬时超时把该
        消息永久污染为 unknown（在 TTL 内不再重试）。只有 LLM 真实返回的结果
        才缓存，保证缓存是"有效分类"而非"失败标记"。
        """
        if not self.enabled or not text:
            return {"intent": INTENT_UNKNOWN, "confidence": 0.0}
        key = self._cache_key(text, after_sale_hint)
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        try:
            result = await asyncio.wait_for(
                self._call_llm(text, after_sale_hint), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning("意图分类超时，保守返回 unknown（不转人工），本次不缓存以便重试")
            return {"intent": INTENT_UNKNOWN, "confidence": 0.0}
        except Exception as e:  # pragma: no cover
            logger.warning(f"意图分类异常，保守返回 unknown（不转人工），本次不缓存: {e}")
            return {"intent": INTENT_UNKNOWN, "confidence": 0.0}
        self._set_cache(key, result)
        return result

    async def _call_llm(self, text: str, after_sale_hint: bool) -> Dict[str, Any]:
        client = self._ensure_client()
        if getattr(client, "_client", None) is None:
            await client.initialize()
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self._build_user(text, after_sale_hint)},
        ]
        resp = await client.chat(messages)
        return self._parse(resp)

    def _build_user(self, text: str, after_sale_hint: bool) -> str:
        user = f"消息：{text}"
        if after_sale_hint:
            user += "\n提示：该消息命中售后相关词，请重点判断是咨询还是操作/投诉/情绪。"
        return user

    def _parse(self, resp: Any) -> Dict[str, Any]:
        content = getattr(resp, "content", None) or ""
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start:end + 1])
                intent = data.get("intent")
                conf = data.get("confidence", 0.0)
                if intent in (
                    INTENT_CONSULT, INTENT_OPERATION, INTENT_COMPLAINT,
                    INTENT_NEGATIVE_EMOTION, INTENT_OTHER, INTENT_UNKNOWN,
                ):
                    return {"intent": intent, "confidence": float(conf or 0.0)}
        except Exception:  # pragma: no cover
            pass
        return {"intent": INTENT_UNKNOWN, "confidence": 0.0}

    @staticmethod
    def should_transfer(intent: str, confidence: float, threshold: float) -> bool:
        """判断给定意图是否应转人工（操作/投诉/负面情绪且置信度达标）。"""
        if intent not in _TRANSFER_INTENTS:
            return False
        return float(confidence or 0.0) >= float(threshold)


_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """获取意图分类器单例（从 config.json 的 intent 段读取配置）。"""
    global _classifier_instance
    if _classifier_instance is None:
        cfg = get_config("intent", {}) or {}
        _classifier_instance = IntentClassifier(cfg)
    return _classifier_instance


def reset_intent_classifier() -> None:
    """重置单例（测试用）。"""
    global _classifier_instance
    _classifier_instance = None
