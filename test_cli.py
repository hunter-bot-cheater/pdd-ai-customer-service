#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent-Customer 消息处理离线测试 CLI
====================================

用途：
    快速验证消息处理逻辑（AI 回复、转人工判断、意图分类），
    无需启动 GUI、无需登录拼多多账号。

为什么把项目依赖放进 if __name__ == "__main__"
    本文件名为 test_*.py，部分 IDE/pytest 会用“收集测试模块”的方式 import 它。
    若顶层直接 import 项目模块（config 等会进一步 import loguru），
    而运行 pytest 的解释器又缺少这些第三方依赖时，收集阶段就会 ImportError。
    把项目依赖与依赖它们的代码全部放进 __main__ 守卫，pytest 收集时只执行
    标准库 import 与纯标准库类（DummySender / _dummy_transfer），不再报缺失依赖。

与线上的一致性（关键）
    本工具调用的是程序的【真实 CustomerAgent】——即真实 message_builder 系统提示词、
    真实 LLM、真实意图分类、真实知识库工具、真实关键词/转人工 handler 链。
    唯一被替换的是“发送”与“转接人工”两个副作用：
      - get_sender        -> DummySender（只打印，不真发）
      - transfer_conversation -> 同步桩（返回“会话转接成功”，不真触拼多多转接 API）
    因此你能看到：某句话 LLM 会怎么答、经过系统后最终呈现什么回复，且不会打扰真实账号。

运行：
    cd D:/ai客服/Customer-Agent-main
    python test_cli.py                      # 交互模式
    python test_cli.py --debug              # 详细日志 + 耗时
    python test_cli.py --case-file cases.txt  # 批量
    python test_cli.py --business-hours 08:00 23:00
    python test_cli.py --real-delay         # 恢复真人延迟

退出：输入 quit / exit / q
"""

import os
import sys
import asyncio
import argparse
import time

# --debug 需要在导入项目模块前把日志级别调高（项目 logger 在 import 时读取该环境变量）
if "--debug" in sys.argv:
    os.environ["LOG_LEVEL"] = "DEBUG"
else:
    os.environ.setdefault("LOG_LEVEL", "WARNING")

# ============================== 测试用常量 ==============================
# shop_id 用字符串：Context.create_pinduoduo_context 的 PinduoduoKwargs 要求 str 型 shop_id。
# 知识库工具的 int 校验失败会在运行时被捕获并走通用指导，不影响回复。
DEFAULT_SHOP_ID = "test_shop"
DEFAULT_USER_ID = "test_user"
DEFAULT_SHOP_NAME = "测试店铺"
DEFAULT_BUYER_UID = "test_buyer_001"

QUIT_COMMANDS = {"quit", "exit", "q"}


# ============================ 模拟发送器 ============================
class DummySender:
    """替代真实拼多多发送器：只打印回复内容，不发起任何网络请求。"""

    def __init__(self):
        self.sent = 0  # 记录“发送”次数，用于判断某条消息是否被拦截（未发送）

    def reset_count(self):
        self.sent = 0

    def send_text(self, shop_id, user_id, recipient_uid, text: str):
        self.sent += 1
        print(f"[模拟发送] 给 {recipient_uid}: {text}")
        # 返回业务码为 0 的成功结构，与项目 _check_send_result 的判定一致
        return {"success": True, "result": {"error_code": 0}}

    def send_image(self, shop_id, user_id, recipient_uid, image_url: str):
        self.sent += 1
        print(f"[模拟发送-图片] 给 {recipient_uid}: {image_url}")
        return {"success": True, "result": {"error_code": 0}}

    def send_product_card(self, shop_id, user_id, recipient_uid, goods_id, biz_type: int = 2):
        return {"success": True, "result": {"error_code": 0}}

    def get_cs_list(self, shop_id, user_id):
        return {}

    def transfer_to_cs(self, shop_id, user_id, recipient_uid, cs_uid):
        return {"success": True, "result": {"error_code": 0}}


_dummy_sender = DummySender()


# ====================== 离线安全的转人工桩函数 ======================
def _dummy_transfer(params, reason, silent, content):
    """替换项目内真实的 transfer_conversation，避免触碰拼多多转接 API。

    项目以 asyncio.to_thread(transfer_conversation, ...) 在线程中同步调用，
    因此此处必须是同步函数（不能是 async）。
    签名与项目调用点一致：(params, reason, send_notification, last_message)。
    返回“会话转接成功”使关键词/意图处理器按成功路径走完，便于离线验证整条链路。
    """
    return "会话转接成功"


# ======================================================================
# 以下为“项目依赖”逻辑：仅在直接运行（python test_cli.py）时加载。
# pytest 以 import 方式收集本模块时不会进入此分支，避免触发项目依赖
# （如 config -> loguru）导致的 ImportError。
# ======================================================================
if __name__ == "__main__":
    # ========================= 项目模块（既有，未改动） =========================
    import config  # 全局配置，import 即加载 config.json
    from bridge.context import Context, ContextType, ChannelType
    from bridge.sender import get_sender
    from bridge.reply import Reply, ReplyType
    from Agent.bot import Bot
    from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
    from Agent.CustomerAgent.tools import move_conversation as _mc_module
    from Message.handlers.keyword_handler import KeywordDetectionHandler
    from Message.handlers import keyword_handler as _kh_module
    from Message.handlers import ai_handler as _ah_module
    from Message.handlers.ai_handler import AIReplyHandler
    from Message.core.handlers import CatchAllHandler
    from core.di_container import configure_standard_services

    # ============================ 依赖注入 ============================
    def install_test_doubles(real_delay: bool):
        """在测试环境中安装所有模拟对象，全部局限在本 CLI 内。"""
        # 1) 发送器：所有渠道均返回 DummySender
        import bridge.sender as _sender_mod
        _sender_mod.get_sender = lambda *a, **k: _dummy_sender

        # 2) 转人工：用桩函数替换真实 transfer_conversation。
        #    - 关键词处理器在模块顶部 import，需替换其模块全局；
        #    - AI 处理器的意图转人工在函数内局部 import，需替换来源模块本身，
        #      局部 import 才会取到桩函数（避免真实调用拼多多转接 API）。
        _mc_module.transfer_conversation = _dummy_transfer
        _kh_module.transfer_conversation = _dummy_transfer

        # 3) 真人时序模拟：默认把 asyncio.sleep 置为瞬时，加速迭代；
        #    --real-delay 时保留项目原有时延（约 10~14 秒/条）。
        if not real_delay:
            async def _instant(*_a, **_k):
                return None
            asyncio.sleep = _instant

    # ============================ 消息处理 ============================
    async def process_message(raw: str, handlers, bot: CustomerAgent, default_uid: str):
        """解析输入 -> 构造 Context/metadata -> 跑真实 handler 链。

        返回简单状态串，供调用方打印。
        """
        # 支持 “消息内容|用户UID” 指定买家身份
        if "|" in raw:
            msg, uid = raw.rsplit("|", 1)
            uid = uid.strip()
        else:
            msg, uid = raw, default_uid

        msg = msg.strip()
        if not msg:
            return

        context = Context.create_pinduoduo_context(
            content=msg,
            from_uid=uid,
            shop_id=DEFAULT_SHOP_ID,
            user_id=DEFAULT_USER_ID,
            shop_name=DEFAULT_SHOP_NAME,
            channel_type=ChannelType.PINDUODUO,
        )
        metadata = {
            "shop_id": DEFAULT_SHOP_ID,
            "user_id": DEFAULT_USER_ID,
            "from_uid": uid,
            "shop_name": DEFAULT_SHOP_NAME,
        }

        _dummy_sender.reset_count()
        sent_before = _dummy_sender.sent

        # 复刻项目 consumer 的遍历逻辑：can_handle 命中即 handle，成功后 break
        for h in handlers:
            if not h.can_handle(context):
                continue

            if isinstance(h, KeywordDetectionHandler):
                # 关键词硬短路：真实执行 handle（转接已被桩函数替代，离线安全）
                await h.handle(context, metadata)
                print("[结果] 被关键词拦截，触发转人工（静默）")
            elif isinstance(h, AIReplyHandler):
                handled = await h.handle(context, metadata)
                sent_after = _dummy_sender.sent
                if sent_after > sent_before:
                    print("[结果] AI 回复已处理")
                else:
                    # 未发送文本：大概率是意图识别转人工 / 非营业时间静默拦截
                    print("[结果] 触发转人工（静默，未发送文本回复）")
                _ = handled
            else:
                await h.handle(context, metadata)
            return

        print("[结果] 无处理器命中（理论上不会到这，CatchAllHandler 兜底）")

    # ============================ 主流程 ============================
    async def main():
        parser = argparse.ArgumentParser(
            description="Agent-Customer 消息处理离线测试 CLI（真实 CustomerAgent）"
        )
        parser.add_argument("--debug", action="store_true", help="显示详细日志与耗时")
        parser.add_argument("--case-file", metavar="PATH", help="批量用例文件，每行一条消息（支持 消息|UID）")
        parser.add_argument("--business-hours", nargs=2, metavar=("START", "END"),
                            default=["00:00", "23:59"],
                            help="营业时间窗口，默认 00:00-23:59（始终营业，便于随时测试）")
        parser.add_argument("--real-delay", action="store_true",
                            help="保留真人已读/打字延迟（每条约 10~14 秒）")
        args = parser.parse_args()

        # 安装测试专用 DI（模拟发送器 + 转人工桩 + 时序加速）
        install_test_doubles(args.real_delay)

        # 初始化 DI 容器：让知识库工具等能正确解析服务（如 KnowledgeService）。
        # 真实 CustomerAgent 走它自己的 initialize_async，不依赖容器取实例，
        # 但知识库工具内部使用全局 container，必须先用标准服务配置填充。
        configure_standard_services()

        # 构造与线上一致的真实 handler 链，仅把 bot 换成真实 CustomerAgent
        business_hours = {"start": args.business_hours[0], "end": args.business_hours[1]}
        bot = CustomerAgent()
        handlers = [
            KeywordDetectionHandler(business_hours=business_hours),
            AIReplyHandler(bot=bot),
            CatchAllHandler(),
        ]

        # 预热真实 CustomerAgent（懒初始化 LLM/工具链 + 首次请求），
        # 避免首个用户消息因冷启动触发意图分类 5 秒超时。
        probe_ctx = Context.create_pinduoduo_context(
            content="ping",
            from_uid=DEFAULT_BUYER_UID,
            shop_id=DEFAULT_SHOP_ID,
            user_id=DEFAULT_USER_ID,
            shop_name=DEFAULT_SHOP_NAME,
            channel_type=ChannelType.PINDUODUO,
        )
        try:
            await bot.initialize_async()
            await bot.async_reply("ping", probe_ctx)  # 预热真实 LLM + 工具链
        except Exception as e:
            print(f"[warn] 真实 CustomerAgent 预热失败，回复将使用占位文本: {e}")

        try:
            from Message.handlers.intent_classifier import get_intent_classifier
            clf = get_intent_classifier()
            if clf is not None and clf.enabled:
                await clf.classify("warmup-ping")  # 预热意图分类器客户端
        except Exception as e:
            print(f"[warn] 意图分类器预热失败（不影响后续，仅可能首次慢）: {e}")

        print("=" * 50)
        print("Agent-Customer 消息处理测试 CLI（真实 CustomerAgent）")
        print("输入消息内容，按回车测试；格式 消息|UID 指定买家")
        print("输入 quit 退出")
        print(f"营业时间窗口: {business_hours['start']}~{business_hours['end']}"
              f"（{'保留真实检查时延' if args.real_delay else '时序模拟已加速'}）")
        print("=" * 50)
        print("[说明] 本工具走真实 CustomerAgent 逻辑；仅发送/转接人工被替换为测试桩，不触真实账号。")

        # 批量用例模式：跑完即退出
        if args.case_file:
            with open(args.case_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    print(f"\n> {line}")
                    t0 = time.perf_counter()
                    await process_message(line, handlers, bot, DEFAULT_BUYER_UID)
                    if args.debug:
                        print(f"[debug] 处理耗时 {time.perf_counter() - t0:.2f}s")
            return

        # 交互模式
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            line = line.strip()
            if line.lower() in QUIT_COMMANDS:
                print("再见。")
                break
            if not line:
                continue
            t0 = time.perf_counter()
            await process_message(line, handlers, bot, DEFAULT_BUYER_UID)
            if args.debug:
                print(f"[debug] 处理耗时 {time.perf_counter() - t0:.2f}s")

    asyncio.run(main())
