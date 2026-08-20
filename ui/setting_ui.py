# 设置界面

from PyQt6.QtCore import Qt, QTime
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QVBoxLayout, QWidget,
                            QMessageBox, QStackedWidget)
from PyQt6.QtGui import QFont
from qfluentwidgets import (CardWidget, SubtitleLabel, CaptionLabel, StrongBodyLabel, BodyLabel,
                           PrimaryPushButton, PushButton,
                           LineEdit, ScrollArea, FluentIcon as FIF,
                           InfoBar, InfoBarPosition, TextEdit, PasswordLineEdit,
                           SwitchButton, SpinBox, DoubleSpinBox, Pivot)
from utils.logger_loguru import get_logger
from config import config, config_base


# ---------------------------------------------------------------------------
# 基础控件辅助函数
# ---------------------------------------------------------------------------

def _double_spin(min_v: float = 0.0, max_v: float = 3600.0, decimals: int = 1, step: float = 0.5):
    """构造一个浮点数字输入框（单位写在标签里，避免依赖 setSuffix）。"""
    w = DoubleSpinBox()
    w.setRange(min_v, max_v)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setFixedWidth(140)
    return w


def _int_spin(min_v: int = 0, max_v: int = 100000, step: int = 1):
    """构造一个整数输入框。"""
    w = SpinBox()
    w.setRange(min_v, max_v)
    w.setSingleStep(step)
    w.setFixedWidth(140)
    return w


def _switch():
    """构造一个开关按钮。"""
    w = SwitchButton()
    w.setFixedWidth(80)
    return w


def _add_row(layout: QVBoxLayout, label: str, widget):
    """向纵向布局添加一行：左侧/上方中文标签 + 右侧/下方控件。

    使用 QFormLayout 时标签会被固定宽度的控件挤出，因此改用手动
    水平布局，确保标签始终可见。
    """
    row = QHBoxLayout()
    row.setSpacing(12)
    row.setContentsMargins(0, 0, 0, 0)
    row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    lbl = BodyLabel(label)
    lbl.setFixedWidth(180)
    lbl.setWordWrap(True)
    row.addWidget(lbl)

    # 让文本类输入框随窗口拉伸，数字/开关保持固定宽度
    if isinstance(widget, (LineEdit, PasswordLineEdit, TextEdit)):
        widget.setMinimumWidth(240)
        row.addWidget(widget, 1)
    else:
        row.addWidget(widget, 0, Qt.AlignmentFlag.AlignLeft)

    row.addStretch(1)
    layout.addLayout(row)
    return widget


def _add_time_row(layout: QVBoxLayout, label: str, hour_spin, minute_spin):
    """添加一行"标签 + 小时 : 分钟"的营业时间控件，SpinBox 启用循环。"""
    row = QHBoxLayout()
    row.setSpacing(12)
    row.setContentsMargins(0, 0, 0, 0)
    row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    lbl = BodyLabel(label)
    lbl.setFixedWidth(180)
    lbl.setWordWrap(True)
    row.addWidget(lbl)

    hour_spin.setWrapping(True)
    minute_spin.setWrapping(True)

    row.addWidget(hour_spin, 0, Qt.AlignmentFlag.AlignLeft)
    row.addWidget(BodyLabel(":"), 0, Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(minute_spin, 0, Qt.AlignmentFlag.AlignLeft)
    row.addStretch(1)
    layout.addLayout(row)
    return hour_spin, minute_spin


def _join_lines(value) -> str:
    """把列表/字符串规整为换行分隔的多行文本（用于 TextEdit 显示）。"""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    if isinstance(value, str):
        return value
    return ""


# ---------------------------------------------------------------------------
# 配置卡片：基础设置（营业时间 / 数据库路径）
# ---------------------------------------------------------------------------

class BasicConfigCard(CardWidget):
    """基础设置卡片：营业时间、数据库路径"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("基础设置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        self.start_hour = _int_spin(0, 23, 1)
        self.start_minute = _int_spin(0, 59, 1)
        _add_time_row(form, "开始时间", self.start_hour, self.start_minute)

        self.end_hour = _int_spin(0, 23, 1)
        self.end_minute = _int_spin(0, 59, 1)
        _add_time_row(form, "结束时间", self.end_hour, self.end_minute)

        self.db_path_edit = LineEdit()
        self.db_path_edit.setPlaceholderText("./temp/channel_shop.db")
        _add_row(form, "数据库路径:", self.db_path_edit)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "设置 AI 客服的工作时间与数据文件位置。\n"
            "非工作时间系统不会自动回复。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        def _fmt(h, m):
            return f"{int(h):02d}:{int(m):02d}"
        return {
            "business_hours": {
                "start": _fmt(self.start_hour.value(), self.start_minute.value()),
                "end": _fmt(self.end_hour.value(), self.end_minute.value()),
            },
            "db_path": (self.db_path_edit.text().strip() or "./temp/channel_shop.db"),
        }

    def setConfig(self, data: dict):
        bh = data.get("business_hours", {}) or {}
        start = QTime.fromString(str(bh.get("start", "00:00")), "HH:mm")
        if start.isValid():
            self.start_hour.setValue(start.hour())
            self.start_minute.setValue(start.minute())
        end = QTime.fromString(str(bh.get("end", "23:59")), "HH:mm")
        if end.isValid():
            self.end_hour.setValue(end.hour())
            self.end_minute.setValue(end.minute())

        self.db_path_edit.setText(data.get("db_path", "./temp/channel_shop.db") or "")


# ---------------------------------------------------------------------------
# 配置卡片：LLM 模型配置
# ---------------------------------------------------------------------------

class LLMConfigCard(CardWidget):
    """LLM配置卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("LLM 模型配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        self.api_base_edit = LineEdit()
        self.api_base_edit.setPlaceholderText("https://ark.cn-beijing.volces.com/api/v3")
        _add_row(form, "API 地址:", self.api_base_edit)

        self.api_key_edit = PasswordLineEdit()
        self.api_key_edit.setPlaceholderText("输入您的 API Key")
        _add_row(form, "API 密钥:", self.api_key_edit)

        self.model_name_edit = LineEdit()
        self.model_name_edit.setPlaceholderText("输入模型名称，如：doubao-seed-1-6-flash-250828")
        _add_row(form, "模型名称:", self.model_name_edit)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "配置 LLM 模型的连接参数。\n"
            "支持 OpenAI 兼容的 API 接口，包括豆包、通义千问等模型。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "api_base": self.api_base_edit.text().strip() or "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": self.api_key_edit.text().strip(),
            "model_name": self.model_name_edit.text().strip()
        }

    def setConfig(self, data: dict):
        self.api_base_edit.setText(data.get("api_base", "https://ark.cn-beijing.volces.com/api/v3"))
        self.api_key_edit.setText(data.get("api_key", ""))
        self.model_name_edit.setText(data.get("model_name", ""))


# ---------------------------------------------------------------------------
# 配置卡片：AI 回复行为
# ---------------------------------------------------------------------------

class AIReplyConfigCard(CardWidget):
    """AI 回复行为配置卡片：已读/打字延迟、拆分间隔、消息长度、合并、防重复等"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("AI 回复行为")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        # 已读延迟
        self.read_min = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "已读最短时间（秒）:", self.read_min)
        self.read_max = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "已读最长时间（秒）:", self.read_max)

        # 打字延迟
        self.typing_min = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "打字最短时间（秒）:", self.typing_min)
        self.typing_max = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "打字最长时间（秒）:", self.typing_max)

        # 分条间隔
        self.split_min = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "分句最短间隔（秒）:", self.split_min)
        self.split_max = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "分句最长间隔（秒）:", self.split_max)

        # 消息长度与句数
        self.max_message_len = _int_spin(1, 2000, 1)
        _add_row(form, "单条消息最大字数:", self.max_message_len)
        self.uid_min_interval = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "同用户最小回复间隔（秒）:", self.uid_min_interval)
        self.max_sentences = _int_spin(1, 50, 1)
        _add_row(form, "单次回复最多句数:", self.max_sentences)

        # 消息合并
        self.enable_coalesce = _switch()
        _add_row(form, "启用消息合并:", self.enable_coalesce)
        self.coalesce_window_sec = _double_spin(0, 60, 1, 0.5)
        _add_row(form, "合并窗口时长（秒）:", self.coalesce_window_sec)

        # 防重复发送
        self.repeat_cache_ttl_sec = _int_spin(0, 86400, 10)
        _add_row(form, "重复缓存有效期（秒）:", self.repeat_cache_ttl_sec)
        self.repeat_cache_max = _int_spin(0, 1000, 1)
        _add_row(form, "重复缓存最大条数:", self.repeat_cache_max)
        self.repeat_rewrite_max = _int_spin(0, 10, 1)
        _add_row(form, "重复改写最大次数:", self.repeat_rewrite_max)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "控制 AI 回复的拟人节奏与消息拆分策略。\n"
            "单条字数过小时回复会被拆成多条；开启合并可在买家连发时合并为一条回答。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "read_seconds_min": round(self.read_min.value(), 1),
            "read_seconds_max": round(self.read_max.value(), 1),
            "typing_seconds_min": round(self.typing_min.value(), 1),
            "typing_seconds_max": round(self.typing_max.value(), 1),
            "split_interval_min": round(self.split_min.value(), 1),
            "split_interval_max": round(self.split_max.value(), 1),
            "max_message_len": int(self.max_message_len.value()),
            "uid_min_interval": round(self.uid_min_interval.value(), 1),
            "max_sentences": int(self.max_sentences.value()),
            "enable_coalesce": bool(self.enable_coalesce.isChecked()),
            "coalesce_window_sec": round(self.coalesce_window_sec.value(), 1),
            "repeat_cache_ttl_sec": int(self.repeat_cache_ttl_sec.value()),
            "repeat_cache_max": int(self.repeat_cache_max.value()),
            "repeat_rewrite_max": int(self.repeat_rewrite_max.value()),
        }

    def setConfig(self, data: dict):
        def num(key, default):
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def inum(key, default):
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return int(default)

        self.read_min.setValue(num("read_seconds_min", 6))
        self.read_max.setValue(num("read_seconds_max", 8))
        self.typing_min.setValue(num("typing_seconds_min", 4))
        self.typing_max.setValue(num("typing_seconds_max", 6))
        self.split_min.setValue(num("split_interval_min", 3))
        self.split_max.setValue(num("split_interval_max", 6))
        self.max_message_len.setValue(inum("max_message_len", 25))
        self.uid_min_interval.setValue(num("uid_min_interval", 4))
        self.max_sentences.setValue(inum("max_sentences", 4))
        self.enable_coalesce.setChecked(bool(data.get("enable_coalesce", True)))
        self.coalesce_window_sec.setValue(num("coalesce_window_sec", 6))
        self.repeat_cache_ttl_sec.setValue(inum("repeat_cache_ttl_sec", 300))
        self.repeat_cache_max.setValue(inum("repeat_cache_max", 12))
        self.repeat_rewrite_max.setValue(inum("repeat_rewrite_max", 1))


# ---------------------------------------------------------------------------
# 配置卡片：通知设置
# ---------------------------------------------------------------------------

class NotificationConfigCard(CardWidget):
    """通知设置卡片：企业微信 Webhook、通知冷却、转人工有效时长"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("通知设置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        self.webhook_edit = LineEdit()
        self.webhook_edit.setPlaceholderText("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...")
        _add_row(form, "企业微信 Webhook 地址:", self.webhook_edit)

        self.cooldown = _int_spin(0, 86400, 10)
        _add_row(form, "转人工通知冷却时间（秒）:", self.cooldown)

        self.handoff_valid_hours = _double_spin(0.1, 8760, 1, 0.5)
        _add_row(form, "转人工有效时长（小时）:", self.handoff_valid_hours)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "转人工/紧急通知会推送到企业微信群机器人。\n"
            "留空则不发送群消息（仍会转接，但不发通知）。冷却时间避免同一会话反复刷屏；\n"
            "转人工有效时长到期后会话恢复由 AI 处理。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "wechat_webhook": self.webhook_edit.text().strip(),
            "handoff_cooldown_seconds": int(self.cooldown.value()),
            "handoff": {
                "valid_hours": round(self.handoff_valid_hours.value(), 1),
            },
        }

    def setConfig(self, data: dict):
        self.webhook_edit.setText(data.get("wechat_webhook", "") or "")
        try:
            self.cooldown.setValue(int(data.get("handoff_cooldown_seconds", 300)))
        except (TypeError, ValueError):
            self.cooldown.setValue(300)

        hv = data.get("handoff", {}) or {}
        try:
            self.handoff_valid_hours.setValue(float(hv.get("valid_hours", 4)))
        except (TypeError, ValueError):
            self.handoff_valid_hours.setValue(4)


# ---------------------------------------------------------------------------
# 配置卡片：意图分类
# ---------------------------------------------------------------------------

class IntentConfigCard(CardWidget):
    """意图分类配置卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("意图分类")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        self.enabled = _switch()
        _add_row(form, "启用意图分类:", self.enabled)

        self.threshold = _double_spin(0.0, 1.0, 2, 0.05)
        _add_row(form, "意图置信度阈值（0~1）:", self.threshold)

        self.cache_ttl = _int_spin(0, 86400, 60)
        _add_row(form, "意图分类缓存时间（秒）:", self.cache_ttl)

        self.timeout = _double_spin(0.5, 60, 1, 0.5)
        _add_row(form, "意图分类超时时间（秒）:", self.timeout)

        self.max_tokens = _int_spin(1, 4096, 1)
        _add_row(form, "意图分类最大 Token 数:", self.max_tokens)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "意图分类决定消息是否需要转人工（投诉/负面情绪/售后操作等）。\n"
            "置信度阈值越高越保守；低于阈值会保守转人工。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "enabled": bool(self.enabled.isChecked()),
            "threshold": round(self.threshold.value(), 2),
            "cache_ttl_seconds": int(self.cache_ttl.value()),
            "timeout_seconds": round(self.timeout.value(), 1),
            "max_tokens": int(self.max_tokens.value()),
        }

    def setConfig(self, data: dict):
        self.enabled.setChecked(bool(data.get("enabled", True)))
        try:
            self.threshold.setValue(float(data.get("threshold", 0.6)))
        except (TypeError, ValueError):
            self.threshold.setValue(0.6)
        try:
            self.cache_ttl.setValue(int(data.get("cache_ttl_seconds", 3600)))
        except (TypeError, ValueError):
            self.cache_ttl.setValue(3600)
        try:
            self.timeout.setValue(float(data.get("timeout_seconds", 5.0)))
        except (TypeError, ValueError):
            self.timeout.setValue(5.0)
        try:
            self.max_tokens.setValue(int(data.get("max_tokens", 32)))
        except (TypeError, ValueError):
            self.max_tokens.setValue(32)


# ---------------------------------------------------------------------------
# 配置卡片：提示词配置
# ---------------------------------------------------------------------------

class PromptConfigCard(CardWidget):
    """提示词配置卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("AI 提示词配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form = QVBoxLayout()
        form.setSpacing(12)

        self.instructions_edit = TextEdit()
        self.instructions_edit.setPlaceholderText("输入行为指令，每行一条")
        self.instructions_edit.setMaximumHeight(260)
        _add_row(form, "行为指令:", self.instructions_edit)

        layout.addLayout(form)

        description_label = CaptionLabel(
            "配置 AI 助手的行为指令，每行一条。\n"
            "角色描述和工具说明由系统自动管理，无需手动配置。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "instructions": [
                line.strip() for line in self.instructions_edit.toPlainText().splitlines() if line.strip()
            ]
        }

    def setConfig(self, data: dict):
        instructions = data.get("instructions", [])
        self.instructions_edit.setPlainText(_join_lines(instructions))


# ---------------------------------------------------------------------------
# 设置主界面
# ---------------------------------------------------------------------------

class SettingUI(QFrame):
    """设置界面：分页覆盖全部配置项"""

    # 各分页对应的卡片 key
    PAGE_KEYS = ["basic", "llm", "ai_reply", "notification", "intent", "prompt"]
    PAGE_TEXTS = {
        "basic": "基础设置",
        "llm": "LLM 设置",
        "ai_reply": "AI 回复行为",
        "notification": "通知设置",
        "intent": "意图分类",
        "prompt": "提示词",
    }

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = get_logger("SettingUI")
        self.cards = {}
        self._pages = {}
        self.setupUI()
        self.loadConfig()
        self.setObjectName("设置")

    # ---- UI 骨架 ----------------------------------------------------------

    def setupUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        main_layout.addWidget(self.createHeaderWidget())

        self.pivot = Pivot(self)
        self.stack = QStackedWidget(self)

        self._build_pages()

        self.pivot.currentItemChanged.connect(
            lambda key: self.stack.setCurrentWidget(self._pages[key])
        )
        self.pivot.setCurrentItem(self.PAGE_KEYS[0])

        main_layout.addWidget(self.pivot)
        main_layout.addWidget(self.stack, 1)

        self.save_btn.clicked.connect(self.onSaveConfig)
        self.reset_btn.clicked.connect(self.onResetConfig)

    def _build_pages(self):
        """为每个分页创建滚动容器并放入对应卡片。"""
        self.cards["basic"] = BasicConfigCard()
        self.cards["llm"] = LLMConfigCard()
        self.cards["ai_reply"] = AIReplyConfigCard()
        self.cards["notification"] = NotificationConfigCard()
        self.cards["intent"] = IntentConfigCard()
        self.cards["prompt"] = PromptConfigCard()

        for key in self.PAGE_KEYS:
            scroll = ScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setStyleSheet("ScrollArea { border: none; background-color: transparent; }")

            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setSpacing(20)
            cl.setContentsMargins(20, 20, 20, 20)
            cl.setAlignment(Qt.AlignmentFlag.AlignTop)
            cl.addWidget(self.cards[key])
            cl.addStretch()
            container.setStyleSheet("QWidget { background-color: transparent; border: none; }")

            scroll.setWidget(container)

            self._pages[key] = scroll
            self.pivot.addItem(key, self.PAGE_TEXTS[key])
            self.stack.addWidget(scroll)

    def createHeaderWidget(self):
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(20)

        title_label = SubtitleLabel("系统设置")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))

        description_label = CaptionLabel("配置 AI 客服的模型、回复行为、通知、意图与提示词等参数")
        description_label.setStyleSheet("color: #666;")

        title_area = QWidget()
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addWidget(description_label)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)

        self.reset_btn = PushButton("重置")
        self.reset_btn.setIcon(FIF.UPDATE)
        self.reset_btn.setFixedSize(80, 40)

        self.save_btn = PrimaryPushButton("保存")
        self.save_btn.setIcon(FIF.SAVE)
        self.save_btn.setFixedSize(100, 40)

        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.save_btn)

        header_layout.addWidget(title_area)
        header_layout.addStretch()
        header_layout.addWidget(buttons_widget)

        return header_widget

    # ---- 加载：config -> UI ----------------------------------------------

    def loadConfig(self):
        """从 config 模块加载全部配置并填充到对应控件（缺失字段用默认值补齐）。"""
        try:
            data = {
                "basic": {
                    "business_hours": {
                        "start": config.get("business_hours.start", "00:00"),
                        "end": config.get("business_hours.end", "23:59"),
                    },
                    "db_path": config.get("db_path", "./temp/channel_shop.db"),
                },
                "llm": {
                    "api_base": config.get("llm.api_base", "https://ark.cn-beijing.volces.com/api/v3"),
                    "api_key": config.get("llm.api_key", ""),
                    "model_name": config.get("llm.model_name", ""),
                },
                "ai_reply": {
                    "read_seconds_min": config.get("ai_reply.read_seconds_min", 6),
                    "read_seconds_max": config.get("ai_reply.read_seconds_max", 8),
                    "typing_seconds_min": config.get("ai_reply.typing_seconds_min", 4),
                    "typing_seconds_max": config.get("ai_reply.typing_seconds_max", 6),
                    "split_interval_min": config.get("ai_reply.split_interval_min", 3),
                    "split_interval_max": config.get("ai_reply.split_interval_max", 6),
                    "max_message_len": config.get("ai_reply.max_message_len", 25),
                    "uid_min_interval": config.get("ai_reply.uid_min_interval", 4),
                    "max_sentences": config.get("ai_reply.max_sentences", 4),
                    "enable_coalesce": config.get("ai_reply.enable_coalesce", True),
                    "coalesce_window_sec": config.get("ai_reply.coalesce_window_sec", 6),
                    "repeat_cache_ttl_sec": config.get("ai_reply.repeat_cache_ttl_sec", 300),
                    "repeat_cache_max": config.get("ai_reply.repeat_cache_max", 12),
                    "repeat_rewrite_max": config.get("ai_reply.repeat_rewrite_max", 1),
                },
                "notification": {
                    "wechat_webhook": config.get("notification.wechat_webhook", ""),
                    "handoff_cooldown_seconds": config.get("notification.handoff_cooldown_seconds", 300),
                    "handoff": {"valid_hours": config.get("handoff.valid_hours", 4)},
                },
                "intent": {
                    "enabled": config.get("intent.enabled", True),
                    "threshold": config.get("intent.threshold", 0.6),
                    "cache_ttl_seconds": config.get("intent.cache_ttl_seconds", 3600),
                    "timeout_seconds": config.get("intent.timeout_seconds", 5.0),
                    "max_tokens": config.get("intent.max_tokens", 32),
                },
                "prompt": {
                    "instructions": config.get("prompt.instructions", []),
                },
            }

            self.cards["basic"].setConfig(data["basic"])
            self.cards["llm"].setConfig(data["llm"])
            self.cards["ai_reply"].setConfig(data["ai_reply"])
            self.cards["notification"].setConfig(data["notification"])
            self.cards["intent"].setConfig(data["intent"])
            self.cards["prompt"].setConfig(data["prompt"])

            self.logger.info("配置加载成功")
        except Exception as e:
            self.logger.error(f"加载配置失败: error_type={type(e).__name__}")
            QMessageBox.warning(self, "加载失败", f"加载配置失败：{str(e)}")

    # ---- 收集与校验：UI -> dict ------------------------------------------

    def _collect(self) -> dict:
        """从界面收集各分页配置。"""
        return {key: self.cards[key].getConfig() for key in self.PAGE_KEYS}

    @staticmethod
    def _merge_section(section_key: str, ui_dict: dict) -> dict:
        """以磁盘现有配置为基底，仅覆盖 UI 管理的字段，避免丢失未显示的子字段。"""
        existing = config.get(section_key, {}) or {}
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(ui_dict)
        return merged

    def _validate(self, parts: dict) -> str | None:
        """返回错误信息字符串；无错误返回 None。"""
        ar = parts["ai_reply"]
        pairs = [
            ("read_seconds_min", "read_seconds_max"),
            ("typing_seconds_min", "typing_seconds_max"),
            ("split_interval_min", "split_interval_max"),
        ]
        for lo, hi in pairs:
            if ar.get(lo, 0) > ar.get(hi, 0):
                return f"「{lo}」不能大于「{hi}」，请检查 AI 回复行为分页。"

        if ar.get("enable_coalesce") and ar.get("coalesce_window_sec", 0) <= 0:
            return "启用消息合并时，合并窗口时长必须大于 0。"

        thr = parts["intent"].get("threshold", 0.6)
        if thr < 0 or thr > 1:
            return "意图置信度阈值必须在 0~1 之间。"

        bh = parts["basic"]["business_hours"]
        if bh.get("start") == bh.get("end"):
            return "营业开始时间和结束时间不能相同。"

        return None

    # ---- 保存：UI -> config -> 磁盘 --------------------------------------

    def onSaveConfig(self):
        try:
            parts = self._collect()

            # 必填项检查
            if not parts["llm"].get("api_key"):
                QMessageBox.warning(self, "配置错误", "请输入 LLM API Key！")
                return
            if not parts["llm"].get("model_name"):
                QMessageBox.warning(self, "配置错误", "请输入 LLM 模型名称！")
                return

            err = self._validate(parts)
            if err:
                QMessageBox.warning(self, "配置错误", err)
                return

            # 构建完整配置，按分组与磁盘现有值合并（不丢字段）
            new_config = {
                "business_hours": self._merge_section("business_hours", parts["basic"]["business_hours"]),
                "db_path": parts["basic"]["db_path"],
                "handoff": self._merge_section("handoff", parts["notification"]["handoff"]),
                "llm": self._merge_section("llm", parts["llm"]),
                "ai_reply": self._merge_section("ai_reply", parts["ai_reply"]),
                "notification": self._merge_section("notification", {
                    "wechat_webhook": parts["notification"]["wechat_webhook"],
                    "handoff_cooldown_seconds": parts["notification"]["handoff_cooldown_seconds"],
                }),
                "intent": self._merge_section("intent", parts["intent"]),
                "prompt": self._merge_section("prompt", parts["prompt"]),
            }

            config.update(new_config, save=True)
            self.logger.info("配置保存成功")

            InfoBar.success(
                title="保存成功",
                content="配置已写入 config.json",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )

            QMessageBox.information(
                self,
                "重启提示",
                "配置已保存。\n\n"
                "营业时间、LLM、提示词等部分参数需重启程序后才能完全生效，"
                "建议重启本应用以使改动生效。",
            )

        except Exception as e:
            self.logger.error(f"保存配置失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "保存失败", f"保存配置时发生错误：{str(e)}")

    def onResetConfig(self):
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有配置吗？\n这将重新加载配置文件中的原始设置。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                config.reload()
                self.loadConfig()
                self.logger.info("配置已重置")

                InfoBar.success(
                    title="重置成功",
                    content="配置已重置为配置文件中的设置！",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self,
                )
            except Exception as e:
                self.logger.error(f"重置配置失败: error_type={type(e).__name__}")
                QMessageBox.critical(self, "重置失败", f"重置配置失败：{str(e)}")
