"""
会话历史管理模块

提供 Agent 会话历史的 SQLite 持久化和上下文压缩功能。
"""
from __future__ import annotations

import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from utils.logger_loguru import get_logger
from utils.db_pragma import setup_sqlite_pragmas
from utils.runtime_path import resolve_data_path

logger = get_logger("SessionManager")

Base = declarative_base()


# ==============================================================================
# 模型定义
# ==============================================================================

class AgentMessage(Base):
    """Agent 会话消息模型"""
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user | assistant | tool | summary
    content = Column(Text, nullable=True)
    tool_call_id = Column(String(128), nullable=True)  # 关联 LLM 请求中的 tool_call id
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        Index("ix_agent_messages_session_timestamp", "session_id", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ==============================================================================
# Token 估算
# ==============================================================================

class TokenEstimator:
    """Token 数量估算器"""

    def __init__(self, model_name: str = "gpt-4"):
        self._encoding = None
        self._available = False
        self._model_name = model_name
        self._init_encoding()

    def _init_encoding(self) -> None:
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(self._model_name)
            self._available = True
            logger.debug("tiktoken 初始化成功")
        except ImportError:
            logger.warning("tiktoken 未安装，使用字符数估算 tokens")
            self._available = False
        except Exception:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self._available = True
                logger.debug("tiktoken cl100k_base 编码器初始化成功")
            except Exception as e:
                logger.warning(
                    f"tiktoken 编码器初始化失败: error_type={type(e).__name__}"
                )
                self._available = False

    def estimate(self, text: str) -> int:
        """估算文本的 token 数量"""
        if not text:
            return 0
        if self._available and self._encoding:
            return len(self._encoding.encode(text))
        # 降级：字符数估算（中文字符 ≈ 2 tokens，英文 ≈ 0.25 tokens/字符）
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 2 + other_chars * 0.25)

    def estimate_messages(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息列表的总 token 数量"""
        total = 0
        for msg in messages:
            # 每个消息有 role overhead
            total += 4  # approximate per-message overhead
            if msg.get("content"):
                total += self.estimate(str(msg["content"]))
            if msg.get("tool_calls"):
                total += self.estimate(str(msg["tool_calls"]))
        return total


# ==============================================================================
# SessionManager
# ==============================================================================

class SessionManager:
    """
    会话历史管理器

    提供：
    - 消息持久化（SQLite）
    - 历史加载和追加
    - 基于 token_window 的上下文压缩
    """

    def __init__(
        self,
        db_path: str,
        token_window: int = 131072,
        compress_ratio: float = 0.7,
        retain_count: int = 5,
        model_name: str = "gpt-4",
    ):
        """
        Args:
            db_path: SQLite 数据库文件路径
            token_window: 模型 context window 大小（默认 128k = 131072）
            compress_ratio: 压缩触发比例（默认 0.7，即超过 70% 时触发）
            retain_count: 压缩后保留的最近消息条数
            model_name: 用于 tiktoken 的模型名称
        """
        configured_path = Path(db_path)
        resolved_path = Path(resolve_data_path(configured_path))

        # The default filename changed during account-isolation migration.
        # Reuse an initialized legacy agent.db without deleting or overwriting
        # either database.
        if resolved_path.name == "channel_shop.db" and not self._has_messages_table(resolved_path):
            legacy_path = (
                Path(resolve_data_path(configured_path.with_name("agent.db")))
                if not configured_path.is_absolute()
                else configured_path.with_name("agent.db")
            )
            if self._has_messages_table(legacy_path):
                resolved_path = legacy_path

        self.db_path = str(resolved_path)
        self.token_window = token_window
        self.compress_ratio = compress_ratio
        self.retain_count = retain_count
        self.threshold = int(token_window * compress_ratio)

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        setup_sqlite_pragmas(self.engine)
        self._Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        self._token_estimator = TokenEstimator(model_name=model_name)
        logger.info(
            f"SessionManager 初始化完成: db={self.db_path}, "
            f"threshold={self.threshold}, retain={retain_count}"
        )

    @staticmethod
    def _has_messages_table(path: Path) -> bool:
        """Return whether *path* contains an initialized session table."""
        if not path.is_file():
            return False
        connection = None
        try:
            connection = sqlite3.connect(str(path), timeout=1)
            row = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='agent_messages' LIMIT 1"
            ).fetchone()
            return row is not None
        except (OSError, sqlite3.Error):
            return False
        finally:
            if connection is not None:
                connection.close()

    def _session(self):
        return self._Session()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: Optional[str] = None,
    ) -> bool:
        """追加消息到会话历史"""
        if role not in {"user", "assistant", "tool", "summary"}:
            logger.warning(
                f"unknown session role rejected: role_type={type(role).__name__}"
            )
            return False
        session = self._session()
        try:
            msg = AgentMessage(
                session_id=session_id,
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                timestamp=datetime.now(),
            )
            session.add(msg)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"添加消息失败: error_type={type(e).__name__}")
            return False
        finally:
            session.close()

    def dispose(self) -> None:
        """Release the SQLAlchemy engine and its connection pool."""
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.dispose()

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取会话历史消息（按时间升序）"""
        session = self._session()
        try:
            query = session.query(AgentMessage).filter(
                AgentMessage.session_id == session_id
            ).order_by(AgentMessage.timestamp.asc(), AgentMessage.id.asc())

            if limit is not None:
                query = query.limit(limit)

            messages = query.all()
            return [msg.to_dict() for msg in messages]
        except SQLAlchemyError as e:
            logger.error(f"获取历史消息失败: error_type={type(e).__name__}")
            return []
        finally:
            session.close()

    def get_message_count(self, session_id: str) -> int:
        """获取会话消息总数"""
        session = self._session()
        try:
            return session.query(AgentMessage).filter(
                AgentMessage.session_id == session_id
            ).count()
        except SQLAlchemyError as e:
            logger.error(f"获取消息数量失败: error_type={type(e).__name__}")
            return 0
        finally:
            session.close()

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息列表的 token 数量"""
        return self._token_estimator.estimate_messages(messages)

    def should_compress(self, session_id: str) -> bool:
        """检查是否需要压缩"""
        messages = self.get_history(session_id)
        if not messages:
            return False
        token_count = self.estimate_tokens(messages)
        return token_count > self.threshold

    async def compress_history(
        self,
        session_id: str,
        llm_callable,  # 可调用的 LLM 函数，接收消息列表返回摘要字符串
    ) -> bool:
        """
        压缩会话历史：保留最近 N 条 + 生成摘要

        Args:
            session_id: 会话 ID
            llm_callable: 接受 messages 列表，返回摘要字符串的可调用对象

        Returns:
            压缩是否成功
        """
        session = self._session()
        try:
            # 1. 获取全部历史
            messages = session.query(AgentMessage).filter(
                AgentMessage.session_id == session_id
            ).order_by(AgentMessage.timestamp.asc(), AgentMessage.id.asc()).all()

            if len(messages) <= self.retain_count + 1:
                logger.debug(f"消息数量 {len(messages)} 未达到压缩阈值，跳过")
                return False

            # 2. 分离要压缩的旧消息和要保留的新消息
            old_messages = messages[:-self.retain_count]
            retain_messages = messages[-self.retain_count:]

            # 3. 生成摘要
            summary_input = [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in old_messages
            ]

            try:
                summary = await llm_callable(summary_input)
            except Exception as e:
                logger.error(f"生成摘要失败: error_type={type(e).__name__}")
                # Never delete history when the summary is unavailable.
                session.rollback()
                return False

            # 4. 删除旧消息
            for msg in old_messages:
                session.delete(msg)

            # 5. 在顶部插入不可信摘要消息
            summary_timestamp = retain_messages[0].timestamp - timedelta(microseconds=1)
            summary_msg = AgentMessage(
                session_id=session_id,
                # A model-generated summary is conversation data, never a new
                # system instruction. MessageBuilder renders this role with a
                # user-level safety wrapper.
                role="summary",
                content=str(summary),
                tool_call_id=None,
                timestamp=summary_timestamp,
            )
            session.add(summary_msg)

            # 6. 更新保留消息的时间戳（保持顺序）
            # 保留消息已经在数据库中，无需移动

            session.commit()
            logger.info(
                f"压缩完成: session={session_id}, "
                f"删除={len(old_messages)} 条, 保留={len(retain_messages)} 条"
            )
            return True

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"压缩历史失败: error_type={type(e).__name__}")
            return False
        finally:
            session.close()
