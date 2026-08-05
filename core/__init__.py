"""
核心模块 - 提供系统基础设施功能
包括依赖注入、连接状态管理、工具类等
"""

from .di_container import DIContainer, container, configure_standard_services
from .base_service import BaseService
from .connection_status import ConnectionStatusManager, ConnectionState, ConnectionStatus

__all__ = [
    'DIContainer',
    'container',
    'configure_standard_services',
    'BaseService',
    'ConnectionStatusManager',
    'ConnectionState',
    'ConnectionStatus',
]
