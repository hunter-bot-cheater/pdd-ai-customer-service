"""服务层 - 封装数据访问与渠道操作，供 UI 调用，避免 UI 直接依赖 database/Channel。

薄封装：仅转发到既有的 db_manager / 渠道模块，不改变业务逻辑。
"""
from .keyword_service import keyword_service
from .account_service import account_service

__all__ = ["keyword_service", "account_service"]
