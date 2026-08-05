"""关键词服务层 - 封装关键词持久化访问，解耦 UI 与数据库层。"""
from database.db_manager import db_manager


class KeywordService:
    """关键词服务的薄封装，转发到 db_manager。"""

    def get_all_keywords(self):
        return db_manager.get_all_keywords()

    def add_keyword(self, keyword: str) -> bool:
        return db_manager.add_keyword(keyword)

    def update_keyword(self, old_keyword: str, new_keyword: str) -> bool:
        return db_manager.update_keyword(old_keyword, new_keyword)

    def delete_keyword(self, keyword: str) -> bool:
        return db_manager.delete_keyword(keyword)


keyword_service = KeywordService()
