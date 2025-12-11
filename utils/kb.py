from pathlib import Path
from config.settings import settings
from utils.logging import logger
import os
def get_available_knowledge_bases():
    """获取可用的知识库列表"""
    try:
        kb_path = Path(settings.KB_PATH)
        if kb_path.exists():
            collections = os.listdir(kb_path)
            return collections if collections else ["default"]
        return ["default"]
    except Exception as e:
        logger.error(f"Error getting knowledge bases: {e}")
        return ["default"]