from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES
import os
from pathlib import Path
from typing import List

def parse_weights(weight_str):
    """解析权重字符串为列表"""
    try:
        # 移除空格并解析列表
        weight_str = weight_str.replace(" ", "")
        if weight_str.startswith('[') and weight_str.endswith(']'):
            weights = [float(x) for x in weight_str[1:-1].split(',')]
            if len(weights) == 2:
                return weights
    except:
        pass
    # 默认权重
    return [0.2, 0.8]

class Settings(BaseSettings):
    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    DOC_CACHE_PATH : str = "./document_cache"
    KB_PATH :str = "./kb_cache"
    KEY_JSON_NAME:str = "hase2name.json" # 存储文件哈希值到文件名称的映射。
    # Retrieval settings - 增加检索的文档数量以提高召回率
    VECTOR_SEARCH_K: int = 20
    HYBRID_RETRIEVER_WEIGHTS: list = [0.2, 0.8]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # 缓存最大总大小（字节），默认1GB
    MAX_CACHE_SIZE: int = 1024 * 1024 * 1024
    
    # SiliconFlow settings
    SILICONFLOW_KEY: str = ""
    SILICONFLOW_URL: str = ""
    
    # Research model settings
    RESEARCH_MODEL_SERVER: str = ""
    RESEARCH_MODEL_NAME: str = ""
    
    # Checker model settings
    CHECKER_MODEL_SERVER: str = ""
    CHECKER_MODEL_NAME: str = ""
    
    # Verification model settings
    VERIFICATION_MODEL_SERVER: str = ""
    VERIFICATION_MODEL_NAME: str = ""

    EMBEDDING_MODEL_SERVER: str = ""
    EMBEDDING_MODEL_NAME: str = ""

    # Core paths - 整合到配置中方便统一管理
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    CACHE_DIR_PATH: str = str(PROJECT_ROOT / "cache")
    CHROMA_DB_DEFAULT_PATH: str = str(PROJECT_ROOT / "chroma_db")
    CHROMA_DEFAULT_COLLECTION_NAME: str = "docchat-collection"
    # 解析器相关配置
    PROCESSOR: str = ""
    # 检索器相关配置
    RETRIEVER: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局设置实例
settings = Settings()