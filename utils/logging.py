from loguru import logger
import sys
import os
from config.settings import settings

# 从环境变量获取日志级别，默认使用配置文件中的设置
log_level = os.getenv("LOG_LEVEL", settings.LOG_LEVEL)

# 移除默认的日志处理器
logger.remove()

# 添加标准输出处理器，支持动态日志级别
logger.add(
    sys.stdout,
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# 添加文件输出处理器
logger.add(
    "run.log",
    rotation="10 MB",
    retention="30 days",
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def set_log_level(level: str):
    """动态设置日志级别"""
    # 移除所有现有的处理器
    logger.remove()
    
    # 添加新的处理器，使用新的日志级别
    logger.add(
        sys.stdout,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.add(
        "app.log",
        rotation="10 MB",
        retention="30 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info(f"Log level changed to {level}")