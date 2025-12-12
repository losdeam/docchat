from loguru import logger
import sys
import os
from config.settings import settings
from typing import Callable, Any
import time
import functools
from datetime import datetime
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

def log_execution(name: str = None):
    """
    函数执行日志装饰器
    
    参数:
    name: 日志中显示的函数名称。如果为None，则使用被装饰函数的名称
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 确定日志中使用的名称
            log_name = name if name else func.__name__
            
            # 开始时间
            start_time = time.time()
            start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 输出开始日志
            logger.info(f"[{log_name}] 开始执行")
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 结束时间
                end_time = time.time()
                end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                execution_time = end_time - start_time
                
                # 输出结束日志
                logger.info(f"[{log_name}] 执行结束，执行耗时: {execution_time:.4f}秒")
                
                return result
                
            except Exception as e:
                # 发生异常时的处理
                end_time = time.time()
                end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                execution_time = end_time - start_time
                
                logger.info(f"[{log_name}] 执行出错: {str(e)}，执行耗时: {execution_time:.4f}秒")

                raise
        
        return wrapper
    
    return decorator