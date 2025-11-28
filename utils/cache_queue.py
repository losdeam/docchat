import asyncio
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from threading import Thread, Lock
from collections import deque
from config.settings import settings
from utils.logging import logger


class CacheQueueManager:
    """
    消息队列管理器，用于处理文档缓存的自动清理
    
    该类使用生产者-消费者模式来管理缓存文件的生命周期。
    当文件被访问时，会将其加入队列，当队列中的文件达到过期时间后，
    自动将其从磁盘上删除以释放空间。
    """
    
    def __init__(self, cache_dir: str = None, expire_days: int = None, max_total_size: int = None):
        """
        初始化缓存队列管理器
        
        Args:
            cache_dir: 缓存目录路径
            expire_days: 缓存过期天数
            max_total_size: 缓存最大总大小（字节），默认为None表示无限制
        """
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.expire_days = expire_days or settings.CACHE_EXPIRE_DAYS
        self.expire_seconds = self.expire_days * 24 * 60 * 60
        # 默认最大缓存大小为1GB，可以通过settings配置
        self.max_total_size = max_total_size or getattr(settings, "MAX_CACHE_SIZE", 1024*1024*1024)
        
        # 使用deque作为队列存储(文件路径, 过期时间戳, 文件大小)
        self.queue = deque()
        self.lock = Lock()
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 控制清理线程的标志
        self.running = False
        self.cleanup_thread = None
        
        # 初始化时加载现有缓存文件
        self._load_existing_cache_files()
        
        logger.info(f"CacheQueueManager initialized with cache_dir: {self.cache_dir}, "
                   f"expire_days: {self.expire_days}, max_total_size: {self.max_total_size}")

    def _load_existing_cache_files(self):
        """
        初始化时加载已存在的缓存文件到队列中
        """
        try:
            cache_files = []
            for file_path in self.cache_dir.glob("*.pkl"):
                if file_path.is_file():
                    stat = file_path.stat()
                    cache_files.append((
                        str(file_path),
                        stat.st_mtime + self.expire_seconds,  # 过期时间
                        stat.st_size  # 文件大小
                    ))
            
            # 按照修改时间排序（最早的在前面）
            cache_files.sort(key=lambda x: x[1] - self.expire_seconds)
            
            with self.lock:
                self.queue.extend(cache_files)
                
            logger.info(f"Loaded {len(cache_files)} existing cache files")
        except Exception as e:
            logger.error(f"Error loading existing cache files: {e}")

    def start_cleanup_loop(self):
        """
        启动后台清理循环线程
        """
        if not self.running:
            self.running = True
            self.cleanup_thread = Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Cache cleanup loop started")
    
    def stop_cleanup_loop(self):
        """
        停止后台清理循环线程
        """
        if self.running:
            self.running = False
            if self.cleanup_thread:
                self.cleanup_thread.join(timeout=5)  # 等待最多5秒
            logger.info("Cache cleanup loop stopped")
    
    def add_file(self, file_path: str):
        """
        将文件添加到清理队列中
        
        Args:
            file_path: 要跟踪的缓存文件路径
        """
        try:
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            expire_time = time.time() + self.expire_seconds
            
            with self.lock:
                # 检查文件是否已经在队列中，避免重复添加
                for existing_path, _, _ in self.queue:
                    if existing_path == file_path:
                        logger.debug(f"File already in cache queue: {file_path}")
                        return
                
                # 添加到队列
                self.queue.append((file_path, expire_time, file_size))
                logger.debug(f"Added file to cache queue: {file_path}, size: {file_size}, expires at: {expire_time}")
                
                # 检查并维护总大小限制
                self._maintain_size_limit()
                
        except Exception as e:
            logger.error(f"Error adding file to cache queue: {e}")
    
    def _maintain_size_limit(self):
        """
        维护缓存总大小限制，必要时删除最老的文件
        """
        if self.max_total_size <= 0:
            return  # 无大小限制
            
        try:
            # 计算当前总大小
            total_size = sum(size for _, _, size in self.queue)
            
            # 如果超出限制，删除最老的文件直到满足限制
            while total_size > self.max_total_size and self.queue:
                oldest_file_path, _, oldest_file_size = self.queue.popleft()
                total_size -= oldest_file_size
                
                # 删除实际文件
                try:
                    if os.path.exists(oldest_file_path):
                        os.remove(oldest_file_path)
                        logger.info(f"Removed oldest cache file to maintain size limit: {oldest_file_path} ({oldest_file_size} bytes)")
                    else:
                        logger.debug(f"Oldest cache file already removed: {oldest_file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {oldest_file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error maintaining cache size limit: {e}")

    def remove_file(self, file_path: str):
        """
        从队列中移除特定文件（当文件被主动使用时）
        
        Args:
            file_path: 要从队列中移除的文件路径
        """
        with self.lock:
            removed_items = [(fp, et, sz) for fp, et, sz in self.queue if fp != file_path]
            removed_count = len(self.queue) - len(removed_items)
            self.queue = deque(removed_items)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} instances of file from cache queue: {file_path}")
    
    def _cleanup_loop(self):
        """
        后台清理循环，定期检查并删除过期的缓存文件
        """
        while self.running:
            try:
                current_time = time.time()
                expired_files = []
                
                # 收集所有已过期的文件
                with self.lock:
                    while self.queue and self.queue[0][1] <= current_time:
                        expired_files.append(self.queue.popleft())
                
                # 删除过期文件
                for file_path, expire_time, file_size in expired_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Removed expired cache file: {file_path} ({file_size} bytes)")
                        else:
                            logger.debug(f"Cache file already removed: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {file_path}: {e}")
                
                # 如果没有待处理的任务，短暂休眠
                if not expired_files:
                    time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                time.sleep(60)
    
    def get_queue_stats(self) -> Dict:
        """
        获取队列统计信息
        
        Returns:
            包含队列统计信息的字典
        """
        with self.lock:
            total_files = len(self.queue)
            total_size = sum(size for _, _, size in self.queue)
            if total_files > 0:
                soonest_expiry = min(expiry for _, expiry, _ in self.queue)
                latest_expiry = max(expiry for _, expiry, _ in self.queue)
                next_expiry_in = soonest_expiry - time.time()
            else:
                soonest_expiry = latest_expiry = next_expiry_in = 0
                
        return {
            "total_files": total_files,
            "total_size": total_size,
            "max_size": self.max_total_size,
            "size_utilization": total_size / self.max_total_size if self.max_total_size > 0 else 0,
            "soonest_expiry": soonest_expiry,
            "latest_expiry": latest_expiry,
            "next_expiry_in_seconds": max(0, next_expiry_in)
        }
    
    def cleanup_all_expired(self):
        """
        立即清理所有已过期的文件
        """
        current_time = time.time()
        expired_files = []
        
        with self.lock:
            # 找出所有已过期的文件
            expired_items = [(fp, et, sz) for fp, et, sz in self.queue if et <= current_time]
            expired_files = [fp for fp, _, _ in expired_items]
            
            # 从队列中移除已过期的项目
            for item in expired_items:
                self.queue.remove(item)
        
        # 删除过期文件
        for file_path in expired_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed expired cache file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove cache file {file_path}: {e}")
        
        return len(expired_files)

# 全局实例
cache_queue_manager = None

def initialize_cache_queue(max_total_size: int = None):
    """
    初始化全局缓存队列管理器
    
    Args:
        max_total_size: 缓存最大总大小（字节）
    """
    global cache_queue_manager
    if cache_queue_manager is None:
        cache_queue_manager = CacheQueueManager(max_total_size=max_total_size)
        cache_queue_manager.start_cleanup_loop()
    return cache_queue_manager

def get_cache_queue_manager():
    """
    获取全局缓存队列管理器实例
    
    Returns:
        CacheQueueManager实例
    """
    global cache_queue_manager
    if cache_queue_manager is None:
        cache_queue_manager = initialize_cache_queue()
    return cache_queue_manager