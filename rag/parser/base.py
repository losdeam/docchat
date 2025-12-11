# 抽象出一个基类以便于后续的多解析器切换
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime, timedelta
import os
import hashlib
import pickle
from pathlib import Path
from config import constants
from config.settings import settings
from utils.logging import logger


class Baseparser(ABC):
    def __init__(self):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        
    def validate_files(self, files: List) -> None:
        """验证上传的文件的大小是否超出限制"""
        total_size = sum(os.path.getsize(f.name) for f in files) # 获取上传文件的总大小
        if total_size > constants.MAX_TOTAL_SIZE: # 如果超过设定值，返回报错
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")
    
    def process(self, files: List) -> List:
        """通用处理流程，包含缓存机制"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            try:
                # Generate content-based hash for caching
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())
                cache_path = self.cache_dir / f"{file_hash}.pkl"
                if self._is_cache_valid(cache_path): # 如果缓存存在则从缓存处加载，而不需要重新解析
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")

                    chunks = self._process_file(file.name)
                    self._save_to_cache(chunks, cache_path)

                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                        
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue
                
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks
    
    def _generate_hash(self, content: bytes) -> str:
        """生成内容的哈希值"""
        return hashlib.sha256(content).hexdigest()
    
    def _save_to_cache(self, chunks: List, cache_path: Path):
        """保存处理结果到缓存"""
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)
        # 将新创建的缓存文件添加到队列管理器中
        self.cache_queue.add_file(str(cache_path))
        
    def _load_from_cache(self, cache_path: Path) -> List:
        """从缓存加载处理结果"""
        # 当从缓存加载文件时，移除它以免被过早删除
        self.cache_queue.remove_file(str(cache_path))
        
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]
        
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False
            
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

    @abstractmethod
    def _process_file(self, file_bytes: bytes) -> List[Any]:
        """
        使用具体的解析器处理单个文件

        Args:
            file_bytes (bytes): 单个文件的字节数据
        """
        pass