import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self):
        # 增加更多层级的标题分块以提高召回率
        self.headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.cache_dir = Path(settings.CACHE_DIR) #获取缓存文件夹
        self.cache_dir.mkdir(parents=True, exist_ok=True)# parents=True 参数表示如果任何父目录不存在，则创建它们
# exist_ok=True 参数表示如果目录已经存在，不会抛出异常
        
    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files) # 获取上传文件的总大小
        if total_size > constants.MAX_TOTAL_SIZE: # 如果超过设定值，返回报错
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
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
                    chunks = self._process_file(file)
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

    def _process_file(self, file) -> List:
        """Original processing logic with Docling"""
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')): # 检测格式是否支持
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

        # 为Docling配置OCR选项，添加中文支持
        pipeline_options = PdfPipelineOptions()
        ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])  # 支持简体中文和英文
        pipeline_options.ocr_options = ocr_options
        
        converter = DocumentConverter(      format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }) # 加载Docling的文本加载器
        result = converter.convert(file.name)
        markdown = result.document.export_to_markdown()
        print("==================docling export markdown==================")
        print(markdown)
        print("==================docling export markdown==================")
        splitter = MarkdownHeaderTextSplitter(self.headers) # 将Docling识别完毕保存为markdown格式的文件使用langchain重新读取
        return splitter.split_text(markdown)

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest() # 生成哈希值

    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f) # 保存本地作为缓存

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
            
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)