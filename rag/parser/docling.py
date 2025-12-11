from pathlib import Path
from typing import List, Any
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from config.settings import settings
from utils.logging import logger
from .base import Baseparser


class Doclingparser(Baseparser):
    def __init__(self):
        # 增加更多层级的标题分块以提高召回率
        self.headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.cache_dir = Path(settings.CACHE_DIR_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _process_file(self, file_path: str) -> List[Any]:
        """Original processing logic with Docling"""
        if not file_path.endswith(('.pdf', '.docx', '.txt', '.md')):  # 检测格式是否支持
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []

        # 为Docling配置OCR选项，添加中文支持
        pipeline_options = PdfPipelineOptions()
        ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])  # 支持简体中文和英文
        pipeline_options.ocr_options = ocr_options
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })  # 加载Docling的文本加载器
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        # 检查是否有内容
        if not markdown or not markdown.strip():
            logger.warning(f"Document {file_path} has no content after processing")
            # 即使没有内容也返回一个空的文档而不是空列表
            return [Document(page_content="")] 
        
        splitter = MarkdownHeaderTextSplitter(self.headers)  # 将Docling识别完毕保存为markdown格式的文件使用langchain重新读取
        chunks = splitter.split_text(markdown)
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            # 如果 chunk 不是 Document，转换为 Document
            if not isinstance(chunk, Document):
                chunk = Document(
                    page_content=str(chunk),
                    metadata={"sort_id": idx}
                )
            else:
                # 确保 metadata 存在
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                # 添加 sort_id，保留原有元数据
                chunk.metadata["sort_id"] = idx
                    
            processed_chunks.append(chunk)

        # 使用 processed_chunks 继续处理
        chunks = processed_chunks
        # 如果没有生成块，创建一个包含整个内容的文档
        if not chunks:
            logger.warning(f"No chunks created for document {file_path}, creating single chunk")
            chunks = [Document(page_content=markdown[:2000] if markdown else "")]  # 限制长度避免embedding问题
        
        return chunks




