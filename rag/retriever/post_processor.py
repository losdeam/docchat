from typing import List, Dict, Any, Callable, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
from pydantic import Field

logger = logging.getLogger(__name__)


class PostProcessingRetriever(BaseRetriever):
    """
    检索后处理器包装器，允许对检索结果进行后处理
    """
    
    base_retriever: BaseRetriever = Field(description="基础检索器")
    post_processors: List[Callable[[List[Document]], List[Document]]] = Field(
        default_factory=list, 
        description="后处理器函数列表，每个函数接收文档列表并返回处理后的文档列表"
    )
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档并应用后处理
        
        Args:
            query: 查询字符串
            
        Returns:
            处理后的文档列表
        """
        # 使用基础检索器获取文档
        docs = self.base_retriever.invoke(query)
        
        # 应用所有后处理器
        for processor in self.post_processors:
            try:
                docs = processor(docs)
                logger.debug(f"Applied post-processor, now have {len(docs)} documents")
            except Exception as e:
                logger.error(f"Error applying post-processor: {e}")
                
        return docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        异步获取相关文档并应用后处理
        
        Args:
            query: 查询字符串
            
        Returns:
            处理后的文档列表
        """
        # 使用基础检索器获取文档
        docs = await self.base_retriever.ainvoke(query)
        
        # 应用所有后处理器
        for processor in self.post_processors:
            try:
                docs = processor(docs)
                logger.debug(f"Applied post-processor, now have {len(docs)} documents")
            except Exception as e:
                logger.error(f"Error applying post-processor: {e}")
                
        return docs


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """
    去除重复文档
    
    Args:
        docs: 文档列表
        
    Returns:
        去重后的文档列表
    """
    seen_content = set()
    unique_docs = []
    
    for doc in docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
            
    logger.debug(f"Deduplicated documents: {len(docs)} -> {len(unique_docs)}")
    return unique_docs


def filter_by_metadata(docs: List[Document], metadata_filter: Dict[str, Any]) -> List[Document]:
    """
    根据元数据过滤文档
    
    Args:
        docs: 文档列表
        metadata_filter: 元数据过滤条件
        
    Returns:
        过滤后的文档列表
    """
    filtered_docs = []
    
    for doc in docs:
        include_doc = True
        for key, value in metadata_filter.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                include_doc = False
                break
                
        if include_doc:
            filtered_docs.append(doc)
            
    logger.debug(f"Filtered documents by metadata: {len(docs)} -> {len(filtered_docs)}")
    return filtered_docs


def sort_by_relevance_score(docs: List[Document], reverse: bool = True) -> List[Document]:
    """
    根据相关性得分排序文档
    
    Args:
        docs: 文档列表
        reverse: 是否降序排列（默认为True，相关性高的在前）
        
    Returns:
        排序后的文档列表
    """
    # 只有当文档包含相关性得分时才进行排序
    if docs and 'relevance_score' in docs[0].metadata:
        sorted_docs = sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=reverse)
        logger.debug("Sorted documents by relevance score")
        return sorted_docs
    
    logger.debug("Documents do not contain relevance scores, returning original order")
    return docs


def limit_documents(docs: List[Document], limit: int) -> List[Document]:
    """
    限制文档数量
    
    Args:
        docs: 文档列表
        limit: 最大文档数量
        
    Returns:
        截断后的文档列表
    """
    limited_docs = docs[:limit]
    logger.debug(f"Limited documents: {len(docs)} -> {len(limited_docs)}")
    return limited_docs