from agents.workflow import AgentWorkflow
from document_processor import DoclingProcessor
from retriever import RetrieverBuilder
from config.settings import settings
# 会话状态
session_states = {}
# 初始化处理器和工作流
processor = DoclingProcessor()
workflow = AgentWorkflow()


# 存储后处理配置的全局变量
post_processing_config = {
    "enable_deduplication": True,
    "max_results": settings.VECTOR_SEARCH_K
}