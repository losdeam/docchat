from agents.workflow import AgentWorkflow
from config.settings import settings
import gradio as gr
# 会话状态
session_states = {}
# 初始化处理器和工作流
workflow = AgentWorkflow()


# 存储后处理配置的全局变量
post_processing_config = {
    "enable_deduplication": True,
    "max_results": settings.VECTOR_SEARCH_K
}

def state_init():
    """
    初始化用户会话状态
    """
    global kb_state
    global log_state
    kb_state=gr.State({
            "VECTOR_SEARCH_K":  settings.VECTOR_SEARCH_K,
            "HYBRID_RETRIEVER_WEIGHTS":  settings.HYBRID_RETRIEVER_WEIGHTS,
            "CACHE_EXPIRE_DAYS":  settings.CACHE_EXPIRE_DAYS,
            "CHROMA_COLLECTION_NAME": settings.CHROMA_DEFAULT_COLLECTION_NAME
        })
    log_state = gr.State({            
        "LOG_LEVEL":  settings.LOG_LEVEL})
    

state_init()