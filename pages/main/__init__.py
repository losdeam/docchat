import gradio as gr 
from pages.utils import *
from typing import List, Dict, Any, Tuple
from utils import logger, get_available_knowledge_bases
import hashlib
import traceback

from config import constants
from rag.retriever.base import kb_manager

def process_message(message: str, history: List[List[str]], 
                   kb_selector: str) -> str:
    """处理用户消息的核心函数"""
    try:
        # 获取或创建会话状态
        session_id = "default"
        if session_id not in session_states:
            session_states[session_id] = {
                "file_hashes": frozenset(),
                "retriever": None
            }
        state = session_states[session_id]
        
        # 检查是否有选择的知识库
        kb = None
        if kb_selector and kb_selector in kb_manager.kb_dict:
            kb = kb_manager.kb_dict[kb_selector]
        

        
        # 如果文件发生变化，或者没有retriever且选择了知识库，重新处理
        if state["retriever"] is None or (kb is not None and state.get("current_kb") != kb_selector):
            logger.info("Processing new/changed documents or switching knowledge base...")
  
            # 如果没有上传文件但选择了知识库，则使用知识库
            if kb is not None:
                # 确保知识库已经激活
                kb.activate_beforeUse()
                # 获取知识库的检索器
                retriever = kb.build_retriever()
                state.update({
                    "file_hashes": frozenset(),
                    "retriever": retriever,
                    "current_kb": kb_selector  # 记录当前使用的知识库
                })
            else:
                return "❌ 没有可用的文档或知识库"
        
        # 使用工作流处理问题
        result = workflow.full_pipeline(
            question=message,
            retriever=state["retriever"]
        )
        
        # 格式化回答
        answer = result["draft_answer"]
        verification = result["verification_report"]
        
        response = f"{answer}\n\n---\n**验证报告**:\n{verification}"
        return response
    
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Processing error: {str(e)}")
        return f"❌ 错误: {str(e)}"

def refresh_kb_list():
    """刷新知识库列表"""
    kb_manager.kb_load_local()
    return gr.update(choices=kb_manager.list_kb())

def main_page(demo=None):
    with gr.TabItem("🏠 主界面"):
        gr.Markdown("# 🏠 DocChat 主界面")
        gr.Markdown("与您的文档进行对话。上传文档或选择已有知识库开始对话。")
        
        # 知识库选择区域
        with gr.Row():
            with gr.Column(scale=3):
                kb_selector = gr.Dropdown(
                    label="📚 选择知识库",
                    choices=kb_manager.list_kb(),
                    value=kb_manager.list_kb()[0] if kb_manager.list_kb() else "default"
                )
            with gr.Column(scale=1):
                refresh_kb_btn = gr.Button("🔄 刷新")


        # Chat Interface
        chatbot = gr.ChatInterface(
            fn=process_message,
            additional_inputs=[
                kb_selector
            ],
            examples=[
                ["请总结文档的主要内容"],
                ["文档中提到了哪些关键技术？"],
                ["文档的结论是什么？"]
            ],
            title="",
            description="",
            cache_examples=False
        )
        
        # 刷新按钮事件
        refresh_kb_btn.click(
            fn=refresh_kb_list,
            inputs=[],
            outputs=[kb_selector]
        )