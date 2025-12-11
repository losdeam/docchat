import gradio as gr 
from pages.utils import *
from typing import List, Dict, Any, Tuple
from utils import logger,get_available_knowledge_bases
import hashlib
import traceback

from config import constants
def process_message(message: str, history: List[List[str]], 
                   uploaded_files: List[Any], kb_selector: str) -> str:
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
        
        # 如果没有上传文件，使用示例文件
        all_files = uploaded_files if uploaded_files else []

        
        if not all_files:
            return "❌ 请上传文档或确保示例文档存在"
        
        # 处理文件哈希
        current_hashes = frozenset([hashlib.sha256(open(f.name, "rb").read()).hexdigest() 
                                   for f in all_files])
        
        # 如果文件发生变化，重新处理
        if state["retriever"] is None or current_hashes != state["file_hashes"]:
            logger.info("Processing new/changed documents...")
            chunks = processor.process(all_files)
            
            if not chunks:
                return "❌ 文档处理后没有生成任何内容，请检查文档格式是否支持"
            
            # 创建检索器
            local_retriever_builder = Chroma_Builder()
            retriever = local_retriever_builder.build_retriever(docs=chunks)
            state.update({
                "file_hashes": current_hashes,
                "retriever": retriever
            })
        
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


def main_page(demo=None):
    with gr.TabItem("🏠 主界面"):
        # 知识库选择器
        with gr.Row():
            kb_selector = gr.Dropdown(
                label="📚 选择知识库",
                choices=get_available_knowledge_bases(),
                value="default",
                scale=4
            )
            refresh_kb_btn = gr.Button("🔄 刷新", scale=1)
        
        # 文件上传组件
        with gr.Accordion("📎 附件", open=False):
            files = gr.Files(label="上传文档", file_types=constants.ALLOWED_TYPES)

        
        # Chat Interface
        chatbot = gr.ChatInterface(
            fn=process_message,
            additional_inputs=[
                files,
                kb_selector
            ],
            examples=[
                ["请总结文档的主要内容"],
                ["文档中提到了哪些关键技术？"],
                ["文档的结论是什么？"]
            ],
            title="",
            description="与您的文档进行对话。上传文档或选择示例开始对话。",
            cache_examples=False
        )
        
        
        # Refresh knowledge base list
        refresh_kb_btn.click(
            fn=lambda: gr.update(choices=get_available_knowledge_bases()),
            inputs=[],
            outputs=[kb_selector]
        )