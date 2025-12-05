# 验证检索效果
import gradio as gr 
from pages.utils import *
from utils.logging import logger
from typing import List, Dict, Any, Tuple
def test_kb_page(demo=None):
    with gr.TabItem("🔍 查询知识库"):
        gr.Markdown("# 🔍 从知识库查询")
        gr.Markdown("直接从已有的知识库中查询信息，无需重新上传文档")
        
        with gr.Row():
            with gr.Column():
                kb_question = gr.Textbox(label="❓ 问题", lines=3)
                query_btn = gr.Button("🔍 查询知识库", variant="primary")
                
            with gr.Column():
                kb_answer_output = gr.Textbox(label="🐥 答案", interactive=False)
                kb_verification_output = gr.Textbox(label="✅ 验证报告")
        
        # 查询知识库的状态
        kb_session_state = gr.State({
            "retriever": None
        })
        
        # 用户配置状态
        kb_user_config_state = gr.State({
            "settings": {},
            "post_processing_config": {
                "enable_deduplication": True,
                "max_results": settings.VECTOR_SEARCH_K
            }
        })
        def query_knowledge_base(question_text: str, state: Dict, user_session: Dict):
            """直接从知识库查询"""
            try:
                if not question_text.strip():
                    raise ValueError("❌ Question cannot be empty")
                    
                # 如果还没有加载检索器，则从知识库加载
                if state["retriever"] is None:
                    logger.info("Loading retriever from knowledge base...")
                    # 使用用户配置创建检索器
                    local_retriever_builder = Chroma_Builder(user_settings=user_session.get("settings"))
                    state["retriever"] = local_retriever_builder.build_retriever()
                
                # 使用已有的检索器处理问题
                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"]
                )
                
                return result["draft_answer"], result["verification_report"], state, user_session
            
            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                return f"❌ Error: {str(e)}", "", state
        
        query_btn.click(
            fn=query_knowledge_base,
            inputs=[kb_question, kb_session_state, kb_user_config_state],
            outputs=[kb_answer_output, kb_verification_output, kb_session_state, kb_user_config_state]
        )
        