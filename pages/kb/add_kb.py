import gradio as gr 
from pages.utils import *
from utils.logging import logger
from typing import List, Dict, Any, Tuple
from datetime import datetime
from rag.retriever.base import kb_manager
import os
import json
from config.settings import settings

def create_new_knowledge_base(name: str, description: str, embedding_model: str, kb_type: str) -> str:
    """创建新的知识库"""
    try:
        # 检查输入
        if not name.strip():
            return "❌ 知识库名称不能为空"
        
        # 检查知识库是否已存在
        if name in kb_manager.kb_dict:
            return f"❌ 知识库 '{name}' 已存在"
        
        # 创建知识库目录
        kb_path = os.path.join(settings.KB_PATH, name)
        os.makedirs(kb_path, exist_ok=True)
        
        # 创建配置文件
        config = {
            "name": name,
            "description": description,
            "KB_TYPE": kb_type,
            "EMBEDDING_MODEL_SERVER": "siliconflow",  # 默认使用siliconflow
            "EMBEDDING_MODEL": embedding_model,
            "PROCESSOR": "Docling",
            "FILE_LIST": {},
            "HYBRID_RETRIEVER_WEIGHTS": [0.5, 0.5]
        }
        
        config_path = os.path.join(kb_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        # 重新加载知识库
        kb_manager.kb_load_local()
        
        return f"✅ 知识库 '{name}' 创建成功!\n\n配置详情:\n- 名称: {name}\n- 描述: {description}\n- 类型: {kb_type}\n- 嵌入模型: {embedding_model}"
        
    except Exception as e:
        logger.error(f"创建知识库时出错: {str(e)}")
        return f"❌ 创建知识库时出错: {str(e)}"

def refresh_kb_list_for_creation():
    """刷新知识库列表"""
    kb_manager.kb_load_local()
    kb_list = kb_manager.list_kb()
    return gr.update(choices=kb_list)

def refresh_kb_list_no_update():
    """刷新知识库列表，不返回任何更新"""
    kb_manager.kb_load_local()

def add_kb_page(demo=None):
    with gr.TabItem("➕ 添加知识库"):
        gr.Markdown("# ➕ 添加新知识库")
        gr.Markdown("创建一个新的持久化知识库用于存储和检索文档")
        
        with gr.Row():
            with gr.Column(scale=1):
                kb_name = gr.Textbox(
                    label="📘 知识库名称",
                    placeholder="输入知识库名称，如：tech_docs"
                )
                kb_description = gr.Textbox(
                    label="📝 知识库描述",
                    placeholder="简要描述该知识库的用途...",
                    lines=3
                )
                kb_type = gr.Dropdown(
                    label="🗃️ 知识库类型",
                    choices=["chroma"],
                    value="chroma",
                    interactive=False  # 暂时只支持chroma
                )
                kb_embedding_model = gr.Dropdown(
                    label="🤖 嵌入模型",
                    choices=[
                        "BAAI/bge-large-zh-v1.5", 
                        "BAAI/bge-m3", 
                        "sentence-transformers/all-MiniLM-L6-v2"
                    ],
                    value="BAAI/bge-m3"
                )
                create_btn = gr.Button("✨ 创建知识库", variant="primary")
                refresh_btn = gr.Button("🔄 刷新知识库列表")
                
            with gr.Column(scale=1):
                creation_result = gr.Textbox(
                    label="📋 创建结果",
                    interactive=False,
                    lines=10
                )
        
        # 事件处理
        create_btn.click(
            fn=create_new_knowledge_base,
            inputs=[kb_name, kb_description, kb_embedding_model, kb_type],
            outputs=[creation_result]
        )
        
        refresh_btn.click(
            fn=refresh_kb_list_no_update,
            inputs=[],
            outputs=[]
        )
        
        # 页面加载时刷新一次知识库列表
        demo.load(
            fn=refresh_kb_list_no_update,
            inputs=[],
            outputs=[]
        )