import gradio as gr 
from pages.utils import *
from utils.logging import logger
from typing import List, Dict, Any, Tuple
from datetime import datetime
from rag.retriever.base import kb_manager
import os
import json
from config.settings import settings

def list_knowledge_bases():
    """列出所有知识库"""
    try:
        kb_list = kb_manager.list_kb()
        if not kb_list:
            return "当前没有任何知识库"
        
        result = "📚 知识库列表:\n\n"
        for i, kb_name in enumerate(kb_list, 1):
            result += f"{i}. {kb_name}\n"
        result += f"\n总计: {len(kb_list)} 个知识库"
        return result
    except Exception as e:
        logger.error(f"列出知识库时出错: {str(e)}")
        return f"❌ 列出知识库时出错: {str(e)}"

def show_kb_details(kb_name: str):
    """显示特定知识库的详细信息"""
    try:
        if not kb_name:
            return "请输入知识库名称"
        
        if kb_name not in kb_manager.kb_dict:
            return f"❌ 知识库 '{kb_name}' 不存在"
        
        kb = kb_manager.kb_dict[kb_name]
        docs = kb.list_docs()
        
        result = f"📘 知识库详情: {kb_name}\n\n"
        result += f"配置文件路径: {kb.config_path}\n"
        result += f"文档存储路径: {kb.docs_dir}\n"
        result += f"激活状态: {'已激活' if kb.activate_status else '未激活'}\n"
        result += f"初始化状态: {'成功' if kb.init_status else '失败'}\n\n"
        
        if kb.config:
            result += "⚙️ 配置信息:\n"
            result += f"  名称: {kb.config.name}\n"
            result += f"  描述: {kb.config.description}\n"
            result += f"  类型: {kb.config.KB_TYPE}\n"
            result += f"  嵌入模型服务商: {kb.config.EMBEDDING_MODEL_SERVER}\n"
            result += f"  嵌入模型: {kb.config.EMBEDDING_MODEL}\n"
            result += f"  文档处理器: {kb.config.PROCESSOR}\n"
            result += f"  混合检索权重: {kb.config.HYBRID_RETRIEVER_WEIGHTS}\n\n"
        
        result += f"📄 文档列表 ({len(docs)} 个):\n"
        if docs:
            for i, doc in enumerate(docs, 1):
                result += f"  {i}. {doc}\n"
        else:
            result += "  暂无文档\n"
            
        return result
    except Exception as e:
        logger.error(f"获取知识库详情时出错: {str(e)}")
        return f"❌ 获取知识库详情时出错: {str(e)}"

def refresh_knowledge_bases():
    """刷新知识库列表"""
    try:
        kb_manager.kb_load_local()
        kb_list = kb_manager.list_kb()
        return [
            gr.update(choices=kb_list),
            list_knowledge_bases()
        ]
    except Exception as e:
        logger.error(f"刷新知识库时出错: {str(e)}")
        return [gr.update(choices=[]), f"❌ 刷新知识库时出错: {str(e)}"]

def kb_manage_page(demo=None):
    with gr.TabItem("📚 知识库管理"):
        gr.Markdown("# 📚 知识库管理")
        gr.Markdown("查看和管理知识库中的文档数据")
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 刷新知识库列表", variant="secondary")
                list_output = gr.Textbox(
                    label="📚 知识库列表", 
                    interactive=False, 
                    lines=10
                )
                
                gr.Markdown("### 📘 查看详细信息")
                kb_selector = gr.Dropdown(
                    label="选择知识库",
                    choices=kb_manager.list_kb(),
                    interactive=True
                )
                show_details_btn = gr.Button("📖 显示详细信息", variant="primary")
                
            with gr.Column(scale=1):
                details_output = gr.Textbox(
                    label="📘 知识库详情", 
                    interactive=False, 
                    lines=20
                )
        
        # 事件处理
        refresh_btn.click(
            fn=refresh_knowledge_bases,
            inputs=[],
            outputs=[kb_selector, list_output]
        )
        
        show_details_btn.click(
            fn=show_kb_details,
            inputs=[kb_selector],
            outputs=[details_output]
        )
        
        kb_selector.change(
            fn=show_kb_details,
            inputs=[kb_selector],
            outputs=[details_output]
        )
        
        # 页面加载时自动刷新知识库列表
        demo.load(
            fn=lambda: refresh_knowledge_bases(),
            inputs=[],
            outputs=[kb_selector, list_output]
        )