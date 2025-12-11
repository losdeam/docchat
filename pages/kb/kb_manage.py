import gradio as gr 
from pages.utils import *
from utils import logger, get_available_knowledge_bases
from utils.file_manage import file_manager_activate
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import json
from rag.retriever.base import kb_manager

def list_knowledge_base_contents(selected_kb=None):
    """列出知识库中的文档"""
    if not selected_kb:
        return "请先选择一个知识库"
    
    try:
        # 使用KB_manager获取知识库实例
        if selected_kb not in kb_manager.kb_dict:
            return f"知识库 '{selected_kb}' 不存在"
        
        kb_builder = kb_manager.kb_dict[selected_kb]
        docs = kb_builder.list_docs()
        if not docs:
            return f"知识库 '{selected_kb}' 中没有文档"
        
        # 格式化输出
        result = f"📚 知识库 '{selected_kb}' 中的文档:\n\n"
        for doc_name in docs:
            result += f"📄 {doc_name}\n"
        
        result += f"\n总计: {len(docs)} 个文档"
        return result
        
    except Exception as e:
        logger.error(f"查询知识库内容时出错: {str(e)}")
        return f"❌ 查询知识库内容时出错: {str(e)}"

def show_document_details(selected_kb, selected_doc):
    """显示文档详情"""
    if not selected_kb or not selected_doc:
        return "请先选择知识库和文档"
    
    try:
        # 使用KB_manager获取知识库实例
        if selected_kb not in kb_manager.kb_dict:
            return f"知识库 '{selected_kb}' 不存在"
        
        kb_builder = kb_manager.kb_dict[selected_kb]
        
        # 获取文档分块
        try:
            chunks = kb_builder.list_chunks(selected_doc)
            chunk_count = len(chunks) if isinstance(chunks, list) else "未知"
        except Exception as e:
            chunk_count = f"无法获取 (错误: {str(e)})"
        
        # 显示文档详情
        result = f"📄 文档详情:\n\n"
        result += f"名称: {selected_doc}\n"
        result += f"知识库: {selected_kb}\n"
        result += f"分块数量: {chunk_count}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"查询文档详情时出错: {str(e)}")
        return f"❌ 查询文档详情时出错: {str(e)}"
def kb_close():
    kb_manager.raise_()
def kb_manage_page(demo=None):
    demo.unload(kb_close)
    with gr.TabItem("📚 知识库管理"):
        gr.Markdown("# 📚 知识库管理")
        gr.Markdown("查看和管理知识库中的文档数据")
        
        with gr.Row():
            with gr.Column():
                kb_selector = gr.Dropdown(
                    label="📚 选择知识库",
                    choices=kb_manager.list_kb(),
                    value=kb_manager.list_kb()[0] if kb_manager.list_kb() else "default",
                    scale=4
                )


        
        # 创建一个交互式文档列表来显示文档详情
        with gr.Row():
            kb_status_output = gr.Textbox(label="知识库状态", interactive=False, lines=10)
                
            # 文档详情部分
            with gr.Group():
                gr.Markdown("### 📄 文档详情")
                doc_selector = gr.Dropdown(label="选择文档", choices=[], interactive=True)
                doc_detail_output = gr.Textbox(label="", interactive=False, lines=8)
        
        # 添加文档选择器的change事件
        def update_doc_selector(selected_kb):
            """更新文档选择器选项"""
            if not selected_kb:
                return gr.update(choices=[])
            
            try:
                if selected_kb not in kb_manager.kb_dict:
                    return gr.update(choices=[])
                
                kb_builder = kb_manager.kb_dict[selected_kb]
                docs = kb_builder.list_docs()
                return gr.update(choices=docs, value=docs[0] if docs else None)
            except Exception as e:
                logger.error(f"更新文档选择器时出错: {str(e)}")
                return gr.update(choices=[])
        
        # 修改原有函数，支持从文档选择器获取文档名
        def show_document_details_from_selector(selected_kb, selected_doc):
            """从选择器显示文档详情"""
            if not selected_kb or not selected_doc:
                return "请先选择知识库和文档"
            return show_document_details(selected_kb, selected_doc)
        
        # 设置事件监听
        kb_selector.change(
            fn=lambda kb: [
                list_knowledge_base_contents(kb),
                update_doc_selector(kb)
            ],
            inputs=[kb_selector],
            outputs=[kb_status_output, doc_selector]
        )
        
        
        doc_selector.change(
            fn=show_document_details_from_selector,
            inputs=[kb_selector, doc_selector],
            outputs=[doc_detail_output]
        )

        
        # 页面加载时自动显示知识库内容
        demo.load(
            fn=lambda: [
                gr.update(choices=kb_manager.list_kb()),
                kb_manager.list_kb()[0] if kb_manager.list_kb() else "default",
                list_knowledge_base_contents(kb_manager.list_kb()[0] if kb_manager.list_kb() else "default"),
                update_doc_selector(kb_manager.list_kb()[0] if kb_manager.list_kb() else "default")
            ],
            inputs=[],
            outputs=[
                kb_selector,
                kb_selector,
                kb_status_output,
                doc_selector
            ]
        )