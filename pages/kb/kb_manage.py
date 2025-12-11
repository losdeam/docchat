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
        return []
    
    try:
        # 使用KB_manager获取知识库实例
        if selected_kb not in kb_manager.kb_dict:
            return [["错误", f"知识库 '{selected_kb}' 不存在"]]
        
        kb_builder = kb_manager.kb_dict[selected_kb]
        docs = kb_builder.list_docs()
        if not docs:
            return [["信息", f"知识库 '{selected_kb}' 中没有文档"]]
        
        # 返回文档列表用于表格展示
        doc_data = []
        for i, doc_name in enumerate(docs, 1):
            doc_data.append([i, doc_name])
        
        return doc_data
        
    except Exception as e:
        logger.error(f"查询知识库内容时出错: {str(e)}")
        return [["错误", f"查询知识库内容时出错: {str(e)}"]]

def show_document_details(selected_kb, selected_doc):
    """显示文档详情"""
    if not selected_kb or not selected_doc:
        return []
    
    try:
        # 使用KB_manager获取知识库实例
        if selected_kb not in kb_manager.kb_dict:
            return [[1, f"知识库 '{selected_kb}' 不存在"]]
        
        kb_builder = kb_manager.kb_dict[selected_kb]
        
        # 获取文档分块
        try:
            chunks = kb_builder.list_chunks(selected_doc)
        except Exception as e:
            chunks = []
        
        # 准备分块数据用于表格展示
        chunk_data = []
        if isinstance(chunks, list):
            for i, chunk in enumerate(chunks, 1):
                if hasattr(chunk, 'page_content'):
                    content = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                    chunk_data.append([i, content])
                else:
                    chunk_data.append([i, str(chunk)[:100]])
        else:
            chunk_data.append([1, "无法获取分块信息"])
        
        return chunk_data
        
    except Exception as e:
        logger.error(f"查询文档详情时出错: {str(e)}")
        return [[1, f"查询文档详情时出错: {str(e)}"]]

def refresh_kb_list():
    """刷新知识库列表"""
    kb_manager.kb_load_local()
    kb_list = kb_manager.list_kb()
    default_value = kb_list[0] if kb_list else "default"
    return [
        gr.update(choices=kb_list),
        default_value,
        list_knowledge_base_contents(default_value),
        update_doc_selector(default_value)
    ]

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

def show_document_details_from_selector(selected_kb, selected_doc):
    """从选择器显示文档详情"""
    if not selected_kb or not selected_doc:
        return []
    return show_document_details(selected_kb, selected_doc)

def kb_close():
    kb_manager.raise_()

def kb_manage_page(demo=None):
    demo.unload(kb_close)
    with gr.TabItem("📚 知识库管理"):
        gr.Markdown("# 📚 知识库管理")
        gr.Markdown("查看和管理知识库中的文档数据")
        
        with gr.Row():
            # 左侧列：知识库选择和文档列表
            with gr.Column(scale=1):
                with gr.Row():
                    kb_selector = gr.Dropdown(
                        label="📚 选择知识库",
                        choices=kb_manager.list_kb(),
                        value=kb_manager.list_kb()[0] if kb_manager.list_kb() else "default",
                        scale=4
                    )
                    refresh_kb_btn = gr.Button("🔄 刷新", scale=1)
                
                gr.Markdown("### 📚 文档列表")
                doc_table = gr.Dataframe(
                    label="",
                    headers=["#", "文档名称"],
                    datatype=["number", "str"],
                    interactive=False
                )
                
            # 右侧列：文档详情和分块列表
            with gr.Column(scale=1):
                gr.Markdown("### 📄 文档分块详情")
                doc_selector = gr.Dropdown(label="选择文档", choices=[], interactive=True)
                chunk_table = gr.Dataframe(
                    label="",
                    headers=["#", "分块内容"],
                    datatype=["number", "str"],
                    interactive=False
                )
        
        # 设置事件监听
        kb_selector.change(
            fn=lambda kb: [
                list_knowledge_base_contents(kb),
                update_doc_selector(kb)
            ],
            inputs=[kb_selector],
            outputs=[doc_table, doc_selector]
        )
        
        doc_selector.change(
            fn=show_document_details_from_selector,
            inputs=[kb_selector, doc_selector],
            outputs=[chunk_table]
        )
        
        refresh_kb_btn.click(
            fn=refresh_kb_list,
            inputs=[],
            outputs=[kb_selector, kb_selector, doc_table, doc_selector]
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
                doc_table,
                doc_selector
            ]
        )