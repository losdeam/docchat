import gradio as gr
import hashlib
from typing import List, Dict
import os
from datetime import datetime

from document_processor import DoclingProcessor
from retriever import RetrieverBuilder
from retriever.post_processor import deduplicate_documents, limit_documents
from agents.workflow import AgentWorkflow
from config import constants
from config.settings import settings
from utils.logging import logger, set_log_level
from utils.cache_queue import initialize_cache_queue
from langchain_community.vectorstores import Chroma

# 1) Define some example data 
#    (i.e., question + paths to documents relevant to that question).
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]  
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}

# 存储后处理配置的全局变量
post_processing_config = {
    "enable_deduplication": True,
    "max_results": settings.VECTOR_SEARCH_K
}

def parse_weights(weight_str):
    """解析权重字符串为列表"""
    try:
        # 移除空格并解析列表
        weight_str = weight_str.replace(" ", "")
        if weight_str.startswith('[') and weight_str.endswith(']'):
            weights = [float(x) for x in weight_str[1:-1].split(',')]
            if len(weights) == 2:
                return weights
    except:
        pass
    return [0.5, 0.5]  # 默认权重

def get_current_settings():
    """获取当前配置设置"""
    return {
        "LOG_LEVEL": settings.LOG_LEVEL,
        "VECTOR_SEARCH_K": settings.VECTOR_SEARCH_K,
        "HYBRID_RETRIEVER_WEIGHTS": settings.HYBRID_RETRIEVER_WEIGHTS,
        "CACHE_EXPIRE_DAYS": settings.CACHE_EXPIRE_DAYS,
        "CHROMA_DB_PATH": settings.CHROMA_DB_DEFAULT_PATH,
        "CHROMA_COLLECTION_NAME": settings.CHROMA_DEFAULT_COLLECTION_NAME
    }

def update_settings(log_level, vector_search_k, hybrid_weights, cache_expire_days, 
                   chroma_db_path, chroma_collection_name, enable_deduplication, max_results):
    """更新配置设置"""
    try:
        # 解析混合检索权重
        weights = parse_weights(hybrid_weights)
        
        # 更新日志级别
        settings.LOG_LEVEL = log_level
        set_log_level(log_level)
        
        # 更新检索设置
        settings.VECTOR_SEARCH_K = int(vector_search_k)
        settings.HYBRID_RETRIEVER_WEIGHTS = weights
        settings.CACHE_EXPIRE_DAYS = int(cache_expire_days)
        
        # 更新后处理设置
        post_processing_config["enable_deduplication"] = enable_deduplication
        post_processing_config["max_results"] = int(max_results)
        
        # 更新数据库设置
        settings.CHROMA_DB_DEFAULT_PATH = chroma_db_path
        settings.CHROMA_DEFAULT_COLLECTION_NAME = chroma_collection_name
        
        # 保存到环境变量，以便其他进程可以访问
        os.environ["LOG_LEVEL"] = log_level
        os.environ["VECTOR_SEARCH_K"] = str(vector_search_k)
        os.environ["HYBRID_RETRIEVER_WEIGHTS"] = str(weights)
        os.environ["CACHE_EXPIRE_DAYS"] = str(cache_expire_days)
        os.environ["CHROMA_DB_PATH"] = chroma_db_path
        os.environ["CHROMA_COLLECTION_NAME"] = chroma_collection_name
        
        return "✅ 设置已成功更新！请注意：某些设置可能需要重启应用才能完全生效。", get_current_settings()
    except Exception as e:
        return f"❌ 更新设置时出错: {str(e)}", get_current_settings()

def import_documents_to_kb(files: List) -> str:
    """导入文档到知识库"""
    try:
        if not files:
            return "❌ 没有选择文件"
            
        # importer = KnowledgeBaseImporter()
        result = ""
        return f"✅ {result}"
    except Exception as e:
        logger.error(f"导入文档时出错: {str(e)}")
        return f"❌ 导入文档时出错: {str(e)}"

def list_knowledge_base_contents():
    """列出知识库中的文档"""
    # TODO: 这是一个占位符函数，具体实现将在后续开发中完成
    return "📚 知识库内容列表功能正在开发中..."

def clear_knowledge_base():
    """清空知识库"""
    # TODO: 这是一个占位符函数，具体实现将在后续开发中完成
    return "🗑️ 清空知识库功能正在开发中..."

def create_knowledge_base_config(name, description, embedding_model):
    """创建新的知识库配置"""
    # TODO: 这是一个占位符函数，具体实现将在后续开发中完成
    if not name.strip():
        return "❌ 知识库名称不能为空"
    
    return f"✅ 知识库配置创建功能正在开发中...\n\n配置详情:\n- 名称: {name}\n- 描述: {description}\n- 嵌入模型: {embedding_model}"

def main():
    # 初始化缓存队列管理器
    cache_queue_manager = initialize_cache_queue()
    
    # 创建文档处理器
    processor = DoclingProcessor()
    workflow = AgentWorkflow()

    # Define custom CSS for styling
    css = """
    .title {
        font-size: 1.5em !important; 
        text-align: center !important;
        color: #FFD700; 
    }

    .subtitle {
        font-size: 1em !important; 
        text-align: center !important;
        color: #FFD700; 
    }

    .text {
        text-align: center;
    }
    
    .tabs {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
    }
    
    #config-button, #back-button {
        margin-bottom: 15px;
        align-self: flex-end;
    }
    
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';

        var text = 'Welcome to DocChat 🐥!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];

                    container.appendChild(letter);

                    setTimeout(function() {
                        letter.style.opacity = '0.9';
                    }, 50);
                }, i * 250);
            })(i);
        }

        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);

        return 'Animation created';
    }
    """

    with gr.Blocks( title="DocChat 🐥") as demo:
        # 注入 CSS
        gr.HTML(f"<style>{css}</style>")

        # 注入 JS
        gr.HTML(f"<script>{js}</script>")
        with gr.Tabs():
            with gr.TabItem("🏠 主界面"):
                gr.Markdown("## DocChat: powered by Docling 🐥 and LangGraph", elem_classes="subtitle")
                gr.Markdown("# How it works ✨:", elem_classes="title")
                gr.Markdown("📤 Upload your document(s), enter your query then press Submit 📝", elem_classes="text")
                gr.Markdown("Or you can select one of the examples from the drop-down menu, select Load Example then press Submit 📝", elem_classes="text")
                gr.Markdown("⚠️ **Note:** DocChat only accepts documents in these formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")

                # 2) Maintain the session state for retrieving doc changes
                session_state = gr.State({
                    "file_hashes": frozenset(),
                    "retriever": None
                })

                # 3) Layout 
                with gr.Row():
                    with gr.Column():
                        # Section for Examples
                        gr.Markdown("### Example 📂")
                        example_dropdown = gr.Dropdown(
                            label="Select an Example 🐥",
                            choices=list(EXAMPLES.keys()),
                            value=None,  # initially unselected
                        )
                        load_example_btn = gr.Button("Load Example 🛠️")

                        # Standard input components
                        files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                        question = gr.Textbox(label="❓ Question", lines=3)
                        
                        # Log level control
                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            log_level = gr.Radio(
                                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                                value=settings.LOG_LEVEL,
                                label="Log Level"
                            )
                            def change_log_level(level):
                                set_log_level(level)
                                # 同步更新到环境变量
                                os.environ["LOG_LEVEL"] = level
                                return f"Log level changed to {level}"
                            
                            log_level.change(
                                fn=change_log_level,
                                inputs=log_level,
                                outputs=gr.Textbox(label="Status", interactive=False)
                            )

                        submit_btn = gr.Button("Submit 🚀")
                        
                    with gr.Column():
                        answer_output = gr.Textbox(label="🐥 Answer", interactive=False)
                        verification_output = gr.Textbox(label="✅ Verification Report")

                # 4) Helper function to load example into the UI
                def load_example(example_key: str):
                    """
                    Given a key like 'Example 1', 
                    read the relevant docs from disk and return
                    them as file-like objects, plus the example question.
                    """
                    if not example_key or example_key not in EXAMPLES:
                        return [], ""  # blank if not found

                    ex_data = EXAMPLES[example_key]
                    question = ex_data["question"]
                    file_paths = ex_data["file_paths"]

                    # Prepare the file list to return. We read them from disk to
                    # give Gradio something it can handle as "uploaded" files.
                    loaded_files = []
                    for path in file_paths:
                        if os.path.exists(path):
                            # Gradio can accept a path directly, or a file-like object
                            loaded_files.append(path)
                        else:
                            logger.warning(f"File not found: {path}")

                    # The function can return lists matching the outputs we define below
                    return loaded_files, question

                load_example_btn.click(
                    fn=load_example,
                    inputs=[example_dropdown],
                    outputs=[files, question]
                )

                # 5) Standard flow for question submission
                def process_question(question_text: str, uploaded_files: List, state: Dict):
                    """Handle questions with document caching."""
                    
                    try:
                        if not question_text.strip():
                            raise ValueError("❌ Question cannot be empty")
                        if not uploaded_files:
                            raise ValueError("❌ No documents uploaded")

                        current_hashes = _get_file_hashes(uploaded_files)
                        
                        if state["retriever"] is None or current_hashes != state["file_hashes"]:
                            logger.info("Processing new/changed documents...")
                            chunks = processor.process(uploaded_files)
                            # 重新创建检索器构建器以应用最新的后处理配置
                            local_retriever_builder = RetrieverBuilder()
                            # 使用新的build_retriever方法
                            retriever = local_retriever_builder.build_retriever(chunks)
                            
                            state.update({
                                "file_hashes": current_hashes,
                                "retriever": retriever
                            })
                        
                        result = workflow.full_pipeline(
                            question=question_text,
                            retriever=state["retriever"]
                        )
                        
                        return result["draft_answer"], result["verification_report"], state
                    
                    except Exception as e:
                        logger.error(f"Processing error: {str(e)}")
                        return f"❌ Error: {str(e)}", "", state

                submit_btn.click(
                    fn=process_question,
                    inputs=[question, files, session_state],
                    outputs=[answer_output, verification_output, session_state]
                )
            
            with gr.TabItem("📥 导入知识库"):
                gr.Markdown("# 🗃️ 导入文档到知识库")
                gr.Markdown("将文档导入到持久化知识库中，以便后续查询使用")
                
                with gr.Row():
                    with gr.Column():
                        kb_files = gr.Files(label="📄 选择要导入的文档", file_types=constants.ALLOWED_TYPES)
                        import_btn = gr.Button("📥 导入到知识库", variant="primary")
                        import_output = gr.Textbox(label="导入结果", interactive=False)
                        
                        import_btn.click(
                            fn=import_documents_to_kb,
                            inputs=[kb_files],
                            outputs=[import_output]
                        )
                        
                        gr.Markdown("## 📖 使用说明")
                        gr.Markdown("""
                        1. 选择要导入的文档文件
                        2. 点击"导入到知识库"按钮
                        3. 等待导入完成，查看导入结果
                        4. 导入的文档将保存在向量数据库中，供后续查询使用
                        
                        **注意**: 导入的文档将被处理并存储在配置的Chroma数据库路径中。
                        """)
            
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
                
                def query_knowledge_base(question_text: str, state: Dict):
                    """直接从知识库查询"""
                    try:
                        if not question_text.strip():
                            raise ValueError("❌ Question cannot be empty")
                            
                        # 如果还没有加载检索器，则从知识库加载
                        if state["retriever"] is None:
                            logger.info("Loading retriever from knowledge base...")
                            state["retriever"] = None
                        
                        # 使用已有的检索器处理问题
                        result = workflow.full_pipeline(
                            question=question_text,
                            retriever=state["retriever"]
                        )
                        
                        return result["draft_answer"], result["verification_report"], state
                        
                    except Exception as e:
                        logger.error(f"Query error: {str(e)}")
                        return f"❌ Error: {str(e)}", "", state
                
                query_btn.click(
                    fn=query_knowledge_base,
                    inputs=[kb_question, kb_session_state],
                    outputs=[kb_answer_output, kb_verification_output, kb_session_state]
                )
            
            with gr.TabItem("📚 知识库管理"):
                gr.Markdown("# 📚 知识库管理")
                gr.Markdown("查看和管理知识库中的文档数据")
                
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("🔄 刷新知识库内容", variant="secondary")
                        clear_btn = gr.Button("🗑️ 清空知识库", variant="stop")
                        kb_status_output = gr.Textbox(label="知识库状态", interactive=False, lines=10)
                    
                    with gr.Column():
                        gr.Markdown("## 🆕 新建知识库配置")
                        new_kb_name = gr.Textbox(label="知识库名称")
                        new_kb_description = gr.Textbox(label="知识库描述", lines=3)
                        new_kb_embedding_model = gr.Dropdown(
                            label="嵌入模型",
                            choices=["BAAI/bge-large-zh-v1.5", "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"],
                            value="BAAI/bge-large-zh-v1.5"
                        )
                        create_kb_btn = gr.Button("➕ 创建知识库配置", variant="primary")
                        kb_config_output = gr.Textbox(label="配置结果", interactive=False)
                
                def create_knowledge_base_config(name, description, embedding_model):
                    """创建新的知识库配置"""
                    if not name.strip():
                        return "❌ 知识库名称不能为空"
                    
                    # 在实际应用中，这里会保存配置到数据库或文件
                    # 目前我们只是模拟这个过程
                    config_info = f"""
知识库配置已创建:
📌 名称: {name}
📝 描述: {description}
🤖 嵌入模型: {embedding_model}
🕒 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    return f"✅ {config_info.strip()}"
                
                def clear_knowledge_base():
                    """清空知识库"""
                    try:
                        chroma_path = settings.CHROMA_DB_PATH
                        if os.path.exists(chroma_path):
                            # 删除Chroma数据库目录
                            import shutil
                            shutil.rmtree(chroma_path)
                            return "✅ 知识库已清空"
                        else:
                            return "ℹ️ 知识库已经为空"
                    except Exception as e:
                        logger.error(f"清空知识库时出错: {str(e)}")
                        return f"❌ 清空知识库时出错: {str(e)}"
                
                # 设置按钮点击事件
                refresh_btn.click(
                    fn=list_knowledge_base_contents,
                    inputs=[],
                    outputs=[kb_status_output]
                )
                
                clear_btn.click(
                    fn=clear_knowledge_base,
                    inputs=[],
                    outputs=[kb_status_output]
                )
                
                create_kb_btn.click(
                    fn=create_knowledge_base_config,
                    inputs=[new_kb_name, new_kb_description, new_kb_embedding_model],
                    outputs=[kb_config_output]
                )
                
                # 页面加载时自动显示知识库内容
                demo.load(
                    fn=list_knowledge_base_contents,
                    inputs=[],
                    outputs=[kb_status_output]
                )
            
            with gr.TabItem("⚙️ 配置管理"):
                gr.Markdown("# 🛠️ DocChat 配置管理")
                gr.Markdown("调整应用的各项配置参数。请注意，某些设置可能需要重启应用才能完全生效。")
                
                with gr.Row():
                    with gr.Column():
                        # 日志设置
                        gr.Markdown("## 📝 日志设置")
                        config_log_level = gr.Radio(
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            value=settings.LOG_LEVEL,
                            label="日志级别"
                        )
                        
                        # 检索设置
                        gr.Markdown("## 🔍 检索设置")
                        config_vector_search_k = gr.Number(
                            value=settings.VECTOR_SEARCH_K,
                            label="向量检索返回结果数量 (VECTOR_SEARCH_K)",
                            precision=0
                        )
                        
                        config_hybrid_weights = gr.Textbox(
                            value=str(settings.HYBRID_RETRIEVER_WEIGHTS),
                            label="混合检索权重 [BM25, Vector] (如: [0.5, 0.5])",
                            placeholder="请输入权重列表，例如: [0.4, 0.6]"
                        )
                        
                        # 后处理设置
                        gr.Markdown("## 🔄 后处理设置")
                        config_enable_deduplication = gr.Checkbox(
                            value=post_processing_config["enable_deduplication"],
                            label="启用文档去重"
                        )
                        
                        config_max_results = gr.Number(
                            value=post_processing_config["max_results"],
                            label="最大返回结果数",
                            precision=0
                        )
                        
                        # 缓存设置
                        gr.Markdown("## 💾 缓存与存储设置")
                        config_cache_expire_days = gr.Number(
                            value=settings.CACHE_EXPIRE_DAYS,
                            label="缓存过期天数 (CACHE_EXPIRE_DAYS)",
                            precision=0
                        )
                        
                        config_chroma_db_path = gr.Textbox(
                            value=settings.CHROMA_DB_PATH,
                            label="Chroma 数据库路径 (CHROMA_DB_PATH)"
                        )
                        
                        config_chroma_collection_name = gr.Textbox(
                            value=settings.CHROMA_COLLECTION_NAME,
                            label="Chroma 集合名称 (CHROMA_COLLECTION_NAME)"
                        )
                        
                        config_update_btn = gr.Button("🔄 更新设置", variant="primary")
                        gr.Markdown("*注意：部分设置如数据库路径等需要重启应用才能完全生效*")
                        
                    with gr.Column():
                        # 显示当前设置
                        gr.Markdown("## ⚙️ 当前设置")
                        current_settings_display = gr.JSON(
                            value=get_current_settings(),
                            label="当前配置值"
                        )
                        
                        # 更新状态
                        config_status = gr.Textbox(label="状态", interactive=False)
                
                # 设置更新按钮的点击事件
                config_update_btn.click(
                    fn=update_settings,
                    inputs=[
                        config_log_level, 
                        config_vector_search_k, 
                        config_hybrid_weights, 
                        config_cache_expire_days, 
                        config_chroma_db_path, 
                        config_chroma_collection_name,
                        config_enable_deduplication,
                        config_max_results
                    ],
                    outputs=[config_status, current_settings_display]
                )
                
                # 页面加载时显示当前设置
                demo.load(
                    fn=get_current_settings,
                    inputs=[],
                    outputs=[current_settings_display]
                )

    demo.launch(server_name="127.0.0.1", server_port=5000, share=False)

def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        with open(file.name, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)

if __name__ == "__main__":
    main()