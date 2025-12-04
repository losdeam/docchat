import gradio as gr
from utils.logging import logger
from typing import List
from config import constants
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

def add_doc_page(demo=None):
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
    