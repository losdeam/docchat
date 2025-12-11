from .add_doc import add_doc_page
from .kb_manage import kb_manage_page
from .test_kb  import test_kb_page
import gradio as gr

def kb_page(demo=None):
    
    with gr.TabItem("📚 知识库管理"):
        with gr.Tabs():
            kb_manage_page(demo)
            add_doc_page(demo)
            test_kb_page(demo)