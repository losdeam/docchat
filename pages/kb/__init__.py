from .add_doc import add_doc_page
from .kb_manage import kb_manage_page
from .add_kb  import add_kb_page
from .read_kb import read_kb_page
import gradio as gr

def kb_page(demo=None):
    
    with gr.TabItem("📚 知识库管理"):
        with gr.Tabs():
            read_kb_page(demo)
            
            add_doc_page(demo)
            add_kb_page(demo)
            kb_manage_page(demo)