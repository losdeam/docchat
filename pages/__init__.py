from .main import main_page
from .setting import setting_page
from .kb import kb_page
import gradio as gr

def page_init():

    with gr.Blocks(title="DocChat 🐥") as demo:
        gr.Markdown("# DocChat: powered by Docling 🐥 and LangGraph")
        
        with gr.Tabs():
            main_page(demo)
            kb_page(demo)
            setting_page(demo)
    demo.launch(server_name="127.0.0.1", server_port=5050, share=False)
            