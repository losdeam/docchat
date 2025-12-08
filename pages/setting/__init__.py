import gradio as gr 
from config.settings import settings
from pages.utils import post_processing_config
from utils.logging import logger, set_log_level
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
def update_settings(log_level, vector_search_k, hybrid_weights, cache_expire_days,
                    chroma_collection_name, enable_deduplication, max_results,
                   user_session):
    """更新配置设置"""
    try:
        weights = parse_weights(hybrid_weights)
        
        # 创建全新的配置字典，而不是修改原字典
        new_session = {
            "settings": {
                "LOG_LEVEL": log_level,
                "VECTOR_SEARCH_K": int(vector_search_k),
                "HYBRID_RETRIEVER_WEIGHTS": weights,
                "CACHE_EXPIRE_DAYS": int(cache_expire_days),
                "CHROMA_COLLECTION_NAME": chroma_collection_name
            },
            "post_processing_config": {
                "enable_deduplication": enable_deduplication,
                "max_results": int(max_results)
            }
        }
        
        set_log_level(log_level)
        
        # 返回全新的会话对象
        return "✅ 设置已成功更新！", get_current_settings(new_session["settings"]), new_session
    except Exception as e:
        # 出错时返回原始状态，确保状态不被破坏
        return f"❌ 更新设置时出错: {str(e)}", get_current_settings(), user_session

def get_current_settings(user_settings=None):
    """获取当前配置设置"""
    if user_settings:
        return {
            "LOG_LEVEL": user_settings.get("LOG_LEVEL", settings.LOG_LEVEL),
            "VECTOR_SEARCH_K": user_settings.get("VECTOR_SEARCH_K", settings.VECTOR_SEARCH_K),
            "HYBRID_RETRIEVER_WEIGHTS": user_settings.get("HYBRID_RETRIEVER_WEIGHTS", settings.HYBRID_RETRIEVER_WEIGHTS),
            "CACHE_EXPIRE_DAYS": user_settings.get("CACHE_EXPIRE_DAYS", settings.CACHE_EXPIRE_DAYS),
            "CHROMA_COLLECTION_NAME": user_settings.get("CHROMA_COLLECTION_NAME", settings.CHROMA_DEFAULT_COLLECTION_NAME)
        }
    return {
        "LOG_LEVEL": settings.LOG_LEVEL,
        "VECTOR_SEARCH_K": settings.VECTOR_SEARCH_K,
        "HYBRID_RETRIEVER_WEIGHTS": settings.HYBRID_RETRIEVER_WEIGHTS,
        "CACHE_EXPIRE_DAYS": settings.CACHE_EXPIRE_DAYS,
        "CHROMA_COLLECTION_NAME": settings.CHROMA_DEFAULT_COLLECTION_NAME
    }
def setting_page(demo=None):
    with gr.TabItem("⚙️ 配置管理"):
        gr.Markdown("# 🛠️ DocChat 配置管理")
        gr.Markdown("调整应用的各项配置参数。请注意，某些设置可能需要重启应用才能完全生效。")
        
        # 用户配置状态
        config_user_state = gr.State({
            "settings": {},
            "post_processing_config": {
                "enable_deduplication": True,
                "max_results": settings.VECTOR_SEARCH_K
            }
        })
        
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
                config_chroma_collection_name,
                config_enable_deduplication,
                config_max_results,
                config_user_state
            ],
            outputs=[config_status, current_settings_display, config_user_state]
        )
                    
        # 页面加载时显示当前设置
        demo.load(
            fn=get_current_settings,
            inputs=[],
            outputs=[current_settings_display] )