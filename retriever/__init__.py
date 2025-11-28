# 检索器模块初始化文件


from config.settings import settings
if settings.RETRIEVER == "Chroma":
    from .chroma import RetrieverBuilder as RetrieverBuilder
    from .chroma import Chroma_Retriever as Retriever
else:
    raise ValueError(f"Unsupported RETRIEVER type: {settings.RETRIEVER}")