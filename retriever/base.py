from abc import ABC, abstractmethod
from typing import Any, List
import logging,os,json
from config.settings import settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pathlib import Path
import hashlib
from document_processor import DoclingProcessor
logger = logging.getLogger(__name__)
# 使用pydantic构建一个检索器构建器的config模型
class BaseKBConfig(BaseModel):
    """
    定义检索器的配置模型
    """
    name: str = Field(...,description="知识库名称")
    description: str = Field(
        default="",
        description="知识库描述",
    )
    KB_TYPE: str = Field(default="chroma",description="知识库类型,如Chroma,Weaviate等")
    EMBEDDING_MODEL_SERVER: str = Field(...,description="使用的嵌入层服务商")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3",description="使用的嵌入层模型名称")
    PROCESSOR: str = Field(default="Docling",description="使用的文档处理器名称")



class BaseRetriever(ABC):
    """
    检索器的抽象基类
    定义检索器的标准接口
    """
    def __init__(self):
        pass
    @abstractmethod
    def invoke(self, query: str) -> List[Document]:
        """
        同步获取相关文档的抽象方法
        
        Args:
            query: 查询字符串
            
        Returns:
            相关文档列表
        """
        pass

class BASE_KB(ABC):
    """
    知识库的抽象基类
    定义构建检索器的标准接口
    """
    def __init__(self,config:BaseKBConfig):
        """初始化检索器构建器"""
        self.init_status = True # 初始化状态，默认为成功
        self.status_msg = ""
        if not config.name:
            raise SyntaxError("知识库名称不能为空")
        if not  config.EMBEDDING_MODEL_SERVER:
            raise SyntaxError("必须要添加嵌入模型才能使用知识库")
        self.post_processors = [] # 用于保存后处理器列表
        self.name = config.name
        # 加载嵌入模型
        if config.EMBEDDING_MODEL_SERVER == "siliconflow":
            embedding = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_KEY"),
            )
        else:
            self.status_msg =f"尚不支持嵌入层服务商: {config.EMBEDDING_MODEL_SERVER}"
            self.init_status = False
            return 
        self.embeddings = embedding
        if config.KB_TYPE == "chroma":
            # 获取本地缓存地址
            self.cache_dir = Path(settings.CHROMA_DB_PATH) / config.name
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.status_msg =f"尚不支持知识库类型: {config.KB_TYPE}"
            self.init_status = False
            return
        # 获取文件解析器
        self.parser = DoclingProcessor()
    @abstractmethod
    def build_retriever(self):
        """
        构建检索器的抽象方法
        
        Args:
            docs: 文档列表
            
        Returns:
            构建的检索器对象
        """
        pass
    def delete_docs(self,doc_name:str):
        """
        删除所有文档的抽象方法
        """
    def add_doc(self, doc: str):
        """
        向检索器中添加单个文档的抽象方法
        
        Args:
            doc: 单个文档对象
        """
        pass
    def update_post_processors(self, processors: List[Any]):
        """
        更新后处理器列表
        
        Args:
            processors: 后处理器列表
        """
        self.post_processors = processors
    def parse_doc(self, docs: List[Any]) -> List[Any]:
        """
        解析文档并应用后处理器
        
        Args:
            docs: 文档列表
            
        Returns:
            处理后的文档列表
        """
        # 应用所有后处理器
        for processor in self.post_processors:
            try:
                docs = processor(docs)
                logger.debug(f"Applied post-processor, now have {len(docs)} documents")
            except Exception as e:
                logger.error(f"Error applying post-processor: {e}")
                
        return docs
    
    # @abstractmethod
    # def save_local():
    #     """
    #     保存本地知识库
    #     """
    #     pass 