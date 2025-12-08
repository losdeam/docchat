from abc import ABC, abstractmethod
from typing import Any, List
import logging,os,json
from config.settings import settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pathlib import Path
from utils import get_single_hash,file_manager_activate
from parser import Doclingparser
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
    # KB_TYPE: str = Field(default="chroma",description="知识库类型,如Chroma,Weaviate等")
    EMBEDDING_MODEL_SERVER: str = Field(default="siliconflow",description="使用的嵌入层服务商")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3",description="使用的嵌入层模型名称")
    PROCESSOR: str = Field(default="Docling",description="使用的文档处理器名称")
    FILE_LIST: Any = Field(default={},description="{文件哈希值: 文件原始名称}")
    HYBRID_RETRIEVER_WEIGHTS: List[float] = Field(
        default=[0.5, 0.5],
        description="混合检索器的权重列表",
    )



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
    def __init__(self,name:str = "default"):
        """初始化检索器构建器"""
        self.init_status = True # 初始化状态，默认为成功
        self.cache_dir = os.path.join(settings.KB_PATH,name)
        self.config_path = os.path.join(self.cache_dir,"config.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        # 加载配置文件，并保存为BaseKBConfig格式以实现验证,如果不存在则使用默认参数进行初始化
        if not os.path.exists(self.config_path):
            self.config = BaseKBConfig(name=name)
        else:
            with open(self.config_path, "r", encoding="utf-8") as f :
                self.config = BaseKBConfig.model_validate(json.load(f))
        self.status_msg = ""
        if not self.config.name:
            raise SyntaxError("知识库名称不能为空")
        if not  self.config.EMBEDDING_MODEL_SERVER:
            raise SyntaxError("必须要添加嵌入模型才能使用知识库")
        self.post_processors = [] # 用于保存后处理器列表
        # 加载嵌入模型
        if self.config.EMBEDDING_MODEL_SERVER == "siliconflow":
            # print(os.getenv("SILICONFLOW_URL"))
            # print(os.getenv("SILICONFLOW_KEY"))
            embedding = OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                base_url=settings.SILICONFLOW_URL,
                openai_api_key=settings.SILICONFLOW_KEY,
            )
        else:
            self.status_msg =f"尚不支持嵌入层服务商: {self.config.EMBEDDING_MODEL_SERVER}"
            self.init_status = False
            return 
        self.embeddings = embedding

        # 获取文件解析器
        self.parser = Doclingparser()

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
    def add_doc(self, doc_list: list[str]):
        """
        向检索器中添加文档列表的抽象方法
        
        Args:
            doc_list: 文档路径列表
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
        # for processor in self.post_processors:
        #     try:
        #         docs = processor(docs)
        #         logger.debug(f"Applied post-processor, now have {len(docs)} documents")
        #     except Exception as e:
        #         logger.error(f"Error applying post-processor: {e}")
        result = []
        file_manager_activate.add_docs(docs) # 将文件保存至本地

        
        for file_path in docs:
            # 保存文件信息至文件的配置文件中
            doc_hash = get_single_hash(file_path) + "." + file_path.split(".")[-1]
            doc_name = os.path.basename(file_path)
            if doc_hash not in self.config.FILE_LIST:
                chunks = self.parser._process_file(file_path)
                result.extend(chunks)
                self.config.FILE_LIST[doc_hash] = doc_name

        return result
    def save_config(self):
        """
        保存本地知识库的配置文件
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=4)
    @abstractmethod
    def save_local():
        """
        保存本地知识库
        """
        pass 