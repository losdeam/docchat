from abc import ABC, abstractmethod
from typing import Any, List
import logging,os,json,pickle
from config.settings import settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pathlib import Path
from utils import get_single_hash,file_manager_activate
from rag.parser import Doclingparser

# 动态导入所需的类
def get_kb_class(kb_type: str):
    """根据知识库类型获取对应的类"""
    if kb_type.lower() == "chroma":
        from .chroma import Chroma_Builder
        return Chroma_Builder
    else:
        raise ValueError(f"Unsupported KB type: {kb_type}")

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
    EMBEDDING_MODEL_SERVER: str = Field(default="siliconflow",description="使用的嵌入层服务商")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3",description="使用的嵌入层模型名称")
    PROCESSOR: str = Field(default="Docling",description="使用的文档处理器名称")
    FILE_LIST: Any = Field(default={},description="{文件哈希值: 文件原始名称}")
    HYBRID_RETRIEVER_WEIGHTS: List[float] = Field(
        default=[0.5, 0.5],
        description="混合检索器的权重列表",
    )



class BASE_KB(ABC):
    """
    定义一个抽象的基类，用于定义知识库的接口
    """
    def __init__(self, name: str = "default"):
        """初始化基本属性"""
        self.name = name
        self.config = None
        self.embeddings = None
        self.parser = None
        self.docs = {}
        self.cache_dir = None
        self.config_path = None
        self.docs_dir = None
        self.retriever = None
    @abstractmethod
    def delete_docs(self,doc_name:str):
        """
        删除文档的抽象方法
        """
        pass 
    @abstractmethod
    def add_doc(self, doc_list: list[str]):
        """
        向检索器中添加文档列表的抽象方法
        
        Args:
            doc_list: 文档路径列表
        """
        pass
    @abstractmethod
    def save_local(self):
        """
        保存本地知识库
        """
        pass 
    @abstractmethod
    def invoke(self, query: str):
        pass
    @abstractmethod
    def list_docs(self):
        """
        列出知识库中的文档列表
        """
        pass
    @abstractmethod
    def list_chunks(self,doc_name):
        """
        列出知识库中指定文档的分块列表
        """
        pass
    def parse_doc(self, docs: list) -> dict:
        """
        解析文档
        
        Args:
            docs: 文档路径列表
            
        Returns:
            处理后的 Document 列表
        """
        from langchain_core.documents import Document
        from utils import get_single_hash,file_manager_activate

        result = {}
        file_manager_activate.add_docs(docs)  # 将文件保存至本地

        for file_path in docs:
            # 生成唯一哈希标识
            ext = file_path.split(".")[-1]
            doc_hash = get_single_hash(file_path) + "." + ext
            doc_name = os.path.basename(file_path)

            if doc_hash not in self.config.FILE_LIST:
                try:
                    chunks = self.parser._process_file(file_path)
                    document_chunks = [
                        Document(page_content=chunk, metadata={"sort_id": idx})
                        for idx, chunk in enumerate(chunks)
                    ]
                    result[doc_hash] = document_chunks
                    self.docs[doc_hash] = document_chunks
                    self.config.FILE_LIST[doc_hash] = doc_name
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
        return result

class KB_factory(BASE_KB):
    """
    用于构建kb的基础功能
    """
    def __init__(self,name:str = "default"):
        """初始化检索器构建器"""
        # 先调用父类初始化
        super().__init__(name)
        self.activate_status = False 
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

    #用于在正式使用知识库时将其激活
    def activate_beforeUse(self):
        if self.activate_status == False:
            self.activate()

    
    def activate(self):
        """
        加载高内存消耗的参数到内存中
        """
        logging.info(f"正在激活知识库: {self.config.name}")
        # 加载嵌入模型
        if self.config.EMBEDDING_MODEL_SERVER == "siliconflow":
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
        self.retriever = None
        self.docs_dir = os.path.join(self.cache_dir,"docs.pkl") #用于存储处理后文档的目录(bm25不支持本地持久化),使用pkl格式文件保存
        if os.path.exists(self.docs_dir): # 加载处理后的文档
            with open(self.docs_dir, 'rb') as f:
                docs = pickle.load(f)
            self.docs = docs
        else:
            self.docs = {}

        # 根据配置文件初始化检索器
        kb_class = get_kb_class(self.config.KB_TYPE)
        self.kb_instance = kb_class(name=self.name)
        # 将当前实例的属性复制到加载的实例中
        self.kb_instance.config = self.config
        self.kb_instance.embeddings = self.embeddings
        self.kb_instance.parser = self.parser
        self.kb_instance.docs = self.docs
        self.kb_instance.cache_dir = self.cache_dir
        self.kb_instance.config_path = self.config_path
        self.kb_instance.docs_dir = self.docs_dir
        self.activate_status = True
        pass
    def deactivate(self):
        """
        释放高内存消耗的参数
        """
        self.kb_instance = None 
        self.embeddings = None 
        self.parser = None 
        self.activate_status = False

    def save_config(self):
        """
        保存本地知识库的配置文件
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=4)
    def build_retriever(self, docs=None):
        """
        构建检索器
        """
        self.activate_beforeUse()
        return self.kb_instance.build_retriever(docs)

    def invoke(self, query: str):
        """
        调用检索器进行查询
        """
        self.activate_beforeUse()
        return self.kb_instance.invoke(query)

    def add_doc(self, doc_list: list[str]):
        """
        添加文档
        """
        self.activate_beforeUse()
        return self.kb_instance.add_doc(doc_list)

    def save_local(self):
        """
        保存到本地
        """
        self.activate_beforeUse()
        self.kb_instance.docs = self.docs
        self.kb_instance.config = self.config
        return self.kb_instance.save_local()

    def list_docs(self):
        """
        列出文档
        """
        self.activate_beforeUse()
        logger.info("正在列出文档")
        return self.kb_instance.list_docs()

    def list_chunks(self, doc_name):
        """
        列出文档的分块
        """
        self.activate_beforeUse()
        return self.kb_instance.list_chunks(doc_name)

    def delete_docs(self, doc_name: str):
        """
        删除文档
        """
        self.activate_beforeUse()
        if doc_name in self.docs:
            del self.docs[doc_name]
        return self.kb_instance.delete_docs(doc_name)


class KB_manager():
    """
    知识库管理器
    """
    def __init__(self):
        self.kb_dict = {}
        self.kb_load_local()
        # 添加一个变量用于保存当前激活的知识库
        self.activate_kb = None


        
    def list_kb(self):
        """
        列出知识库
        """
        return list(self.kb_dict.keys())
        
    def kb_load_local(self):
        """
        加载本地知识库
        """
        # 确保KB_PATH目录存在
        if not os.path.exists(settings.KB_PATH):
            os.makedirs(settings.KB_PATH)
            
        for kb_name in os.listdir(settings.KB_PATH):
            kb_path = os.path.join(settings.KB_PATH, kb_name)
            if os.path.isdir(kb_path):
                self.kb_dict[kb_name] = KB_factory(kb_name)

    def raise_(self):
        """
        关闭所有知识库，并保存对其的修改
        """
        logger.info("正在关闭所有知识库")
        for kb_name in self.kb_dict:
            self.kb_dict[kb_name].save_local()
            logging.info(f"已保存知识库 {kb_name}")
            self.kb_dict[kb_name].deactivate()

kb_manager = KB_manager()