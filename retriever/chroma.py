from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from config.settings import settings
import logging,os,pickle
from .base import BASE_KB,BaseRetriever,BaseKBConfig
logger = logging.getLogger(__name__)

class Chroma_Retriever(BaseRetriever):
    def __init__(self, retrievers,weights,flags):
        """初始化检索器列表，以及对应的权重，以及对应标记"""
        self.retrievers = retrievers
        self.weights = weights
        self.flags = flags

    def invoke(self, query: str):
        """进行混合检索"""
        combined = [] 
        # 遍历每个检索器，并使用权重进行混合
        for retriever, weight,flag in zip(self.retrievers, self.weights,self.flags):
            if flag in ["vector"]:
                docs = retriever.similarity_search_with_score(query)
            else:
                docs = retriever.invoke(query)
            print(docs)
            for doc in docs:
                if type(doc) is tuple:
                    adjusted_score = doc[1] * weight
                    combined.append((doc[0], adjusted_score,flag))
                else:
                    adjusted_score = doc.metadata.get("score", 0) * weight
                    combined.append((doc, adjusted_score,flag))
        # 按调整后的分数降序排序
        combined.sort(key=lambda x: x[1], reverse=True)
        # 去重：基于文档内容
        seen_content = set()
        final_docs = []
        for doc, score, source in combined:
            content_snippet = doc.page_content[:100]  # 使用内容前100字符作为标识
            if content_snippet not in seen_content: # 如果没有见过，则添加
                seen_content.add(content_snippet)
                # 处理doc没有metadata的情况
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                # 可以选择将合并后的分数存入metadata
                doc.metadata["score"] = score
                doc.metadata["retrieval_source"] = source
                final_docs.append(doc)
        return final_docs

class RetrieverBuilder(BASE_KB):
    def __init__(self, base_setting: BaseKBConfig= None,user_settings=None):
        """Initialize the retriever builder with embeddings."""
        if not base_setting:
            super().__init__(BaseKBConfig(name="default",EMBEDDING_MODEL_SERVER="siliconflow"))
        else:
            super().__init__(base_setting)
        self.retriever = None
        self.docs_dir = os.path.join(self.cache_dir,"docs.pkl") #用于存储处理后文档的目录(bm25不支持本地持久化),使用pkl格式文件保存
        self.file_dir =os.path.join(self.cache_dir,"files") #存储原始文档的目录
        if os.path.exists(self.docs_dir): # 加载处理后的文档
            with open(self.docs_dir, 'rb') as f:
                docs = pickle.load(f)
            self.docs = docs #由于bm25不支持持久化，所以本质上需要即插即用没法预先构建检索器在后续操作，所以利用保存文档列表的形式变相的保存
        else:
            self.docs = []
        
        # 使用用户设置或默认设置
        self.user_settings = user_settings or {}

    def build_retriever(self, docs=None):
        """构建一个结合BM25与向量检索的混合检索器。"""
        try:
            # 使用用户设置或默认设置
            hybrid_retriever_weights = self.user_settings.get("HYBRID_RETRIEVER_WEIGHTS", settings.HYBRID_RETRIEVER_WEIGHTS)
            
            # 如果提供了docs则使用它，否则使用self.docs
            documents = docs if docs is not None else self.docs
            
            # 检查是否有文档，如果没有则返回一个空的检索器
            if not documents:
                logger.warning("No documents provided for retriever construction")
                # 创建一个空的BM25检索器
                bm25 = BM25Retriever.from_texts([""])
                # 创建一个空的Chroma检索器
                vector_store = Chroma(embedding_function=self.embeddings, persist_directory=str(self.cache_dir))
                
                hybrid_retriever = Chroma_Retriever(
                    retrievers=[bm25, vector_store],
                    weights=hybrid_retriever_weights,
                    flags=["bm25", "vector"]
                )
                self.retriever = hybrid_retriever
                return self.retriever
            
            # 创建Chroma向量存储的新方式
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.cache_dir)
            )

            bm25 = BM25Retriever.from_documents(documents)
            hybrid_retriever = Chroma_Retriever(
                    retrievers=[bm25, vector_store],
                    weights=hybrid_retriever_weights,
                    flags=["bm25","vector"]
                )
            self.retriever = hybrid_retriever
            return self.retriever
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    def save_local():
        pass 