from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from config.settings import settings
import logging,os,pickle
from .base import BASE_KB,BaseRetriever
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
    def __init__(self):
        """Initialize the retriever builder with embeddings."""
        super().__init__()
        self.retriever = None
        self.docs_dir = os.path.join(self.cache_dir,"docs.pkl") #用于存储处理后文档的目录(bm25不支持本地持久化),使用pkl格式文件保存
        self.file_dir =os.path.join(self.cache_dir,"files") #存储原始文档的目录
        if self.docs_dir: # 加载处理后的文档
            with open(self.docs_dir, 'rb') as f:
                docs = pickle.load(f)
        self.docs = docs #由于bm25不支持持久化，所以本质上需要即插即用没法预先构建检索器在后续操作，所以利用保存文档列表的形式变相的保存
    def build_retriever(self):
        """构建一个结合BM25与向量检索的混合检索器。"""

        try:
            vector_store = Chroma(
                documents=self.docs,
                embedding=self.embeddings,
            )

            bm25 = BM25Retriever.from_documents(self.docs)
            hybrid_retriever = Chroma_Retriever(
                    retrievers=[bm25, vector_store],
                    weights=settings.HYBRID_RETRIEVER_WEIGHTS,
                    flags=["bm25","vector"]
                )
            self.retriever = hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

                
