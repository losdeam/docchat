from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from utils import get_single_hash,file_manager_activate
from config.settings import settings
import logging, os, pickle,json
from typing import List, Any
from langchain_core.documents import Document
from .base import BASE_KB
logger = logging.getLogger(__name__)

class Chroma_Retriever():
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

class Chroma_Builder(BASE_KB):
    def __init__(self, name: str = "default"):
        """Initialize the retriever builder with embeddings."""
        super().__init__(name)  # 调用基类初始化，即使 BASE_KB 是空实现也保持一致性
        self.name = name
        self.retriever = None
        self.config = None
        self.embeddings = None
        self.parser = None
        self.docs = {}
        self.cache_dir = None
        self.config_path = None
        self.docs_dir = None



    def build_retriever(self, docs=None):
        """构建一个结合BM25与向量检索的混合检索器。"""
        try:
            # 使用用户设置或默认设置
            hybrid_retriever_weights = self.config.HYBRID_RETRIEVER_WEIGHTS
            
            # 如果提供了docs则使用它，否则使用self.docs
            documents = docs if docs is not None else list(self.docs.values()) if isinstance(self.docs, dict) else self.docs

            # 展平文档：如果 self.docs 是字典，values() 是列表的列表，则需展平
            if isinstance(documents, dict):
                documents = [doc for chunk_list in documents.values() for doc in chunk_list]
            elif isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list):
                documents = [doc for sublist in documents for doc in sublist]

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

    def invoke(self, query: str) -> list[str]:
        """使用检索器进行查询。"""
        try:
            if self.retriever is None:
                self.build_retriever()
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Failed to invoke retriever: {e}")
            raise
    
    def add_doc(self, doc_list: list[str]):
        
        result = {}
        file_manager_activate.add_docs(doc_list) # 将文件保存至本地

        for file_path in doc_list:
            # 保存文件信息至文件的配置文件中
            doc_hash = get_single_hash(file_path) + "." + file_path.split(".")[-1]
            doc_name = os.path.basename(file_path)
            if doc_hash not in self.config.FILE_LIST:
                chunks = self.parser._process_file(file_path)
                result[doc_name] = chunks
                self.docs[doc_name] = chunks
                self.config.FILE_LIST[doc_hash] = doc_name
        return result

    def save_local(self):
        # 将self.docs以pkl格式保存到self.docs_dir
        with open(self.docs_dir, 'wb') as f:
            pickle.dump(self.docs, f)
        self.save_config()

    def list_docs(self):
        return list(self.docs.keys()) if isinstance(self.docs, dict) else []

    def list_chunks(self, doc_name):
        return self.docs.get(doc_name, []) if isinstance(self.docs, dict) else []
        
    def delete_docs(self, doc_name: str):
        """删除指定文档"""
        if isinstance(self.docs, dict) and doc_name in self.docs:
            del self.docs[doc_name]
        

        
    def save_config(self):
        """
        保存本地知识库的配置文件
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=4)