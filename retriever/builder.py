from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
# from langchain_ibm import WatsonxEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import logging,os
from langchain_openai import ChatOpenAI
logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with embeddings."""
        # embed_params = {
        #     EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        #     EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        # }

        # embedding = WatsonxEmbeddings(
        #     model_id="ibm/slate-125m-english-rtrvr-v2",
        #     url="https://us-south.ml.cloud.ibm.com",
        #     project_id="skills-network",
        #     params=embed_params
        # )
        # 使用siliconflow的embedding模型
        embedding = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5"),
            base_url=os.getenv("SILICONFLOW_URL"),
            openai_api_key=os.getenv("SILICONFLOW_KEY"),

        )
        self.embeddings = embedding
        
    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
        try:
            # print(docs)
            # Create Chroma vector store
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )
            logger.info("Vector store created successfully.")
            
            # Create BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")
            
            # Create vector-based retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")
            
            # Combine retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise