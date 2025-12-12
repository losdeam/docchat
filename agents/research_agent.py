from config.settings import settings
from typing import List, Dict
from langchain_core.documents import Document
import re,os
import logging
from langchain_openai import ChatOpenAI
from utils import logger,log_execution
# logger = logging.getLogger(__name__)

class ResearchAgent:
    @log_execution("rag问题回复节点——初始化")
    def __init__(self):
        """
        Initialize the research agent 
        """
        # logger.info("正在初始化生成模型...")
        model_server = os.getenv("RESEARCH_MODEL_SERVER")
        model_name = os.getenv("RESEARCH_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        
        if model_server == "siliconflow":
            self.model = ChatOpenAI(
                model=model_name,
                base_url=os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_KEY"),
                max_tokens=2000,
                temperature=0.3
            )
        else:
            logger.info("未配置有效的模型服务器。")
        # logger.info("生成模型初始化成功.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        你是一个基于给定上下文提供准确且事实性回答的人工智能助手。
        **说明：**
        - 仅根据提供的上下文回答以下问题。
        - 言辞明确、简洁且基于事实。
        尽可能从上下文中提取所有信息。
        **问题：** {question}
        **上下文:**
        {context}
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        logger.info(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.info(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        logger.info("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            logger.info("Sending prompt to the model...")
            response = self.model.invoke(prompt)
            logger.info("LLM response received.")
        except Exception as e:
            logger.info(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        # Extract and process the LLM's response
        try:
            llm_response = response.content.strip()
            logger.info(f"Raw LLM response:\n{llm_response}")
        except (IndexError, KeyError) as e:
            logger.info(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        logger.info(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }