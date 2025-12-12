from config.settings import settings
import re,os
import logging
from langchain_openai import ChatOpenAI
from utils import logger,log_execution

class RelevanceChecker: # 构建一个类来获取检索的状态
    @log_execution("结果相关性判断节点——初始化")
    def __init__(self):
        """
        Initialize the research agent 
        """
        # logger.info("正在初始化判别模型...")
        model_server = os.getenv("CHECKER_MODEL_SERVER")
        if model_server == "siliconflow":
            self.model = ChatOpenAI(
                model=os.getenv("CHECKER_MODEL_NAME"),  # 指定硅基流动平台上的模型，例如DeepSeek-V3[citation:5]
                base_url= os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_KEY"),# 传入API密钥
                max_tokens=1000,            # Adjust based on desired response length
                temperature=0           # Controls randomness; lower values make output more deterministic
            )
        # logger.info("判别模型初始化成功.")


    def check(self, question: str, documents=None,k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        if not documents:
            logger.debug("No documents returned from rag.retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in documents[:k])

        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        你是一个AI相关性检查器，用于检查用户问题和提供的文档内容之间的相关性。

        **说明：**
        - 对文档内容与用户问题的相关程度进行分类。
        - 只能回复以下标签之一：CAN_ANSWER、PARTIAL、NO_MATCH。
        - 不要包含任何额外的文字或解释。

        **标签含义：**
        1) "CAN_ANSWER"：文档包含足够信息来完整回答问题。
        2) "PARTIAL"：文档提到了或讨论了问题的主题，但没有提供完整回答所需的全部细节。
        3) "NO_MATCH"：文档完全没有讨论或提到问题的主题。

        **重要：** 如果文档以任何方式提及或关联到问题的主题或时间范围，即使不完整，也要回复"PARTIAL"而不是"NO_MATCH"。

        **问题：** {question}
        **文档片段：** {document_content}

        **仅回复以下标签之一：CAN_ANSWER、PARTIAL、NO_MATCH**
        """

        # Call the LLM
        try:
            response = self.model.invoke(prompt)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"

        # Extract the content from the response
        try:
            llm_response = response.content.strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except (IndexError, KeyError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"
        logger.info(prompt)
        logger.info(f"Checker response: {llm_response}")

        # Validate the response
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug(f"Classification recognized as '{llm_response}'.")
            classification = llm_response

        return classification