import json  # Import for JSON serialization
from typing import Dict, List
from langchain_core.documents import Document
import re,os
import logging
from langchain_openai import ChatOpenAI


class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with the IBM WatsonX ModelInference.
        """
        print("正在初始化判别模型...")
        model_server = os.getenv("VERIFICATION_MODEL_SERVER")
        if model_server == "siliconflow":
            self.model = ChatOpenAI(
                model=os.getenv("VERIFICATION_MODEL_NAME"),  # 指定硅基流动平台上的模型，例如DeepSeek-V3[citation:5]
                base_url=os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_KEY"),  # 传入API密钥
                max_tokens=2000,  # Adjust based on desired response length
                temperature=0  # Controls randomness; lower values make output more deterministic
            )
            print("判别模型初始化成功.")
        else:
            print("未配置有效的模型服务器。")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        你是一个AI助手，专门用于根据提供的上下文验证答案的准确性和相关性。

        **说明：**
        - 根据提供的上下文验证以下答案。
        - 检查以下内容：
        1. 是否有直接/间接的事实支持（是/否）
        2. 未经证实的声明（如果存在，请列出）
        3. 矛盾之处（如果存在，请列出）
        4. 与问题的相关性（是/否）
        - 在适当的地方提供额外的细节或解释。
        - 严格按照下面指定的确切格式回复，不要添加任何无关信息。

        **格式：**
        Supported: 是/否
        Unsupported Claims: [项目1, 项目2, ...]
        Contradictions: [项目1, 项目2, ...]
        Relevant: 是/否
        Additional Details: [任何额外的信息或解释]

        **答案：** {answer}
        **上下文：**
        {context}

        **仅按照上述格式回复。**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # 处理中英文关键字映射
                    key_mapping = {
                        "支持": "Supported",
                        "未经证实的声明": "Unsupported Claims",
                        "矛盾": "Contradictions",
                        "相关": "Relevant",
                        "额外细节": "Additional Details"
                    }
                    
                    normalized_key = key_mapping.get(key, key)
                    if normalized_key in {"Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"}:
                        if normalized_key in {"Unsupported Claims", "Contradictions"}:
                            # Convert string list to actual list
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                # Remove any surrounding quotes and whitespace
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[normalized_key] = items
                            else:
                                verification[normalized_key] = []
                        elif normalized_key == "Additional Details":
                            verification[normalized_key] = value
                        else:
                            # Handle both YES/NO and 是/否
                            if value.upper() in ["YES", "是"]:
                                verification[normalized_key] = "YES"
                            elif value.upper() in ["NO", "否"]:
                                verification[normalized_key] = "NO"
                            else:
                                verification[normalized_key] = value.upper()
            # Ensure all keys are present
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        # 支持中英文显示
        supported_text = "是" if supported == "YES" else "否" if supported == "NO" else supported
        relevant_text = "是" if relevant == "YES" else "否" if relevant == "NO" else relevant
        
        report = f"**支持情况 (Supported):** {supported_text}\n"
        if unsupported_claims:
            report += f"**未经证实的声明 (Unsupported Claims):** {', '.join(unsupported_claims)}\n"
        else:
            report += f"**未经证实的声明 (Unsupported Claims):** 无\n"

        if contradictions:
            report += f"**矛盾之处 (Contradictions):** {', '.join(contradictions)}\n"
        else:
            report += f"**矛盾之处 (Contradictions):** 无\n"

        report += f"**相关性 (Relevant):** {relevant_text}\n"

        if additional_details:
            report += f"**额外细节 (Additional Details):** {additional_details}\n"
        else:
            report += f"**额外细节 (Additional Details):** 无\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents.")

        # Combine all document contents into one string without truncation
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM to verify the answer
        prompt = self.generate_prompt(answer, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the verification report
        try:
            print("Sending prompt to the model...")
            response = self.model.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        # Extract and process the LLM's response
        try:
            llm_response = response.content.strip()
            print(f"Raw LLM response:\n{llm_response}")
        except Exception as e:
            print(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            print(f"Verification report:\n{verification_report_formatted}")
            print(f"Context used: {context}")
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        # Sanitize the response
        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            print("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            # Parse the response into the expected format
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                print("LLM did not respond with the expected format. Using default verification report.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        # Format the verification report into a paragraph
        verification_report_formatted = self.format_verification_report(verification_report)
        print(f"Verification report:\n{verification_report_formatted}")
        print(f"Context used: {context}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }