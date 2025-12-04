from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
# 核心是以下三个智能体
from .research_agent import ResearchAgent   # 使用相关文档生成草拟答案
from .verification_agent import VerificationAgent # 评估草拟答案的准确性和相关性
from .relevance_checker import RelevanceChecker # 确定查询是否够可以根据检索到的文档进行回答

from retriever import Retriever
from langchain_core.documents import Document
import logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever :Retriever

class AgentWorkflow:
    def __init__(self, config=None):
        self.config = config or {}
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()  # Compile once during initialization
        
    def build_workflow(self):
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        
        # Define edges
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )
        return workflow.compile()
    def _retriever(self, state: AgentState) -> Retriever:
        """
        Retrieve relevant documents for the given question.
        """
        retriever = state["retriever"]
        top_docs = retriever.invoke(state["question"])
        return top_docs
    def _check_relevance_step(self, state: AgentState) -> Dict:
        classification = self.relevance_checker.check(
            question=state["question"], 
            documents=self._retriever(state), 
            k=30  # 提高k值以增强召回率，确保更多潜在相关的文档被考虑
        )

        if classification == "CAN_ANSWER":
            # We have enough info to proceed
            return {"is_relevant": True}

        elif classification == "PARTIAL":
            # There's partial coverage, but we can still proceed
            return {
                "is_relevant": True
            }

        else:  # classification == "NO_MATCH"
            return {
                "is_relevant": False,
                "draft_answer": "知识库中查询不到与您的查询相关的数据。请就上传的文档提出其他相关问题"
            }


    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        print(f"[DEBUG] _decide_after_relevance_check -> {decision}")
        return decision
    
    def full_pipeline(self, question: str, retriever: Retriever):
        try:
            print(f"[DEBUG] Starting full_pipeline with question='{question}'")
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents (from .invoke)")

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever
            )
            
            final_state = self.compiled_workflow.invoke(initial_state)
            
            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"]
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _research_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _research_step with question='{state['question']}'")
        result = self.researcher.generate(state["question"], state["documents"])
        print("[DEBUG] Researcher returned draft answer.")
        return {"draft_answer": result["draft_answer"]}
    
    def _verification_step(self, state: AgentState) -> Dict:
        print("[DEBUG] Entered _verification_step. Verifying the draft answer...")
        result = self.verifier.check(state["draft_answer"], state["documents"])
        print("[DEBUG] VerificationAgent returned a verification report.")
        return {"verification_report": result["verification_report"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        verification_report = state["verification_report"]
        print(f"[DEBUG] _decide_next_step with verification_report='{verification_report}'")
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("[DEBUG] Verification indicates re-research needed.")
            return "re_research"
        else:
            logger.info("[DEBUG] Verification successful, ending workflow.")
            return "end"