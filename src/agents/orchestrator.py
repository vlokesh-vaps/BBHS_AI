"""Agent Orchestrator: Multi-intent classifier and LangGraph coordinator."""

from typing import Any
from langchain_chroma import Chroma

from src.agents.graph import (
    AgentGraphState,
    admissions_graph,
    classify_intent,
)
from src.agents.state import (
    AgentIntent,
    AgentResponse,
    SessionState,
)
from src.core.retrievers import BM25Index, EnsembleRetriever


class AgentOrchestrator:
    """Central orchestrator for the BBHS Conversational Admissions Agent using LangGraph."""

    def __init__(self, graph: Any = None) -> None:
        self.graph = graph or admissions_graph

    def classify_intent(self, question: str, session_state: SessionState) -> AgentIntent:
        """Determine user intent based on the current turn and active conversation state."""
        return classify_intent(question, session_state)

    def process_turn(
        self,
        question: str,
        chat_history: list[dict[str, str]],
        session_state: SessionState | None = None,
        vector_db: Chroma | None = None,
        bm25_index: BM25Index | None = None,
        llm: Any = None,
        cache: dict[str, dict[str, object]] | None = None,
        ensemble_retriever: EnsembleRetriever | None = None,
        answer_question_fn: Any = None,
    ) -> AgentResponse:
        """Process a conversational turn through the compiled LangGraph pipeline.

        Internal decision making is structured; no chain-of-thought is revealed to the user.
        Transcripts remain strictly in memory per Invariant D3.
        """
        state = session_state or SessionState()

        graph_input: AgentGraphState = {
            "question": question,
            "chat_history": chat_history,
            "session_state": state,
            "vector_db": vector_db,
            "bm25_index": bm25_index,
            "llm": llm,
            "cache": cache,
            "ensemble_retriever": ensemble_retriever,
            "answer_question_fn": answer_question_fn,
        }

        output = self.graph.invoke(graph_input)

        return AgentResponse(
            answer=output.get("response_answer", ""),
            actions=output.get("response_actions", []),
            state=output.get("session_state", state),
        )
