"""Prompt building utilities for the chatbot using LangChain templates."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from src.config import (
    ASSISTANT_NAME,
    DIRECT_CONTENT_HINTS,
    GREETING_PATTERN,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGE_CHARS,
    MAX_HISTORY_MESSAGES,
    TIME_QUERY_PATTERN,
    normalize_question,
)


def _bounded_history(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep the newest history within the configured prompt-size budget."""
    if not chat_history:
        return []

    max_messages = MAX_HISTORY_MESSAGES
    max_history_chars = MAX_HISTORY_CHARS
    try:
        from src.admin.config_service import load_config

        performance = load_config().performance
        max_messages = performance.max_history_messages
        max_history_chars = performance.max_history_chars
    except (OSError, ValueError):
        pass

    bounded: list[dict[str, str]] = []
    used_chars = 0
    for item in reversed(chat_history[-max_messages:]):
        content = str(item.get("content", ""))[:MAX_HISTORY_MESSAGE_CHARS]
        message = {"role": str(item.get("role", "")), "content": content}
        message_chars = len(content)
        if bounded and used_chars + message_chars > max_history_chars:
            break
        bounded.append(message)
        used_chars += message_chars
    bounded.reverse()
    return bounded


def to_langchain_messages(chat_history: list[dict[str, str]] | None) -> list[BaseMessage]:
    """
    Convert a list of role/content dictionaries into native LangChain BaseMessage objects.

    Args:
        chat_history: List of message dictionaries with 'role' and 'content'

    Returns:
        List of HumanMessage and AIMessage objects
    """
    if not chat_history:
        return []

    messages: list[BaseMessage] = []
    for item in _bounded_history(chat_history):
        role = item.get("role", "").lower()
        content = item.get("content", "")
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai", "bot"):
            messages.append(AIMessage(content=content))
    return messages


def format_history(chat_history: list[dict[str, str]]) -> str:
    """
    Format chat history for legacy string-based inclusion in prompts.

    Args:
        chat_history: List of message dictionaries with 'role' and 'content'

    Returns:
        Formatted conversation string
    """
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in _bounded_history(chat_history):
        role = message.get("role", "User").capitalize()
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def classify_query(question: str) -> str:
    """
    Classify a query into one of four categories:
    - small_talk: Greeting or casual message
    - tool_call: Request for live information supplied by a tool
    - direct_context: User provides content or asks for structured output
    - retrieval: Standard question requiring knowledge base lookup

    Args:
        question: User question

    Returns:
        Query classification string
    """
    normalized_question = normalize_question(question)
    if TIME_QUERY_PATTERN.search(normalized_question):
        return "tool_call"
    if GREETING_PATTERN.match(normalized_question):
        return "small_talk"
    if any(hint in normalized_question for hint in DIRECT_CONTENT_HINTS):
        return "direct_context"
    if len(question) > 1200:
        return "direct_context"
    return "retrieval"


# System instructions for BBHS AI Assistant
SYSTEM_RAG_INSTRUCTIONS = (
    f"You are a knowledgeable and helpful assistant, {ASSISTANT_NAME}.\n"
    "Your role is to answer user questions accurately and comprehensively using the provided knowledge base.\n"
    "Use the conversation history to understand follow-up questions and maintain context.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided Context. Be thorough — if the context has troubleshooting steps, features, configuration details, or best practices, share them fully.\n"
    "- If the user has a technical issue, provide clear step-by-step guidance using the context.\n"
    "- Be helpful and professional when responding to all queries.\n"
    "- If the question is casual or a greeting, respond in a friendly manner.\n"
    '- If the answer is not in the context, simply state: "I don\'t have information about that in the knowledge base. Please provide more details or rephrase your question."\n'
    "- Just focus on answering the question."
)

SYSTEM_DIRECT_INSTRUCTIONS = (
    f"You are a helpful assistant, {ASSISTANT_NAME}.\n"
    "The user may provide full technical details, error messages, or documentation directly inside the request.\n"
    "When that happens, answer from the user-provided content to help resolve their issue or answer their question.\n\n"
    "Rules:\n"
    "- Answer the user's question fully and clearly first.\n"
    "- Provide step-by-step solutions for technical issues when applicable.\n"
    "- Be professional and helpful in your responses.\n"
    "- Do not ask for contact information - just focus on answering the question.\n"
    "- If you cannot fully answer from the provided information, suggest what additional details would help you assist better."
)

# LangChain ChatPromptTemplates
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_RAG_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
    ]
)

DIRECT_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_DIRECT_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("User request:\n{question}\n\nAnswer:"),
    ]
)


def _active_prompt_config() -> dict[str, str]:
    """Load editable prompt text while preserving source-code defaults."""
    try:
        from src.admin.config_service import load_config

        return load_config().prompts.model_dump()
    except (OSError, ValueError):
        return {
            "system_prompt": SYSTEM_RAG_INSTRUCTIONS,
            "rag_prompt": "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
            "context_instructions": SYSTEM_DIRECT_INSTRUCTIONS,
        }


def build_active_qa_prompt() -> ChatPromptTemplate:
    """Build the RAG prompt from the validated admin configuration."""
    values = _active_prompt_config()
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(values["system_prompt"]),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(values["rag_prompt"]),
        ]
    )


def build_active_direct_prompt() -> ChatPromptTemplate:
    """Build the direct-context prompt from the validated admin configuration."""
    values = _active_prompt_config()
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(values["context_instructions"]),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template("User request:\n{question}\n\nAnswer:"),
        ]
    )


def active_response_text(name: str, fallback: str) -> str:
    """Return a configured response string such as greeting or no-answer text."""
    values = _active_prompt_config()
    return values.get(name, fallback)


def active_guardrail_text(fallback: str) -> str:
    """Return the configured pure-RAG out-of-scope response."""
    try:
        from src.admin.config_service import load_config

        return load_config().guardrails.out_of_scope_response
    except (OSError, ValueError):
        return fallback


def build_prompt(question: str, context: str, history: str) -> str:
    """Backward-compatible helper returning a raw formatted prompt string."""
    return f"""{SYSTEM_RAG_INSTRUCTIONS}

Conversation:
{history}
Context:
{context}
Question:
{question}
Answer:"""


def build_direct_context_prompt(question: str, history: str) -> str:
    """Backward-compatible helper returning a raw formatted direct prompt string."""
    return f"""{SYSTEM_DIRECT_INSTRUCTIONS}

Conversation:
{history}
User request:
{question}
Answer:"""
