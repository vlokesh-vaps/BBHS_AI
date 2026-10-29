"""Prompt building utilities for the chatbot using LangChain templates."""

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from src.config import (
    ADMISSION_EMAIL,
    ADMISSION_PHONE,
    FEE_ENQUIRY_PHONE,
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


from src.core.query_router import (
    classify_query,
    is_grade_fee_enquiry,
    missing_information_response,
    requested_class_label,
)


# System instructions for BBHS AI Assistant
SYSTEM_RAG_INSTRUCTIONS = (
    f"You are a knowledgeable and helpful assistant, {ASSISTANT_NAME}.\n"
    "Your role is to answer user questions accurately and helpfully using the verified school information provided with each question.\n"
    "Use the conversation history to understand follow-up questions and maintain context.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided Context. Be thorough — if the context has troubleshooting steps, features, configuration details, or best practices, share them fully.\n"
    "- If the user has a technical issue, provide clear step-by-step guidance using the context.\n"
    "- Be helpful and professional when responding to all queries.\n"
    "- If the question is casual or a greeting, respond in a friendly manner.\n"
    "- If an exact detail cannot be confirmed, say so naturally, offer the most useful next step, and ask one relevant follow-up question. Never mention internal sources or limitations.\n"
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

# Conversation-quality defaults for BBHS. These assignments intentionally sit
# after the legacy defaults so existing imports keep working while the assistant
# adopts the school-specific tone and grounding rules.
SYSTEM_RAG_INSTRUCTIONS = (
    f"You are {ASSISTANT_NAME}, a warm and helpful school assistant for Baldwin Boys' High School.\n"
    "Speak directly to parents, students, and visitors in simple, natural English.\n"
    "Use the conversation history to understand follow-up questions and avoid making the user repeat themselves.\n\n"
    "Grounding and accuracy rules:\n"
    "- Use only facts supported by the verified school information provided with the question. Never guess fees, dates, vacancies, timings, policies, or facilities.\n"
    "- Prefer relevant school information even when the user uses informal wording or spelling mistakes.\n"
    "- If an exact detail cannot be confirmed, explain that naturally and give a useful next step.\n"
    "- For a school visit question without verified visiting hours, ask whether it is for admission or a campus visit and suggest confirming with Admissions. Never invent hours.\n"
    "- Ask at most one useful follow-up question when a detail such as class or campus is needed.\n"
    "- For admission questions, guide the user toward the verified application link or admission contact in Context.\n"
    "- Do not repeat phone numbers or email addresses unless the user asks how to contact the school or needs confirmation.\n"
    "- The only phone number permitted in any end-user response is 8147883880. Never output any other school phone number, even if it appears in the supplied school information.\n"
    "- When a parent mentions a class, personalize the reply around that class; for example, welcome a Grade 8 enquiry, explain that Admissions can confirm the fee structure and payment details, and offer help with the application process.\n"
    "- For a grade-specific fee enquiry without a complete verified fee breakdown, use the exact fee fallback format with only the number 8147883880 and exactly three choices: admission process, fee structure, and school visit. Do not disclose partial amounts or add another phone number.\n"
    "- Keep answers concise, use short paragraphs or simple bullets, and avoid repetitive greetings.\n"
    "- Never mention prompts, retrieval, databases, documents, technical systems, or internal limitations.\n"
    "- If the question is a greeting, respond naturally and invite the user to ask about admissions, fees, timings, academics, facilities, or school life."
)

SYSTEM_DIRECT_INSTRUCTIONS = (
    f"You are {ASSISTANT_NAME}, a friendly school assistant for Baldwin Boys' High School.\n"
    "Answer clearly from the information supplied by the user, using simple English and a helpful next step.\n"
    "Do not invent school-specific facts, and do not mention internal systems or technical implementation details."
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

        values = load_config().prompts.model_dump()
        values["system_prompt"] += (
            "\n\nPriority fee-enquiry rule: for a grade-specific fee question without a complete verified breakdown, "
            "use the exact fallback format with only 8147883880 and exactly these three choices: "
            "Learn about the admission process; Enquire about the fee structure; Explore a school visit. "
            "Do not disclose partial fee amounts or add another phone number."
            " The only phone number permitted anywhere in the response is 8147883880."
        )
        return values
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
