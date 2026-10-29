"""Query intent classification, institutional route dispatching, and fallback responses."""

import re

from src.config import (
    ADMISSION_EMAIL,
    ADMISSION_PHONE,
    DIRECT_CONTENT_HINTS,
    FEE_ENQUIRY_PHONE,
    GREETING_PATTERN,
    TIME_QUERY_PATTERN,
    normalize_question,
)


def classify_query(question: str) -> str:
    """Classify user query into routing categories.

    Returns:
        One of: 'small_talk', 'time_query', 'direct_context', or 'retrieval'.
    """
    normalized_question = normalize_question(question)
    if not normalized_question:
        return "small_talk"
    if TIME_QUERY_PATTERN.search(normalized_question):
        return "tool_call"
    if GREETING_PATTERN.search(normalized_question):
        return "small_talk"
    if any(hint in normalized_question for hint in DIRECT_CONTENT_HINTS):
        return "direct_context"
    if len(question) > 1200:
        return "direct_context"
    return "retrieval"


def requested_class_label(question: str) -> str | None:
    """Extract a parent-friendly class label from common school wording."""
    normalized = question.lower()
    named_class = re.search(r"\b(pre[- ]?kg|lkg|ukg)\b", normalized)
    if named_class:
        return named_class.group(1).upper().replace("-", "-")

    grade_number = re.search(
        r"\b(?:grade|class|std\.?|standard)\s*(\d{1,2})\b"
        r"|\b(\d{1,2})\s*(?:st|nd|rd|th)\s*(?:std\.?|standard|class)\b",
        normalized,
    )
    if grade_number:
        number = grade_number.group(1) or grade_number.group(2)
        return f"Grade {number}"
    return None


def is_grade_fee_enquiry(question: str) -> bool:
    """Identify a grade-specific fee question before cache or LLM generation."""
    return bool(
        requested_class_label(question)
        and re.search(r"\b(fee|fees|tuition|cost|price|charges?)\b", question.lower())
    )


def missing_information_response(question: str) -> str:
    """Give a warm, grounded next step directing parents to school contacts."""
    normalized = question.lower()
    contact_requested = bool(
        re.search(r"\b(contact|phone|call|email|number|reach|speak|talk)\b", normalized)
    )
    contact_line = (
        f" You can reach Admissions at {ADMISSION_PHONE} or {ADMISSION_EMAIL}."
        if contact_requested
        else ""
    )
    class_label = requested_class_label(question)
    if contact_requested and not re.search(
        r"\b(fee|fees|tuition|cost|price|charges?|visit|visiting|tour|campus|admission|enrol|enroll|apply|application)\b",
        normalized,
    ):
        return (
            "Of course. You can reach the Baldwin Boys' High School Admissions team at "
            f"{ADMISSION_PHONE} or {ADMISSION_EMAIL}. What would you like to ask them about?"
        )
    if re.search(r"\b(fee|fees|tuition|cost|price|charges?)\b", normalized):
        if class_label:
            class_name = class_label
            return (
                f"**Enquiring about {class_name.title()} at BBHS?**\n\n"
                "We’d be happy to help you with the fee details for your child’s admission to "
                "Baldwin Boys’ High School.\n\n"
                f"**Contact our admissions team:** {FEE_ENQUIRY_PHONE} for the latest fee structure "
                "and applicable charges.\n\n"
                "**How would you like to proceed?**\n\n"
                "- Learn about the admission process\n"
                "- Enquire about the fee structure\n"
                "- Explore a school visit"
            )
        return (
            "I'd be happy to help with the fee details for Baldwin Boys' High School. "
            "The exact fee structure can vary by class. Which class are you enquiring about?"
            f"{contact_line}"
        )
    if re.search(r"\b(visit|visiting|tour|campus|come to school)\b", normalized):
        visit_next_step = (
            f"You can confirm them with Admissions at {ADMISSION_PHONE} or {ADMISSION_EMAIL} before you come."
            if contact_requested
            else "Please confirm them with Admissions before you come."
        )
        return (
            "We'd be happy to welcome you to Baldwin Boys' High School. Are you planning "
            "to visit for an admission enquiry or to explore the campus? The visiting hours "
            f"can vary. {visit_next_step}"
        )
    if re.search(r"\b(admission|enrol|enroll|apply|application)\b", normalized):
        return (
            "I'd be happy to help you with admission at Baldwin Boys' High School. "
            "Which class are you looking for? I can guide you through the next steps."
            f"{contact_line}"
        )
    return (
        "I'd love to help with that. Could you tell me a little more? For example, "
        "are you asking about admissions, fees, school timings, facilities, or school life?"
    )
