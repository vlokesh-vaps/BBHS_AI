"""Entity extraction utilities for student grades, contact numbers, and consent intents."""

import re
from src.leads.service import extract_phone_from_text


WORD_TO_NUMBER = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
}

ROMAN_TO_NUMBER = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
    "xi": "11",
    "xii": "12",
}

AFFIRMATIVE_PATTERN = re.compile(
    r"\b(yes|yeah|yep|sure|ok|okay|please do|call me|contact me|reach out|agreed?|consent|callback)\b",
    re.IGNORECASE,
)

NEGATIVE_PATTERN = re.compile(
    r"\b(no|nope|nah|don'?t call|do not call|no need|cancel|never mind|stop|not now|not yet)\b",
    re.IGNORECASE,
)


def extract_grade(text: str) -> str | None:
    """Extract a standardized grade/class label from conversational user text.

    Examples:
        - 'What is the Grade 8 fee?' -> 'Grade 8'
        - 'Fee for 3 th std' -> 'Grade 3'
        - 'Admissions in LKG' -> 'LKG'
        - 'Looking for class VIII' -> 'Grade 8'
        - 'Enquiring for first standard' -> 'Grade 1'

    Returns:
        Standardized label (e.g. 'Grade 8', 'LKG', 'Pre-KG') or None.
    """
    normalized = text.strip().lower()

    # 1. Early childhood / kindergarten
    pre_kg_match = re.search(r"\b(pre[- ]?kg|pre[- ]?primary|nursery)\b", normalized)
    if pre_kg_match:
        return "Pre-KG"

    kg_match = re.search(r"\b(lkg|ukg)\b", normalized)
    if kg_match:
        return kg_match.group(1).upper()

    # 2. Word numbers: "third standard", "eighth grade"
    for word, num in WORD_TO_NUMBER.items():
        if re.search(rf"\b{word}\s*(?:grade|class|std\.?|standard)\b", normalized) or \
           re.search(rf"\b(?:grade|class|std\.?|standard)\s*{word}\b", normalized):
            return f"Grade {num}"

    # 3. Numeric matches: "Grade 8", "Class 10", "3 th std", "8th standard"
    numeric_match = re.search(
        r"\b(?:grade|class|std\.?|standard)\s*(\d{1,2})\b"
        r"|\b(\d{1,2})\s*(?:st|nd|rd|th|\s*th)?\s*(?:std\.?|standard|class|grade)?\b",
        normalized,
    )
    # Check for direct "Grade X" or "Class 8"
    direct = re.search(r"\b(?:grade|class|std\.?|standard)\s*(\d{1,2})\b", normalized)
    if direct:
        num = int(direct.group(1))
        if 1 <= num <= 12:
            return f"Grade {num}"

    suffixed = re.search(r"\b(\d{1,2})\s*(?:st|nd|rd|th)\s*(?:std\.?|standard|class|grade)?\b", normalized)
    if suffixed:
        num = int(suffixed.group(1))
        if 1 <= num <= 12:
            return f"Grade {num}"

    # Standard / std with number
    std_num = re.search(r"\b(\d{1,2})\s*(?:std\.?|standard)\b", normalized)
    if std_num:
        num = int(std_num.group(1))
        if 1 <= num <= 12:
            return f"Grade {num}"

    # 4. Roman numerals: "Class VIII", "Grade X"
    roman_match = re.search(r"\b(?:grade|class|std\.?|standard)\s*(xii|xi|ix|x|viii|vii|vi|iv|v|iii|ii|i)\b", normalized)
    if roman_match:
        roman = roman_match.group(1)
        num = ROMAN_TO_NUMBER.get(roman)
        if num:
            return f"Grade {num}"

    return None


def is_consent_affirmative(text: str) -> bool:
    """Return True if the text expresses explicit agreement or consent."""
    return bool(AFFIRMATIVE_PATTERN.search(text)) and not bool(NEGATIVE_PATTERN.search(text))


def is_consent_negative(text: str) -> bool:
    """Return True if the text declines or refuses consent."""
    return bool(NEGATIVE_PATTERN.search(text))


def extract_entities(text: str) -> dict[str, str | None]:
    """Extract grade and phone entities from a turn."""
    return {
        "grade": extract_grade(text),
        "phone": extract_phone_from_text(text),
    }

