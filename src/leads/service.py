"""Consent-based lead capture and storage service for admissions callback requests."""

import json
import re
import threading
from datetime import datetime, timezone
import uuid
from pathlib import Path

from src.config import INDIAN_PHONE_PATTERN, LEADS_FILE
from src.leads.schemas import LeadRecord
from src.utils.logging import log_error, log_step

_lead_lock = threading.Lock()


def clean_phone_number(raw_phone: str) -> str | None:
    """Validate and extract a standard 10-digit Indian phone number from free-form text.

    Args:
        raw_phone: User supplied phone string (e.g. '+91 9880012345', '98800 12345')

    Returns:
        10-digit normalized phone string (e.g. '9880012345') or None if invalid.
    """
    digits = re.sub(r"\D", "", raw_phone)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    if len(digits) == 10 and digits[0] in "6789":
        return digits
    return None


def extract_phone_from_text(text: str) -> str | None:
    """Search for a potential 10-digit mobile number embedded within a user sentence."""
    matches = re.findall(r"(?:(?:\+91|91|0)[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", text)
    for match in matches:
        cleaned = clean_phone_number(match)
        if cleaned:
            return cleaned
    return None


def record_lead(
    phone: str,
    grade: str | None = None,
    enquiry_type: str = "admission",
    consent_granted: bool = True,
    destination_file: Path = LEADS_FILE,
) -> LeadRecord | None:
    """Store a callback enquiry record with explicit parent consent.

    In compliance with Invariant D3, chat transcripts or raw messages are strictly
    NEVER stored in this record or anywhere on disk.

    Args:
        phone: Validated 10-digit contact number
        grade: Student grade/class (e.g., 'Grade 8')
        enquiry_type: Type of enquiry (e.g., 'fee', 'admission', 'visit')
        consent_granted: Explicit user consent confirmation
        destination_file: JSONL destination path

    Returns:
        The created LeadRecord or None if consent was not granted or saving failed.
    """
    if not consent_granted:
        log_step("LEAD_REJECTED_NO_CONSENT", enquiry_type=enquiry_type)
        return None

    cleaned_phone = clean_phone_number(phone)
    if not cleaned_phone:
        log_step("LEAD_REJECTED_INVALID_PHONE", enquiry_type=enquiry_type)
        return None

    now_iso = datetime.now(timezone.utc).isoformat()
    record = LeadRecord(
        lead_id=f"lead_{uuid.uuid4().hex[:10]}",
        timestamp=now_iso,
        grade=grade,
        phone=cleaned_phone,
        enquiry_type=enquiry_type,
        consent_granted=True,
        consent_timestamp=now_iso,
    )

    try:
        with _lead_lock:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            with open(destination_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump()) + "\n")
        log_step("LEAD_SAVED", lead_id=record.lead_id, grade=grade or "unspecified")
        return record
    except Exception as exc:
        log_error("LEAD_PERSIST_ERROR", exc)
        return None

