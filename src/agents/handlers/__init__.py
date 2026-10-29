"""Modular agent handlers package."""

from src.agents.handlers.admissions import handle_admissions_process
from src.agents.handlers.contact import handle_contact_request
from src.agents.handlers.fees import handle_fee_enquiry
from src.agents.handlers.general import handle_out_of_scope, handle_small_talk, handle_time_query
from src.agents.handlers.knowledge import handle_knowledge_query
from src.agents.handlers.lead import handle_lead_callback
from src.agents.handlers.visit import handle_school_visit

__all__ = [
    "handle_admissions_process",
    "handle_contact_request",
    "handle_fee_enquiry",
    "handle_knowledge_query",
    "handle_lead_callback",
    "handle_out_of_scope",
    "handle_school_visit",
    "handle_small_talk",
    "handle_time_query",
]

