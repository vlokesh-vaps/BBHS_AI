"""Conversation-quality regression tests for the BBHS school assistant."""

from src.core.prompts import classify_query, is_grade_fee_enquiry, missing_information_response
from src.core.rag import _retrieval_query


def test_greeting_is_small_talk():
    assert classify_query("Hi there, can you help me?") == "small_talk"


def test_misspelled_admission_query_is_expanded_for_search():
    query = _retrieval_query("What is the admisson fee?", [])
    assert "admission" in query
    assert "fee structure" in query


def test_fee_fallback_is_friendly_and_asks_for_class():
    answer = missing_information_response("I want to know the fee details")
    assert "Which class" in answer
    assert "+91 81479 23880" not in answer
    assert "retrieval" not in answer.lower()


def test_contact_details_are_given_when_requested():
    answer = missing_information_response("What phone number can I call for admissions?")
    assert "8147883880" in answer
    assert "+91 81479 23880" not in answer
    assert "admission@baldwinboyshighschool.edu.in" in answer


def test_class_fee_question_has_personalized_next_steps():
    answer = missing_information_response("Planning for Grade 8 and want the fee details")
    assert "**Enquiring about Grade 8 at BBHS?**" in answer
    assert "8147883880" in answer
    assert "+91 81479 23880" not in answer
    assert "Learn about the admission process" in answer
    assert "Explore a school visit" in answer


def test_informal_third_standard_fee_question_uses_exact_grade():
    question = "i want to know the fee the school the for 3 th std"
    assert is_grade_fee_enquiry(question)
    answer = missing_information_response(question + " fee")
    assert "Grade 3" in answer
    assert "Grade 8" not in answer
    assert "8147883880" in answer
    assert "080-49363866" not in answer
    assert "₹700" not in answer


def test_visit_fallback_does_not_invent_visiting_hours():
    answer = missing_information_response("Can I visit the campus?")
    assert "visiting hours" in answer
    assert "confirm" in answer.lower()
    assert "8:15" not in answer


def test_follow_up_keeps_previous_user_topic():
    query = _retrieval_query(
        "What about the timings?",
        [{"role": "user", "content": "Tell me about Grade 5"}],
    )
    assert "Tell me about Grade 5" in query
    assert "school timings" in query
