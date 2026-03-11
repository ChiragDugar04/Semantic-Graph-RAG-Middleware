"""
tests/test_entity_extractor.py

Phase 2 test suite for entity_extractor.py

Run with:
    pytest tests/test_entity_extractor.py -v

These tests use ONLY the rule-based tier (Tier 1) so they run
instantly without needing Ollama running. The LLM tier is tested
separately in the end-to-end pipeline test.
"""

from __future__ import annotations

import pytest
from middleware.entity_extractor import extract_entities
from middleware.models import EntityExtractionResult


# ============================================================
# HELPER
# ============================================================

def extract(q: str) -> EntityExtractionResult:
    """Shorthand for extract_entities()."""
    return extract_entities(q)


# ============================================================
# TEST 1 — Single employee name detection
# ============================================================

class TestSingleEmployee:

    def test_sarah_connor_salary(self) -> None:
        r = extract("What is Sarah Connor's salary?")
        assert "Employee" in r.entities
        assert r.filters.get("employee_name", "").lower() == "sarah connor"
        assert r.question_type in ("lookup", "cross_entity")

    def test_alan_turing_role(self) -> None:
        r = extract("What role does Alan Turing have?")
        assert "Employee" in r.entities
        assert "alan turing" in r.filters.get("employee_name", "").lower()

    def test_grace_hopper_earnings(self) -> None:
        r = extract("How much does Grace Hopper earn?")
        assert "Employee" in r.entities
        assert "grace hopper" in r.filters.get("employee_name", "").lower()

    def test_salary_in_projections(self) -> None:
        r = extract("What is Sarah Connor's salary?")
        assert "salary" in r.projections


# ============================================================
# TEST 2 — Department detection
# ============================================================

class TestDepartmentDetection:

    def test_engineering_department(self) -> None:
        r = extract("What is the Engineering department budget?")
        assert "Department" in r.entities
        assert r.filters.get("department_name", "").lower() == "engineering"

    def test_marketing_list(self) -> None:
        r = extract("List all employees in Marketing")
        assert "Employee" in r.entities
        assert "Department" in r.entities
        assert r.question_type == "list"

    def test_hr_normalized(self) -> None:
        r = extract("Who works in HR?")
        assert "Department" in r.entities
        dept = r.filters.get("department_name", "")
        assert dept in ("Human Resources", "Hr")

    def test_sales_employees(self) -> None:
        r = extract("Show me everyone in Sales")
        assert "Department" in r.entities
        assert r.filters.get("department_name", "").lower() == "sales"


# ============================================================
# TEST 3 — Question type detection
# ============================================================

class TestQuestionTypeDetection:

    def test_comparison_highest_paid(self) -> None:
        r = extract("Who is the highest paid in Engineering?")
        assert r.question_type == "comparison"

    def test_comparison_most_expensive(self) -> None:
        r = extract("What is the most expensive product?")
        assert r.question_type == "comparison"

    def test_list_type(self) -> None:
        r = extract("List all employees in Sales")
        assert r.question_type == "list"

    def test_aggregation_how_many(self) -> None:
        r = extract("How many employees are in each department?")
        assert r.question_type == "aggregation"

    def test_lookup_single_person(self) -> None:
        r = extract("What is Alan Turing's salary?")
        assert r.question_type in ("lookup", "cross_entity")


# ============================================================
# TEST 4 — Cross-entity detection
# ============================================================

class TestCrossEntityDetection:

    def test_employee_and_project(self) -> None:
        r = extract("Which employees are assigned to the API Gateway Rebuild project?")
        assert "Employee" in r.entities
        assert "Project" in r.entities
        assert r.question_type == "cross_entity"

    def test_employee_dept_and_manager(self) -> None:
        r = extract("Which employees in Engineering are working on projects managed by Sarah?")
        assert "Employee" in r.entities
        assert r.question_type == "cross_entity"

    def test_two_entities_always_cross(self) -> None:
        r = extract("Show all Engineering employees on the Platform Migration project")
        assert len(set(r.entities)) >= 2
        assert r.question_type == "cross_entity"


# ============================================================
# TEST 5 — Product detection
# ============================================================

class TestProductDetection:

    def test_laptop_stock(self) -> None:
        r = extract("How many Laptop Pro 15 units are in stock?")
        assert "Product" in r.entities
        assert "laptop pro 15" in r.filters.get("product_name", "").lower()

    def test_electronics_category(self) -> None:
        r = extract("Show all Electronics products")
        assert "Product" in r.entities
        assert r.filters.get("category", "").lower() in ("electronics", "Electronics".lower())

    def test_most_expensive_comparison(self) -> None:
        r = extract("What is the most expensive product?")
        assert "Product" in r.entities
        assert r.question_type == "comparison"


# ============================================================
# TEST 6 — Order detection
# ============================================================

class TestOrderDetection:

    def test_pending_orders(self) -> None:
        r = extract("Show me all pending orders")
        assert "Order" in r.entities
        assert r.filters.get("order_status") == "pending"

    def test_shipped_orders(self) -> None:
        r = extract("List shipped orders")
        assert "Order" in r.entities
        assert r.filters.get("order_status") == "shipped"

    def test_all_orders_no_status(self) -> None:
        r = extract("Show me all recent orders")
        assert "Order" in r.entities

    def test_in_transit_synonym(self) -> None:
        r = extract("Which orders are in transit?")
        assert "Order" in r.entities
        assert r.filters.get("order_status") == "shipped"


# ============================================================
# TEST 7 — Unknown / other questions
# ============================================================

class TestUnknownQuestions:

    def test_weather_question(self) -> None:
        r = extract("What is the weather today?")
        # Should return no entities (falls to LLM) or empty
        # We just verify it doesn't crash and returns a valid model
        assert isinstance(r, EntityExtractionResult)
        assert isinstance(r.entities, list)

    def test_joke_question(self) -> None:
        r = extract("Tell me a joke")
        assert isinstance(r, EntityExtractionResult)

    def test_no_crash_on_empty(self) -> None:
        r = extract("")
        assert isinstance(r, EntityExtractionResult)


# ============================================================
# TEST 8 — Extraction method tracking
# ============================================================

class TestExtractionMethod:

    def test_rules_used_for_known_employee(self) -> None:
        r = extract("What is Linus Torvalds's salary?")
        assert r.extraction_method == "rules"

    def test_rules_used_for_known_department(self) -> None:
        r = extract("List all employees in Engineering")
        assert r.extraction_method == "rules"

    def test_result_is_correct_type(self) -> None:
        r = extract("What is Sarah Connor's salary?")
        assert isinstance(r, EntityExtractionResult)
        assert isinstance(r.entities, list)
        assert isinstance(r.filters, dict)
        assert isinstance(r.projections, list)
        assert isinstance(r.question_type, str)
        assert isinstance(r.latency_ms, float)