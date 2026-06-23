"""The 15 features used by both predictive models (COPD and ALT)."""
from __future__ import annotations

FEATURE_SPEC: dict[str, dict] = {
    "age":                    {"kind": "numeric"},
    "sex":                    {"kind": "categorical", "ordered": False, "choices": ["Female", "Male"]},
    "bmi":                    {"kind": "numeric"},
    "smoker":                 {"kind": "categorical", "ordered": False, "choices": ["No", "Yes"]},
    "diagnosis_code":         {"kind": "categorical", "ordered": False, "choices": ["D1", "D2", "D3", "D4", "D5"]},
    "medication_count":       {"kind": "numeric"},
    "days_hospitalized":      {"kind": "numeric"},
    "readmitted":             {"kind": "binary", "choices": [0, 1]},
    "last_lab_glucose":       {"kind": "numeric"},
    # exercise_frequency / education_level have empty cells in the CSV that
    # are MISSING DATA, not a "None" category. Choices list real categories
    # only; missing rows get imputed at predict time (median/mode).
    "exercise_frequency":     {"kind": "categorical", "ordered": True,  "choices": ["Low", "Moderate", "High"]},
    "diet_quality":           {"kind": "categorical", "ordered": True,  "choices": ["Poor", "Average", "Good"]},
    "income_bracket":         {"kind": "categorical", "ordered": True,  "choices": ["Low", "Middle", "High"]},
    "education_level":        {"kind": "categorical", "ordered": True,  "choices": ["Primary", "Secondary", "Tertiary"]},
    "urban":                  {"kind": "binary", "choices": [0, 1]},
    "albumin_globulin_ratio": {"kind": "numeric"},
}

FEATURE_NAMES: list[str] = list(FEATURE_SPEC.keys())

TARGET_NAMES: list[str] = ["chronic_opd", "alt"]  # short aliases used in code paths
TARGET_CSV_COLUMNS: dict[str, str] = {
    "chronic_opd": "chronic_obstructive_pulmonary_disease",
    "alt": "alanine_aminotransferase",
}
