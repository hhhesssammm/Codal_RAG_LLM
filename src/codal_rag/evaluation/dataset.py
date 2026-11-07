from __future__ import annotations

import re
from codal_rag.storage.json_io import load_json, save_json

LEVELS = ("explicit", "implicit", "latent")


def create_question(number: int, question_type: str) -> dict:
    return {"id": f"Q{number:03d}", "question": "", "type": question_type,
            "expected_answer": "", "gold_evidence": [{"document_id": "", "text": ""}], "notes": ""}


def create_template(path, counts=(20, 15, 15)):
    questions = []
    for level, count in zip(LEVELS, counts):
        for _ in range(count):
            questions.append(create_question(len(questions) + 1, level))
    return save_json(questions, path)


def validate_question(question: dict) -> dict:
    if not isinstance(question, dict):
        raise ValueError("Evaluation question must be an object")
    if not isinstance(question.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', question['id']):
        raise ValueError("Question id must contain 1-100 ASCII letters, digits, underscores, or hyphens")
    if question.get('type') not in LEVELS:
        raise ValueError(f"Question {question['id']}: type must be explicit, implicit, or latent")
    for key in ('question', 'expected_answer'):
        if not isinstance(question.get(key), str) or not question[key].strip():
            raise ValueError(f"Question {question['id']}: fill {key} before evaluation")
    gold = question.get('gold_evidence')
    if not isinstance(gold, list):
        raise ValueError("gold_evidence must be a list (empty means retrieval is unscored)")
    for evidence in gold:
        if not isinstance(evidence, dict) or any(not isinstance(evidence.get(key), str) or not evidence[key].strip() for key in ('document_id', 'text')):
            raise ValueError("Gold evidence requires a document_id and a nonempty exact evidence passage")
    return question


def load_questions(path) -> list[dict]:
    questions = load_json(path)
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a nonempty evaluation question list")
    ids = set()
    for question in questions:
        validate_question(question)
        if question['id'] in ids:
            raise ValueError(f"Duplicate question id: {question['id']}")
        ids.add(question['id'])
    return questions
