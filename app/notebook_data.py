from __future__ import annotations

from app.evaluation import EvaluationExample


DEFAULT_EVAL_SET = [
    EvaluationExample(question="What problem is the document solving?"),
    EvaluationExample(question="What method or algorithm is proposed?"),
    EvaluationExample(question="Which limitations are discussed?"),
]
