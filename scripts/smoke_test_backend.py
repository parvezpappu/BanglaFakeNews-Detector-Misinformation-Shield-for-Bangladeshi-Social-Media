from __future__ import annotations

import json

from app1.backend.predictor import EnsemblePredictor


def main() -> None:
    predictor = EnsemblePredictor()
    result = predictor.predict(
        category="National",
        headline="ঢাকায় নতুন প্রকল্প উদ্বোধন করলেন প্রধানমন্ত্রী",
        content="সরকারি সূত্রে জানা গেছে, রাজধানীতে একটি নতুন অবকাঠামো প্রকল্পের উদ্বোধন করা হয়েছে।",
    )
    print(
        json.dumps(
            {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "branch_probabilities": result.branch_probabilities,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
