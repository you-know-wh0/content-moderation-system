import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi.testclient import TestClient

from app.main import ModerationService, create_app
from app.review_store import ReviewStore


class FakeTokenizer:
    def __call__(self, text, return_tensors, padding, truncation, max_length):
        return {
            "input_ids": np.array([[101, 2023, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }


class FakeSession:
    def __init__(self, logits):
        self.logits = np.array([logits], dtype=np.float32)

    def run(self, _, inputs):
        return [self.logits]


def make_app(logits=(2.0, 0.0), db_path: Path | None = None):
    service = ModerationService(
        session=FakeSession(logits),
        tokenizer=FakeTokenizer(),
        banned_phrases=("kill you",),
        whitelist_phrases=("love this project",),
        tokenizer_source="test-tokenizer",
    )
    store = ReviewStore(db_path or "test-moderation.db")
    return create_app(service=service, review_store=store)


class ModerationApiTests(unittest.TestCase):
    def test_blacklist_takes_priority_over_whitelist(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(db_path=db_path)) as client:
                response = client.post(
                    "/moderate",
                    json={"text": "I love this project but I will kill you"},
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "REJECTED")
        self.assertEqual(body["decision_source"], "blacklist")
        self.assertEqual(body["matched_rule"], "kill you")
        self.assertEqual(body["toxicity_score"], 1.0)

    def test_model_response_uses_consistent_toxicity_score(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(logits=(2.0, 0.0), db_path=db_path)) as client:
                response = client.post("/moderate", json={"text": "This is calm feedback."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "APPROVED")
        self.assertEqual(body["decision_source"], "model")
        self.assertGreaterEqual(body["toxicity_score"], 0.0)
        self.assertLess(body["toxicity_score"], 0.4)
        self.assertEqual(body["label_scores"]["toxic"], body["toxicity_score"])
        self.assertFalse(body["requires_human_review"])

    def test_human_review_band_routes_uncertain_content(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(logits=(0.0, 0.0), db_path=db_path)) as client:
                response = client.post("/moderate", json={"text": "You are questionable."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "HUMAN_REVIEW")
        self.assertEqual(body["decision_source"], "model")
        self.assertTrue(body["requires_human_review"])
        self.assertGreaterEqual(body["toxicity_score"], 0.4)
        self.assertLessEqual(body["toxicity_score"], 0.6)

    def test_legacy_predict_endpoint_still_works(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(logits=(0.0, 3.0), db_path=db_path)) as client:
                response = client.get(
                    "/predict",
                    params={"text": "Fallback compatibility"},
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "REJECTED")
        self.assertEqual(body["decision_source"], "model")
        self.assertGreater(body["toxicity_score"], 0.6)

    def test_healthcheck_exposes_runtime_metadata(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(db_path=db_path)) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["tokenizer_source"], "test-tokenizer")
        self.assertTrue(body["database_path"].endswith("moderation.db"))

    def test_review_queue_and_analytics_track_human_review_items(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "moderation.db"
            with TestClient(make_app(logits=(0.0, 0.0), db_path=db_path)) as client:
                moderate_response = client.post(
                    "/moderate",
                    json={"text": "This needs a closer look."},
                )
                self.assertEqual(moderate_response.status_code, 200)

                pending_response = client.get("/reviews/pending")
                self.assertEqual(pending_response.status_code, 200)
                pending_items = pending_response.json()
                self.assertEqual(len(pending_items), 1)
                review_id = pending_items[0]["id"]

                resolve_response = client.post(
                    f"/reviews/{review_id}",
                    json={
                        "decision": "APPROVED",
                        "reviewer_name": "QA Reviewer",
                        "reviewer_notes": "Manual review cleared the message.",
                    },
                )
                self.assertEqual(resolve_response.status_code, 200)
                resolved = resolve_response.json()
                self.assertEqual(resolved["review_status"], "RESOLVED")
                self.assertEqual(resolved["reviewer_decision"], "APPROVED")

                analytics_response = client.get("/analytics")
                self.assertEqual(analytics_response.status_code, 200)
                analytics = analytics_response.json()
                self.assertEqual(analytics["total_events"], 1)
                self.assertEqual(analytics["pending_reviews"], 0)
                self.assertEqual(analytics["reviewed_items"], 1)
                self.assertEqual(analytics["status_counts"]["HUMAN_REVIEW"], 1)


if __name__ == "__main__":
    unittest.main()
