from __future__ import annotations

import logging
import os
import re
import time
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from huggingface_hub import try_to_load_from_cache
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from app.review_store import ReviewStore


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "toxic_model.onnx"))
DATABASE_PATH = Path(os.getenv("MODERATION_DB_PATH", BASE_DIR / "moderation.db"))
TOKENIZER_SOURCE = os.getenv("TOKENIZER_PATH") or os.getenv(
    "TOKENIZER_NAME", "distilbert-base-uncased"
)
TOKENIZER_LOCAL_ONLY = os.getenv("TOKENIZER_LOCAL_ONLY", "false").lower() == "true"
LOWER_REVIEW_THRESHOLD = float(os.getenv("LOWER_REVIEW_THRESHOLD", "0.40"))
UPPER_REVIEW_THRESHOLD = float(os.getenv("UPPER_REVIEW_THRESHOLD", "0.60"))
BANNED_PHRASES = tuple(
    phrase.strip()
    for phrase in os.getenv(
        "BANNED_PHRASES", "racialslur1,beat you up,kill you,punch you"
    ).split(",")
    if phrase.strip()
)
WHITELIST_PHRASES = tuple(
    phrase.strip()
    for phrase in os.getenv(
        "WHITELIST_PHRASES",
        "love this project,machine learning,awesome,great",
    ).split(",")
    if phrase.strip()
)
MODERATION_API_KEY = os.getenv("MODERATION_API_KEY", "").strip()


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

logger = logging.getLogger("content_moderation.api")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    normalized = normalize_text(phrase)
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def tokenizer_available_locally(source: str) -> bool:
    source_path = Path(source)
    if source_path.exists():
        return True

    cached_file = try_to_load_from_cache(source, "tokenizer_config.json")
    return isinstance(cached_file, str) and Path(cached_file).exists()


class ModerationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User supplied text that should be moderated.",
    )


class LabelScores(BaseModel):
    non_toxic: float = Field(..., ge=0.0, le=1.0)
    toxic: float = Field(..., ge=0.0, le=1.0)


class ModerationResponse(BaseModel):
    text: str
    status: Literal["APPROVED", "REJECTED", "HUMAN_REVIEW"]
    decision_source: Literal["blacklist", "whitelist", "model"]
    toxicity_score: float = Field(..., ge=0.0, le=1.0)
    ai_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human_review: bool
    matched_rule: str | None = None
    label_scores: LabelScores
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    database_path: str
    tokenizer_source: str
    review_thresholds: dict[str, float]


class AnalyticsResponse(BaseModel):
    total_events: int
    average_latency_ms: float
    human_review_rate: float
    pending_reviews: int
    reviewed_items: int
    status_counts: dict[str, int]
    decision_source_counts: dict[str, int]


class ReviewQueueItem(BaseModel):
    id: int
    text: str
    status: Literal["HUMAN_REVIEW"]
    toxicity_score: float = Field(..., ge=0.0, le=1.0)
    created_at: str
    review_status: Literal["PENDING", "RESOLVED"]
    reviewer_decision: str | None = None
    reviewer_name: str | None = None
    reviewer_notes: str | None = None
    reviewed_at: str | None = None


class ReviewDecisionRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    reviewer_name: str | None = Field(default=None, max_length=120)
    reviewer_notes: str | None = Field(default=None, max_length=1000)


@dataclass(frozen=True)
class PhraseMatcher:
    phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_patterns",
            tuple((phrase, _compile_phrase_pattern(phrase)) for phrase in self.phrases),
        )

    def match(self, text: str) -> str | None:
        normalized = normalize_text(text)
        for phrase, pattern in self._patterns:
            if pattern.search(normalized):
                return phrase
        return None


class ModerationService:
    def __init__(
        self,
        *,
        session: Any,
        tokenizer: Any,
        banned_phrases: tuple[str, ...] = BANNED_PHRASES,
        whitelist_phrases: tuple[str, ...] = WHITELIST_PHRASES,
        lower_review_threshold: float = LOWER_REVIEW_THRESHOLD,
        upper_review_threshold: float = UPPER_REVIEW_THRESHOLD,
        tokenizer_source: str = TOKENIZER_SOURCE,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.tokenizer_source = tokenizer_source
        self.lower_review_threshold = lower_review_threshold
        self.upper_review_threshold = upper_review_threshold
        self.blacklist = PhraseMatcher(tuple(banned_phrases))
        self.whitelist = PhraseMatcher(tuple(whitelist_phrases))

    @classmethod
    def from_artifacts(cls) -> "ModerationService":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Expected ONNX model at {MODEL_PATH}, but the file is missing."
            )

        providers = cls._preferred_providers()
        local_only = TOKENIZER_LOCAL_ONLY or tokenizer_available_locally(TOKENIZER_SOURCE)
        logger.info("Loading ONNX model from %s with providers=%s", MODEL_PATH, providers)
        session = ort.InferenceSession(MODEL_PATH.as_posix(), providers=providers)
        logger.info(
            "Loading tokenizer from %s (local_only=%s)",
            TOKENIZER_SOURCE,
            local_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_SOURCE,
            local_files_only=local_only,
        )

        return cls(
            session=session,
            tokenizer=tokenizer,
            tokenizer_source=TOKENIZER_SOURCE,
        )

    @staticmethod
    def _preferred_providers() -> list[str]:
        available = ort.get_available_providers()
        requested = os.getenv("ORT_PROVIDERS")
        if requested:
            preferred = [provider.strip() for provider in requested.split(",") if provider.strip()]
            configured = [provider for provider in preferred if provider in available]
            if configured:
                return configured

        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        return available

    def moderate(self, text: str) -> ModerationResponse:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        start = time.perf_counter()

        matched_blacklist = self.blacklist.match(cleaned_text)
        if matched_blacklist:
            return self._build_rule_response(
                text=cleaned_text,
                status="REJECTED",
                decision_source="blacklist",
                toxicity_score=1.0,
                matched_rule=matched_blacklist,
                start=start,
            )

        matched_whitelist = self.whitelist.match(cleaned_text)
        if matched_whitelist:
            return self._build_rule_response(
                text=cleaned_text,
                status="APPROVED",
                decision_source="whitelist",
                toxicity_score=0.0,
                matched_rule=matched_whitelist,
                start=start,
            )

        probabilities = self.predict_probabilities(cleaned_text)
        toxicity_score = float(probabilities["toxic"])
        non_toxic_score = float(probabilities["non_toxic"])
        ai_confidence = float(max(toxicity_score, non_toxic_score))
        requires_human_review = (
            self.lower_review_threshold <= toxicity_score <= self.upper_review_threshold
        )

        if requires_human_review:
            status = "HUMAN_REVIEW"
        elif toxicity_score > self.upper_review_threshold:
            status = "REJECTED"
        else:
            status = "APPROVED"

        return ModerationResponse(
            text=cleaned_text,
            status=status,
            decision_source="model",
            toxicity_score=toxicity_score,
            ai_confidence=ai_confidence,
            requires_human_review=requires_human_review,
            matched_rule=None,
            label_scores=LabelScores(
                non_toxic=non_toxic_score,
                toxic=toxicity_score,
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )

    def predict_probabilities(self, text: str) -> dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        logits = np.asarray(outputs[0], dtype=np.float64)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return {
            "non_toxic": float(probabilities[0][0]),
            "toxic": float(probabilities[0][1]),
        }

    def _build_rule_response(
        self,
        *,
        text: str,
        status: Literal["APPROVED", "REJECTED"],
        decision_source: Literal["blacklist", "whitelist"],
        toxicity_score: float,
        matched_rule: str,
        start: float,
    ) -> ModerationResponse:
        non_toxic_score = 1.0 - toxicity_score
        return ModerationResponse(
            text=text,
            status=status,
            decision_source=decision_source,
            toxicity_score=toxicity_score,
            ai_confidence=1.0,
            requires_human_review=False,
            matched_rule=matched_rule,
            label_scores=LabelScores(
                non_toxic=non_toxic_score,
                toxic=toxicity_score,
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )


def get_service(request: Request) -> ModerationService:
    service = getattr(request.app.state, "moderation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Moderation service is not ready.")
    return service


def get_review_store(request: Request) -> ReviewStore:
    review_store = getattr(request.app.state, "review_store", None)
    if review_store is None:
        raise HTTPException(status_code=503, detail="Review store is not ready.")
    return review_store


def _extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()

    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()

    return None


def require_api_key(request: Request) -> None:
    if not MODERATION_API_KEY:
        return

    provided_key = _extract_api_key(request)
    if not provided_key or not hmac.compare_digest(provided_key, MODERATION_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def create_app(
    service: ModerationService | None = None,
    review_store: ReviewStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.review_store = review_store or ReviewStore(DATABASE_PATH)
        app.state.review_store.initialize()
        if service is not None:
            app.state.moderation_service = service
            yield
            return

        app.state.moderation_service = ModerationService.from_artifacts()
        yield

    app = FastAPI(
        title="Real-Time Toxicity Guard",
        version="2.0.0",
        description=(
            "Hybrid content moderation API with deterministic rules, an ONNX "
            "DistilBERT classifier, and a human-review band for uncertain cases."
        ),
        lifespan=lifespan,
    )

    @app.get("/", tags=["system"])
    def home() -> dict[str, str]:
        return {
            "message": "Moderation API is online.",
            "docs": "/docs",
            "health": "/health",
            "primary_endpoint": "/moderate",
        }

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def healthcheck(
        moderation_service: ModerationService = Depends(get_service),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            database_path=review_store.db_path,
            tokenizer_source=moderation_service.tokenizer_source,
            review_thresholds={
                "lower": moderation_service.lower_review_threshold,
                "upper": moderation_service.upper_review_threshold,
            },
        )

    @app.get("/analytics", response_model=AnalyticsResponse, tags=["system"])
    def analytics(
        _: None = Depends(require_api_key),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> AnalyticsResponse:
        return AnalyticsResponse(**review_store.analytics())

    @app.post("/moderate", response_model=ModerationResponse, tags=["moderation"])
    def moderate(
        payload: ModerationRequest,
        _: None = Depends(require_api_key),
        moderation_service: ModerationService = Depends(get_service),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> ModerationResponse:
        result = moderation_service.moderate(payload.text)
        review_store.record_event(result.model_dump())
        return result

    @app.get(
        "/predict",
        response_model=ModerationResponse,
        deprecated=True,
        tags=["moderation"],
    )
    def predict(
        text: str = Query(..., min_length=1, max_length=5000),
        _: None = Depends(require_api_key),
        moderation_service: ModerationService = Depends(get_service),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> ModerationResponse:
        result = moderation_service.moderate(text)
        review_store.record_event(result.model_dump())
        return result

    @app.get(
        "/reviews/pending",
        response_model=list[ReviewQueueItem],
        tags=["reviews"],
    )
    def pending_reviews(
        limit: int = Query(50, ge=1, le=200),
        _: None = Depends(require_api_key),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> list[ReviewQueueItem]:
        rows = review_store.pending_reviews(limit=limit)
        return [ReviewQueueItem(**row) for row in rows]

    @app.post(
        "/reviews/{review_id}",
        response_model=ReviewQueueItem,
        tags=["reviews"],
    )
    def resolve_review(
        review_id: int,
        payload: ReviewDecisionRequest,
        _: None = Depends(require_api_key),
        review_store: ReviewStore = Depends(get_review_store),
    ) -> ReviewQueueItem:
        row = review_store.submit_review(
            review_id,
            decision=payload.decision,
            reviewer_name=payload.reviewer_name,
            reviewer_notes=payload.reviewer_notes,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Review item not found.")
        return ReviewQueueItem(**row)

    return app


app = create_app()
