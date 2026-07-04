from datetime import UTC, datetime
from typing import Any, Dict, List

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mrds.infrastructure.db.models.base import Base


class RunModel(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    triggered_by: Mapped[str] = mapped_column(String(255))
    environment: Mapped[str] = mapped_column(String(255))
    git_commit_hash: Mapped[str] = mapped_column(String(40), nullable=True)

    evaluations: Mapped[List["EvaluationModel"]] = relationship(
        "EvaluationModel", back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationModel(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)

    # Nested configurations and metadata stored as JSON to prevent over-normalization
    prompt_config: Mapped[Dict[str, Any]] = mapped_column(JSON)
    response_raw_text: Mapped[str] = mapped_column(String)
    response_token_usage: Mapped[Dict[str, Any]] = mapped_column(JSON)
    response_latency: Mapped[Dict[str, Any]] = mapped_column(JSON)
    response_finish_reason: Mapped[str] = mapped_column(String(50), nullable=True)

    run: Mapped["RunModel"] = relationship("RunModel", back_populates="evaluations")
    scores: Mapped[List["ScoreModel"]] = relationship(
        "ScoreModel", back_populates="evaluation", cascade="all, delete-orphan"
    )


class ScoreModel(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[float] = mapped_column(Float)
    max_value: Mapped[float] = mapped_column(Float)

    evaluation: Mapped["EvaluationModel"] = relationship("EvaluationModel", back_populates="scores")
