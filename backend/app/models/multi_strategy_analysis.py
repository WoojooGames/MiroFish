"""
Multi-Strategy Analysis 데이터 모델
단타/스윙/장기 등 복수 전략 병렬 시뮬레이션 & 비교 분석 상태 관리
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..config import Config


class AnalysisStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PARTIAL_COMPLETED = "partial_completed"
    COMPLETED = "completed"
    FAILED = "failed"


class StrategyStatus(str, Enum):
    PENDING = "pending"
    CREATING = "creating"
    PREPARING = "preparing"
    RUNNING = "running"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


STRATEGY_LABELS = {
    "short_term": "Short-Term / 단타",
    "swing": "Swing / 스윙",
    "long_term": "Long-Term / 장기",
}


@dataclass
class StrategyState:
    """개별 전략 실행 상태"""
    strategy_type: str
    label: str = ""
    simulation_id: Optional[str] = None
    report_id: Optional[str] = None
    status: StrategyStatus = StrategyStatus.PENDING
    progress: int = 0
    message: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if not self.label:
            self.label = STRATEGY_LABELS.get(self.strategy_type, self.strategy_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "label": self.label,
            "simulation_id": self.simulation_id,
            "report_id": self.report_id,
            "status": self.status.value if isinstance(self.status, StrategyStatus) else self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyState':
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = StrategyStatus(status)
        return cls(
            strategy_type=data["strategy_type"],
            label=data.get("label", ""),
            simulation_id=data.get("simulation_id"),
            report_id=data.get("report_id"),
            status=status,
            progress=data.get("progress", 0),
            message=data.get("message", ""),
            error=data.get("error"),
        )


@dataclass
class MultiStrategyAnalysis:
    """멀티 전략 분석 상태"""
    analysis_id: str
    project_id: str
    ticker: str
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    comparison_summary: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.CREATED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "project_id": self.project_id,
            "ticker": self.ticker,
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "comparison_summary": self.comparison_summary,
            "status": self.status.value if isinstance(self.status, AnalysisStatus) else self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiStrategyAnalysis':
        status = data.get("status", "created")
        if isinstance(status, str):
            status = AnalysisStatus(status)
        strategies = {}
        for k, v in data.get("strategies", {}).items():
            strategies[k] = StrategyState.from_dict(v)
        return cls(
            analysis_id=data["analysis_id"],
            project_id=data["project_id"],
            ticker=data.get("ticker", ""),
            strategies=strategies,
            comparison_summary=data.get("comparison_summary"),
            status=status,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


class MultiStrategyManager:
    """멀티 전략 분석 상태 영속화 관리"""

    ANALYSES_DIR = os.path.join(Config.UPLOAD_FOLDER, 'multi_strategy')

    @classmethod
    def _ensure_dir(cls):
        os.makedirs(cls.ANALYSES_DIR, exist_ok=True)

    @classmethod
    def _get_analysis_dir(cls, analysis_id: str) -> str:
        return os.path.join(cls.ANALYSES_DIR, analysis_id)

    @classmethod
    def _get_meta_path(cls, analysis_id: str) -> str:
        return os.path.join(cls._get_analysis_dir(analysis_id), 'analysis.json')

    @classmethod
    def save(cls, analysis: MultiStrategyAnalysis) -> None:
        analysis_dir = cls._get_analysis_dir(analysis.analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        meta_path = cls._get_meta_path(analysis.analysis_id)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def get(cls, analysis_id: str) -> Optional[MultiStrategyAnalysis]:
        meta_path = cls._get_meta_path(analysis_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return MultiStrategyAnalysis.from_dict(data)

    @classmethod
    def list_analyses(cls, limit: int = 50) -> List[MultiStrategyAnalysis]:
        cls._ensure_dir()
        analyses = []
        if not os.path.exists(cls.ANALYSES_DIR):
            return analyses
        for aid in os.listdir(cls.ANALYSES_DIR):
            if aid.startswith('.'):
                continue
            analysis = cls.get(aid)
            if analysis:
                analyses.append(analysis)
        analyses.sort(key=lambda a: a.created_at, reverse=True)
        return analyses[:limit]
