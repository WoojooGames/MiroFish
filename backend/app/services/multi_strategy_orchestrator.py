"""
Multi-Strategy Orchestrator
선택된 전략(단타/스윙/장기)에 대한 병렬 시뮬레이션 실행 및 비교 분석 요약 생성
"""

import os
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..models.project import ProjectManager
from ..models.multi_strategy_analysis import (
    MultiStrategyAnalysis, StrategyState, MultiStrategyManager,
    AnalysisStatus, StrategyStatus, STRATEGY_LABELS,
)
from .simulation_manager import SimulationManager, SimulationStatus
from .simulation_runner import SimulationRunner, RunnerStatus
from .report_agent import ReportAgent, ReportManager, ReportStatus
from .stock_data_provider import StockDataProvider

logger = get_logger('mirofish.multi_strategy')


# 전략별 프리셋 설정
STRATEGY_PRESETS: Dict[str, Dict[str, Any]] = {
    "short_term": {
        "label": "Short-Term / 단타",
        "requirement_suffix": (
            "\n\n[투자 전략: 단타 (1-3일)]\n"
            "1-3일 단기 트레이딩 관점에서 분석하세요. "
            "일중 가격 패턴, 기술적 지표(RSI, MACD, 볼린저밴드), "
            "거래량 급증, 모멘텀 신호를 중점적으로 분석하세요. "
            "매수/매도 타이밍, 목표가, 손절가를 제시하세요."
        ),
        "time_override": {
            "total_simulation_hours": 48,
            "minutes_per_round": 30,
        },
        "max_rounds": 96,  # 48h / 30min
    },
    "swing": {
        "label": "Swing / 스윙",
        "requirement_suffix": (
            "\n\n[투자 전략: 스윙 (1-3주)]\n"
            "1-3주 스윙 트레이딩 관점에서 분석하세요. "
            "추세 패턴, 지지/저항, 섹터 로테이션, "
            "기관 자금 흐름, 실적 모멘텀을 중점적으로 분석하세요. "
            "매수/매도 타이밍, 목표가, 손절가를 제시하세요."
        ),
        "time_override": {
            "total_simulation_hours": 336,
            "minutes_per_round": 60,
        },
        "max_rounds": 200,
    },
    "long_term": {
        "label": "Long-Term / 장기",
        "requirement_suffix": (
            "\n\n[투자 전략: 장기 (3-6개월)]\n"
            "3-6개월 장기 투자 관점에서 분석하세요. "
            "펀더멘털 밸류에이션, 거시경제 트렌드, "
            "경쟁 포지셔닝, 실적 성장 궤적, 장기 촉매를 중점적으로 분석하세요. "
            "목표가, 적정 매수 구간, 리스크 요인을 제시하세요."
        ),
        "time_override": {
            "total_simulation_hours": 2880,
            "minutes_per_round": 240,
        },
        "max_rounds": 150,
    },
}

VALID_STRATEGIES = set(STRATEGY_PRESETS.keys())


class MultiStrategyOrchestrator:
    """멀티 전략 병렬 시뮬레이션 오케스트레이터"""

    # 시뮬레이션 완료 체크 간격(초)
    POLL_INTERVAL = 10
    # 시뮬레이션 최대 대기 시간(초)
    MAX_WAIT_TIME = 3600  # 1시간

    def __init__(self):
        self.sim_manager = SimulationManager()
        self.stock_provider = StockDataProvider()

    def run_analysis(
        self,
        analysis: MultiStrategyAnalysis,
        progress_callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> MultiStrategyAnalysis:
        """
        선택된 전략들에 대해 병렬로 시뮬레이션 → 리포트 → 비교 요약을 실행

        Args:
            analysis: 분석 상태 객체 (strategies에 요청된 전략들이 포함)
            progress_callback: 진행 콜백 (strategy_type, info_dict)

        Returns:
            완료된 MultiStrategyAnalysis
        """
        analysis.status = AnalysisStatus.RUNNING
        MultiStrategyManager.save(analysis)

        strategy_types = list(analysis.strategies.keys())
        logger.info(f"멀티 전략 분석 시작: {analysis.analysis_id}, 전략={strategy_types}")

        # 프로젝트 정보 가져오기
        project = ProjectManager.get_project(analysis.project_id)
        if not project:
            analysis.status = AnalysisStatus.FAILED
            analysis.error = f"프로젝트를 찾을 수 없음: {analysis.project_id}"
            MultiStrategyManager.save(analysis)
            return analysis

        base_requirement = project.simulation_requirement or ""
        document_text = ProjectManager.get_extracted_text(analysis.project_id) or ""
        graph_id = project.graph_id

        if not graph_id:
            analysis.status = AnalysisStatus.FAILED
            analysis.error = "그래프 ID가 없습니다. 먼저 그래프를 빌드하세요."
            MultiStrategyManager.save(analysis)
            return analysis

        # 병렬 실행
        results: Dict[str, bool] = {}

        def run_single_strategy(strategy_type: str) -> bool:
            try:
                return self._run_strategy_pipeline(
                    analysis=analysis,
                    strategy_type=strategy_type,
                    graph_id=graph_id,
                    base_requirement=base_requirement,
                    document_text=document_text,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.error(f"전략 {strategy_type} 실행 실패: {e}")
                ss = analysis.strategies[strategy_type]
                ss.status = StrategyStatus.FAILED
                ss.error = str(e)
                MultiStrategyManager.save(analysis)
                return False

        with ThreadPoolExecutor(max_workers=len(strategy_types)) as executor:
            futures = {
                executor.submit(run_single_strategy, st): st
                for st in strategy_types
            }
            for future in as_completed(futures):
                st = futures[future]
                results[st] = future.result()

        # 결과 집계
        completed_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        if completed_count == total_count:
            analysis.status = AnalysisStatus.COMPLETED
        elif completed_count > 0:
            analysis.status = AnalysisStatus.PARTIAL_COMPLETED
        else:
            analysis.status = AnalysisStatus.FAILED
            analysis.error = "모든 전략 실행 실패"

        # 비교 요약 생성 (2개 이상 성공 시)
        if completed_count >= 2:
            try:
                summary = self._generate_comparison_summary(analysis)
                analysis.comparison_summary = summary
            except Exception as e:
                logger.error(f"비교 요약 생성 실패: {e}")

        from datetime import datetime
        analysis.completed_at = datetime.now().isoformat()
        MultiStrategyManager.save(analysis)

        logger.info(
            f"멀티 전략 분석 완료: {analysis.analysis_id}, "
            f"성공={completed_count}/{total_count}"
        )
        return analysis

    def _run_strategy_pipeline(
        self,
        analysis: MultiStrategyAnalysis,
        strategy_type: str,
        graph_id: str,
        base_requirement: str,
        document_text: str,
        progress_callback: Optional[Callable] = None,
    ) -> bool:
        """단일 전략의 전체 파이프라인: create → prepare → start → wait → report"""

        preset = STRATEGY_PRESETS[strategy_type]
        ss = analysis.strategies[strategy_type]

        # 전략별 simulation_requirement 조합
        strategy_requirement = base_requirement + preset["requirement_suffix"]

        # 주가 데이터 조회 & 주입
        try:
            stock_context = self.stock_provider.get_stock_context(
                ticker=analysis.ticker,
                strategy_type=strategy_type,
            )
            if stock_context:
                strategy_requirement += f"\n\n{stock_context}"
                logger.info(f"[{strategy_type}] 주가 데이터 컨텍스트 주입 완료 ({len(stock_context)}자)")
        except Exception as e:
            logger.warning(f"[{strategy_type}] 주가 데이터 조회 실패 (계속 진행): {e}")

        # 1. 시뮬레이션 생성
        ss.status = StrategyStatus.CREATING
        ss.message = "시뮬레이션 생성 중..."
        MultiStrategyManager.save(analysis)

        state = self.sim_manager.create_simulation(
            project_id=analysis.project_id,
            graph_id=graph_id,
            enable_twitter=True,
            enable_reddit=True,
        )
        ss.simulation_id = state.simulation_id

        # 2. 시뮬레이션 준비
        ss.status = StrategyStatus.PREPARING
        ss.message = "시뮬레이션 준비 중 (엔티티 로드 & 프로필 생성)..."
        ss.progress = 10
        MultiStrategyManager.save(analysis)

        self.sim_manager.prepare_simulation(
            simulation_id=state.simulation_id,
            simulation_requirement=strategy_requirement,
            document_text=document_text,
            time_override=preset.get("time_override"),
        )

        # 3. 시뮬레이션 시작
        ss.status = StrategyStatus.RUNNING
        ss.message = "시뮬레이션 실행 중..."
        ss.progress = 30
        MultiStrategyManager.save(analysis)

        SimulationRunner.start_simulation(
            simulation_id=state.simulation_id,
            platform="parallel",
            max_rounds=preset.get("max_rounds"),
        )

        # 4. 시뮬레이션 완료 대기
        if not self._wait_for_simulation_complete(state.simulation_id, ss, analysis):
            return False

        # 5. 리포트 생성
        ss.status = StrategyStatus.GENERATING_REPORT
        ss.message = "리포트 생성 중..."
        ss.progress = 70
        MultiStrategyManager.save(analysis)

        report_id = f"report_{uuid.uuid4().hex[:12]}"
        ss.report_id = report_id

        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=state.simulation_id,
            simulation_requirement=strategy_requirement,
        )
        report = agent.generate_report(report_id=report_id)
        ReportManager.save_report(report)

        if report.status != ReportStatus.COMPLETED:
            ss.status = StrategyStatus.FAILED
            ss.error = report.error or "리포트 생성 실패"
            ss.progress = 0
            MultiStrategyManager.save(analysis)
            return False

        # 6. 완료
        ss.status = StrategyStatus.COMPLETED
        ss.message = "완료"
        ss.progress = 100
        MultiStrategyManager.save(analysis)
        return True

    def _wait_for_simulation_complete(
        self,
        simulation_id: str,
        ss: StrategyState,
        analysis: MultiStrategyAnalysis,
    ) -> bool:
        """시뮬레이션 프로세스 완료를 폴링으로 대기"""
        elapsed = 0
        while elapsed < self.MAX_WAIT_TIME:
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

            run_state = SimulationRunner.get_run_state(simulation_id)
            if not run_state:
                ss.status = StrategyStatus.FAILED
                ss.error = "시뮬레이션 상태를 확인할 수 없음"
                MultiStrategyManager.save(analysis)
                return False

            status = run_state.runner_status
            if status == RunnerStatus.COMPLETED:
                ss.progress = 65
                ss.message = "시뮬레이션 완료"
                MultiStrategyManager.save(analysis)
                return True
            elif status in (RunnerStatus.FAILED, RunnerStatus.STOPPED):
                ss.status = StrategyStatus.FAILED
                ss.error = f"시뮬레이션 {status.value}"
                MultiStrategyManager.save(analysis)
                return False

            # 진행률 업데이트
            if run_state.total_rounds > 0:
                sim_progress = min(
                    60,
                    30 + int(run_state.current_round / run_state.total_rounds * 35)
                )
                ss.progress = sim_progress
                ss.message = f"시뮬레이션 라운드 {run_state.current_round}/{run_state.total_rounds}"
                MultiStrategyManager.save(analysis)

        # 타임아웃
        ss.status = StrategyStatus.FAILED
        ss.error = f"시뮬레이션 타임아웃 ({self.MAX_WAIT_TIME}초)"
        MultiStrategyManager.save(analysis)
        return False

    def _generate_comparison_summary(self, analysis: MultiStrategyAnalysis) -> str:
        """완료된 리포트들을 비교 분석하는 요약 생성"""

        # 완료된 전략의 리포트 수집
        reports_text = []
        for st, ss in analysis.strategies.items():
            if ss.status != StrategyStatus.COMPLETED or not ss.report_id:
                continue
            report = ReportManager.get_report(ss.report_id)
            if report and report.markdown_content:
                reports_text.append(
                    f"=== {ss.label} 전략 리포트 ===\n{report.markdown_content}\n"
                )

        if len(reports_text) < 2:
            return ""

        combined = "\n\n".join(reports_text)

        # LLM으로 비교 요약 생성
        client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)
        response = client.chat.completions.create(
            model=Config.LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 전문 투자 애널리스트입니다. "
                        "여러 투자 전략별 분석 리포트를 비교하여 "
                        "종합적인 투자 의견을 마크다운 형식으로 작성하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"다음은 종목 {analysis.ticker}에 대한 "
                        f"전략별 분석 리포트입니다.\n\n{combined}\n\n"
                        "위 리포트들을 비교 분석하여 다음을 포함하는 종합 요약을 작성하세요:\n"
                        "1. 전략별 핵심 의견 비교표 (전략, 투자의견, 목표가, 리스크)\n"
                        "2. 전략 간 공통점과 차이점\n"
                        "3. 현재 시장 상황에서 추천 전략과 그 근거\n"
                        "4. 종합 리스크 요인"
                    ),
                },
            ],
            temperature=0.5,
            max_tokens=4000,
        )

        return response.choices[0].message.content or ""
