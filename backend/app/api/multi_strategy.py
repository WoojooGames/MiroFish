"""
Multi-Strategy Analysis API
선택적 전략(단타/스윙/장기) 병렬 시뮬레이션 & 비교 분석 엔드포인트
"""

import uuid
import threading
import traceback
from flask import request, jsonify

from . import multi_strategy_bp
from ..config import Config
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..models.multi_strategy_analysis import (
    MultiStrategyAnalysis, StrategyState, MultiStrategyManager,
    AnalysisStatus, StrategyStatus,
)
from ..services.multi_strategy_orchestrator import (
    MultiStrategyOrchestrator, VALID_STRATEGIES, STRATEGY_PRESETS,
)
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.multi_strategy')


@multi_strategy_bp.route('/analyze', methods=['POST'])
def analyze():
    """
    멀티 전략 투자 분석 시작 (비동기)

    요청(JSON):
        {
            "project_id": "proj_xxxx",           // 필수
            "ticker": "AAPL",                    // 필수, 종목 심볼
            "strategies": ["short_term", "swing"] // 필수, 1개 이상
        }

    strategies 가능한 값: "short_term" (단타), "swing" (스윙), "long_term" (장기)

    반환:
        {
            "success": true,
            "data": {
                "analysis_id": "msa_xxxx",
                "task_id": "task_xxxx",
                "ticker": "AAPL",
                "strategies": ["short_term", "swing"],
                "status": "running"
            }
        }
    """
    try:
        data = request.get_json() or {}

        project_id = data.get('project_id')
        if not project_id:
            return jsonify({"success": False, "error": "project_id가 필요합니다"}), 400

        ticker = data.get('ticker', '').upper()
        if not ticker:
            return jsonify({"success": False, "error": "ticker가 필요합니다"}), 400

        strategies = data.get('strategies', [])
        if not strategies or not isinstance(strategies, list):
            return jsonify({
                "success": False,
                "error": "strategies 배열이 필요합니다 (예: [\"short_term\", \"swing\", \"long_term\"])"
            }), 400

        # 유효성 검사
        invalid = [s for s in strategies if s not in VALID_STRATEGIES]
        if invalid:
            return jsonify({
                "success": False,
                "error": f"유효하지 않은 전략: {invalid}. 가능한 값: {list(VALID_STRATEGIES)}"
            }), 400

        # 프로젝트 확인
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({"success": False, "error": f"프로젝트 없음: {project_id}"}), 404

        if not project.graph_id:
            return jsonify({
                "success": False,
                "error": "그래프가 빌드되지 않았습니다. 먼저 그래프를 빌드하세요."
            }), 400

        # 분석 객체 생성
        analysis_id = f"msa_{uuid.uuid4().hex[:12]}"
        strategy_states = {
            st: StrategyState(strategy_type=st)
            for st in strategies
        }
        analysis = MultiStrategyAnalysis(
            analysis_id=analysis_id,
            project_id=project_id,
            ticker=ticker,
            strategies=strategy_states,
        )
        MultiStrategyManager.save(analysis)

        # 비동기 태스크 생성
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="multi_strategy_analyze",
            metadata={
                "analysis_id": analysis_id,
                "ticker": ticker,
                "strategies": strategies,
            }
        )

        def run_analysis():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="멀티 전략 분석 시작...",
                )
                orchestrator = MultiStrategyOrchestrator()

                def on_progress(strategy_type, info):
                    # 전체 진행률 계산
                    total = len(strategies)
                    done = sum(
                        1 for ss in analysis.strategies.values()
                        if ss.status in (StrategyStatus.COMPLETED, StrategyStatus.FAILED)
                    )
                    overall = int(done / total * 100)
                    task_manager.update_task(
                        task_id,
                        progress=overall,
                        message=f"[{strategy_type}] {info.get('message', '')}",
                        progress_detail={
                            st: ss.to_dict()
                            for st, ss in analysis.strategies.items()
                        },
                    )

                result = orchestrator.run_analysis(
                    analysis=analysis,
                    progress_callback=on_progress,
                )

                if result.status in (AnalysisStatus.COMPLETED, AnalysisStatus.PARTIAL_COMPLETED):
                    task_manager.complete_task(task_id, result={
                        "analysis_id": analysis_id,
                        "status": result.status.value,
                    })
                else:
                    task_manager.fail_task(task_id, result.error or "분석 실패")
            except Exception as e:
                logger.error(f"멀티 전략 분석 실패: {e}")
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "analysis_id": analysis_id,
                "task_id": task_id,
                "ticker": ticker,
                "strategies": strategies,
                "status": "running",
                "message": "멀티 전략 분석이 시작되었습니다. /api/multi-strategy/status 로 진행 상황을 확인하세요.",
            }
        })

    except Exception as e:
        logger.error(f"멀티 전략 분석 요청 실패: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@multi_strategy_bp.route('/status', methods=['POST'])
def status():
    """
    멀티 전략 분석 진행 상황 조회

    요청(JSON):
        {
            "analysis_id": "msa_xxxx"   // 또는
            "task_id": "task_xxxx"
        }

    반환:
        {
            "success": true,
            "data": {
                "analysis_id": "msa_xxxx",
                "status": "running",
                "strategies": {
                    "short_term": { "status": "completed", "progress": 100, ... },
                    "swing": { "status": "running", "progress": 45, ... }
                },
                "task": { ... }
            }
        }
    """
    try:
        data = request.get_json() or {}

        analysis_id = data.get('analysis_id')
        task_id = data.get('task_id')

        if not analysis_id and not task_id:
            return jsonify({
                "success": False,
                "error": "analysis_id 또는 task_id가 필요합니다"
            }), 400

        # task_id로 analysis_id 찾기
        task_data = None
        if task_id:
            task_manager = TaskManager()
            task = task_manager.get_task(task_id)
            if task:
                task_data = task.to_dict()
                if not analysis_id:
                    analysis_id = task.metadata.get('analysis_id')

        if not analysis_id:
            return jsonify({"success": False, "error": "분석을 찾을 수 없습니다"}), 404

        analysis = MultiStrategyManager.get(analysis_id)
        if not analysis:
            return jsonify({"success": False, "error": f"분석 없음: {analysis_id}"}), 404

        result = {
            "analysis_id": analysis.analysis_id,
            "ticker": analysis.ticker,
            "status": analysis.status.value if isinstance(analysis.status, AnalysisStatus) else analysis.status,
            "strategies": {k: v.to_dict() for k, v in analysis.strategies.items()},
            "created_at": analysis.created_at,
            "completed_at": analysis.completed_at,
        }
        if task_data:
            result["task"] = task_data

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@multi_strategy_bp.route('/<analysis_id>', methods=['GET'])
def get_result(analysis_id: str):
    """
    멀티 전략 분석 결과 조회

    반환:
        {
            "success": true,
            "data": {
                "analysis_id": "msa_xxxx",
                "ticker": "AAPL",
                "status": "completed",
                "strategies": {
                    "short_term": {
                        "status": "completed",
                        "simulation_id": "sim_xxx",
                        "report_id": "report_xxx",
                        "label": "Short-Term / 단타"
                    },
                    ...
                },
                "comparison_summary": "# 비교 분석...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        analysis = MultiStrategyManager.get(analysis_id)
        if not analysis:
            return jsonify({"success": False, "error": f"분석 없음: {analysis_id}"}), 404

        return jsonify({"success": True, "data": analysis.to_dict()})

    except Exception as e:
        logger.error(f"결과 조회 실패: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@multi_strategy_bp.route('/list', methods=['GET'])
def list_analyses():
    """멀티 전략 분석 목록 조회"""
    try:
        limit = request.args.get('limit', 20, type=int)
        analyses = MultiStrategyManager.list_analyses(limit=limit)
        return jsonify({
            "success": True,
            "data": [a.to_dict() for a in analyses],
        })
    except Exception as e:
        logger.error(f"목록 조회 실패: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@multi_strategy_bp.route('/strategies', methods=['GET'])
def list_strategies():
    """사용 가능한 전략 목록 조회"""
    return jsonify({
        "success": True,
        "data": {
            st: {"label": preset["label"], "description": preset["requirement_suffix"].strip()}
            for st, preset in STRATEGY_PRESETS.items()
        }
    })
