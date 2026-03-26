"""
Stock Data Provider
FMP(Financial Modeling Prep) API를 통한 실시간 주가 데이터 조회 서비스
전략별(단타/스윙/장기) 적합한 기간·간격의 데이터를 가져와 LLM 컨텍스트용 텍스트로 변환
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.stock_data_provider')

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# 전략별 데이터 조회 설정
STRATEGY_DATA_CONFIG: Dict[str, Dict[str, Any]] = {
    "short_term": {
        "label": "단기(1-3일)",
        "price_days": 5,
        "price_interval": "1hour",       # 1시간봉
        "sma_period": 20,
        "rsi_period": 14,
        "include_intraday": True,
    },
    "swing": {
        "label": "스윙(1-3주)",
        "price_days": 90,
        "price_interval": "daily",        # 일봉
        "sma_period": 50,
        "rsi_period": 14,
        "include_intraday": False,
    },
    "long_term": {
        "label": "장기(3-6개월)",
        "price_days": 365,
        "price_interval": "daily",        # 일봉 (주봉은 별도 집계)
        "sma_period": 200,
        "rsi_period": 14,
        "include_intraday": False,
    },
}


class StockDataProvider:
    """FMP API 기반 주가 데이터 조회 및 포맷팅"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.FMP_API_KEY
        if not self.api_key:
            logger.warning("FMP_API_KEY가 설정되지 않았습니다. 주가 데이터를 조회할 수 없습니다.")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """FMP API GET 요청"""
        params = params or {}
        params["apikey"] = self.api_key
        url = f"{FMP_BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"FMP API 요청 실패: {endpoint} - {e}")
            return None

    # ------------------------------------------------------------------
    # 개별 데이터 조회
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> Optional[Dict]:
        """실시간 시세 조회"""
        data = self._get(f"quote/{ticker}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """기업 프로필 조회"""
        data = self._get(f"profile/{ticker}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_historical_daily(self, ticker: str, days: int = 90) -> Optional[List[Dict]]:
        """일별 종가 데이터 조회"""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        data = self._get(f"historical-price-full/{ticker}", {
            "from": from_date,
            "to": to_date,
        })
        if data and "historical" in data:
            return data["historical"]
        return None

    def get_intraday(self, ticker: str, interval: str = "1hour", days: int = 5) -> Optional[List[Dict]]:
        """인트라데이 데이터 조회 (1min, 5min, 15min, 30min, 1hour, 4hour)"""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        data = self._get(f"historical-chart/{interval}/{ticker}", {
            "from": from_date,
            "to": to_date,
        })
        if data and isinstance(data, list):
            return data
        return None

    def get_key_metrics(self, ticker: str) -> Optional[Dict]:
        """핵심 재무 지표 조회"""
        data = self._get(f"key-metrics-ttm/{ticker}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_technical_indicator(self, ticker: str, indicator: str, period: int = 14,
                                indicator_type: str = "daily") -> Optional[List[Dict]]:
        """기술적 지표 조회 (sma, ema, rsi, etc.)"""
        data = self._get(f"technical_indicator/{indicator_type}/{ticker}", {
            "type": indicator,
            "period": period,
        })
        if data and isinstance(data, list):
            return data
        return None

    # ------------------------------------------------------------------
    # 전략별 통합 조회 & 포맷팅
    # ------------------------------------------------------------------

    def get_stock_context(self, ticker: str, strategy_type: str) -> str:
        """
        전략 타입에 맞는 주가 컨텍스트 텍스트를 생성

        Args:
            ticker: 종목 심볼 (e.g. "AAPL")
            strategy_type: "short_term" | "swing" | "long_term"

        Returns:
            LLM 프롬프트에 삽입할 주가 컨텍스트 마크다운 문자열
            (API 키 미설정 또는 전체 실패 시 빈 문자열)
        """
        if not self.api_key:
            return ""

        config = STRATEGY_DATA_CONFIG.get(strategy_type)
        if not config:
            logger.warning(f"알 수 없는 전략 타입: {strategy_type}")
            return ""

        sections: List[str] = []
        sections.append(f"\n[실시간 주가 데이터: {ticker} / 전략: {config['label']}]")

        # 1. 현재 시세
        quote = self.get_quote(ticker)
        if quote:
            sections.append(self._format_quote(quote))

        # 2. 기업 프로필 (장기만)
        if strategy_type == "long_term":
            profile = self.get_company_profile(ticker)
            if profile:
                sections.append(self._format_profile(profile))

        # 3. 가격 히스토리
        if config["include_intraday"]:
            prices = self.get_intraday(ticker, config["price_interval"], config["price_days"])
            if prices:
                sections.append(self._format_intraday_prices(prices, config["label"]))
        else:
            prices = self.get_historical_daily(ticker, config["price_days"])
            if prices:
                sections.append(self._format_daily_prices(
                    prices, config["label"], weekly=(strategy_type == "long_term")
                ))

        # 4. 핵심 재무 지표 (스윙/장기)
        if strategy_type in ("swing", "long_term"):
            metrics = self.get_key_metrics(ticker)
            if metrics:
                sections.append(self._format_key_metrics(metrics))

        if len(sections) <= 1:
            # 헤더만 있고 데이터 없음
            logger.warning(f"주가 데이터 조회 실패: {ticker}")
            return ""

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # 포맷팅 헬퍼
    # ------------------------------------------------------------------

    def _format_quote(self, q: Dict) -> str:
        """현재 시세 포맷팅"""
        lines = ["\n### 현재 시세"]
        lines.append(f"- 현재가: ${q.get('price', 'N/A')}")
        change = q.get('change', 0)
        change_pct = q.get('changesPercentage', 0)
        direction = "▲" if change >= 0 else "▼"
        lines.append(f"- 등락: {direction} ${abs(change):.2f} ({change_pct:+.2f}%)")
        lines.append(f"- 거래량: {q.get('volume', 'N/A'):,}")
        lines.append(f"- 평균 거래량: {q.get('avgVolume', 'N/A'):,}")
        lines.append(f"- 시가총액: ${q.get('marketCap', 0):,.0f}")
        lines.append(f"- 52주 최고: ${q.get('yearHigh', 'N/A')} / 최저: ${q.get('yearLow', 'N/A')}")
        lines.append(f"- PE 비율: {q.get('pe', 'N/A')}")
        lines.append(f"- EPS: ${q.get('eps', 'N/A')}")
        return "\n".join(lines)

    def _format_profile(self, p: Dict) -> str:
        """기업 프로필 포맷팅"""
        lines = ["\n### 기업 개요"]
        lines.append(f"- 기업명: {p.get('companyName', 'N/A')}")
        lines.append(f"- 섹터: {p.get('sector', 'N/A')} / 산업: {p.get('industry', 'N/A')}")
        lines.append(f"- 직원수: {p.get('fullTimeEmployees', 'N/A'):,}")
        desc = p.get('description', '')
        if desc and len(desc) > 300:
            desc = desc[:300] + "..."
        if desc:
            lines.append(f"- 설명: {desc}")
        return "\n".join(lines)

    def _format_intraday_prices(self, prices: List[Dict], label: str) -> str:
        """인트라데이 가격 데이터 포맷팅"""
        lines = [f"\n### 최근 가격 추이 ({label})"]
        lines.append("| 시각 | 시가 | 고가 | 저가 | 종가 | 거래량 |")
        lines.append("|------|------|------|------|------|--------|")

        # 최근 48개 (2일치 1시간봉)로 제한, 시간순 정렬
        recent = sorted(prices[:48], key=lambda x: x.get("date", ""))
        for p in recent:
            dt = p.get("date", "")
            lines.append(
                f"| {dt} | {p.get('open', ''):.2f} | {p.get('high', ''):.2f} "
                f"| {p.get('low', ''):.2f} | {p.get('close', ''):.2f} "
                f"| {p.get('volume', 0):,} |"
            )

        # 간단한 통계
        if recent:
            closes = [p.get("close", 0) for p in recent if p.get("close")]
            if closes:
                lines.append(f"\n- 기간 최고가: ${max(closes):.2f}")
                lines.append(f"- 기간 최저가: ${min(closes):.2f}")
                lines.append(f"- 변동폭: ${max(closes) - min(closes):.2f} ({(max(closes) - min(closes)) / min(closes) * 100:.1f}%)")

        return "\n".join(lines)

    def _format_daily_prices(self, prices: List[Dict], label: str, weekly: bool = False) -> str:
        """일별/주별 가격 데이터 포맷팅"""
        if weekly:
            prices = self._aggregate_weekly(prices)
            interval_label = "주봉"
        else:
            interval_label = "일봉"

        lines = [f"\n### 최근 가격 추이 ({label}, {interval_label})"]
        lines.append("| 날짜 | 시가 | 고가 | 저가 | 종가 | 거래량 | 변동률 |")
        lines.append("|------|------|------|------|------|--------|--------|")

        # 최신순 → 오래된순으로 정렬, 최대 60행
        sorted_prices = sorted(prices[:60], key=lambda x: x.get("date", ""))
        for p in sorted_prices:
            change_pct = p.get("changePercent", 0) or 0
            lines.append(
                f"| {p.get('date', '')} | {p.get('open', 0):.2f} | {p.get('high', 0):.2f} "
                f"| {p.get('low', 0):.2f} | {p.get('close', 0):.2f} "
                f"| {p.get('volume', 0):,} | {change_pct:+.2f}% |"
            )

        # 통계 요약
        if sorted_prices:
            closes = [p.get("close", 0) for p in sorted_prices if p.get("close")]
            if closes:
                first_close = closes[0]
                last_close = closes[-1]
                total_return = (last_close - first_close) / first_close * 100 if first_close else 0
                lines.append(f"\n- 기간 수익률: {total_return:+.1f}%")
                lines.append(f"- 기간 최고가: ${max(closes):.2f}")
                lines.append(f"- 기간 최저가: ${min(closes):.2f}")

        return "\n".join(lines)

    def _aggregate_weekly(self, daily_prices: List[Dict]) -> List[Dict]:
        """일봉 → 주봉 집계"""
        if not daily_prices:
            return []

        # 오래된순 정렬
        sorted_prices = sorted(daily_prices, key=lambda x: x.get("date", ""))
        weekly = []
        current_week: List[Dict] = []

        for p in sorted_prices:
            date_str = p.get("date", "")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if current_week:
                prev_dt = datetime.strptime(current_week[0]["date"], "%Y-%m-%d")
                if dt.isocalendar()[1] != prev_dt.isocalendar()[1] or dt.isocalendar()[0] != prev_dt.isocalendar()[0]:
                    weekly.append(self._merge_week(current_week))
                    current_week = []

            current_week.append(p)

        if current_week:
            weekly.append(self._merge_week(current_week))

        return weekly

    def _merge_week(self, week_prices: List[Dict]) -> Dict:
        """한 주의 일봉을 하나의 주봉으로 병합"""
        opens = [p.get("open", 0) for p in week_prices]
        highs = [p.get("high", 0) for p in week_prices]
        lows = [p.get("low", 0) for p in week_prices]
        closes = [p.get("close", 0) for p in week_prices]
        volumes = [p.get("volume", 0) for p in week_prices]

        first_open = opens[0] if opens else 0
        last_close = closes[-1] if closes else 0
        change_pct = (last_close - first_open) / first_open * 100 if first_open else 0

        return {
            "date": week_prices[0]["date"],
            "open": first_open,
            "high": max(highs) if highs else 0,
            "low": min(lows) if lows else 0,
            "close": last_close,
            "volume": sum(volumes),
            "changePercent": change_pct,
        }

    def _format_key_metrics(self, m: Dict) -> str:
        """핵심 재무 지표 포맷팅"""
        lines = ["\n### 핵심 재무 지표 (TTM)"]

        def _fmt(val, prefix="", suffix="", decimals=2):
            if val is None:
                return "N/A"
            try:
                return f"{prefix}{float(val):,.{decimals}f}{suffix}"
            except (ValueError, TypeError):
                return str(val)

        lines.append(f"- PER: {_fmt(m.get('peRatioTTM'), decimals=1)}")
        lines.append(f"- PBR: {_fmt(m.get('pbRatioTTM'), decimals=2)}")
        lines.append(f"- PSR: {_fmt(m.get('priceToSalesRatioTTM'), decimals=2)}")
        lines.append(f"- EV/EBITDA: {_fmt(m.get('enterpriseValueOverEBITDATTM'), decimals=1)}")
        lines.append(f"- 배당수익률: {_fmt(m.get('dividendYieldTTM'), suffix='%', decimals=2)}")
        lines.append(f"- ROE: {_fmt(m.get('roeTTM'), suffix='%', decimals=1)}")
        lines.append(f"- ROA: {_fmt(m.get('roaTTM'), suffix='%', decimals=1)}")
        lines.append(f"- 부채비율: {_fmt(m.get('debtToEquityTTM'), decimals=2)}")
        lines.append(f"- 유동비율: {_fmt(m.get('currentRatioTTM'), decimals=2)}")
        lines.append(f"- 영업이익률: {_fmt(m.get('operatingProfitMarginTTM'), suffix='%', decimals=1)}")
        lines.append(f"- 순이익률: {_fmt(m.get('netProfitMarginTTM'), suffix='%', decimals=1)}")
        lines.append(f"- 매출 성장률: {_fmt(m.get('revenuePerShareTTM'), prefix='$', decimals=2)} (주당)")

        return "\n".join(lines)
