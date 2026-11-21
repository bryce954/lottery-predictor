"""
Cloud & Scale Module for Lottery Predictor (v2.0)
FastAPI REST API, Docker, Database backend, User accounts, A/B testing.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
import re
import io
import numpy as np
from pathlib import Path
from PIL import Image

# Optional imports for database
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


def _to_py(value):
    """Recursively convert numpy / non-standard types to JSON-safe Python natives."""
    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_py(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_py(v) for k, v in value.items()}
    return value


# Pydantic models for API
class PredictionRequest(BaseModel):
    lottery: str = "powerball"
    method: str = "ensemble"
    count: int = 5
    last_n: int = 200
    train: bool = False


class PredictionResponse(BaseModel):
    lottery: str
    method: str
    predictions: List[Dict[str, Any]]
    timestamp: str
    disclaimer: str


class AnalysisRequest(BaseModel):
    lottery: str = "powerball"
    last_n: int = 200


class AnalysisResponse(BaseModel):
    lottery: str
    total_draws: int
    date_range: Dict[str, str]
    hot_numbers: List[Dict[str, Any]]
    cold_numbers: List[Dict[str, Any]]
    sum_statistics: Dict[str, float]
    top_pairs: List[Dict[str, Any]]


class TrainingRequest(BaseModel):
    lottery: str = "powerball"
    lookback: int = 10


class TrainingResponse(BaseModel):
    lottery: str
    models: Dict[str, Dict[str, float]]
    timestamp: str


class LotteryInfo(BaseModel):
    key: str
    name: str
    main_numbers: int
    main_max: int
    bonus_numbers: int
    bonus_max: int
    draw_days: List[str]
    start_date: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# User profile models
class UserCreateRequest(BaseModel):
    username: str
    email: Optional[str] = None


class UserLoginRequest(BaseModel):
    username: str


class UserPreferenceUpdate(BaseModel):
    default_lottery: Optional[str] = None
    default_method: Optional[str] = None
    default_prediction_count: Optional[int] = None
    language: Optional[str] = None
    theme: Optional[str] = None
    auto_detect_state: Optional[bool] = None
    use_all_history: Optional[bool] = None


class FavoriteLotteryRequest(BaseModel):
    lottery: str


# Export models
class ExportPredictionsRequest(BaseModel):
    lottery: str
    method: str
    count: int = 5
    format: str = "json"  # json or csv


# Odds models
class OddsResponse(BaseModel):
    lottery: str
    jackpot_odds: int
    jackpot_odds_str: str
    expected_value: float
    break_even_jackpot: float
    prize_tiers: List[Dict[str, Any]]


class OddsRequest(BaseModel):
    lottery: str
    jackpot: float = 0
    ticket_cost: float = 2.0


class TicketCheckRequest(BaseModel):
    lottery: str
    main_numbers: List[int]
    bonus_numbers: List[int] = []
    check_recent: int = 100


# Syndicate models
class SyndicateMemberRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    discord_id: Optional[str] = None
    share_percentage: float = 0.0


class SyndicateCreateRequest(BaseModel):
    name: str
    lottery_type: str
    members: List[SyndicateMemberRequest]
    budget: float
    ticket_cost: float = 2.0


class SyndicateResponse(BaseModel):
    name: str
    lottery_type: str
    draw_date: Optional[str]
    total_cost: float
    total_shares: float
    members: List[Dict[str, Any]]
    tickets: List[Dict[str, Any]]


# Notification models
class EmailConfigRequest(BaseModel):
    smtp_server: str
    smtp_port: int = 587
    smtp_user: str
    smtp_pass: str


class DiscordConfigRequest(BaseModel):
    webhook_url: str


class PushbulletConfigRequest(BaseModel):
    pushbullet_token: str


class NotificationStatusResponse(BaseModel):
    email_configured: bool
    discord_configured: bool
    pushbullet_configured: bool


class NotificationTestRequest(BaseModel):
    channel: str  # email, discord, pushbullet
    recipient: Optional[str] = None
    message: Optional[str] = None


class NotificationTestResponse(BaseModel):
    channel: str
    success: bool
    message: str


# Data health models
class DataHealthResponse(BaseModel):
    lottery: str
    timestamp: str
    total_draws: int
    date_range_start: str
    date_range_end: str
    days_since_last_draw: int
    expected_draws: int
    missing_draws: int
    gaps: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    completeness_score: str
    freshness_score: str


# Cross-lottery correlation models
class CorrelationResponse(BaseModel):
    lottery_a: str
    lottery_b: str
    correlation_type: str
    correlation_coefficient: float
    p_value: float
    significant: bool
    sample_size: int
    details: Dict[str, Any]


class CorrelatedLotteryResponse(BaseModel):
    lottery: str
    correlation_coefficient: float
    p_value: float
    significant: bool
    details: Dict[str, Any]


class UserProfileResponse(BaseModel):
    username: str
    email: Optional[str]
    created_at: str
    last_login: Optional[str]
    total_predictions: int
    total_spent: float
    total_won: float
    roi: Optional[float]
    favorite_lotteries: List[str]
    default_lottery: str
    default_method: str


# Database models (if SQLAlchemy available)
if HAS_SQLALCHEMY:
    Base = declarative_base()

    class User(Base):
        __tablename__ = 'users'
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, index=True, nullable=False)
        email = Column(String(100), unique=True, index=True)
        hashed_password = Column(String(255), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        is_active = Column(Integer, default=1)
        predictions = relationship("PredictionHistory", back_populates="user")

    class PredictionHistory(Base):
        __tablename__ = 'prediction_history'
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey('users.id'))
        lottery = Column(String(50), nullable=False)
        method = Column(String(50))
        main_numbers = Column(Text)  # JSON array
        bonus_numbers = Column(Text)  # JSON array
        confidence = Column(Float)
        created_at = Column(DateTime, default=datetime.utcnow)
        user = relationship("User", back_populates="predictions")

    class ABTest(Base):
        __tablename__ = 'ab_tests'
        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), unique=True)
        variant_a = Column(String(50))  # Method A
        variant_b = Column(String(50))  # Method B
        traffic_split = Column(Float, default=0.5)
        is_active = Column(Integer, default=1)
        created_at = Column(DateTime, default=datetime.utcnow)

    class ABTestResult(Base):
        __tablename__ = 'ab_test_results'
        id = Column(Integer, primary_key=True, index=True)
        test_id = Column(Integer, ForeignKey('ab_tests.id'))
        variant = Column(String(50))  # 'A' or 'B'
        lottery = Column(String(50))
        prediction = Column(Text)  # JSON
        actual = Column(Text)  # JSON
        match_count = Column(Integer)
        created_at = Column(DateTime, default=datetime.utcnow)


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Lottery Predictor API",
        description="AI-Powered Universal Lottery Number Predictor",
        version="2.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import here to avoid circular imports
    from .data import get_data_manager, LOTTERY_CONFIGS
    from .models import get_predictor, StatisticalPredictor, MLModelPredictor, EnsemblePredictor
    from .advanced import UserProfileManager, SyndicateManager, SyndicateMember, NotificationSystem, HistoricalDataHealthDashboard, CrossLotteryCorrelationAnalyzer
    from .validation import OddsCalculator

    # In-memory cache for models
    model_cache = {}

    # User profile manager
    user_manager = UserProfileManager()

    # Syndicate manager
    syndicate_manager = SyndicateManager()

    # Notification system
    notification_system = NotificationSystem()

    # Data health dashboard
    health_dashboard = HistoricalDataHealthDashboard()

    # Cross-lottery correlation analyzer
    correlation_analyzer = CrossLotteryCorrelationAnalyzer()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.utcnow().isoformat()
        )

    @app.get("/lotteries", response_model=List[LotteryInfo])
    async def list_lotteries():
        """List all supported lotteries."""
        lotteries = []
        for key, config in LOTTERY_CONFIGS.items():
            lotteries.append(LotteryInfo(
                key=key,
                name=config['name'],
                main_numbers=config['main_numbers'],
                main_max=config['main_max'],
                bonus_numbers=config['bonus_numbers'],
                bonus_max=config['bonus_max'],
                draw_days=config['draw_days'],
                start_date=config['start_date']
            ))
        return lotteries

    @app.get("/lotteries/{lottery_key}", response_model=LotteryInfo)
    async def get_lottery(lottery_key: str):
        """Get details for a specific lottery."""
        if lottery_key not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail="Lottery not found")
        config = LOTTERY_CONFIGS[lottery_key]
        return LotteryInfo(
            key=lottery_key,
            name=config['name'],
            main_numbers=config['main_numbers'],
            main_max=config['main_max'],
            bonus_numbers=config['bonus_numbers'],
            bonus_max=config['bonus_max'],
            draw_days=config['draw_days'],
            start_date=config['start_date']
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Generate lottery predictions."""
        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail="Lottery not found")

        try:
            predictor = get_predictor(request.lottery, request.method)

            if request.method == "statistical":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_balanced(last_n=request.last_n, num_predictions=request.count)
            elif request.method == "frequency":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_frequency_based(last_n=request.last_n, num_predictions=request.count)
            elif request.method == "cold":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_cold_numbers(last_n=request.last_n, num_predictions=request.count)
            elif request.method == "pairs":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_pair_based(last_n=request.last_n, num_predictions=request.count)
            elif request.method == "ml":
                pred = MLModelPredictor(request.lottery)
                if not pred.is_trained or request.train:
                    pred.train()
                predictions = pred.predict(num_predictions=request.count)
            else:  # ensemble
                pred = EnsemblePredictor(request.lottery)
                predictions = pred.get_consensus_prediction(num_predictions=request.count)

            pred_data = [
                {
                    "main_numbers": _to_py(p.main_numbers),
                    "bonus_numbers": _to_py(p.bonus_numbers),
                    "confidence": _to_py(p.confidence),
                    "method": p.method,
                    "metadata": _to_py(p.metadata),
                }
                for p in predictions
            ]

            return PredictionResponse(
                lottery=request.lottery,
                method=request.method,
                predictions=pred_data,
                timestamp=datetime.utcnow().isoformat(),
                disclaimer="This tool is for entertainment and educational purposes only. Lottery draws are random events. No algorithm or AI can predict future lottery numbers with certainty."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(request: AnalysisRequest):
        """Analyze historical lottery data."""
        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail="Lottery not found")

        try:
            dm = get_data_manager(request.lottery)
            df = dm.load_data().tail(request.last_n)

            freq = dm.get_number_frequency(request.last_n)
            hot, cold = dm.get_hot_cold_numbers(request.last_n)
            stats = dm.get_sum_statistics(request.last_n)
            # 'recent' is a list; AnalysisResponse requires all float values
            stats.pop("recent", None)
            stats = _to_py(stats)
            pairs = dm.get_pair_frequency(request.last_n)
            top_pairs = list(pairs.items())[:10]

            return AnalysisResponse(
                lottery=request.lottery,
                total_draws=len(df),
                date_range={
                    "start": df['date'].min().strftime('%Y-%m-%d'),
                    "end": df['date'].max().strftime('%Y-%m-%d')
                },
                hot_numbers=[{"number": n, "frequency": freq[n]} for n in hot],
                cold_numbers=[{"number": n, "frequency": freq[n]} for n in cold],
                sum_statistics=stats,
                top_pairs=[{"pair": f"{a}-{b}", "count": c} for (a, b), c in top_pairs]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/train", response_model=TrainingResponse)
    async def train(request: TrainingRequest):
        """Train ML models."""
        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail="Lottery not found")

        try:
            predictor = MLModelPredictor(request.lottery)
            results = predictor.train(lookback=request.lookback)

            return TrainingResponse(
                lottery=request.lottery,
                models=results,
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/history/{lottery}")
    async def history(lottery: str, count: int = 10):
        """Get recent draw history."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail="Lottery not found")

        try:
            dm = get_data_manager(lottery)
            df = dm.load_data()
            recent = dm.get_recent_draws(count)

            return {
                "lottery": lottery,
                "draws": [
                    {
                        "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        "main_numbers": row['main_numbers'],
                        "bonus_numbers": row['bonus_numbers'],
                        "jackpot": row['jackpot'],
                        "draw_number": row['draw_number']
                    }
                    for _, row in recent.iterrows()
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # User Profile Endpoints
    # =========================================================================

    @app.get("/users", response_model=List[str])
    async def list_users():
        """List all user profiles."""
        return user_manager.list_profiles()

    @app.post("/users", response_model=UserProfileResponse)
    async def create_user(request: UserCreateRequest):
        """Create a new user profile."""
        try:
            profile = user_manager.create_profile(request.username, request.email)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _profile_to_response(profile)

    @app.post("/users/login")
    async def login_user(request: UserLoginRequest):
        """Set the current active user."""
        if not user_manager.set_current_user(request.username):
            raise HTTPException(status_code=404, detail=f"User '{request.username}' not found")
        profile = user_manager.get_current_profile()
        return {"message": f"Logged in as {request.username}", "profile": _profile_to_response(profile)}

    @app.get("/users/me", response_model=UserProfileResponse)
    async def get_current_user():
        """Get the current active user profile."""
        profile = user_manager.get_current_profile()
        if not profile:
            raise HTTPException(status_code=401, detail="No user logged in")
        return _profile_to_response(profile)

    @app.delete("/users/me")
    async def logout_user():
        """Logout the current user."""
        user_manager.current_user = None
        user_manager._save_current_user()
        return {"message": "Logged out"}

    @app.put("/users/me/preferences")
    async def update_preferences(request: UserPreferenceUpdate):
        """Update current user preferences."""
        profile = user_manager.get_current_profile()
        if not profile:
            raise HTTPException(status_code=401, detail="No user logged in")

        updates = request.model_dump(exclude_unset=True)
        if not user_manager.update_preferences(profile.username, **updates):
            raise HTTPException(status_code=400, detail="Failed to update preferences")
        return {"message": "Preferences updated"}

    @app.post("/users/me/favorites")
    async def add_favorite_lottery(request: FavoriteLotteryRequest):
        """Add a lottery to the current user's favorites."""
        profile = user_manager.get_current_profile()
        if not profile:
            raise HTTPException(status_code=401, detail="No user logged in")

        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Unknown lottery '{request.lottery}'")

        if not user_manager.add_favorite_lottery(profile.username, request.lottery):
            raise HTTPException(status_code=400, detail="Failed to add favorite lottery")
        return {"message": f"Added {request.lottery} to favorites"}

    @app.delete("/users/me/favorites/{lottery}")
    async def remove_favorite_lottery(lottery: str):
        """Remove a lottery from the current user's favorites."""
        profile = user_manager.get_current_profile()
        if not profile:
            raise HTTPException(status_code=401, detail="No user logged in")

        if not user_manager.remove_favorite_lottery(profile.username, lottery):
            raise HTTPException(status_code=400, detail=f"'{lottery}' is not a favorite")
        return {"message": f"Removed {lottery} from favorites"}

    @app.get("/users/me/roi")
    async def get_user_roi():
        """Get the current user's ROI."""
        profile = user_manager.get_current_profile()
        if not profile:
            raise HTTPException(status_code=401, detail="No user logged in")
        roi = user_manager.get_roi(profile.username)
        return {"roi": roi}

    def _profile_to_response(profile) -> UserProfileResponse:
        """Convert a UserProfile to a UserProfileResponse."""
        roi = user_manager.get_roi(profile.username)
        return UserProfileResponse(
            username=profile.username,
            email=profile.email,
            created_at=profile.created_at,
            last_login=profile.last_login,
            total_predictions=len(profile.prediction_history),
            total_spent=profile.total_spent,
            total_won=profile.total_won,
            roi=roi,
            favorite_lotteries=profile.preferences.favorite_lotteries,
            default_lottery=profile.preferences.default_lottery,
            default_method=profile.preferences.default_method,
        )

    # =========================================================================
    # Export Endpoints
    # =========================================================================

    @app.post("/export/predictions")
    async def export_predictions(request: ExportPredictionsRequest):
        """Export predictions as JSON or CSV."""
        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{request.lottery}'")

        if request.format not in ("json", "csv"):
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

        try:
            predictor = get_predictor(request.lottery, request.method)

            if request.method == "statistical":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_balanced(num_predictions=request.count)
            elif request.method == "frequency":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_frequency_based(num_predictions=request.count)
            elif request.method == "cold":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_cold_numbers(num_predictions=request.count)
            elif request.method == "pairs":
                pred = StatisticalPredictor(request.lottery)
                predictions = pred.predict_pair_based(num_predictions=request.count)
            elif request.method == "ml":
                pred = MLModelPredictor(request.lottery)
                if not pred.is_trained:
                    pred.train()
                predictions = pred.predict(num_predictions=request.count)
            else:  # ensemble
                pred = EnsemblePredictor(request.lottery)
                predictions = pred.get_consensus_prediction(num_predictions=request.count)

            pred_data = [
                {
                    "main_numbers": _to_py(p.main_numbers),
                    "bonus_numbers": _to_py(p.bonus_numbers),
                    "confidence": _to_py(p.confidence),
                    "method": p.method,
                    "metadata": _to_py(p.metadata),
                }
                for p in predictions
            ]

            if request.format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["main_numbers", "bonus_numbers", "confidence", "method"])
                for p in pred_data:
                    writer.writerow(
                        [
                            " ".join(str(n) for n in p["main_numbers"]),
                            " ".join(str(n) for n in p["bonus_numbers"]),
                            f"{p['confidence']:.4f}",
                            p["method"],
                        ]
                    )
                csv_content = output.getvalue()

                return StreamingResponse(
                    iter([csv_content]),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f'attachment; filename="{request.lottery}_{request.method}_predictions.csv"'
                    },
                )

            return {
                "lottery": request.lottery,
                "method": request.method,
                "count": len(pred_data),
                "predictions": pred_data,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/export/history/{lottery}")
    async def export_history(lottery: str, count: int = 100, format: str = "json"):
        """Export historical draw data as JSON or CSV."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery}'")

        if format not in ("json", "csv"):
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

        try:
            dm = get_data_manager(lottery)
            df = dm.load_data()
            recent = df.tail(count)

            rows = [
                {
                    "date": row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"]),
                    "draw_number": row["draw_number"],
                    "main_numbers": row["main_numbers"],
                    "bonus_numbers": row["bonus_numbers"],
                    "jackpot": row["jackpot"],
                }
                for _, row in recent.iterrows()
            ]

            if format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["date", "draw_number", "main_numbers", "bonus_numbers", "jackpot"])
                for r in rows:
                    writer.writerow(
                        [
                            r["date"],
                            r["draw_number"],
                            " ".join(str(n) for n in r["main_numbers"]),
                            " ".join(str(n) for n in r["bonus_numbers"]),
                            r["jackpot"],
                        ]
                    )
                csv_content = output.getvalue()

                return StreamingResponse(
                    iter([csv_content]),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f'attachment; filename="{lottery}_history.csv"'
                    },
                )

            return {"lottery": lottery, "count": len(rows), "draws": rows}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Odds Calculator Endpoints
    # =========================================================================

    @app.get("/odds/{lottery}", response_model=OddsResponse)
    async def get_odds(
        lottery: str,
        jackpot: float = 0,
        ticket_cost: float = 2.0,
    ):
        """Calculate odds, prize tiers, and expected value for a lottery."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery}'")

        try:
            calc = OddsCalculator(lottery)
            result = calc.full_odds_analysis(current_jackpot=jackpot, ticket_cost=ticket_cost)

            return OddsResponse(
                lottery=result.lottery,
                jackpot_odds=result.jackpot_odds,
                jackpot_odds_str=result.jackpot_odds_str,
                expected_value=result.expected_value,
                break_even_jackpot=result.break_even_jackpot,
                prize_tiers=result.prize_tiers,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/check-ticket")
    async def check_ticket(request: TicketCheckRequest):
        """Check a ticket's numbers against recent historical draws."""
        if request.lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{request.lottery}'")

        if not request.main_numbers:
            raise HTTPException(status_code=400, detail="main_numbers is required")

        try:
            dm = get_data_manager(request.lottery)
            df = dm.load_data()
            recent = df.tail(request.check_recent)

            config = LOTTERY_CONFIGS[request.lottery]
            main_set = set(request.main_numbers)
            bonus_set = set(request.bonus_numbers) if request.bonus_numbers else set()

            matches = []
            for _, row in recent.iterrows():
                draw_main = set(row['main_numbers'])
                draw_bonus = set(row['bonus_numbers']) if row['bonus_numbers'] else set()
                main_match = len(main_set & draw_main)
                bonus_match = len(bonus_set & draw_bonus)
                matches.append({
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "draw_number": int(row['draw_number']),
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "total_match": main_match + bonus_match,
                    "is_winner": main_match >= config['main_numbers']
                })

            matches.sort(key=lambda x: x['total_match'], reverse=True)
            best = matches[0] if matches else None

            calc = OddsCalculator(request.lottery)
            tiers = calc.calculate_prize_tiers()

            return _to_py({
                "lottery": request.lottery,
                "ticket": {
                    "main_numbers": sorted(request.main_numbers),
                    "bonus_numbers": sorted(request.bonus_numbers) if request.bonus_numbers else []
                },
                "draws_checked": len(recent),
                "best_match": best,
                "all_matches": matches[:10],
                "total_wins": sum(1 for m in matches if m['is_winner']),
                "prize_tiers": tiers[:6],
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ocr-ticket")
    async def ocr_ticket(lottery: str = File(...), file: UploadFile = File(...)):
        """OCR a ticket photo and extract numbers."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery}'")

        config = LOTTERY_CONFIGS[lottery]
        main_max = config['main_max']
        main_n = config['main_numbers']
        bonus_n = config['bonus_numbers']
        bonus_max = config['bonus_max']

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Preprocess: upscale 2x + grayscale + contrast (helps small phone-camera digits)
            from PIL import ImageOps, ImageEnhance
            w, h = image.size
            image = image.resize((w * 2, h * 2), Image.LANCZOS)
            gray = ImageOps.grayscale(image)
            enhanced = ImageEnhance.Contrast(gray).enhance(1.5)

            # Lazy-load easyocr (heavy dependency)
            try:
                import easyocr
            except ImportError:
                raise HTTPException(status_code=500, detail="easyocr not installed. Run: pip install easyocr")

            reader = easyocr.Reader(['en'], gpu=False)

            # Try both grayscale and enhanced, pick the variant with more digit detections
            best = None
            for img_variant in (gray, enhanced):
                result = reader.readtext(np.array(img_variant), allowlist='0123456789 ')
                detected = []
                for bbox, text, conf in result:
                    # bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    detected.append({
                        "text": text,
                        "confidence": round(conf, 2),
                        "bbox": [round(xs[0]), round(ys[0]), round(xs[2]), round(ys[2])],
                    })
                nums = [int(n) for n in re.findall(r'\d+', " ".join(d["text"] for d in detected)) if 1 <= int(n) <= max(main_max, bonus_max)]
                if best is None or len(nums) > len(best[0]):
                    best = (nums, detected, " ".join(d["text"] for d in detected))

            numbers, detected, all_text = best

            # Be lenient: return whatever digits OCR found (even partial), let user fill in the rest
            if len(numbers) == 0:
                raise HTTPException(status_code=400, detail=f"Could not read any numbers from ticket. All detections: {[d['text'] for d in detected]}. Try a clearer photo with better lighting, make sure numbers are fully in frame, and avoid glare.")

            # Use whatever we found: first main_n as main, rest as bonus
            main_numbers = sorted(numbers[:main_n])
            bonus_numbers = sorted(numbers[main_n:main_n + bonus_n]) if bonus_n > 0 else []

            return _to_py({
                "lottery": lottery,
                "main_numbers": main_numbers,
                "bonus_numbers": bonus_numbers,
                "ocr_text": all_text[:500],
                "raw_numbers": numbers,
                "detections": detected[:30],
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    # =========================================================================
    # Syndicate Endpoints
    # =========================================================================

    @app.post("/syndicates", response_model=SyndicateResponse)
    async def create_syndicate(request: SyndicateCreateRequest):
        """Create a new lottery syndicate with group play."""
        if request.lottery_type not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{request.lottery_type}'")

        if not request.members:
            raise HTTPException(status_code=400, detail="At least one member required")

        if request.budget <= 0:
            raise HTTPException(status_code=400, detail="Budget must be positive")

        try:
            member_list = [
                SyndicateMember(
                    name=m.name,
                    email=m.email,
                    phone=m.phone,
                    discord_id=m.discord_id,
                    share_percentage=m.share_percentage,
                )
                for m in request.members
            ]

            syndicate = syndicate_manager.create_syndicate(
                name=request.name,
                lottery_type=request.lottery_type,
                members=member_list,
                budget=request.budget,
                ticket_cost=request.ticket_cost,
            )

            return SyndicateResponse(
                name=syndicate.name,
                lottery_type=syndicate.lottery_type,
                draw_date=syndicate.draw_date,
                total_cost=_to_py(syndicate.total_cost),
                total_shares=_to_py(syndicate.total_shares),
                members=[
                    {
                        "name": m.name,
                        "email": m.email,
                        "phone": m.phone,
                        "discord_id": m.discord_id,
                        "share_percentage": _to_py(m.share_percentage),
                        "contribution": _to_py(m.contribution),
                    }
                    for m in syndicate.members
                ],
                tickets=[
                    {
                        "main_numbers": _to_py(t.main_numbers),
                        "bonus_numbers": _to_py(t.bonus_numbers),
                        "method": t.method,
                        "confidence": _to_py(t.confidence),
                        "cost": _to_py(t.cost),
                    }
                    for t in syndicate.tickets
                ],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/syndicates")
    async def list_syndicates():
        """List all syndicates."""
        return [
            {
                "name": s.name,
                "lottery_type": s.lottery_type,
                "total_cost": _to_py(s.total_cost),
                "num_members": len(s.members),
                "num_tickets": len(s.tickets),
            }
            for s in syndicate_manager.syndicates.values()
        ]

    @app.get("/syndicates/{name}", response_model=SyndicateResponse)
    async def get_syndicate(name: str):
        """Get syndicate details."""
        syndicate = syndicate_manager.get_syndicate(name)
        if not syndicate:
            raise HTTPException(status_code=404, detail=f"Syndicate '{name}' not found")

        return SyndicateResponse(
            name=syndicate.name,
            lottery_type=syndicate.lottery_type,
            draw_date=syndicate.draw_date,
            total_cost=_to_py(syndicate.total_cost),
            total_shares=_to_py(syndicate.total_shares),
            members=[
                {
                    "name": m.name,
                    "email": m.email,
                    "phone": m.phone,
                    "discord_id": m.discord_id,
                    "share_percentage": _to_py(m.share_percentage),
                    "contribution": _to_py(m.contribution),
                }
                for m in syndicate.members
            ],
            tickets=[
                {
                    "main_numbers": _to_py(t.main_numbers),
                    "bonus_numbers": _to_py(t.bonus_numbers),
                    "method": t.method,
                    "confidence": _to_py(t.confidence),
                    "cost": _to_py(t.cost),
                }
                for t in syndicate.tickets
            ],
        )

    @app.delete("/syndicates/{name}")
    async def delete_syndicate(name: str):
        """Delete a syndicate."""
        if name not in syndicate_manager.syndicates:
            raise HTTPException(status_code=404, detail=f"Syndicate '{name}' not found")
        del syndicate_manager.syndicates[name]
        return {"message": f"Syndicate '{name}' deleted"}

    @app.post("/syndicates/{name}/winnings")
    async def distribute_winnings(name: str, winnings: float):
        """Distribute winnings among syndicate members."""
        if winnings < 0:
            raise HTTPException(status_code=400, detail="Winnings must be positive")

        distributions = syndicate_manager.distribute_winnings(name, winnings)
        if not distributions:
            raise HTTPException(status_code=404, detail=f"Syndicate '{name}' not found")

        return {
            "syndicate": name,
            "total_winnings": winnings,
            "distributions": distributions,
        }

    # =========================================================================
    # Notification Endpoints
    # =========================================================================

    @app.get("/notifications/status", response_model=NotificationStatusResponse)
    async def notification_status():
        """Check which notification channels are configured."""
        return NotificationStatusResponse(
            email_configured=notification_system.email_config is not None,
            discord_configured=notification_system.discord_webhook is not None,
            pushbullet_configured=notification_system.pushbullet_token is not None,
        )

    @app.post("/notifications/configure/email")
    async def configure_email(request: EmailConfigRequest):
        """Configure email notifications via SMTP."""
        try:
            notification_system.configure_email(
                request.smtp_server, request.smtp_port, request.smtp_user, request.smtp_pass
            )
            return {"message": "Email configured successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/notifications/configure/discord")
    async def configure_discord(request: DiscordConfigRequest):
        """Configure Discord webhook notifications."""
        try:
            notification_system.configure_discord(request.webhook_url)
            return {"message": "Discord configured successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/notifications/configure/pushbullet")
    async def configure_pushbullet(request: PushbulletConfigRequest):
        """Configure Pushbullet push notifications."""
        try:
            notification_system.configure_pushbullet(request.pushbullet_token)
            return {"message": "Pushbullet configured successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/notifications/test", response_model=NotificationTestResponse)
    async def test_notification(request: NotificationTestRequest):
        """Send a test notification to verify configuration."""
        try:
            if request.channel == "email":
                if not notification_system.email_config:
                    raise HTTPException(status_code=400, detail="Email not configured")
                recipient = request.recipient or notification_system.email_config.get("smtp_user")
                success = notification_system.send_email(
                    recipient,
                    "Lottery Predictor — Test Notification",
                    "This is a test notification from your Lottery Predictor app.",
                )
            elif request.channel == "discord":
                if not notification_system.discord_webhook:
                    raise HTTPException(status_code=400, detail="Discord not configured")
                success = notification_system.send_discord(
                    request.message or "🎲 Lottery Predictor — Test notification"
                )
            elif request.channel == "pushbullet":
                if not notification_system.pushbullet_token:
                    raise HTTPException(status_code=400, detail="Pushbullet not configured")
                success = notification_system.send_pushbullet(
                    "Lottery Predictor — Test",
                    request.message or "This is a test notification from your Lottery Predictor app.",
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown channel '{request.channel}'")

            return NotificationTestResponse(
                channel=request.channel,
                success=success,
                message="Sent successfully" if success else "Failed to send",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Data Health Dashboard Endpoints
    # =========================================================================

    @app.get("/health-dashboard/{lottery}", response_model=DataHealthResponse)
    async def get_lottery_health(lottery: str):
        """Get data health metrics for a specific lottery."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery}'")

        try:
            record = health_dashboard.analyze_lottery_health(lottery)

            return DataHealthResponse(
                lottery=record.lottery,
                timestamp=record.timestamp,
                total_draws=record.total_draws,
                date_range_start=record.date_range_start,
                date_range_end=record.date_range_end,
                days_since_last_draw=record.days_since_last_draw,
                expected_draws=record.expected_draws,
                missing_draws=record.missing_draws,
                gaps=record.gaps,
                anomalies=record.anomalies,
                completeness_score=f"{record.completeness_score:.1%}",
                freshness_score=f"{record.freshness_score:.1%}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health-dashboard", response_model=List[DataHealthResponse])
    async def get_all_lotteries_health():
        """Get data health metrics for all lotteries."""
        try:
            records = health_dashboard.get_all_lotteries_health()

            return [
                DataHealthResponse(
                    lottery=r["lottery"],
                    timestamp=r.get("last_checked", ""),
                    total_draws=r.get("total_draws", 0),
                    date_range_start=r.get("date_range", "").split(" to ")[0] if r.get("date_range") else "",
                    date_range_end=r.get("date_range", "").split(" to ")[1] if r.get("date_range") else "",
                    days_since_last_draw=r.get("days_since_last_draw", 0),
                    expected_draws=r.get("expected_draws", 0),
                    missing_draws=r.get("missing_draws", 0),
                    gaps=[],  # get_health_summary doesn't include gap details
                    anomalies=[],  # get_health_summary doesn't include anomaly details
                    completeness_score=r.get("completeness_score", "0%"),
                    freshness_score=r.get("freshness_score", "0%"),
                )
                for r in records
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Cross-Lottery Correlation Endpoints
    # =========================================================================

    @app.get("/correlations/find-correlated/{lottery}", response_model=List[CorrelatedLotteryResponse])
    async def find_correlated_lotteries(lottery: str, top_n: int = 5):
        """Find lotteries most correlated with a given lottery."""
        if lottery not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery}'")

        try:
            results = correlation_analyzer.find_correlated_lotteries(lottery, top_n=top_n)

            return [
                CorrelatedLotteryResponse(
                    lottery=r["lottery"],
                    correlation_coefficient=r["avg_correlation"],
                    p_value=r["min_p_value"],
                    significant=r["significant"],
                    sample_size=0,
                    details=r.get("details", {}),
                )
                for r in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/correlations/{lottery_a}/{lottery_b}", response_model=CorrelationResponse)
    async def get_correlation(lottery_a: str, lottery_b: str):
        """Analyze correlation between two lottery games."""
        if lottery_a not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery_a}'")
        if lottery_b not in LOTTERY_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown lottery '{lottery_b}'")

        try:
            result = correlation_analyzer.analyze_number_frequency_correlation(lottery_a, lottery_b)

            return CorrelationResponse(
                lottery_a=result.lottery_a,
                lottery_b=result.lottery_b,
                correlation_type=result.correlation_type,
                correlation_coefficient=result.correlation_coefficient,
                p_value=result.p_value,
                significant=result.significant,
                sample_size=result.sample_size,
                details=result.details,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Database setup functions
def init_database(database_url: str = "sqlite:///lottery_predictor.db"):
    """Initialize database."""
    if not HAS_SQLALCHEMY:
        raise ImportError("SQLAlchemy not installed. Run: pip install sqlalchemy")

    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def get_db():
    """Dependency for database session."""
    if not HAS_SQLALCHEMY:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Generate files (copies from repo-root templates; not embedded here to avoid drift)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_K8S_DEPLOYMENT = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: lottery-predictor
  labels:
    app: lottery-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lottery-predictor
  template:
    metadata:
      labels:
        app: lottery-predictor
    spec:
      containers:
      - name: api
        image: lottery-predictor:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: lottery-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: lottery-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: lottery-predictor
spec:
  selector:
    app: lottery-predictor
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lottery-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lottery-predictor
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
'''


def generate_dockerfile(output_path: str = "Dockerfile"):
    """Generate Dockerfile from the repo-root template."""
    src = _PROJECT_ROOT / "Dockerfile"
    Path(output_path).write_text(src.read_text())
    return output_path


def generate_docker_compose(output_path: str = "docker-compose.yml"):
    """Generate docker-compose.yml from the repo-root template."""
    src = _PROJECT_ROOT / "docker-compose.yml"
    Path(output_path).write_text(src.read_text())
    return output_path


def generate_kubernetes(output_dir: str = "k8s"):
    """Generate Kubernetes deployment files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/deployment.yaml").write_text(_K8S_DEPLOYMENT)
    return output_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "dockerfile":
            generate_dockerfile()
            print("Dockerfile generated")
        elif sys.argv[1] == "docker-compose":
            generate_docker_compose()
            print("docker-compose.yml generated")
        elif sys.argv[1] == "k8s":
            generate_kubernetes()
            print("Kubernetes files generated")
        elif sys.argv[1] == "all":
            generate_dockerfile()
            generate_docker_compose()
            generate_kubernetes()
            print("All deployment files generated")
    else:
        print("Usage: python -m lottery_predictor.cloud [dockerfile|docker-compose|k8s|all]")