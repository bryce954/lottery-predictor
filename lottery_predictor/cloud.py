"""
Cloud & Scale Module for Lottery Predictor (v2.0)
FastAPI REST API, Docker, Database backend, User accounts, A/B testing.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import here to avoid circular imports
    from .data import get_data_manager, LOTTERY_CONFIGS
    from .models import get_predictor, StatisticalPredictor, MLModelPredictor, EnsemblePredictor
    from .advanced import UserProfileManager

    # In-memory cache for models
    model_cache = {}

    # User profile manager
    user_manager = UserProfileManager()

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


# Docker configuration
DOCKERFILE_CONTENT = '''# Dockerfile for Lottery Predictor
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "lottery_predictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

DOCKER_COMPOSE_CONTENT = '''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/lottery
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=lottery
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
'''

KUBERNETES_DEPLOYMENT = '''apiVersion: apps/v1
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

# Generate files
def generate_dockerfile(output_path: str = "Dockerfile"):
    """Generate Dockerfile."""
    Path(output_path).write_text(DOCKERFILE_CONTENT)
    return output_path


def generate_docker_compose(output_path: str = "docker-compose.yml"):
    """Generate docker-compose.yml."""
    Path(output_path).write_text(DOCKER_COMPOSE_CONTENT)
    return output_path


def generate_kubernetes(output_dir: str = "k8s"):
    """Generate Kubernetes deployment files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/deployment.yaml").write_text(KUBERNETES_DEPLOYMENT)
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