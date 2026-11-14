"""
Lottery Predictor - AI-powered universal lottery number prediction
Supports all US state lotteries and major international games.
"""

__version__ = "2.0.0"
__author__ = "Lottery Predictor"

# Core modules
from .data import get_data_manager, LOTTERY_CONFIGS
from .models import (
    StatisticalPredictor, MLModelPredictor, EnsemblePredictor,
    TimeSeriesPredictor, TransformerPredictor, OnlineLearningPredictor,
    get_predictor, PredictionResult
)
from .universal import (
    LotteryGame, BUILTIN_LOTTERIES, UniversalLotteryManager,
    get_universal_manager, register_lottery, list_lotteries, get_lottery,
    auto_select_lottery, detect_state_from_ip, get_lottery_for_state,
    STATE_TO_LOTTERY
)
from .visualize import LotteryVisualizer
from .data_integration import (
    APIClient, WebScraper, DataValidator, ScheduledUpdater
)
from .advanced import (
    PortfolioOptimizer, SyndicateManager, NotificationSystem, I18nManager
)
from .cloud import (
    create_app, init_database, generate_dockerfile,
    generate_docker_compose, generate_kubernetes
)
from .validation import (
    PatternAnalyzer, OddsCalculator, ResultsVerifier, SignificanceTester,
    PatternAnalysis, OddsResult, VerificationResult, SignificanceTestResult
)

__all__ = [
    # Data
    'get_data_manager', 'LOTTERY_CONFIGS',
    # Models
    'StatisticalPredictor', 'MLModelPredictor', 'EnsemblePredictor',
    'TimeSeriesPredictor', 'TransformerPredictor', 'OnlineLearningPredictor',
    'get_predictor', 'PredictionResult',
    # Universal
    'LotteryGame', 'BUILTIN_LOTTERIES', 'UniversalLotteryManager',
    'get_universal_manager', 'register_lottery', 'list_lotteries', 'get_lottery',
    'auto_select_lottery', 'detect_state_from_ip', 'get_lottery_for_state',
    'STATE_TO_LOTTERY',
    # Visualize
    'LotteryVisualizer',
    # Data Integration
    'APIClient', 'WebScraper', 'DataValidator', 'ScheduledUpdater',
    # Advanced
    'PortfolioOptimizer', 'SyndicateManager', 'NotificationSystem', 'I18nManager',
    # Cloud
    'create_app', 'init_database', 'generate_dockerfile',
    'generate_docker_compose', 'generate_kubernetes',
    # Validation (v1.7)
    'PatternAnalyzer', 'OddsCalculator', 'ResultsVerifier', 'SignificanceTester',
    'PatternAnalysis', 'OddsResult', 'VerificationResult', 'SignificanceTestResult',
]