"""
AI/ML Models for Lottery Number Prediction
Implements multiple prediction strategies including statistical, ML, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

# Time-series forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# v1.3 Enhanced AI/ML: Transformer, Optuna, SHAP, Online Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .data import LotteryDataManager, get_data_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    main_numbers: List[int]
    bonus_numbers: List[int]
    confidence: float
    method: str
    metadata: Dict[str, Any]


class StatisticalPredictor:
    """Statistical prediction based on frequency analysis and patterns."""

    def __init__(self, lottery_type: str = "powerball"):
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config

    def predict_frequency_based(self, last_n: int = 100, num_predictions: int = 5) -> List[PredictionResult]:
        """Predict based on historical frequency (hot numbers)."""
        freq = self.dm.get_number_frequency(last_n)
        sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        results = []
        for i in range(num_predictions):
            # Weighted random selection favoring hot numbers
            numbers = list(freq.keys())
            weights = np.array(list(freq.values())) + 1  # Add 1 to avoid zero weights
            weights = weights / weights.sum()

            main = sorted(np.random.choice(
                numbers,
                size=self.config['main_numbers'],
                replace=False,
                p=weights
            ).tolist())

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus_freq = self.dm.get_bonus_frequency(last_n)
                b_numbers = list(bonus_freq.keys())
                b_weights = np.array(list(bonus_freq.values())) + 1
                b_weights = b_weights / b_weights.sum()
                bonus = sorted(np.random.choice(
                    b_numbers,
                    size=self.config['bonus_numbers'],
                    replace=False,
                    p=b_weights
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.6,
                method="frequency_weighted",
                metadata={"last_n_draws": last_n, "strategy": "hot_numbers"}
            ))

        return results

    def predict_cold_numbers(self, last_n: int = 100, num_predictions: int = 3) -> List[PredictionResult]:
        """Predict based on cold numbers (due to appear)."""
        freq = self.dm.get_number_frequency(last_n)
        sorted_nums = sorted(freq.items(), key=lambda x: x[1])

        results = []
        cold_pool = [n for n, _ in sorted_nums[:20]]  # 20 coldest numbers

        for i in range(num_predictions):
            main = sorted(np.random.choice(
                cold_pool,
                size=self.config['main_numbers'],
                replace=False
            ).tolist())

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus_freq = self.dm.get_bonus_frequency(last_n)
                b_sorted = sorted(bonus_freq.items(), key=lambda x: x[1])
                cold_bonus = [n for n, _ in b_sorted[:10]]
                bonus = sorted(np.random.choice(
                    cold_bonus,
                    size=self.config['bonus_numbers'],
                    replace=False
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.4,
                method="cold_numbers",
                metadata={"last_n_draws": last_n, "strategy": "due_numbers"}
            ))

        return results

    def predict_balanced(self, last_n: int = 200, num_predictions: int = 5) -> List[PredictionResult]:
        """Balanced prediction using sum statistics and distribution."""
        stats = self.dm.get_sum_statistics(last_n)
        target_sum = stats['mean']
        freq = self.dm.get_number_frequency(last_n)

        results = []
        for i in range(num_predictions):
            # Generate numbers that sum close to historical average
            attempts = 0
            while attempts < 1000:
                numbers = list(freq.keys())
                weights = np.array(list(freq.values())) + 1
                weights = weights / weights.sum()

                main = sorted(np.random.choice(
                    numbers,
                    size=self.config['main_numbers'],
                    replace=False,
                    p=weights
                ).tolist())

                if abs(sum(main) - target_sum) < stats['std']:
                    break
                attempts += 1

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus_freq = self.dm.get_bonus_frequency(last_n)
                b_numbers = list(bonus_freq.keys())
                b_weights = np.array(list(bonus_freq.values())) + 1
                b_weights = b_weights / b_weights.sum()
                bonus = sorted(np.random.choice(
                    b_numbers,
                    size=self.config['bonus_numbers'],
                    replace=False,
                    p=b_weights
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.65,
                method="balanced_statistical",
                metadata={
                    "target_sum": target_sum,
                    "actual_sum": sum(main),
                    "last_n_draws": last_n
                }
            ))

        return results

    def predict_pair_based(self, last_n: int = 100, num_predictions: int = 3) -> List[PredictionResult]:
        """Predict based on frequent number pairs."""
        pairs = self.dm.get_pair_frequency(last_n)
        top_pairs = list(pairs.keys())[:20]

        results = []
        for i in range(num_predictions):
            # Start with a frequent pair
            pair = list(top_pairs[np.random.randint(0, len(top_pairs))])
            remaining = self.config['main_numbers'] - 2

            freq = self.dm.get_number_frequency(last_n)
            numbers = [n for n in freq.keys() if n not in pair]
            weights = np.array([freq[n] for n in numbers]) + 1
            weights = weights / weights.sum()

            extra = np.random.choice(
                numbers,
                size=remaining,
                replace=False,
                p=weights
            ).tolist()

            main = sorted(pair + extra)

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus_freq = self.dm.get_bonus_frequency(last_n)
                b_numbers = list(bonus_freq.keys())
                b_weights = np.array(list(bonus_freq.values())) + 1
                b_weights = b_weights / b_weights.sum()
                bonus = sorted(np.random.choice(
                    b_numbers,
                    size=self.config['bonus_numbers'],
                    replace=False,
                    p=b_weights
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.55,
                method="pair_based",
                metadata={"seed_pair": pair, "last_n_draws": last_n}
            ))

        return results


class MLModelPredictor:
    """Machine Learning based prediction using trained models."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, lookback: int = 10, test_size: float = 0.2) -> Dict[str, float]:
        """Train ML models on historical data."""
        logger.info(f"Training ML models for {self.config['name']}...")

        X, y = self.dm.prepare_ml_features(lookback)
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            # 'gradient_boost': GradientBoostingClassifier(
            #     n_estimators=50, max_depth=5, random_state=42,
            #     subsample=0.8
            # ),  # Disabled - too slow on large datasets
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=42,
                early_stopping=True
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, max_depth=8,
                random_state=42, n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=0
            )
        }

        results = {}
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            try:
                # For multi-output, train one model per number position
                # Simplified: predict top-k numbers
                model.fit(X_train_scaled, y_train.argmax(axis=1))

                # Evaluate
                train_acc = model.score(X_train_scaled, y_train.argmax(axis=1))
                test_acc = model.score(X_test_scaled, y_test.argmax(axis=1))

                self.models[name] = model
                results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
                logger.info(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")
            except Exception as e:
                logger.warning(f"{name} training failed: {e}")
                results[name] = {'train_acc': 0, 'test_acc': 0, 'error': str(e)}
                continue

        self.is_trained = True
        self.save_models()
        return results

    def save_models(self):
        """Save trained models to disk."""
        model_path = MODEL_DIR / f"{self.lottery_type}_models.joblib"
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'config': self.config
        }, model_path)
        logger.info(f"Models saved to {model_path}")

    def load_models(self) -> bool:
        """Load trained models from disk."""
        model_path = MODEL_DIR / f"{self.lottery_type}_models.joblib"
        if model_path.exists():
            data = joblib.load(model_path)
            self.models = data['models']
            self.scaler = data['scaler']
            self.is_trained = True
            logger.info(f"Models loaded from {model_path}")
            return True
        return False

    def predict(self, num_predictions: int = 5) -> List[PredictionResult]:
        """Generate predictions using trained ML models."""
        if not self.is_trained:
            if not self.load_models():
                logger.warning("Models not trained, training now...")
                self.train()

        # Get latest features
        X, _ = self.dm.prepare_ml_features(lookback=10)
        if len(X) == 0:
            return []

        latest_features = X[-1:].reshape(1, -1)
        latest_scaled = self.scaler.transform(latest_features)

        results = []
        for i in range(num_predictions):
            # Ensemble prediction from all models
            predictions = []
            for name, model in self.models.items():
                # Get probability distribution
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(latest_scaled)[0]
                    # Sample from distribution
                    pred = np.random.choice(
                        len(proba),
                        size=self.config['main_numbers'],
                        replace=False,
                        p=proba
                    )
                else:
                    pred = model.predict(latest_scaled)
                    pred = np.random.choice(
                        self.config['main_max'],
                        size=self.config['main_numbers'],
                        replace=False
                    )
                predictions.extend([p + 1 for p in pred])  # Convert to 1-indexed

            # Ensemble: take most frequent predictions
            from collections import Counter
            counter = Counter(predictions)
            main = sorted([n for n, _ in counter.most_common(self.config['main_numbers'])])

            # Bonus prediction (simplified)
            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus = sorted(np.random.choice(
                    range(1, self.config['bonus_max'] + 1),
                    size=self.config['bonus_numbers'],
                    replace=False
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.7,
                method="ml_ensemble",
                metadata={"models_used": list(self.models.keys())}
            ))

        return results


class DeepLearningPredictor:
    """Deep learning model for lottery prediction (experimental)."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, input_dim: int, output_dim: int):
        """Build a neural network model."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, epochs: int = 100, batch_size: int = 32):
        """Train the deep learning model."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow not available, skipping deep learning")
            return

        X, y = self.dm.prepare_ml_features(lookback=10)
        X_scaled = self.scaler.fit_transform(X)

        # Convert y to categorical
        from tensorflow.keras.utils import to_categorical
        y_cat = np.zeros((len(y), self.config['main_max']))
        for i, row in enumerate(y):
            indices = np.where(row == 1)[0]
            for idx in indices:
                y_cat[i, idx] = 1

        self.build_model(X.shape[1], self.config['main_max'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            X_scaled, y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        logger.info("Deep learning model trained")
        return history

    def predict(self, num_predictions: int = 3) -> List[PredictionResult]:
        """Generate predictions using deep learning model."""
        if self.model is None:
            logger.warning("Deep learning model not trained")
            return []

        X, _ = self.dm.prepare_ml_features(lookback=10)
        latest = self.scaler.transform(X[-1:].reshape(1, -1))

        results = []
        for _ in range(num_predictions):
            proba = self.model.predict(latest, verbose=0)[0]
            # Sample from probability distribution
            main = sorted(np.random.choice(
                range(1, self.config['main_max'] + 1),
                size=self.config['main_numbers'],
                replace=False,
                p=proba
            ).tolist())

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus = sorted(np.random.choice(
                    range(1, self.config['bonus_max'] + 1),
                    size=self.config['bonus_numbers'],
                    replace=False
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.65,
                method="deep_learning",
                metadata={}
            ))

        return results


class TimeSeriesPredictor:
    """Time-series forecasting for lottery numbers using ARIMA, Prophet, and Exponential Smoothing."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.models = {}
        self.is_trained = False

    def prepare_series(self, last_n: int = 200) -> pd.DataFrame:
        """Prepare time series data for each number position."""
        df = self.dm.load_data().tail(last_n).copy()

        # Create a series for each position (1st, 2nd, 3rd, etc. drawn number)
        series_data = []
        for i, row in df.iterrows():
            for pos, num in enumerate(row['main_numbers']):
                series_data.append({
                    'date': row['date'],
                    'position': pos,
                    'number': num
                })
        return pd.DataFrame(series_data)

    def train_arima(self, series: pd.Series, order: Tuple = (2, 1, 2)) -> Any:
        """Train ARIMA model on a time series."""
        if not HAS_STATSMODELS:
            return None
        try:
            model = ARIMA(series, order=order)
            return model.fit()
        except Exception as e:
            logger.warning(f"ARIMA training failed: {e}")
            return None

    def train_prophet(self, df: pd.DataFrame) -> Any:
        """Train Prophet model on time series with date column."""
        if not HAS_PROPHET:
            return None
        try:
            prophet_df = df.rename(columns={'date': 'ds', 'number': 'y'})
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_df)
            return model
        except Exception as e:
            logger.warning(f"Prophet training failed: {e}")
            return None

    def train_exponential_smoothing(self, series: pd.Series) -> Any:
        """Train Exponential Smoothing model."""
        if not HAS_STATSMODELS:
            return None
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=52)
            return model.fit()
        except Exception as e:
            logger.warning(f"Exponential Smoothing training failed: {e}")
            return None

    def train(self, last_n: int = 200) -> Dict:
        """Train all time-series models."""
        logger.info(f"Training time-series models for {self.config['name']}")

        series_df = self.prepare_series(last_n)
        results = {}

        # Train per position
        for pos in range(self.config['main_numbers']):
            pos_data = series_df[series_df['position'] == pos]['number'].values
            if len(pos_data) < 10:
                continue

            pos_series = pd.Series(pos_data)

            # ARIMA
            arima_model = self.train_arima(pos_series)
            if arima_model:
                self.models[f'arima_pos_{pos}'] = arima_model
                results[f'arima_pos_{pos}'] = {'aic': arima_model.aic}

            # Exponential Smoothing
            ets_model = self.train_exponential_smoothing(pos_series)
            if ets_model:
                self.models[f'ets_pos_{pos}'] = ets_model
                results[f'ets_pos_{pos}'] = {'aic': ets_model.aic}

        # Prophet on all numbers combined (using date and number)
        if HAS_PROPHET:
            try:
                prophet_df = series_df[['date', 'number']].rename(columns={'date': 'ds', 'number': 'y'})
                prophet_model = self.train_prophet(prophet_df)
                if prophet_model:
                    self.models['prophet'] = prophet_model
                    results['prophet'] = {'trained': True}
            except Exception as e:
                logger.warning(f"Prophet training failed: {e}")

        self.is_trained = True
        logger.info(f"Time-series models trained: {list(self.models.keys())}")
        return results

    def predict_arima(self, pos: int, steps: int = 1) -> Optional[float]:
        """Get ARIMA prediction for a position."""
        model = self.models.get(f'arima_pos_{pos}')
        if model:
            try:
                forecast = model.forecast(steps=steps)
                return float(forecast.iloc[-1])
            except Exception:
                return None
        return None

    def predict_ets(self, pos: int, steps: int = 1) -> Optional[float]:
        """Get ETS prediction for a position."""
        model = self.models.get(f'ets_pos_{pos}')
        if model:
            try:
                forecast = model.forecast(steps=steps)
                return float(forecast.iloc[-1])
            except Exception:
                return None
        return None

    def predict_prophet(self, future_date: pd.Timestamp) -> Optional[float]:
        """Get Prophet prediction for a future date."""
        model = self.models.get('prophet')
        if model:
            try:
                future = pd.DataFrame({'ds': [future_date]})
                forecast = model.predict(future)
                return float(forecast['yhat'].iloc[0])
            except Exception:
                return None
        return None

    def predict(self, num_predictions: int = 5, last_n: int = 200) -> List[PredictionResult]:
        """Generate predictions using time-series models."""
        if not self.is_trained:
            self.train(last_n)

        if not self.models:
            logger.warning("No time-series models trained, falling back to statistical")
            pred = StatisticalPredictor(self.lottery_type)
            return pred.predict_balanced(last_n=last_n, num_predictions=num_predictions)

        results = []
        next_date = pd.Timestamp.now() + pd.Timedelta(days=2)  # Approximate next draw

        for _ in range(num_predictions):
            main_numbers = []

            for pos in range(self.config['main_numbers']):
                preds = []

                # ARIMA prediction
                arima_pred = self.predict_arima(pos)
                if arima_pred is not None:
                    preds.append(('arima', arima_pred))

                # ETS prediction
                ets_pred = self.predict_ets(pos)
                if ets_pred is not None:
                    preds.append(('ets', ets_pred))

                # Combine predictions (weighted average)
                if preds:
                    weighted = sum(p[1] for p in preds) / len(preds)
                    num = int(round(weighted))
                    num = max(1, min(self.config['main_max'], num))
                else:
                    num = np.random.randint(1, self.config['main_max'] + 1)

                main_numbers.append(num)

            # Ensure unique numbers
            main_numbers = list(dict.fromkeys(main_numbers))
            while len(main_numbers) < self.config['main_numbers']:
                num = np.random.randint(1, self.config['main_max'] + 1)
                if num not in main_numbers:
                    main_numbers.append(num)
            main_numbers = sorted(main_numbers[:self.config['main_numbers']])

            # Bonus numbers
            bonus_numbers = []
            if self.config['bonus_numbers'] > 0:
                bonus_numbers = sorted(np.random.choice(
                    range(1, self.config['bonus_max'] + 1),
                    size=self.config['bonus_numbers'],
                    replace=False
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main_numbers,
                bonus_numbers=bonus_numbers,
                confidence=0.55,  # Time-series typically lower confidence
                method="time_series",
                metadata={"models_used": list(self.models.keys())}
            ))

        return results


class EnsemblePredictor:
    """Combines multiple prediction strategies for best results."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.statistical = StatisticalPredictor(lottery_type)
        self.ml = MLModelPredictor(lottery_type)
        self.dl = DeepLearningPredictor(lottery_type)
        self.ts = TimeSeriesPredictor(lottery_type)
        self.transformer = TransformerPredictor(lottery_type)
        self.online = OnlineLearningPredictor(lottery_type)

    def predict_all(self, num_each: int = 3) -> Dict[str, List[PredictionResult]]:
        """Get predictions from all methods."""
        results = {}

        # Statistical methods
        results['frequency'] = self.statistical.predict_frequency_based(num_predictions=num_each)
        results['cold'] = self.statistical.predict_cold_numbers(num_predictions=num_each)
        results['balanced'] = self.statistical.predict_balanced(num_predictions=num_each)
        results['pairs'] = self.statistical.predict_pair_based(num_predictions=num_each)

        # ML methods
        results['ml'] = self.ml.predict(num_predictions=num_each)

        # Time-series methods
        results['time_series'] = self.ts.predict(num_predictions=num_each)

        # v1.3: Transformer (if trained)
        try:
            if HAS_TORCH and self.transformer.model is not None:
                results['transformer'] = self.transformer.predict(num_predictions=num_each)
        except Exception:
            pass

        # v1.3: Online learning
        try:
            results['online_learning'] = self.online.predict(num_predictions=num_each)
        except Exception:
            pass

        # Deep learning (optional)
        # results['dl'] = self.dl.predict(num_predictions=num_each)

        return results

    def get_consensus_prediction(self, num_predictions: int = 5) -> List[PredictionResult]:
        """Get consensus prediction from all methods."""
        all_results = self.predict_all(num_each=3)

        # Collect all predictions with weights
        predictions = []
        weights = {
            'frequency': 1.0,
            'cold': 0.8,
            'balanced': 1.2,
            'pairs': 1.0,
            'ml': 1.5,
            'time_series': 0.9,
            'transformer': 1.1,
            'online_learning': 1.2,
        }

        for method, preds in all_results.items():
            w = weights.get(method, 1.0)
            for p in preds:
                predictions.append((tuple(p.main_numbers), tuple(p.bonus_numbers), w * p.confidence))

        # Weighted voting
        from collections import defaultdict
        vote_counts = defaultdict(float)
        for main, bonus, conf in predictions:
            vote_counts[(main, bonus)] += conf

        # Top predictions
        sorted_preds = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        results = []
        for (main, bonus), score in sorted_preds[:num_predictions]:
            results.append(PredictionResult(
                main_numbers=list(main),
                bonus_numbers=list(bonus),
                confidence=min(score / sum(w for _, _, w in predictions), 0.95),
                method="ensemble_consensus",
                metadata={"vote_score": score, "methods_combined": len(all_results)}
            ))

        return results


class TransformerPredictor:
    """Transformer-based sequence model for lottery prediction (v1.3)."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TransformerModel(nn.Module):
        """Transformer model for sequence prediction."""
        def __init__(self, input_dim: int, output_dim: int, seq_len: int = 10,
                     d_model: int = 128, nhead: int = 8, num_layers: int = 3):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, output_dim)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, input_dim)
            x = self.input_proj(x) + self.pos_encoding.unsqueeze(0)
            x = self.transformer(x)
            x = x[:, -1, :]  # Take last token
            return self.output_head(x)

    def prepare_sequence_data(self, lookback: int = 10, seq_len: int = 10):
        """Prepare sequential data for transformer."""
        X, y = self.dm.prepare_ml_features(lookback=lookback)
        if len(X) < seq_len + 1:
            return None, None

        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        sequences = []
        targets = []
        for i in range(len(X_scaled) - seq_len):
            sequences.append(X_scaled[i:i + seq_len])
            # Target: the numbers in the next draw
            next_draw_idx = i + seq_len
            targets.append(y[next_draw_idx])

        return np.array(sequences), np.array(targets)

    def build_model(self, input_dim: int, output_dim: int, seq_len: int = 10):
        """Build transformer model."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping transformer")
            return None

        self.model = self.TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len
        ).to(self.device)
        return self.model

    def train(self, epochs: int = 50, batch_size: int = 32, seq_len: int = 10, lr: float = 1e-3):
        """Train transformer model."""
        if not HAS_TORCH:
            return

        X_seq, y_seq = self.prepare_sequence_data(seq_len=seq_len)
        if X_seq is None:
            logger.warning("Not enough data for transformer training")
            return

        # Convert y to multi-hot encoding
        output_dim = self.config['main_max']
        y_multihot = np.zeros((len(y_seq), output_dim))
        for i, row in enumerate(y_seq):
            indices = np.where(row == 1)[0]
            for idx in indices:
                y_multihot[i, idx] = 1

        # Split
        split = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_multihot[:split], y_multihot[split:]

        # Build model
        self.build_model(X_seq.shape[2], output_dim, seq_len)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                xb = torch.FloatTensor(X_train[idx]).to(self.device)
                yb = torch.FloatTensor(y_train[idx]).to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                xb = torch.FloatTensor(X_val).to(self.device)
                yb = torch.FloatTensor(y_val).to(self.device)
                val_out = self.model(xb)
                val_loss = criterion(val_out, yb)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                logger.info(f"Transformer epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")

        logger.info("Transformer model trained")

    def predict(self, num_predictions: int = 3, seq_len: int = 10) -> List[PredictionResult]:
        """Generate predictions using transformer."""
        if not HAS_TORCH or self.model is None:
            logger.warning("Transformer not trained")
            return []

        X_seq, _ = self.prepare_sequence_data(seq_len=seq_len)
        if X_seq is None:
            return []

        self.model.eval()
        latest_seq = torch.FloatTensor(X_seq[-1:]).to(self.device)

        results = []
        with torch.no_grad():
            for _ in range(num_predictions):
                logits = self.model(latest_seq)[0]
                probs = torch.sigmoid(logits).cpu().numpy()

                # Sample from distribution
                main = sorted(np.random.choice(
                    range(1, self.config['main_max'] + 1),
                    size=self.config['main_numbers'],
                    replace=False,
                    p=probs / probs.sum() if probs.sum() > 0 else None
                ).tolist())

                bonus = []
                if self.config['bonus_numbers'] > 0:
                    bonus = sorted(np.random.choice(
                        range(1, self.config['bonus_max'] + 1),
                        size=self.config['bonus_numbers'],
                        replace=False
                    ).tolist())

                results.append(PredictionResult(
                    main_numbers=main,
                    bonus_numbers=bonus,
                    confidence=0.6,
                    method="transformer",
                    metadata={"model": "transformer", "seq_len": seq_len}
                ))

        return results


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna (v1.3)."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config

    def optimize_random_forest(self, n_trials: int = 50) -> Dict:
        """Optimize Random Forest hyperparameters."""
        if not HAS_OPTUNA:
            logger.warning("Optuna not available")
            return {}

        X, y = self.dm.prepare_ml_features(lookback=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train.argmax(axis=1))
            score = model.score(X_test, y_test.argmax(axis=1))
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best RF params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")

        return study.best_params

    def optimize_xgboost(self, n_trials: int = 50) -> Dict:
        """Optimize XGBoost hyperparameters."""
        if not HAS_OPTUNA:
            logger.warning("Optuna not available")
            return {}

        X, y = self.dm.prepare_ml_features(lookback=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train.argmax(axis=1))
            score = model.score(X_test, y_test.argmax(axis=1))
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best XGB params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")

        return study.best_params

    def optimize_mlp(self, n_trials: int = 30) -> Dict:
        """Optimize MLP hyperparameters."""
        if not HAS_OPTUNA:
            logger.warning("Optuna not available")
            return {}

        X, y = self.dm.prepare_ml_features(lookback=10)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []
            for i in range(n_layers):
                hidden_sizes.append(trial.suggest_int(f'hidden_{i}', 32, 512))

            params = {
                'hidden_layer_sizes': tuple(hidden_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 200,
                'early_stopping': True,
                'random_state': 42
            }

            model = MLPClassifier(**params)
            model.fit(X_train, y_train.argmax(axis=1))
            score = model.score(X_test, y_test.argmax(axis=1))
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best MLP params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")

        return study.best_params


class ModelExplainer:
    """Model explainability using SHAP (v1.3)."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.explainers = {}

    def explain_random_forest(self, model, X_sample: np.ndarray) -> Dict:
        """Generate SHAP explanations for Random Forest."""
        if not HAS_SHAP:
            logger.warning("SHAP not available")
            return {}

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Feature importance
            if isinstance(shap_values, list):
                # Multi-class - average across classes
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            feature_names = self._get_feature_names(X_sample.shape[1])
            importance = dict(zip(feature_names, mean_shap))

            # Top features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

            return {
                'shap_values': shap_values,
                'feature_importance': importance,
                'top_features': top_features,
                'explainer': explainer
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}

    def explain_any_model(self, model, X_sample: np.ndarray, model_type: str = 'tree') -> Dict:
        """Generate SHAP explanations for any model type."""
        if not HAS_SHAP:
            logger.warning("SHAP not available")
            return {}

        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                # Use KernelExplainer as fallback (slower)
                explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])

            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            feature_names = self._get_feature_names(X_sample.shape[1])
            importance = dict(zip(feature_names, mean_shap))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

            return {
                'shap_values': shap_values,
                'feature_importance': importance,
                'top_features': top_features
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}

    def _get_feature_names(self, n_features: int) -> List[str]:
        """Generate feature names based on data preparation."""
        names = []
        main_max = self.config['main_max']
        bonus_max = self.config['bonus_max']

        # Features: frequency counts for each number position + recency + sum + even/odd + high/low
        # This matches the prepare_ml_features in data.py
        lookback = 10
        for i in range(lookback):
            for j in range(main_max):
                names.append(f'draw_{i}_num_{j+1}_freq')
            for j in range(bonus_max):
                names.append(f'draw_{i}_bonus_{j+1}_freq')

        # Additional features: sum, even_count, high_count for each draw
        for i in range(lookback):
            names.append(f'draw_{i}_sum')
            names.append(f'draw_{i}_even')
            names.append(f'draw_{i}_high')

        return names[:n_features]

    def plot_summary(self, shap_values, X_sample: np.ndarray):
        """Plot SHAP summary plot."""
        if not HAS_SHAP:
            return
        try:
            feature_names = self._get_feature_names(X_sample.shape[1])
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")


class OnlineLearningPredictor:
    """Online/incremental learning for continuous model updates (v1.3)."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config
        self.models = {}
        self.scaler = StandardScaler()
        self.is_initialized = False
        self.update_buffer = []  # Buffer for batch updates

    def initialize(self):
        """Initialize models for online learning."""
        from sklearn.linear_model import SGDClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier

        # Use models that support partial_fit
        self.models = {
            'sgd_logistic': SGDClassifier(loss='log_loss', random_state=42, max_iter=1000),
            'sgd_hinge': SGDClassifier(loss='hinge', random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB(),
            # Random Forest doesn't support partial_fit, retrain periodically
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        }

        # Get initial training data
        X, y = self.dm.prepare_ml_features(lookback=10)
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
            # Initialize with first batch
            for name, model in self.models.items():
                if hasattr(model, 'partial_fit'):
                    classes = np.arange(self.config['main_max'])
                    model.partial_fit(X_scaled, y.argmax(axis=1), classes=classes)
                else:
                    model.fit(X_scaled, y.argmax(axis=1))

            self.is_initialized = True
            logger.info("Online learning models initialized")

    def update(self, new_draws: List[Dict]):
        """Update models with new draw data."""
        if not self.is_initialized:
            self.initialize()

        # Add new draws to buffer
        self.update_buffer.extend(new_draws)

        # Update periodically (every 10 new draws)
        if len(self.update_buffer) >= 10:
            self._retrain_incremental()
            self.update_buffer = []

    def _retrain_incremental(self):
        """Incremental retrain with buffered data."""
        # Reload all data and retrain incremental models
        X, y = self.dm.prepare_ml_features(lookback=10)
        if len(X) == 0:
            return

        X_scaled = self.scaler.transform(X)

        for name, model in self.models.items():
            if hasattr(model, 'partial_fit'):
                classes = np.arange(self.config['main_max'])
                model.partial_fit(X_scaled, y.argmax(axis=1), classes=classes)
            elif name == 'random_forest':
                # Retrain RF periodically
                model.fit(X_scaled, y.argmax(axis=1))

        logger.info(f"Online models updated with {len(X)} samples")

    def predict(self, num_predictions: int = 3) -> List[PredictionResult]:
        """Generate predictions using online models."""
        if not self.is_initialized:
            self.initialize()

        X, _ = self.dm.prepare_ml_features(lookback=10)
        if len(X) == 0:
            return []

        latest = self.scaler.transform(X[-1:].reshape(1, -1))

        results = []
        for _ in range(num_predictions):
            predictions = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(latest)[0]
                    pred = np.random.choice(
                        len(proba),
                        size=self.config['main_numbers'],
                        replace=False,
                        p=proba
                    )
                else:
                    pred = model.predict(latest)
                    pred = np.random.choice(
                        self.config['main_max'],
                        size=self.config['main_numbers'],
                        replace=False
                    )
                predictions.extend([p + 1 for p in pred])

            # Ensemble voting
            from collections import Counter
            counter = Counter(predictions)
            main = sorted([n for n, _ in counter.most_common(self.config['main_numbers'])])

            bonus = []
            if self.config['bonus_numbers'] > 0:
                bonus = sorted(np.random.choice(
                    range(1, self.config['bonus_max'] + 1),
                    size=self.config['bonus_numbers'],
                    replace=False
                ).tolist())

            results.append(PredictionResult(
                main_numbers=main,
                bonus_numbers=bonus,
                confidence=0.65,
                method="online_learning",
                metadata={"models_used": list(self.models.keys())}
            ))

        return results


def get_predictor(lottery_type: str, method: str = "ensemble") -> Any:
    """Factory function to get predictor."""
    if method == "statistical":
        return StatisticalPredictor(lottery_type)
    elif method == "ml":
        return MLModelPredictor(lottery_type)
    elif method == "dl":
        return DeepLearningPredictor(lottery_type)
    elif method == "time_series":
        return TimeSeriesPredictor(lottery_type)
    elif method == "transformer":
        return TransformerPredictor(lottery_type)
    elif method == "online_learning":
        return OnlineLearningPredictor(lottery_type)
    elif method == "ensemble":
        return EnsemblePredictor(lottery_type)
    elif method == "optuna":
        return HyperparameterOptimizer(lottery_type)
    elif method == "shap":
        return ModelExplainer(lottery_type)
    else:
        return EnsemblePredictor(lottery_type)


if __name__ == "__main__":
    # Test predictors
    for lt in ["florida_lotto", "powerball"]:
        print(f"\n{'='*50}")
        print(f"Testing {lt}")
        print(f"{'='*50}")

        ensemble = EnsemblePredictor(lt)
        predictions = ensemble.get_consensus_prediction(3)

        for i, p in enumerate(predictions, 1):
            main_str = ' '.join(f"{n:2d}" for n in p.main_numbers)
            bonus_str = ' '.join(f"{n:2d}" for n in p.bonus_numbers) if p.bonus_numbers else "N/A"
            print(f"  Prediction {i}: [{main_str}]  Bonus: [{bonus_str}]  Confidence: {p.confidence:.2%}  Method: {p.method}")