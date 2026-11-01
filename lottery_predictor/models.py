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
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200, max_depth=10,
                random_state=42, n_jobs=-1,
                eval_metric='mlogloss'
            )
        }

        results = {}
        for name, model in models_config.items():
            logger.info(f"Training {name}...")

            # For multi-output, train one model per number position
            # Simplified: predict top-k numbers
            model.fit(X_train_scaled, y_train.argmax(axis=1))

            # Evaluate
            train_acc = model.score(X_train_scaled, y_train.argmax(axis=1))
            test_acc = model.score(X_test_scaled, y_test.argmax(axis=1))

            self.models[name] = model
            results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
            logger.info(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

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


class EnsemblePredictor:
    """Combines multiple prediction strategies for best results."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.statistical = StatisticalPredictor(lottery_type)
        self.ml = MLModelPredictor(lottery_type)
        self.dl = DeepLearningPredictor(lottery_type)

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


def get_predictor(lottery_type: str, method: str = "ensemble") -> Any:
    """Factory function to get predictor."""
    if method == "statistical":
        return StatisticalPredictor(lottery_type)
    elif method == "ml":
        return MLModelPredictor(lottery_type)
    elif method == "dl":
        return DeepLearningPredictor(lottery_type)
    elif method == "ensemble":
        return EnsemblePredictor(lottery_type)
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