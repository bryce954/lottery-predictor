#!/usr/bin/env python3
"""
Backtesting Framework for Lottery Predictor
Evaluates prediction methods against historical data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from lottery_predictor.data import get_data_manager, LOTTERY_CONFIGS
from lottery_predictor.models import (
    StatisticalPredictor,
    MLModelPredictor,
    EnsemblePredictor,
    TimeSeriesPredictor,
    PredictionResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    method: str
    lottery: str
    test_period: Tuple[str, str]
    total_draws: int
    metrics: Dict[str, float]
    detailed_results: List[Dict]


class Backtester:
    """Backtesting framework for evaluating prediction methods."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config

    def _calculate_metrics(self, predictions: List[List[int]], actual: List[List[int]]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if not predictions or not actual:
            return {}

        total_draws = len(actual)
        hits_at_1 = hits_at_2 = hits_at_3 = hits_at_4 = hits_at_5 = 0
        total_matches = 0

        for pred, act in zip(predictions, actual):
            pred_set = set(pred)
            act_set = set(act)
            matches = len(pred_set & act_set)
            total_matches += matches

            if matches >= 1: hits_at_1 += 1
            if matches >= 2: hits_at_2 += 1
            if matches >= 3: hits_at_3 += 1
            if matches >= 4: hits_at_4 += 1
            if matches >= 5: hits_at_5 += 1

        return {
            'avg_matches': total_matches / total_draws,
            'hit_rate_1+': hits_at_1 / total_draws,
            'hit_rate_2+': hits_at_2 / total_draws,
            'hit_rate_3+': hits_at_3 / total_draws,
            'hit_rate_4+': hits_at_4 / total_draws,
            'hit_rate_5': hits_at_5 / total_draws,
            'total_draws': total_draws
        }

    def run_backtest(
        self,
        method: str,
        train_window: int = 200,
        test_window: int = 50,
        step: int = 10,
        num_predictions: int = 3
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            method: Prediction method ('statistical', 'frequency', 'cold', 'pairs', 'ml', 'time_series', 'ensemble')
            train_window: Number of historical draws to use for training
            test_window: Number of draws to test on
            step: Step size for walk-forward (how many draws to advance each iteration)
            num_predictions: Number of predictions to generate per iteration
        """
        df = self.dm.load_data()
        total_draws = len(df)

        if total_draws < train_window + test_window:
            raise ValueError(f"Not enough data: {total_draws} draws, need {train_window + test_window}")

        all_predictions = []
        all_actual = []
        detailed = []

        # Walk-forward validation
        for start_idx in range(train_window, total_draws - test_window + 1, step):
            end_idx = min(start_idx + test_window, total_draws)

            # Training data
            train_df = df.iloc[start_idx - train_window:start_idx]

            # Test data
            test_df = df.iloc[start_idx:end_idx]
            actual_draws = [row['main_numbers'] for _, row in test_df.iterrows()]

            # Generate predictions using training data
            preds = self._predict_with_method(
                method, train_df, num_predictions
            )

            # For each test draw, evaluate predictions
            for i, (pred, act) in enumerate(zip(preds, actual_draws)):
                all_predictions.append(pred)
                all_actual.append(act)

                # Check matches
                matches = len(set(pred) & set(act))
                detailed.append({
                    'iteration': len(all_predictions),
                    'test_draw_idx': start_idx + i,
                    'date': test_df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(test_df.iloc[i]['date'], 'strftime') else str(test_df.iloc[i]['date']),
                    'predicted': pred,
                    'actual': act,
                    'matches': matches,
                    'matched_numbers': list(set(pred) & set(act))
                })

        metrics = self._calculate_metrics(all_predictions, all_actual)

        return BacktestResult(
            method=method,
            lottery=self.lottery_type,
            test_period=(
                df.iloc[train_window]['date'].strftime('%Y-%m-%d'),
                df.iloc[-1]['date'].strftime('%Y-%m-%d')
            ),
            total_draws=len(all_predictions),
            metrics=metrics,
            detailed_results=detailed
        )

    def _predict_with_method(self, method: str, train_df: pd.DataFrame, num_predictions: int) -> List[List[int]]:
        """Generate predictions using training data."""
        # For time-series, we need to use a cached predictor or retrain
        # This is a simplified approach - in practice you'd retrain models

        # Use a persistent predictor for time-series to avoid retraining
        if not hasattr(self, '_ts_predictor') or self._ts_predictor is None:
            self._ts_predictor = TimeSeriesPredictor(self.lottery_type)

        if method == "statistical":
            pred = StatisticalPredictor(self.lottery_type)
            # Use balanced method
            results = pred.predict_balanced(last_n=len(train_df), num_predictions=num_predictions)
        elif method == "frequency":
            pred = StatisticalPredictor(self.lottery_type)
            results = pred.predict_frequency_based(last_n=len(train_df), num_predictions=num_predictions)
        elif method == "cold":
            pred = StatisticalPredictor(self.lottery_type)
            results = pred.predict_cold_numbers(last_n=len(train_df), num_predictions=num_predictions)
        elif method == "pairs":
            pred = StatisticalPredictor(self.lottery_type)
            results = pred.predict_pair_based(last_n=len(train_df), num_predictions=num_predictions)
        elif method == "ml":
            pred = MLModelPredictor(self.lottery_type)
            if not pred.is_trained:
                pred.train()
            results = pred.predict(num_predictions=num_predictions)
        elif method == "time_series":
            # Use cached predictor to avoid retraining every iteration
            results = self._ts_predictor.predict(num_predictions=num_predictions, last_n=len(train_df))
        else:  # ensemble
            pred = EnsemblePredictor(self.lottery_type)
            results = pred.get_consensus_prediction(num_predictions=num_predictions)

        return [p.main_numbers for p in results]

    def compare_methods(
        self,
        methods: List[str],
        train_window: int = 200,
        test_window: int = 50,
        step: int = 10,
        num_predictions: int = 3
    ) -> Dict[str, BacktestResult]:
        """Compare multiple methods."""
        results = {}
        for method in methods:
            logger.info(f"Backtesting {method}...")
            results[method] = self.run_backtest(
                method, train_window, test_window, step, num_predictions
            )
        return results

    def print_comparison(self, results: Dict[str, BacktestResult]):
        """Print comparison table."""
        print(f"\n{'='*80}")
        print(f"BACKTEST COMPARISON - {self.config['name']}")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'Draws':<8} {'Avg Match':<10} {'Hit 1+':<10} {'Hit 2+':<10} {'Hit 3+':<10} {'Hit 4+':<10} {'Hit 5':<10}")
        print(f"{'-'*80}")

        for method, result in results.items():
            m = result.metrics
            print(f"{method:<20} {m.get('total_draws', 0):<8} {m.get('avg_matches', 0):<10.3f} "
                  f"{m.get('hit_rate_1+', 0):<10.3f} {m.get('hit_rate_2+', 0):<10.3f} "
                  f"{m.get('hit_rate_3+', 0):<10.3f} {m.get('hit_rate_4+', 0):<10.3f} "
                  f"{m.get('hit_rate_5', 0):<10.3f}")

    def save_results(self, results: Dict[str, BacktestResult], filepath: str):
        """Save backtest results to JSON."""
        data = {}
        for method, result in results.items():
            data[method] = {
                'method': result.method,
                'lottery': result.lottery,
                'test_period': result.test_period,
                'total_draws': result.total_draws,
                'metrics': result.metrics,
                'detailed_results': result.detailed_results
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {filepath}")


def run_backtest_cli():
    """CLI entry point for backtesting."""
    import typer
    from rich.console import Console
    from rich.table import Table

    app = typer.Typer()
    console = Console()

    @app.command()
    def backtest(
        lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
        methods: List[str] = typer.Option(["statistical", "frequency", "cold", "pairs", "ml", "time_series", "ensemble"], "--methods", "-m", help="Methods to test"),
        train_window: int = typer.Option(200, "--train-window", help="Training window size"),
        test_window: int = typer.Option(50, "--test-window", help="Test window size"),
        step: int = typer.Option(10, "--step", help="Walk-forward step size"),
        num_predictions: int = typer.Option(3, "--predictions", "-p", help="Predictions per iteration"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to JSON")
    ):
        """Run backtest comparison."""

        if lottery not in LOTTERY_CONFIGS:
            console.print(f"[red]Unknown lottery: {lottery}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Running backtest for {LOTTERY_CONFIGS[lottery]['name']}[/bold]")
        console.print(f"Methods: {methods}")
        console.print(f"Train window: {train_window}, Test window: {test_window}, Step: {step}")

        backtester = Backtester(lottery)
        results = backtester.compare_methods(
            methods, train_window, test_window, step, num_predictions
        )

        backtester.print_comparison(results)

        if output:
            backtester.save_results(results, output)
            console.print(f"[green]Results saved to {output}[/green]")

    app()


if __name__ == "__main__":
    run_backtest_cli()