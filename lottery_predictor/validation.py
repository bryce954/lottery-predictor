"""
Validation & Analytics Module for Lottery Predictor (v1.7)
Pattern analysis, odds calculation, results verification, statistical significance testing.
"""

import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy import stats
from scipy.stats import chi2_contingency

from .data import get_data_manager, LOTTERY_CONFIGS
from .models import (
    StatisticalPredictor, MLModelPredictor, EnsemblePredictor,
    TimeSeriesPredictor, TransformerPredictor, OnlineLearningPredictor,
    get_predictor, PredictionResult
)


@dataclass
class PatternAnalysis:
    """Results of pattern analysis on lottery draws."""
    lottery: str
    total_draws: int
    consecutive_analysis: Dict[str, Any]
    odd_even_analysis: Dict[str, Any]
    high_low_analysis: Dict[str, Any]
    decade_analysis: Dict[str, Any]
    sum_distribution: Dict[str, float]


@dataclass
class OddsResult:
    """Odds calculation result for a lottery."""
    lottery: str
    jackpot_odds: float
    jackpot_odds_str: str
    prize_tiers: List[Dict[str, Any]]
    expected_value: float
    break_even_jackpot: float


@dataclass
class VerificationResult:
    """Results of prediction verification against actual draws."""
    lottery: str
    method: str
    total_predictions: int
    total_draws_verified: int
    matches_by_tier: Dict[int, int]
    hit_rate: float
    roi: float
    accuracy_by_tier: Dict[str, float]


@dataclass
class SignificanceTestResult:
    """Results of statistical significance testing."""
    method_a: str
    method_b: str
    test_type: str
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]


class PatternAnalyzer:
    """Analyze patterns in lottery historical data."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = LOTTERY_CONFIGS[lottery_type]

    def analyze_consecutive_numbers(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Analyze frequency of consecutive numbers in draws."""
        df = self.dm.load_data()
        if last_n:
            df = df.tail(last_n)

        total_draws = len(df)
        consecutive_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        for _, row in df.iterrows():
            main_nums = sorted(row['main_numbers'])
            consecutive = 0
            for i in range(len(main_nums) - 1):
                if main_nums[i+1] == main_nums[i] + 1:
                    consecutive += 1
            consecutive_counts[consecutive] = consecutive_counts.get(consecutive, 0) + 1

        # Convert to percentages
        consecutive_pct = {k: (v / total_draws * 100) for k, v in consecutive_counts.items()}

        # Expected probability for random draws (approximate)
        # For n numbers drawn from 1..N, expected consecutive pairs
        n = self.config['main_numbers']
        N = self.config['main_max']
        expected_pairs = (n - 1) * (N - n + 1) / (N - 1)  # Approximation

        return {
            "total_draws": total_draws,
            "consecutive_counts": consecutive_counts,
            "consecutive_percentages": consecutive_pct,
            "expected_consecutive_pairs": round(expected_pairs, 2),
            "observed_avg_consecutive": round(
                sum(k * v for k, v in consecutive_counts.items()) / total_draws, 2
            )
        }

    def analyze_odd_even(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Analyze odd/even distribution in draws."""
        df = self.dm.load_data()
        if last_n:
            df = df.tail(last_n)

        total_draws = len(df)
        odd_even_counts = {}

        for _, row in df.iterrows():
            main_nums = row['main_numbers']
            odd_count = sum(1 for n in main_nums if n % 2 == 1)
            even_count = len(main_nums) - odd_count
            key = f"{odd_count}odd-{even_count}even"
            odd_even_counts[key] = odd_even_counts.get(key, 0) + 1

        odd_even_pct = {k: (v / total_draws * 100) for k, v in odd_even_counts.items()}

        # Expected distribution (binomial)
        n = self.config['main_numbers']
        p = 0.5
        expected = {}
        for k in range(n + 1):
            exp_count = total_draws * math.comb(n, k) * (p ** k) * ((1-p) ** (n-k))
            expected[f"{k}odd-{n-k}even"] = round(exp_count, 1)

        return {
            "total_draws": total_draws,
            "observed_counts": odd_even_counts,
            "observed_percentages": odd_even_pct,
            "expected_counts": expected,
            "chi2_statistic": self._chi2_test(odd_even_counts, expected)
        }

    def analyze_high_low(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Analyze high/low number distribution."""
        df = self.dm.load_data()
        if last_n:
            df = df.tail(last_n)

        total_draws = len(df)
        mid_point = self.config['main_max'] / 2
        high_low_counts = {}

        for _, row in df.iterrows():
            main_nums = row['main_numbers']
            low_count = sum(1 for n in main_nums if n <= mid_point)
            high_count = len(main_nums) - low_count
            key = f"{low_count}low-{high_count}high"
            high_low_counts[key] = high_low_counts.get(key, 0) + 1

        high_low_pct = {k: (v / total_draws * 100) for k, v in high_low_counts.items()}

        n = self.config['main_numbers']
        expected = {}
        for k in range(n + 1):
            exp_count = total_draws * math.comb(n, k) * (0.5 ** n)
            expected[f"{k}low-{n-k}high"] = round(exp_count, 1)

        return {
            "total_draws": total_draws,
            "mid_point": mid_point,
            "observed_counts": high_low_counts,
            "observed_percentages": high_low_pct,
            "expected_counts": expected,
            "chi2_statistic": self._chi2_test(high_low_counts, expected)
        }

    def analyze_decades(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Analyze number distribution by decade (1-10, 11-20, etc.)."""
        df = self.dm.load_data()
        if last_n:
            df = df.tail(last_n)

        total_draws = len(df)
        max_num = self.config['main_max']
        num_decades = (max_num + 9) // 10
        decade_counts = {f"decade_{i+1}": 0 for i in range(num_decades)}

        for _, row in df.iterrows():
            main_nums = row['main_numbers']
            for n in main_nums:
                decade_idx = (n - 1) // 10
                if 0 <= decade_idx < num_decades:
                    decade_counts[f"decade_{decade_idx+1}"] += 1

        # Average per draw
        decade_avg = {k: round(v / total_draws, 2) for k, v in decade_counts.items()}

        # Expected per draw (uniform distribution)
        expected_per_decade = self.config['main_numbers'] / num_decades

        return {
            "total_draws": total_draws,
            "num_decades": num_decades,
            "total_counts": decade_counts,
            "avg_per_draw": decade_avg,
            "expected_per_draw": round(expected_per_decade, 2),
            "deviation_from_expected": {
                k: round(v - expected_per_decade, 2) for k, v in decade_avg.items()
            }
        }

    def analyze_sum_distribution(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Analyze sum distribution statistics."""
        df = self.dm.load_data()
        if last_n:
            df = df.tail(last_n)

        sums = df['main_numbers'].apply(sum)
        return {
            "mean": round(sums.mean(), 2),
            "median": round(sums.median(), 2),
            "std": round(sums.std(), 2),
            "min": int(sums.min()),
            "max": int(sums.max()),
            "skewness": round(stats.skew(sums), 3),
            "kurtosis": round(stats.kurtosis(sums), 3)
        }

    def _chi2_test(self, observed: Dict, expected: Dict) -> Dict[str, float]:
        """Perform chi-square goodness of fit test."""
        # Use union of keys to ensure alignment
        all_keys = set(observed.keys()) | set(expected.keys())
        obs_vals = [observed.get(k, 0) for k in all_keys]
        exp_vals = [expected.get(k, 0) for k in all_keys]

        # Normalize expected to match observed sum (for proper chi2 test)
        obs_sum = sum(obs_vals)
        exp_sum = sum(exp_vals)
        if exp_sum > 0:
            exp_vals = [e * obs_sum / exp_sum for e in exp_vals]

        # Filter out zero expected
        filtered = [(o, e) for o, e in zip(obs_vals, exp_vals) if e > 0]
        if len(filtered) < 2:
            return {"chi2": 0, "p_value": 1.0, "dof": 0}

        obs_f, exp_f = zip(*filtered)
        chi2, p = stats.chisquare(obs_f, exp_f)
        return {"chi2": round(chi2, 3), "p_value": round(p, 4), "dof": len(filtered) - 1}

    def full_analysis(self, last_n: Optional[int] = None) -> PatternAnalysis:
        """Run complete pattern analysis."""
        return PatternAnalysis(
            lottery=self.lottery_type,
            total_draws=len(self.dm.load_data().tail(last_n)) if last_n else len(self.dm.load_data()),
            consecutive_analysis=self.analyze_consecutive_numbers(last_n),
            odd_even_analysis=self.analyze_odd_even(last_n),
            high_low_analysis=self.analyze_high_low(last_n),
            decade_analysis=self.analyze_decades(last_n),
            sum_distribution=self.analyze_sum_distribution(last_n)
        )


class OddsCalculator:
    """Calculate exact odds and expected values for lottery games."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS[lottery_type]

    def calculate_combinations(self, n: int, k: int) -> int:
        """Calculate C(n, k) = n! / (k! * (n-k)!)."""
        if k > n or k < 0:
            return 0
        return math.comb(n, k)

    def calculate_jackpot_odds(self) -> Tuple[int, str]:
        """Calculate exact jackpot odds."""
        main_combos = self.calculate_combinations(
            self.config['main_max'],
            self.config['main_numbers']
        )

        if self.config['bonus_numbers'] > 0:
            bonus_combos = self.calculate_combinations(
                self.config['bonus_max'],
                self.config['bonus_numbers']
            )
            total_combos = main_combos * bonus_combos
        else:
            total_combos = main_combos

        odds_str = f"1 in {total_combos:,}"
        return total_combos, odds_str

    def calculate_prize_tiers(self) -> List[Dict[str, Any]]:
        """Calculate odds for all prize tiers."""
        # This is a simplified version - actual prize tiers vary by lottery
        # Returns standard tier structure

        tiers = []
        main_n = self.config['main_numbers']
        main_max = self.config['main_max']
        bonus_n = self.config['bonus_numbers']
        bonus_max = self.config['bonus_max']

        # Standard tier structure: match N main + bonus
        for main_match in range(main_n, -1, -1):
            for bonus_match in range(bonus_n, -1, -1):
                if main_match == main_n and bonus_match == bonus_n:
                    tier_name = "Jackpot"
                elif main_match == main_n and bonus_match == bonus_n - 1 and bonus_n > 0:
                    tier_name = "Second Prize"
                elif main_match == main_n:
                    tier_name = f"Match {main_n} (no bonus)"
                elif main_match == main_n - 1 and bonus_match == bonus_n:
                    tier_name = f"Match {main_n-1} + Bonus"
                elif main_match == main_n - 1:
                    tier_name = f"Match {main_n-1}"
                elif main_match >= 2:
                    tier_name = f"Match {main_match}"
                else:
                    continue

                # Calculate odds
                main_ways = self.calculate_combinations(main_n, main_match) * \
                           self.calculate_combinations(main_max - main_n, main_n - main_match)

                if bonus_n > 0:
                    bonus_ways = self.calculate_combinations(bonus_n, bonus_match) * \
                                self.calculate_combinations(bonus_max - bonus_n, bonus_n - bonus_match)
                    total_ways = main_ways * bonus_ways
                else:
                    total_ways = main_ways

                total_combos, _ = self.calculate_jackpot_odds()
                odds = total_combos / total_ways if total_ways > 0 else float('inf')

                tiers.append({
                    "tier": tier_name,
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "odds": round(odds, 0),
                    "odds_str": f"1 in {odds:,.0f}" if odds != float('inf') else "N/A",
                    "probability": round(1 / odds, 10) if odds != float('inf') else 0
                })

        return tiers

    def calculate_expected_value(self, jackpot: float, ticket_cost: float = 2.0) -> Dict[str, Any]:
        """Calculate expected value given jackpot amount."""
        # Typical prize structure (simplified)
        # In reality, these vary by lottery and are pari-mutuel
        tiers = self.calculate_prize_tiers()

        # Estimate prize amounts (very approximate)
        # These would need to be customized per lottery
        prize_estimates = {}
        total_combos, _ = self.calculate_jackpot_odds()

        for tier in tiers:
            if tier["tier"] == "Jackpot":
                prize_estimates[tier["tier"]] = jackpot
            elif tier["tier"] == "Second Prize":
                prize_estimates[tier["tier"]] = 1_000_000  # Typical $1M
            elif "Match" in tier["tier"] and "+" in tier["tier"]:
                prize_estimates[tier["tier"]] = 50_000
            elif "Match" in tier["tier"]:
                if tier["main_match"] == self.config['main_numbers'] - 1:
                    prize_estimates[tier["tier"]] = 1_000
                elif tier["main_match"] == self.config['main_numbers'] - 2:
                    prize_estimates[tier["tier"]] = 100
                elif tier["main_match"] == self.config['main_numbers'] - 3:
                    prize_estimates[tier["tier"]] = 10
                else:
                    prize_estimates[tier["tier"]] = 4
            else:
                prize_estimates[tier["tier"]] = 0

        # Calculate EV
        ev = 0
        for tier in tiers:
            prob = tier["probability"]
            prize = prize_estimates.get(tier["tier"], 0)
            ev += prob * prize

        # Break-even jackpot
        non_jackpot_ev = sum(
            tier["probability"] * prize_estimates.get(tier["tier"], 0)
            for tier in tiers if tier["tier"] != "Jackpot"
        )
        jackpot_prob = next((t["probability"] for t in tiers if t["tier"] == "Jackpot"), 0)

        break_even = (ticket_cost - non_jackpot_ev) / jackpot_prob if jackpot_prob > 0 else float('inf')

        return {
            "expected_value": round(ev, 4),
            "net_expected_value": round(ev - ticket_cost, 4),
            "break_even_jackpot": round(break_even, 0),
            "jackpot_probability": jackpot_prob,
            "ticket_cost": ticket_cost,
            "assumed_prizes": prize_estimates
        }

    def full_odds_analysis(self, current_jackpot: float = 0, ticket_cost: float = 2.0) -> OddsResult:
        """Run complete odds analysis."""
        jackpot_odds, jackpot_odds_str = self.calculate_jackpot_odds()
        prize_tiers = self.calculate_prize_tiers()
        ev_data = self.calculate_expected_value(current_jackpot, ticket_cost)

        return OddsResult(
            lottery=self.lottery_type,
            jackpot_odds=jackpot_odds,
            jackpot_odds_str=jackpot_odds_str,
            prize_tiers=prize_tiers,
            expected_value=ev_data["net_expected_value"],
            break_even_jackpot=ev_data["break_even_jackpot"]
        )


class ResultsVerifier:
    """Verify predictions against actual historical draws."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = LOTTERY_CONFIGS[lottery_type]

    def verify_predictions(
        self,
        predictions: List[PredictionResult],
        actual_draws: List[Dict],
        lookback: int = 100
    ) -> VerificationResult:
        """Verify a set of predictions against actual draws.

        Assumes predictions and actual_draws are aligned (each prediction corresponds
        to the draw at the same index). If lengths differ, uses min length.
        """
        max_main = self.config['main_numbers']
        matches_by_tier = {i: 0 for i in range(max_main + 1)}

        # Match each prediction with its corresponding draw (by index)
        num_comparisons = min(len(predictions), len(actual_draws))

        for i in range(num_comparisons):
            pred = predictions[i]
            draw = actual_draws[i]
            main_match = len(set(pred.main_numbers) & set(draw['main_numbers']))
            matches_by_tier[main_match] += 1

        # Calculate hit rate (at least 1 number matched per prediction)
        matched = sum(matches_by_tier[i] for i in range(1, max_main + 1))
        hit_rate = matched / num_comparisons if num_comparisons > 0 else 0

        # ROI calculation based on actual prize tiers (simplified)
        # $2 per ticket, using typical prize structure
        total_cost = num_comparisons * 2.0

        # Prize estimates per tier (typical values, would be lottery-specific)
        prize_estimates = {
            0: 0,  # No match
            1: 0,  # Match 1 - usually no prize
            2: 4,  # Match 2 - typically $4 (Powerball) or free ticket
            3: 10,  # Match 3 - typically $7-10
            4: 100,  # Match 4 - typically $100
            5: 1_000_000,  # Match 5 - typically $1M (second prize)
            6: 1_000_000,  # Match 5+bonus would be jackpot, but we only track main
        }

        estimated_winnings = sum(matches_by_tier[i] * prize_estimates.get(i, 0) for i in range(max_main + 1))
        roi = (estimated_winnings - total_cost) / total_cost if total_cost > 0 else 0

        return VerificationResult(
            lottery=self.lottery_type,
            method=predictions[0].method if predictions else "unknown",
            total_predictions=len(predictions),
            total_draws_verified=len(actual_draws),
            matches_by_tier=matches_by_tier,
            hit_rate=round(hit_rate * 100, 2),
            roi=round(roi * 100, 2),
            accuracy_by_tier={
                f"match_{i}": round(matches_by_tier[i] / num_comparisons * 100, 2)
                for i in range(max_main + 1)
            }
        )

    def backtest_method(
        self,
        method: str,
        num_predictions: int = 1,
        train_window: int = 200,
        test_window: int = 50,
        step: int = 10
    ) -> VerificationResult:
        """Run walk-forward backtest for a prediction method.

        Generates one prediction per test draw (num_predictions=1 by default)
        to ensure proper alignment with actual draws.
        """
        df = self.dm.load_data()
        total_draws = len(df)

        all_predictions = []
        all_actual = []

        for start in range(train_window, total_draws - test_window, step):
            train_end = start
            test_end = min(start + test_window, total_draws)

            train_df = df.iloc[train_end - train_window:train_end]
            test_df = df.iloc[start:test_end]
            num_test_draws = len(test_df)

            # Generate one prediction per test draw
            preds_per_draw = max(1, num_predictions)
            total_preds_needed = num_test_draws * preds_per_draw

            # Generate predictions using training data
            if method == "statistical":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_balanced(last_n=train_window, num_predictions=total_preds_needed)
            elif method == "frequency":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_frequency_based(last_n=train_window, num_predictions=total_preds_needed)
            elif method == "cold":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_cold_numbers(last_n=train_window, num_predictions=total_preds_needed)
            elif method == "pairs":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_pair_based(last_n=train_window, num_predictions=total_preds_needed)
            elif method == "ml":
                pred = MLModelPredictor(self.lottery_type)
                if not pred.is_trained:
                    pred.train(lookback=10)
                preds = pred.predict(num_predictions=total_preds_needed)
            elif method == "ensemble":
                pred = EnsemblePredictor(self.lottery_type)
                preds = pred.get_consensus_prediction(num_predictions=total_preds_needed)
            else:
                continue

            all_predictions.extend(preds[:total_preds_needed])
            all_actual.extend(test_df.to_dict('records'))

        return self.verify_predictions(all_predictions, all_actual)


class SignificanceTester:
    """Statistical significance testing for prediction methods."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = LOTTERY_CONFIGS[lottery_type]

    def compare_methods(
        self,
        method_a: str,
        method_b: str,
        num_trials: int = 100,
        predictions_per_trial: int = 10,
        lookback: int = 200
    ) -> SignificanceTestResult:
        """Compare two methods using paired t-test on match counts."""
        df = self.dm.load_data()
        total_draws = len(df)

        method_a_matches = []
        method_b_matches = []

        for trial in range(num_trials):
            # Use a random segment of data
            start_idx = np.random.randint(lookback, total_draws - 50)
            train_end = start_idx
            test_end = start_idx + 50

            test_df = df.iloc[train_end:test_end]
            test_draws = test_df.to_dict('records')

            # Generate predictions for each method
            for method_name in [method_a, method_b]:
                if method_name == "statistical":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_balanced(last_n=lookback, num_predictions=predictions_per_trial)
                elif method_name == "frequency":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_frequency_based(last_n=lookback, num_predictions=predictions_per_trial)
                elif method_name == "cold":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_cold_numbers(last_n=lookback, num_predictions=predictions_per_trial)
                elif method_name == "pairs":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_pair_based(last_n=lookback, num_predictions=predictions_per_trial)
                elif method_name == "ml":
                    pred = MLModelPredictor(self.lottery_type)
                    if not pred.is_trained:
                        pred.train(lookback=10)
                    preds = pred.predict(num_predictions=predictions_per_trial)
                elif method_name == "ensemble":
                    pred = EnsemblePredictor(self.lottery_type)
                    preds = pred.get_consensus_prediction(num_predictions=predictions_per_trial)
                else:
                    continue

                # Count matches
                matches = 0
                for p in preds:
                    for d in test_draws:
                        matches += len(set(p.main_numbers) & set(d['main_numbers']))

                if method_name == method_a:
                    method_a_matches.append(matches)
                else:
                    method_b_matches.append(matches)

        if len(method_a_matches) < 2 or len(method_b_matches) < 2:
            return SignificanceTestResult(
                method_a=method_a,
                method_b=method_b,
                test_type="paired_ttest",
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(method_a_matches, method_b_matches)

        # Effect size (Cohen's d)
        diff = np.array(method_a_matches) - np.array(method_b_matches)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        # Confidence interval for mean difference
        se = np.std(diff) / np.sqrt(len(diff))
        ci_low = np.mean(diff) - 1.96 * se
        ci_high = np.mean(diff) + 1.96 * se

        return SignificanceTestResult(
            method_a=method_a,
            method_b=method_b,
            test_type="paired_ttest",
            p_value=round(p_value, 6),
            significant=p_value < 0.05,
            effect_size=round(cohens_d, 3),
            confidence_interval=(round(ci_low, 3), round(ci_high, 3))
        )

    def compare_vs_random(
        self,
        method: str,
        num_trials: int = 100,
        predictions_per_trial: int = 10,
        lookback: int = 200
    ) -> SignificanceTestResult:
        """Compare a method against random selection."""
        df = self.dm.load_data()
        total_draws = len(df)

        method_matches = []
        random_matches = []

        main_max = self.config['main_max']
        main_n = self.config['main_numbers']

        for trial in range(num_trials):
            start_idx = np.random.randint(lookback, total_draws - 50)
            test_df = df.iloc[start_idx:start_idx + 50]
            test_draws = test_df.to_dict('records')

            # Method predictions
            if method == "statistical":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_balanced(last_n=lookback, num_predictions=predictions_per_trial)
            elif method == "frequency":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_frequency_based(last_n=lookback, num_predictions=predictions_per_trial)
            elif method == "cold":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_cold_numbers(last_n=lookback, num_predictions=predictions_per_trial)
            elif method == "pairs":
                pred = StatisticalPredictor(self.lottery_type)
                preds = pred.predict_pair_based(last_n=lookback, num_predictions=predictions_per_trial)
            elif method == "ml":
                pred = MLModelPredictor(self.lottery_type)
                if not pred.is_trained:
                    pred.train(lookback=10)
                preds = pred.predict(num_predictions=predictions_per_trial)
            elif method == "ensemble":
                pred = EnsemblePredictor(self.lottery_type)
                preds = pred.get_consensus_prediction(num_predictions=predictions_per_trial)
            else:
                continue

            # Count method matches
            matches = 0
            for p in preds:
                for d in test_draws:
                    matches += len(set(p.main_numbers) & set(d['main_numbers']))
            method_matches.append(matches)

            # Random predictions (same number of tickets)
            random_matches_this_trial = 0
            for _ in range(predictions_per_trial):
                rand_nums = sorted(np.random.choice(
                    range(1, main_max + 1),
                    size=main_n,
                    replace=False
                ))
                for d in test_draws:
                    random_matches_this_trial += len(set(rand_nums) & set(d['main_numbers']))
            random_matches.append(random_matches_this_trial)

        # Paired t-test vs random
        t_stat, p_value = stats.ttest_rel(method_matches, random_matches)

        diff = np.array(method_matches) - np.array(random_matches)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        se = np.std(diff) / np.sqrt(len(diff))
        ci_low = np.mean(diff) - 1.96 * se
        ci_high = np.mean(diff) + 1.96 * se

        return SignificanceTestResult(
            method_a=method,
            method_b="random",
            test_type="paired_ttest_vs_random",
            p_value=round(p_value, 6),
            significant=p_value < 0.05,
            effect_size=round(cohens_d, 3),
            confidence_interval=(round(ci_low, 3), round(ci_high, 3))
        )