"""
Visualization Module for Lottery Predictions
Creates charts and visualizations for analysis and predictions.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .data import LotteryDataManager, get_data_manager
from .models import PredictionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIZ_DIR = Path(__file__).parent.parent / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class LotteryVisualizer:
    """Creates visualizations for lottery data and predictions."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.dm = get_data_manager(lottery_type)
        self.config = self.dm.config

    def plot_number_frequency(self, last_n: Optional[int] = None, save: bool = True) -> str:
        """Plot frequency of each number."""
        freq = self.dm.get_number_frequency(last_n)

        fig, ax = plt.subplots(figsize=(14, 6))
        numbers = list(freq.keys())
        counts = list(freq.values())

        colors = ['red' if c > np.mean(counts) else 'blue' for c in counts]
        bars = ax.bar(numbers, counts, color=colors, alpha=0.7, edgecolor='black')

        ax.axhline(y=np.mean(counts), color='black', linestyle='--',
                   label=f'Mean: {np.mean(counts):.1f}')
        ax.set_xlabel('Number', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{self.config["name"]} - Number Frequency (Last {last_n or "All"} Draws)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_frequency.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved frequency plot to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def plot_hot_cold_numbers(self, last_n: int = 50, save: bool = True) -> str:
        """Plot hot and cold numbers."""
        hot, cold = self.dm.get_hot_cold_numbers(last_n)
        freq = self.dm.get_number_frequency(last_n)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Hot numbers
        hot_freq = {n: freq[n] for n in hot}
        ax1.barh(range(len(hot)), list(hot_freq.values()), color='red', alpha=0.7)
        ax1.set_yticks(range(len(hot)))
        ax1.set_yticklabels([str(n) for n in hot])
        ax1.set_xlabel('Frequency')
        ax1.set_title(f'Top 10 Hot Numbers (Last {last_n} Draws)')
        ax1.invert_yaxis()

        # Cold numbers
        cold_freq = {n: freq[n] for n in cold}
        ax2.barh(range(len(cold)), list(cold_freq.values()), color='blue', alpha=0.7)
        ax2.set_yticks(range(len(cold)))
        ax2.set_yticklabels([str(n) for n in cold])
        ax2.set_xlabel('Frequency')
        ax2.set_title(f'Top 10 Cold Numbers (Last {last_n} Draws)')
        ax2.invert_yaxis()

        plt.suptitle(f'{self.config["name"]} - Hot vs Cold Numbers', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_hot_cold.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved hot/cold plot to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def plot_sum_distribution(self, last_n: Optional[int] = None, save: bool = True) -> str:
        """Plot distribution of sum of drawn numbers."""
        stats = self.dm.get_sum_statistics(last_n)

        if self.dm.df is None:
            self.dm.load_data()

        df = self.dm.df.tail(last_n) if last_n else self.dm.df
        sums = df['main_numbers'].apply(sum)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(sums, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)

        # Add normal distribution curve
        from scipy import stats as scipy_stats
        x = np.linspace(sums.min(), sums.max(), 100)
        ax.plot(x, scipy_stats.norm.pdf(x, stats['mean'], stats['std']),
                'r-', lw=2, label=f'Normal Fit (μ={stats["mean"]:.1f}, σ={stats["std"]:.1f})')

        ax.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
        ax.axvline(stats['median'], color='blue', linestyle='--', label=f'Median: {stats["median"]:.1f}')

        ax.set_xlabel('Sum of Main Numbers', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{self.config["name"]} - Sum Distribution (Last {last_n or "All"} Draws)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_sum_dist.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved sum distribution plot to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def plot_pair_heatmap(self, last_n: int = 200, top_k: int = 20, save: bool = True) -> str:
        """Plot heatmap of frequent number pairs."""
        pairs = self.dm.get_pair_frequency(last_n)
        top_pairs = list(pairs.items())[:top_k]

        if not top_pairs:
            logger.warning("No pair data available")
            return ""

        # Create matrix
        max_num = self.config['main_max']
        matrix = np.zeros((max_num, max_num))

        for (n1, n2), count in top_pairs:
            matrix[n1-1, n2-1] = count
            matrix[n2-1, n1-1] = count

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xlabel('Number', fontsize=12)
        ax.set_ylabel('Number', fontsize=12)
        ax.set_title(f'{self.config["name"]} - Top {top_k} Number Pairs (Last {last_n} Draws)',
                     fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Co-occurrence Count')
        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_pairs.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved pair heatmap to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def plot_prediction_analysis(self, predictions: List[PredictionResult], save: bool = True) -> str:
        """Visualize prediction analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Predicted numbers distribution
        all_predicted = []
        for p in predictions:
            all_predicted.extend(p.main_numbers)

        ax1 = axes[0, 0]
        freq = pd.Series(all_predicted).value_counts().sort_index()
        ax1.bar(freq.index, freq.values, color='purple', alpha=0.7)
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Times Predicted')
        ax1.set_title('Predicted Numbers Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Sum of predictions
        ax2 = axes[0, 1]
        pred_sums = [sum(p.main_numbers) for p in predictions]
        ax2.hist(pred_sums, bins=10, color='orange', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(pred_sums), color='red', linestyle='--',
                    label=f'Mean: {np.mean(pred_sums):.1f}')
        ax2.set_xlabel('Sum of Predicted Numbers')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Sums Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Confidence by method
        ax3 = axes[1, 0]
        methods = [p.method for p in predictions]
        confidences = [p.confidence for p in predictions]
        method_conf = pd.Series(confidences, index=methods).groupby(level=0).mean()
        ax3.bar(range(len(method_conf)), method_conf.values, color='teal', alpha=0.7)
        ax3.set_xticks(range(len(method_conf)))
        ax3.set_xticklabels(method_conf.index, rotation=45, ha='right')
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Average Confidence by Method')
        ax3.grid(True, alpha=0.3)

        # 4. Even/Odd distribution
        ax4 = axes[1, 1]
        even_counts = []
        odd_counts = []
        for p in predictions:
            evens = sum(1 for n in p.main_numbers if n % 2 == 0)
            odds = len(p.main_numbers) - evens
            even_counts.append(evens)
            odd_counts.append(odds)

        x = np.arange(len(predictions))
        width = 0.35
        ax4.bar(x - width/2, even_counts, width, label='Even', color='blue', alpha=0.7)
        ax4.bar(x + width/2, odd_counts, width, label='Odd', color='red', alpha=0.7)
        ax4.set_xlabel('Prediction #')
        ax4.set_ylabel('Count')
        ax4.set_title('Even/Odd Distribution in Predictions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{self.config["name"]} - Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_prediction_analysis.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved prediction analysis to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def plot_timeline_analysis(self, last_n: int = 100, save: bool = True) -> str:
        """Plot timeline of draws with statistics."""
        if self.dm.df is None:
            self.dm.load_data()

        df = self.dm.df.tail(last_n).copy()
        df['sum'] = df['main_numbers'].apply(sum)
        df['even_count'] = df['main_numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 0))
        df['high_count'] = df['main_numbers'].apply(lambda x: sum(1 for n in x if n > self.config['main_max'] / 2))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Sum over time
        axes[0].plot(df['date'], df['sum'], 'o-', color='green', alpha=0.7, markersize=3)
        axes[0].axhline(y=df['sum'].mean(), color='red', linestyle='--', label='Mean')
        axes[0].set_ylabel('Sum of Numbers')
        axes[0].set_title(f'{self.config["name"]} - Draw Sum Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Even count over time
        axes[1].plot(df['date'], df['even_count'], 'o-', color='blue', alpha=0.7, markersize=3)
        axes[1].axhline(y=df['even_count'].mean(), color='red', linestyle='--', label='Mean')
        axes[1].set_ylabel('Even Numbers Count')
        axes[1].set_title('Even Numbers Count Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # High count over time
        axes[2].plot(df['date'], df['high_count'], 'o-', color='orange', alpha=0.7, markersize=3)
        axes[2].axhline(y=df['high_count'].mean(), color='red', linestyle='--', label='Mean')
        axes[2].set_ylabel('High Numbers Count')
        axes[2].set_xlabel('Date')
        axes[2].set_title('High Numbers Count Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = VIZ_DIR / f"{self.lottery_type}_timeline.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved timeline analysis to {filepath}")

        plt.close()
        return str(filepath) if save else ""

    def generate_all_visualizations(self, last_n: int = 200) -> List[str]:
        """Generate all visualizations."""
        files = []

        files.append(self.plot_number_frequency(last_n))
        files.append(self.plot_hot_cold_numbers(last_n))
        files.append(self.plot_sum_distribution(last_n))
        files.append(self.plot_pair_heatmap(last_n))
        files.append(self.plot_timeline_analysis(last_n))

        return [f for f in files if f]


def visualize_predictions(predictions: List[PredictionResult], lottery_type: str = "powerball") -> str:
    """Quick visualization of predictions."""
    viz = LotteryVisualizer(lottery_type)
    return viz.plot_prediction_analysis(predictions)


if __name__ == "__main__":
    # Test visualizations
    for lt in ["florida_lotto", "powerball"]:
        print(f"\nGenerating visualizations for {lt}...")
        viz = LotteryVisualizer(lt)
        files = viz.generate_all_visualizations(200)
        for f in files:
            print(f"  Created: {f}")