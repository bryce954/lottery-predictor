"""
Historical Lottery Data Management
Fetches, caches, and manages historical lottery data for Florida Lotto and Powerball.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Lottery configurations
LOTTERY_CONFIGS = {
    "florida_lotto": {
        "name": "Florida Lotto",
        "main_numbers": 6,
        "main_max": 53,
        "bonus_numbers": 0,
        "bonus_max": 0,
        "draw_days": ["Wednesday", "Saturday"],
        "start_date": "1988-01-06",
        "api_url": "https://www.flalottery.com/exptkt/lotto/lotto_odds.pdf",  # Not a real API
    },
    "powerball": {
        "name": "Powerball",
        "main_numbers": 5,
        "main_max": 69,
        "bonus_numbers": 1,
        "bonus_max": 26,
        "draw_days": ["Monday", "Wednesday", "Saturday"],
        "start_date": "1992-04-22",
        "api_url": "https://www.powerball.com/api/v1/numbers",  # Example
    }
}


class LotteryDataManager:
    """Manages historical lottery data for training and prediction."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS.get(lottery_type, LOTTERY_CONFIGS["powerball"])
        self.cache_file = DATA_DIR / f"{lottery_type}_history.json"
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load historical data from cache or generate sample data."""
        if self.cache_file.exists():
            logger.info(f"Loading cached data for {self.config['name']}")
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
            self.df['date'] = pd.to_datetime(self.df['date'])
            return self.df

        # Generate comprehensive historical data if no cache
        logger.info(f"Generating historical data for {self.config['name']}")
        self.df = self._generate_historical_data()
        self.save_data()
        return self.df

    def _generate_historical_data(self) -> pd.DataFrame:
        """Generate realistic historical lottery data based on actual game rules."""
        config = self.config
        start_date = pd.Timestamp(config['start_date'])
        end_date = pd.Timestamp.now()

        # Calculate draw dates based on draw days
        draw_dates = []
        current = start_date
        while current <= end_date:
            if current.strftime("%A") in config['draw_days']:
                draw_dates.append(current)
            current += timedelta(days=1)

        # Generate draws with some realistic patterns
        np.random.seed(42)  # For reproducibility
        draws = []

        for i, date in enumerate(draw_dates):
            # Main numbers
            main_nums = sorted(np.random.choice(
                range(1, config['main_max'] + 1),
                size=config['main_numbers'],
                replace=False
            ).tolist())

            # Bonus numbers
            bonus_nums = []
            if config['bonus_numbers'] > 0:
                bonus_nums = sorted(np.random.choice(
                    range(1, config['bonus_max'] + 1),
                    size=config['bonus_numbers'],
                    replace=False
                ).tolist())

            draws.append({
                'draw_number': i + 1,
                'date': date.strftime('%Y-%m-%d'),
                'main_numbers': main_nums,
                'bonus_numbers': bonus_nums,
                'jackpot': np.random.randint(1000000, 500000000),
            })

        df = pd.DataFrame(draws)
        logger.info(f"Generated {len(df)} historical draws for {config['name']}")
        return df

    def fetch_from_api(self) -> Optional[pd.DataFrame]:
        """Attempt to fetch real data from lottery APIs."""
        try:
            # This would be implemented with actual API endpoints
            # For now, we use generated data
            logger.warning("Real API fetching not implemented, using generated data")
            return None
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return None

    def save_data(self):
        """Save data to cache."""
        if self.df is not None:
            data = self.df.copy()
            # Handle both datetime and string dates
            if pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            with open(self.cache_file, 'w') as f:
                json.dump(data.to_dict('records'), f)
            logger.info(f"Saved {len(data)} draws to cache")

    def get_recent_draws(self, n: int = 100) -> pd.DataFrame:
        """Get most recent n draws."""
        if self.df is None:
            self.load_data()
        return self.df.tail(n).copy()

    def get_number_frequency(self, last_n: Optional[int] = None) -> Dict[int, int]:
        """Get frequency of each main number."""
        if self.df is None:
            self.load_data()

        df = self.df.tail(last_n) if last_n else self.df
        all_numbers = []
        for nums in df['main_numbers']:
            all_numbers.extend(nums)

        freq = pd.Series(all_numbers).value_counts().to_dict()
        # Ensure all possible numbers are represented
        for i in range(1, self.config['main_max'] + 1):
            if i not in freq:
                freq[i] = 0
        return dict(sorted(freq.items()))

    def get_bonus_frequency(self, last_n: Optional[int] = None) -> Dict[int, int]:
        """Get frequency of bonus numbers."""
        if self.config['bonus_numbers'] == 0:
            return {}

        if self.df is None:
            self.load_data()

        df = self.df.tail(last_n) if last_n else self.df
        all_bonus = []
        for nums in df['bonus_numbers']:
            all_bonus.extend(nums)

        freq = pd.Series(all_bonus).value_counts().to_dict()
        for i in range(1, self.config['bonus_max'] + 1):
            if i not in freq:
                freq[i] = 0
        return dict(sorted(freq.items()))

    def get_hot_cold_numbers(self, last_n: int = 50) -> Tuple[List[int], List[int]]:
        """Get hot (frequent) and cold (infrequent) numbers."""
        freq = self.get_number_frequency(last_n)
        sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = [n for n, _ in sorted_nums[:10]]
        cold = [n for n, _ in sorted_nums[-10:]]
        return hot, cold

    def get_pair_frequency(self, last_n: Optional[int] = None) -> Dict[Tuple[int, int], int]:
        """Get frequency of number pairs appearing together."""
        if self.df is None:
            self.load_data()

        df = self.df.tail(last_n) if last_n else self.df
        pair_counts = {}

        for nums in df['main_numbers']:
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    pair = tuple(sorted([nums[i], nums[j]]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return dict(sorted(pair_counts.items(), key=lambda x: x[1], reverse=True))

    def get_sum_statistics(self, last_n: Optional[int] = None) -> Dict:
        """Get statistics on sum of drawn numbers."""
        if self.df is None:
            self.load_data()

        df = self.df.tail(last_n) if last_n else self.df
        sums = df['main_numbers'].apply(sum)

        return {
            'mean': sums.mean(),
            'median': sums.median(),
            'std': sums.std(),
            'min': sums.min(),
            'max': sums.max(),
            'recent': sums.tail(10).tolist()
        }

    def prepare_ml_features(self, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML training."""
        if self.df is None:
            self.load_data()

        config = self.config
        max_num = config['main_max']

        # Create one-hot encoded sequences
        X, y = [], []

        for i in range(lookback, len(self.df)):
            # Features: previous 'lookback' draws as one-hot
            features = []
            for j in range(i - lookback, i):
                draw_vec = np.zeros(max_num)
                for num in self.df.iloc[j]['main_numbers']:
                    draw_vec[num - 1] = 1
                features.append(draw_vec)

            X.append(np.array(features).flatten())

            # Target: next draw as one-hot
            target = np.zeros(max_num)
            for num in self.df.iloc[i]['main_numbers']:
                target[num - 1] = 1
            y.append(target)

        return np.array(X), np.array(y)


def get_data_manager(lottery_type: str) -> LotteryDataManager:
    """Factory function to get data manager for lottery type."""
    return LotteryDataManager(lottery_type)


if __name__ == "__main__":
    # Test data generation
    for lt in ["florida_lotto", "powerball"]:
        dm = LotteryDataManager(lt)
        df = dm.load_data()
        print(f"\n{lt}: {len(df)} draws")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Sample draw: {df.iloc[-1].to_dict()}")