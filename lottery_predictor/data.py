"""
Historical Lottery Data Management
Fetches, caches, and manages historical lottery data for Florida Lotto and Powerball.
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import io
import re

from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Lottery configurations - populated from universal system
from .universal import get_universal_manager

# Initialize universal manager to get all lottery configs
_universal_manager = get_universal_manager()
LOTTERY_CONFIGS = {}

for game in _universal_manager.list_games():
    if game.main_numbers > 0:  # Skip "no lottery" entries
        LOTTERY_CONFIGS[game.key] = {
            "name": game.name,
            "main_numbers": game.main_numbers,
            "main_max": game.main_max,
            "bonus_numbers": game.bonus_numbers,
            "bonus_max": game.bonus_max,
            "draw_days": game.draw_days,
            "start_date": game.start_date,
            "api_url": game.website,
        }


class LotteryDataManager:
    """Manages historical lottery data for training and prediction."""

    def __init__(self, lottery_type: str = "powerball"):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS.get(lottery_type, LOTTERY_CONFIGS["powerball"])
        self.cache_file = DATA_DIR / f"{lottery_type}_history.json"
        self.refresh_meta_file = DATA_DIR / f"{lottery_type}_refresh.json"
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load the stored draw history without inventing missing results."""
        if self.cache_file.exists():
            logger.info(f"Loading cached data for {self.config['name']}")
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.df = pd.DataFrame(data)
            if not self.df.empty:
                self.df["date"] = pd.to_datetime(self.df["date"], errors="raise")
        else:
            self.df = pd.DataFrame(columns=["draw_number", "date", "main_numbers", "bonus_numbers", "jackpot"])
            self.df["date"] = pd.to_datetime(self.df["date"])
        self.refresh_status = self.refresh_official_results()
        if self.df.empty:
            logger.warning("No draw data available for %s (%s)", self.config['name'], self.refresh_status["message"])
        return self.df

    def refresh_official_results(self) -> Dict:
        """Refresh draw history from an official lottery authority source."""
        if self.cache_file.exists() and self.refresh_meta_file.exists():
            try:
                previous = json.loads(self.refresh_meta_file.read_text(encoding="utf-8"))
                refreshed_at = datetime.fromisoformat(previous["updated_at"])
                if (datetime.now(timezone.utc) - refreshed_at).total_seconds() < 300:
                    previous["message"] = "Official results refreshed within the last five minutes."
                    return previous
            except (KeyError, TypeError, ValueError, OSError):
                pass
        if self.lottery_type == "florida_lotto":
            url = "https://files.floridalottery.com/exptkt/l6.pdf"
            source = "Florida Lottery Winning Numbers History"
            try:
                response = requests.get(url, timeout=30, headers={"User-Agent": "LotteryPredictor/1.0"})
                response.raise_for_status()
                reader = PdfReader(io.BytesIO(response.content))
                rows = []
                date_pattern = re.compile(r"\d{2}/\d{2}/\d{2}")
                numbers_pattern = re.compile(r"\d{1,2}(?:\s*-\s*\d{1,2}){5}")
                for page in reader.pages:
                    text = page.extract_text(extraction_mode="layout") or ""
                    for line in text.splitlines():
                        dates = list(date_pattern.finditer(line))
                        for index, date_match in enumerate(dates):
                            end = dates[index + 1].start() if index + 1 < len(dates) else len(line)
                            draw_record = line[date_match.end():end]
                            if re.search(r"\bLOTTO\s+DP\b", draw_record):
                                continue
                            numbers_match = numbers_pattern.search(draw_record)
                            if not numbers_match:
                                continue
                            draw_date = pd.to_datetime(date_match.group(), format="%m/%d/%y").normalize()
                            numbers = [int(value) for value in re.findall(r"\d+", numbers_match.group())]
                            if len(numbers) != 6 or len(set(numbers)) != 6 or min(numbers) < 1 or max(numbers) > self.config["main_max"]:
                                raise ValueError(f"Official Florida Lotto record is invalid for {draw_date.date()}")
                            rows.append({"date": draw_date, "main_numbers": sorted(numbers),
                                         "bonus_numbers": [], "jackpot": 0})
                official = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
                if official.empty:
                    raise ValueError("Official Florida Lottery history contained no draw records")
                official["draw_number"] = range(1, len(official) + 1)
                self.df = official.reset_index(drop=True)
                self.save_data()
                status = {"verified": True, "source": source, "source_url": url,
                          "updated_at": datetime.now(timezone.utc).isoformat(),
                          "latest_draw_date": official["date"].max().date().isoformat(),
                          "message": "Results refreshed from the official Florida Lottery history."}
                self.refresh_meta_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
                return status
            except Exception as exc:
                logger.warning("Official Florida Lotto refresh failed: %s", exc)
                return {"verified": False, "source": None, "updated_at": None,
                        "message": f"Official refresh unavailable; cached results retained. ({exc})"}

        sources = {
            "powerball": ("d6yy-54nr", pd.Timestamp("2010-02-03")),
            "mega_millions": ("5xaw-6ayf", pd.Timestamp("2002-05-17")),
        }
        if self.lottery_type not in sources:
            return {"verified": False, "source": None, "updated_at": None,
                    "message": "No official automatic feed is configured; cached results were retained."}
        dataset_id, coverage_start = sources[self.lottery_type]
        url = f"https://data.ny.gov/resource/{dataset_id}.json"
        try:
            response = requests.get(url, params={"$limit": 5000, "$order": "draw_date ASC"},
                                    timeout=20, headers={"User-Agent": "LotteryPredictor/1.0"})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise ValueError("Official source returned no draw records")
            rows = []
            for item in payload:
                values = [int(n) for n in item.get("winning_numbers", "").split() if n.isdigit()]
                if self.lottery_type == "powerball":
                    main, bonus = values[:5], values[5:6]
                else:
                    main = values[:5]
                    bonus = [int(item["mega_ball"])] if item.get("mega_ball") else values[5:6]
                if len(main) != 5 or len(bonus) != 1:
                    continue
                historical_main_max = 69 if self.lottery_type == "powerball" else 75
                historical_bonus_max = 39 if self.lottery_type == "powerball" else 56
                if len(set(main)) != 5 or min(main) < 1 or max(main) > historical_main_max:
                    raise ValueError("Official source returned invalid main numbers")
                if bonus[0] < 1 or bonus[0] > historical_bonus_max:
                    raise ValueError("Official source returned invalid bonus number")
                rows.append({"date": pd.to_datetime(item["draw_date"]).normalize(),
                             "main_numbers": main, "bonus_numbers": bonus, "jackpot": 0})
            official = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
            if official.empty:
                raise ValueError("Official source contained no valid records")
            preserved = self.df[self.df["date"] < coverage_start]
            official["draw_number"] = range(len(preserved) + 1, len(preserved) + len(official) + 1)
            combined = pd.concat([preserved, official], ignore_index=True)
            combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
            combined["draw_number"] = range(1, len(combined) + 1)
            self.df = combined
            self.save_data()
            status = {"verified": True, "source": "New York State Gaming Commission Open Data",
                      "source_url": url, "updated_at": datetime.now(timezone.utc).isoformat(),
                      "latest_draw_date": official["date"].max().date().isoformat(),
                      "message": "Results refreshed from the official state dataset."}
            self.refresh_meta_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return status
        except Exception as exc:
            logger.warning("Official refresh failed for %s: %s", self.lottery_type, exc)
            return {"verified": False, "source": None, "updated_at": None,
                    "message": f"Official refresh unavailable; cached results retained. ({exc})"}

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