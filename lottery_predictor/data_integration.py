"""
Data Integration Module for Lottery Predictor (v1.4)
Handles real API integrations, web scraping, scheduled updates, and data validation.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import schedule
import time
import threading
from urllib.parse import urljoin

from .data import LotteryDataManager, get_data_manager, LOTTERY_CONFIGS
from .universal import get_universal_manager

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    lottery_key: str
    type: str  # 'api', 'scrape', 'csv'
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    parser: Optional[str] = None  # Function name to parse response
    enabled: bool = True
    rate_limit: float = 1.0  # seconds between requests


# Built-in data sources for major lotteries
BUILTIN_SOURCES = [
    # Powerball - official API (if available) or scrape
    DataSource(
        name="Powerball Official",
        lottery_key="powerball",
        type="scrape",
        url="https://www.powerball.com/previous-results",
        parser="parse_powerball_results",
        rate_limit=2.0
    ),
    # Mega Millions
    DataSource(
        name="Mega Millions Official",
        lottery_key="mega_millions",
        type="scrape",
        url="https://www.megamillions.com/winning-numbers",
        parser="parse_megamillions_results",
        rate_limit=2.0
    ),
    # Florida Lotto
    DataSource(
        name="Florida Lottery Official",
        lottery_key="florida_lotto",
        type="scrape",
        url="https://flalottery.com/lotto/winning-numbers",
        parser="parse_florida_lotto_results",
        rate_limit=2.0
    ),
    # California SuperLotto Plus
    DataSource(
        name="California Lottery Official",
        lottery_key="california_superlotto",
        type="scrape",
        url="https://www.calottery.com/draw-games/superlottoplus/winning-numbers",
        parser="parse_california_superlotto_results",
        rate_limit=2.0
    ),
    # New York Lotto
    DataSource(
        name="New York Lottery Official",
        lottery_key="new_york_lotto",
        type="scrape",
        url="https://nylottery.ny.gov/lotto",
        parser="parse_ny_lotto_results",
        rate_limit=2.0
    ),
    # Texas Lotto
    DataSource(
        name="Texas Lottery Official",
        lottery_key="texas_lotto",
        type="scrape",
        url="https://www.texaslottery.com/export/sites/lottery/Games/Lotto_Texas/Winning_Numbers/",
        parser="parse_texas_lotto_results",
        rate_limit=2.0
    ),
]


class APIClient:
    """Async HTTP client for API requests with rate limiting."""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request = 0
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, url: str, headers: Optional[Dict] = None,
                  params: Optional[Dict] = None) -> Optional[Any]:
        """GET request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            self.last_request = time.time()
            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await resp.json()
                    else:
                        return await resp.text()
                else:
                    logger.warning(f"API request failed: {resp.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"API request error for {url}: {e}")
            return None


class WebScraper:
    """Web scraper for lottery result pages."""

    def __init__(self, rate_limit: float = 2.0):
        self.rate_limit = rate_limit
        self.last_request = 0
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': 'Mozilla/5.0 (compatible; LotteryPredictor/1.0)'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch(self, url: str) -> Optional[str]:
        """Fetch page content with rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            self.last_request = time.time()
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    logger.warning(f"Scrape failed: {resp.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Scrape error for {url}: {e}")
            return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML with BeautifulSoup."""
        return BeautifulSoup(html, 'lxml')


class DataValidator:
    """Validate and detect anomalies in lottery data."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS.get(lottery_type, {})
        self.dm = get_data_manager(lottery_type)

    def validate_draw(self, draw: Dict) -> tuple[bool, List[str]]:
        """Validate a single draw. Returns (is_valid, list_of_errors)."""
        errors = []

        # Check required fields
        required = ['date', 'main_numbers', 'bonus_numbers', 'draw_number', 'jackpot']
        for field in required:
            if field not in draw:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate main numbers
        main = draw['main_numbers']
        if not isinstance(main, list) or len(main) != self.config.get('main_numbers', 5):
            errors.append(f"Invalid main numbers count: expected {self.config.get('main_numbers')}, got {len(main) if isinstance(main, list) else 'not a list'}")

        if isinstance(main, list):
            for n in main:
                if not isinstance(n, int) or n < 1 or n > self.config.get('main_max', 69):
                    errors.append(f"Invalid main number: {n} (range 1-{self.config.get('main_max')})")

        # Check for duplicates
        if isinstance(main, list) and len(main) != len(set(main)):
            errors.append("Duplicate main numbers found")

        # Validate bonus numbers
        bonus = draw['bonus_numbers']
        expected_bonus = self.config.get('bonus_numbers', 0)
        if len(bonus) != expected_bonus:
            errors.append(f"Invalid bonus numbers count: expected {expected_bonus}, got {len(bonus)}")

        if isinstance(bonus, list):
            for n in bonus:
                if not isinstance(n, int) or n < 1 or n > self.config.get('bonus_max', 26):
                    errors.append(f"Invalid bonus number: {n} (range 1-{self.config.get('bonus_max')})")

        # Validate date
        try:
            if isinstance(draw['date'], str):
                pd.Timestamp(draw['date'])
        except Exception:
            errors.append(f"Invalid date format: {draw['date']}")

        # Validate jackpot
        if not isinstance(draw['jackpot'], (int, float)) or draw['jackpot'] < 0:
            errors.append(f"Invalid jackpot: {draw['jackpot']}")

        return len(errors) == 0, errors

    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect statistical anomalies in historical data."""
        anomalies = []

        if len(df) == 0:
            return anomalies

        # Check for missing dates
        df = df.sort_values('date')
        expected_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        draw_dates = set(pd.to_datetime(df['date']).dt.date)
        missing = [d for d in expected_dates if d.date() not in draw_dates and
                   d.weekday() in self._get_draw_weekdays()]
        if missing:
            anomalies.append({
                'type': 'missing_draws',
                'count': len(missing),
                'details': f"Missing {len(missing)} expected draw dates"
            })

        # Check for duplicate draws
        dup_main = df[df.duplicated(subset='main_numbers', keep=False)]
        if len(dup_main) > 0:
            anomalies.append({
                'type': 'duplicate_main_numbers',
                'count': len(dup_main),
                'details': f"Found {len(dup_main)} draws with identical main numbers"
            })

        # Statistical anomalies - sum outliers
        df['sum'] = df['main_numbers'].apply(sum)
        sum_mean = df['sum'].mean()
        sum_std = df['sum'].std()
        outliers = df[np.abs(df['sum'] - sum_mean) > 3 * sum_std]
        if len(outliers) > 0:
            anomalies.append({
                'type': 'sum_outliers',
                'count': len(outliers),
                'details': f"Found {len(outliers)} draws with sum > 3 std from mean"
            })

        # Frequency anomalies - numbers that never appear
        all_numbers = set(range(1, self.config.get('main_max', 69) + 1))
        seen_numbers = set()
        for nums in df['main_numbers']:
            seen_numbers.update(nums)
        missing_numbers = all_numbers - seen_numbers
        if missing_numbers:
            anomalies.append({
                'type': 'missing_numbers',
                'count': len(missing_numbers),
                'details': f"Numbers never drawn: {sorted(missing_numbers)}"
            })

        return anomalies

    def _get_draw_weekdays(self) -> List[int]:
        """Get weekday numbers for draw days (0=Monday)."""
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                   'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        return [day_map[d] for d in self.config.get('draw_days', []) if d in day_map]


# Parser functions for specific lottery sites
async def parse_powerball_results(html: str) -> List[Dict]:
    """Parse Powerball results from powerball.com."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    # This is a template - actual selectors depend on site structure
    try:
        rows = soup.select('table.results tbody tr') or soup.select('.results-table tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)
                bonus_str = cells[2].get_text(strip=True)

                # Parse numbers
                main = [int(n) for n in numbers_str.split() if n.isdigit()]
                bonus = [int(n) for n in bonus_str.split() if n.isdigit()]

                if main and bonus:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:5]),
                        'bonus_numbers': bonus[:1],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"Powerball parse error: {e}")
    return results


async def parse_megamillions_results(html: str) -> List[Dict]:
    """Parse Mega Millions results."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    try:
        rows = soup.select('.winning-numbers-table tbody tr') or soup.select('table tbody tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)
                bonus_str = cells[2].get_text(strip=True)

                main = [int(n) for n in numbers_str.replace('-', ' ').split() if n.isdigit()]
                bonus = [int(n) for n in bonus_str.split() if n.isdigit()]

                if main and bonus:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:5]),
                        'bonus_numbers': bonus[:1],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"Mega Millions parse error: {e}")
    return results


async def parse_florida_lotto_results(html: str) -> List[Dict]:
    """Parse Florida Lotto results."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    try:
        rows = soup.select('table.winning-numbers tbody tr') or soup.select('.results tbody tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)

                main = [int(n) for n in numbers_str.replace('-', ' ').split() if n.isdigit()]

                if main:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:6]),
                        'bonus_numbers': [],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"Florida Lotto parse error: {e}")
    return results


async def parse_california_superlotto_results(html: str) -> List[Dict]:
    """Parse California SuperLotto Plus results."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    try:
        rows = soup.select('.winning-numbers tbody tr') or soup.select('table tbody tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)
                bonus_str = cells[2].get_text(strip=True)

                main = [int(n) for n in numbers_str.split() if n.isdigit()]
                bonus = [int(n) for n in bonus_str.split() if n.isdigit()]

                if main and bonus:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:5]),
                        'bonus_numbers': bonus[:1],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"California SuperLotto parse error: {e}")
    return results


async def parse_ny_lotto_results(html: str) -> List[Dict]:
    """Parse New York Lotto results."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    try:
        rows = soup.select('table.results tbody tr') or soup.select('.draw-results tbody tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)

                main = [int(n) for n in numbers_str.split() if n.isdigit()]

                if main:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:6]),
                        'bonus_numbers': [],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"NY Lotto parse error: {e}")
    return results


async def parse_texas_lotto_results(html: str) -> List[Dict]:
    """Parse Texas Lotto results."""
    soup = BeautifulSoup(html, 'lxml')
    results = []
    try:
        rows = soup.select('table.winning-numbers tbody tr') or soup.select('table tbody tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                date_str = cells[0].get_text(strip=True)
                numbers_str = cells[1].get_text(strip=True)

                main = [int(n) for n in numbers_str.split() if n.isdigit()]

                if main:
                    results.append({
                        'date': date_str,
                        'main_numbers': sorted(main[:6]),
                        'bonus_numbers': [],
                        'draw_number': len(results) + 1,
                        'jackpot': 0
                    })
    except Exception as e:
        logger.warning(f"Texas Lotto parse error: {e}")
    return results


# Map of parser functions
PARSERS = {
    'parse_powerball_results': parse_powerball_results,
    'parse_megamillions_results': parse_megamillions_results,
    'parse_florida_lotto_results': parse_florida_lotto_results,
    'parse_california_superlotto_results': parse_california_superlotto_results,
    'parse_ny_lotto_results': parse_ny_lotto_results,
    'parse_texas_lotto_results': parse_texas_lotto_results,
}


class DataIntegrator:
    """Main class for integrating data from multiple sources."""

    def __init__(self):
        self.universal_manager = get_universal_manager()
        self.sources = BUILTIN_SOURCES.copy()
        self.validator = None

    def add_source(self, source: DataSource):
        """Add a custom data source."""
        self.sources.append(source)

    def get_sources_for_lottery(self, lottery_key: str) -> List[DataSource]:
        """Get all enabled sources for a lottery."""
        return [s for s in self.sources if s.lottery_key == lottery_key and s.enabled]

    async def fetch_from_source(self, source: DataSource) -> List[Dict]:
        """Fetch data from a single source."""
        logger.info(f"Fetching from {source.name} for {source.lottery_key}")

        if source.type == 'api':
            async with APIClient(rate_limit=source.rate_limit) as client:
                data = await client.get(source.url, headers=source.headers, params=source.params)
                if data and source.parser and source.parser in PARSERS:
                    return await PARSERS[source.parser](json.dumps(data) if isinstance(data, dict) else data)
                return data if isinstance(data, list) else []

        elif source.type == 'scrape':
            async with WebScraper(rate_limit=source.rate_limit) as scraper:
                html = await scraper.fetch(source.url)
                if html and source.parser and source.parser in PARSERS:
                    return await PARSERS[source.parser](html)
                return []

        return []

    async def fetch_all(self, lottery_key: str) -> List[Dict]:
        """Fetch from all sources for a lottery and merge results."""
        sources = self.get_sources_for_lottery(lottery_key)
        if not sources:
            logger.warning(f"No sources configured for {lottery_key}")
            return []

        all_results = []
        for source in sources:
            try:
                results = await self.fetch_from_source(source)
                if results:
                    all_results.extend(results)
                    logger.info(f"Got {len(results)} draws from {source.name}")
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}")

        # Deduplicate by date and numbers
        seen = set()
        unique = []
        for r in all_results:
            key = (r.get('date'), tuple(r.get('main_numbers', [])), tuple(r.get('bonus_numbers', [])))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return sorted(unique, key=lambda x: x.get('date', ''))

    def validate_and_save(self, lottery_key: str, draws: List[Dict]) -> Dict:
        """Validate draws and save to data manager."""
        self.validator = DataValidator(lottery_key)
        dm = get_data_manager(lottery_key)

        valid_draws = []
        invalid_draws = []
        all_errors = []

        for draw in draws:
            is_valid, errors = self.validator.validate_draw(draw)
            if is_valid:
                valid_draws.append(draw)
            else:
                invalid_draws.append({'draw': draw, 'errors': errors})
                all_errors.extend(errors)

        # Save valid draws
        saved = 0
        if valid_draws:
            df_new = pd.DataFrame(valid_draws)
            df_new['date'] = pd.to_datetime(df_new['date'])
            df_new = df_new.sort_values('date')

            # Load existing data
            existing = dm.load_data()
            if len(existing) > 0:
                # Merge and deduplicate
                combined = pd.concat([existing, df_new]).drop_duplicates(
                    subset=['date', 'main_numbers'], keep='last'
                ).sort_values('date')
            else:
                combined = df_new

            dm.save_data(combined)
            saved = len(valid_draws)

        # Run anomaly detection on full dataset
        full_data = dm.load_data()
        anomalies = self.validator.detect_anomalies(full_data)

        return {
            'lottery': lottery_key,
            'total_fetched': len(draws),
            'valid': len(valid_draws),
            'invalid': len(invalid_draws),
            'saved': saved,
            'errors': all_errors[:10],  # Limit errors
            'anomalies': anomalies
        }


class ScheduledUpdater:
    """Schedule automatic data updates."""

    def __init__(self):
        self.integrator = DataIntegrator()
        self.running = False
        self.thread = None

    def schedule_updates(self, lottery_keys: List[str], interval_hours: int = 24):
        """Schedule updates for specified lotteries."""
        for key in lottery_keys:
            schedule.every(interval_hours).hours.do(
                self._run_update_job, lottery_key=key
            ).tag(key)
        logger.info(f"Scheduled updates for {lottery_keys} every {interval_hours} hours")

    def _run_update_job(self, lottery_key: str):
        """Job to run async update."""
        asyncio.run(self._update_lottery(lottery_key))

    async def _update_lottery(self, lottery_key: str):
        """Update a single lottery."""
        logger.info(f"Scheduled update for {lottery_key}")
        try:
            draws = await self.integrator.fetch_all(lottery_key)
            result = self.integrator.validate_and_save(lottery_key, draws)
            logger.info(f"Update result for {lottery_key}: {result['saved']} saved, "
                        f"{result['invalid']} invalid, {len(result['anomalies'])} anomalies")
        except Exception as e:
            logger.error(f"Scheduled update failed for {lottery_key}: {e}")

    def start(self):
        """Start the scheduler in background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info("Scheduled updater started")

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        schedule.clear()
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Scheduled updater stopped")

    def _run_scheduler(self):
        """Run scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def generate_github_actions_workflow() -> str:
    """Generate GitHub Actions workflow for scheduled updates."""
    workflow = """name: Scheduled Lottery Data Updates

on:
  schedule:
    # Run daily at 6 AM UTC (after most draws complete)
    - cron: '0 6 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  update-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Update lottery data
        run: |
          python -c "
          from lottery_predictor.data_integration import ScheduledUpdater
          updater = ScheduledUpdater()
          # Update all major lotteries
          lotteries = ['powerball', 'mega_millions', 'florida_lotto',
                       'california_superlotto', 'new_york_lotto', 'texas_lotto']
          import asyncio
          for lt in lotteries:
              asyncio.run(updater.integrator.fetch_all(lt))
              # Save logic here...
          print('Data update complete')
          "

      - name: Commit and push if changed
        run: |
          git config user.name 'github-actions[bot]'
          git config user.email 'github-actions[bot]@users.noreply.github.com'
          git add data/
          git diff --quiet && git diff --staged --quiet || git commit -m 'chore: update lottery data [skip ci]' && git push
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
"""
    return workflow


def save_github_workflow():
    """Save GitHub Actions workflow file."""
    workflow = generate_github_actions_workflow()
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = workflow_dir / "update_lottery_data.yml"
    workflow_path.write_text(workflow)
    logger.info(f"GitHub Actions workflow saved to {workflow_path}")
    return str(workflow_path)


# CLI functions
async def update_lottery_data(lottery_key: str) -> Dict:
    """Update data for a single lottery from all sources."""
    integrator = DataIntegrator()
    draws = await integrator.fetch_all(lottery_key)
    return integrator.validate_and_save(lottery_key, draws)


async def update_all_lotteries(lottery_keys: Optional[List[str]] = None) -> Dict:
    """Update data for all or specified lotteries."""
    if lottery_keys is None:
        lottery_keys = list(LOTTERY_CONFIGS.keys())

    integrator = DataIntegrator()
    results = {}

    for key in lottery_keys:
        logger.info(f"Updating {key}...")
        draws = await integrator.fetch_all(key)
        result = integrator.validate_and_save(key, draws)
        results[key] = result

    return results


def run_scheduled_updater(lottery_keys: Optional[List[str]] = None, interval_hours: int = 24):
    """Run the scheduled updater (blocking)."""
    updater = ScheduledUpdater()
    keys = lottery_keys or list(LOTTERY_CONFIGS.keys())
    updater.schedule_updates(keys, interval_hours)
    updater.start()

    try:
        while True:
            time.sleep(3600)  # Sleep 1 hour
    except KeyboardInterrupt:
        updater.stop()


if __name__ == "__main__":
    # Test the module
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "generate-workflow":
            save_github_workflow()
            print("GitHub Actions workflow generated")
        elif sys.argv[1] == "update":
            key = sys.argv[2] if len(sys.argv) > 2 else "powerball"
            result = asyncio.run(update_lottery_data(key))
            print(json.dumps(result, indent=2, default=str))
        elif sys.argv[1] == "update-all":
            results = asyncio.run(update_all_lotteries())
            print(json.dumps(results, indent=2, default=str))
        elif sys.argv[1] == "validate":
            key = sys.argv[2] if len(sys.argv) > 2 else "powerball"
            dm = get_data_manager(key)
            df = dm.load_data()
            validator = DataValidator(key)
            anomalies = validator.detect_anomalies(df)
            print(json.dumps(anomalies, indent=2, default=str))
    else:
        print("Usage: python -m lottery_predictor.data_integration [generate-workflow|update|update-all|validate] [lottery_key]")