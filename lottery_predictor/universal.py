"""
Universal Lottery System
Configurable lottery games with any parameters - supports any lottery worldwide.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

from .data import LotteryDataManager, LOTTERY_CONFIGS, DATA_DIR

logger = logging.getLogger(__name__)

CONFIG_FILE = DATA_DIR / "universal_lotteries.json"


@dataclass
class LotteryGame:
    """Configuration for any lottery game."""
    key: str                      # Unique identifier (e.g., "powerball", "euromillions")
    name: str                     # Display name (e.g., "Powerball")
    main_numbers: int             # How many main numbers to pick
    main_min: int = 1             # Minimum main number (usually 1)
    main_max: int = 69            # Maximum main number
    bonus_numbers: int = 0        # How many bonus/powerball numbers
    bonus_min: int = 1            # Minimum bonus number
    bonus_max: int = 26           # Maximum bonus number
    draw_days: List[str] = field(default_factory=lambda: ["Wednesday", "Saturday"])
    start_date: str = "1992-01-01"  # First draw date (YYYY-MM-DD)
    timezone: str = "UTC"         # Timezone for draw times
    website: str = ""             # Official website
    notes: str = ""               # Additional notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "main_numbers": self.main_numbers,
            "main_min": self.main_min,
            "main_max": self.main_max,
            "bonus_numbers": self.bonus_numbers,
            "bonus_min": self.bonus_min,
            "bonus_max": self.bonus_max,
            "draw_days": self.draw_days,
            "start_date": self.start_date,
            "timezone": self.timezone,
            "website": self.website,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LotteryGame":
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.main_numbers <= 0:
            errors.append("main_numbers must be > 0")
        if self.main_min >= self.main_max:
            errors.append("main_min must be < main_max")
        if self.main_numbers > (self.main_max - self.main_min + 1):
            errors.append(f"main_numbers ({self.main_numbers}) exceeds available range")
        if self.bonus_numbers < 0:
            errors.append("bonus_numbers cannot be negative")
        if self.bonus_numbers > 0 and self.bonus_min >= self.bonus_max:
            errors.append("bonus_min must be < bonus_max")
        if self.bonus_numbers > (self.bonus_max - self.bonus_min + 1):
            errors.append(f"bonus_numbers ({self.bonus_numbers}) exceeds available range")
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in self.draw_days:
            if day not in valid_days:
                errors.append(f"Invalid draw day: {day}")
        return errors


# Built-in popular lottery games
BUILTIN_LOTTERIES = {
    "powerball": LotteryGame(
        key="powerball",
        name="Powerball (US)",
        main_numbers=5,
        main_max=69,
        bonus_numbers=1,
        bonus_max=26,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="1992-04-22",
        website="https://www.powerball.com",
    ),
    "florida_lotto": LotteryGame(
        key="florida_lotto",
        name="Florida Lotto",
        main_numbers=6,
        main_max=53,
        draw_days=["Wednesday", "Saturday"],
        start_date="1988-01-06",
        website="https://www.flalottery.com",
    ),
    "mega_millions": LotteryGame(
        key="mega_millions",
        name="Mega Millions (US)",
        main_numbers=5,
        main_max=70,
        bonus_numbers=1,
        bonus_max=25,
        draw_days=["Tuesday", "Friday"],
        start_date="2002-05-17",
        website="https://www.megamillions.com",
    ),
    "euromillions": LotteryGame(
        key="euromillions",
        name="EuroMillions (EU)",
        main_numbers=5,
        main_max=50,
        bonus_numbers=2,
        bonus_max=12,
        draw_days=["Tuesday", "Friday"],
        start_date="2004-02-13",
        website="https://www.euro-millions.com",
    ),
    "uk_lotto": LotteryGame(
        key="uk_lotto",
        name="UK National Lottery",
        main_numbers=6,
        main_max=59,
        draw_days=["Wednesday", "Saturday"],
        start_date="1994-11-19",
        website="https://www.national-lottery.co.uk",
    ),
    "canada_lotto649": LotteryGame(
        key="canada_lotto649",
        name="Lotto 6/49 (Canada)",
        main_numbers=6,
        main_max=49,
        bonus_numbers=1,
        bonus_max=49,
        draw_days=["Wednesday", "Saturday"],
        start_date="1982-06-12",
        website="https://www.lotoquebec.com",
    ),
    "australia_ozlotto": LotteryGame(
        key="australia_ozlotto",
        name="Oz Lotto (Australia)",
        main_numbers=7,
        main_max=47,
        draw_days=["Tuesday"],
        start_date="1994-02-22",
        website="https://www.ozlotteries.com",
    ),
    "australia_powerball": LotteryGame(
        key="australia_powerball",
        name="Powerball (Australia)",
        main_numbers=7,
        main_max=35,
        bonus_numbers=1,
        bonus_max=20,
        draw_days=["Thursday"],
        start_date="1996-05-23",
        website="https://www.powerball.net.au",
    ),
    "superenalotto": LotteryGame(
        key="superenalotto",
        name="SuperEnalotto (Italy)",
        main_numbers=6,
        main_max=90,
        bonus_numbers=1,
        bonus_max=90,
        draw_days=["Tuesday", "Thursday", "Saturday"],
        start_date="1997-12-03",
        website="https://www.sisal.it",
    ),
    "eurojackpot": LotteryGame(
        key="eurojackpot",
        name="Eurojackpot (EU)",
        main_numbers=5,
        main_max=50,
        bonus_numbers=2,
        bonus_max=12,
        draw_days=["Tuesday", "Friday"],
        start_date="2012-03-23",
        website="https://www.eurojackpot.org",
    ),
}


class UniversalLotteryManager:
    """Manages universal lottery configurations."""

    def __init__(self):
        self.games: Dict[str, LotteryGame] = {}
        self.load_all()

    def load_all(self):
        """Load built-in and custom lotteries."""
        # Load built-in
        self.games = BUILTIN_LOTTERIES.copy()

        # Load custom from file
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    custom = json.load(f)
                for data in custom:
                    game = LotteryGame.from_dict(data)
                    errors = game.validate()
                    if errors:
                        logger.warning(f"Invalid custom lottery {game.key}: {errors}")
                    else:
                        self.games[game.key] = game
            except Exception as e:
                logger.warning(f"Failed to load custom lotteries: {e}")

    def save_custom(self):
        """Save custom (non-builtin) lotteries to file."""
        custom = [g.to_dict() for k, g in self.games.items() if k not in BUILTIN_LOTTERIES]
        DATA_DIR.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(custom, f, indent=2)

    def get_game(self, key: str) -> Optional[LotteryGame]:
        """Get lottery game by key."""
        return self.games.get(key)

    def list_games(self) -> List[LotteryGame]:
        """List all available games."""
        return list(self.games.values())

    def add_game(self, game: LotteryGame) -> bool:
        """Add a custom lottery game."""
        errors = game.validate()
        if errors:
            raise ValueError(f"Invalid lottery: {errors}")
        if game.key in BUILTIN_LOTTERIES:
            raise ValueError(f"Cannot override built-in lottery: {game.key}")
        self.games[game.key] = game
        self.save_custom()
        return True

    def remove_game(self, key: str) -> bool:
        """Remove a custom lottery game."""
        if key in BUILTIN_LOTTERIES:
            raise ValueError(f"Cannot remove built-in lottery: {key}")
        if key in self.games:
            del self.games[key]
            self.save_custom()
            return True
        return False

    def create_data_manager(self, key: str) -> LotteryDataManager:
        """Create a data manager for a lottery game."""
        game = self.get_game(key)
        if not game:
            raise ValueError(f"Unknown lottery: {key}")

        # Temporarily add to LOTTERY_CONFIGS for compatibility
        config = {
            "name": game.name,
            "main_numbers": game.main_numbers,
            "main_max": game.main_max,
            "bonus_numbers": game.bonus_numbers,
            "bonus_max": game.bonus_max,
            "draw_days": game.draw_days,
            "start_date": game.start_date,
            "api_url": game.website,
        }

        # Monkey-patch the config for this session
        LOTTERY_CONFIGS[key] = config
        return LotteryDataManager(key)


# Global instance
_manager = None

def get_universal_manager() -> UniversalLotteryManager:
    """Get the global universal lottery manager."""
    global _manager
    if _manager is None:
        _manager = UniversalLotteryManager()
    return _manager


def register_lottery(game: LotteryGame) -> bool:
    """Register a new lottery game."""
    return get_universal_manager().add_game(game)


def list_lotteries() -> List[LotteryGame]:
    """List all available lottery games."""
    return get_universal_manager().list_games()


def get_lottery(key: str) -> Optional[LotteryGame]:
    """Get a lottery game by key."""
    return get_universal_manager().get_game(key)


if __name__ == "__main__":
    # Demo
    mgr = UniversalLotteryManager()
    print("Available lotteries:")
    for game in mgr.list_games():
        bonus = f" + {game.bonus_numbers} bonus (1-{game.bonus_max})" if game.bonus_numbers > 0 else ""
        print(f"  {game.key}: {game.name} - {game.main_numbers}/{game.main_max}{bonus} - {', '.join(game.draw_days)}")