"""
Universal Lottery System
Configurable lottery games with any parameters - supports any lottery worldwide.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Data directory (defined here to avoid circular import)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
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
    # US State Lotteries
    "california_superlotto": LotteryGame(
        key="california_superlotto",
        name="SuperLotto Plus (CA)",
        main_numbers=5,
        main_max=47,
        bonus_numbers=1,
        bonus_max=27,
        draw_days=["Wednesday", "Saturday"],
        start_date="1986-10-18",
        website="https://www.calottery.com",
    ),
    "new_york_lotto": LotteryGame(
        key="new_york_lotto",
        name="New York Lotto (NY)",
        main_numbers=6,
        main_max=59,
        draw_days=["Wednesday", "Saturday"],
        start_date="1978-09-12",
        website="https://www.nylottery.org",
    ),
    "texas_lotto": LotteryGame(
        key="texas_lotto",
        name="Lotto Texas (TX)",
        main_numbers=6,
        main_max=54,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="1992-11-07",
        website="https://www.texaslottery.com",
    ),
    "georgia_fantasy5": LotteryGame(
        key="georgia_fantasy5",
        name="Fantasy 5 (GA)",
        main_numbers=5,
        main_max=42,
        draw_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        start_date="1994-11-14",
        website="https://www.galottery.com",
    ),
    "illinois_lotto": LotteryGame(
        key="illinois_lotto",
        name="Illinois Lotto (IL)",
        main_numbers=6,
        main_max=52,
        draw_days=["Monday", "Thursday", "Saturday"],
        start_date="1974-07-30",
        website="https://www.illinoislottery.com",
    ),
    "pennsylvania_match6": LotteryGame(
        key="pennsylvania_match6",
        name="Match 6 Lotto (PA)",
        main_numbers=6,
        main_max=49,
        draw_days=["Monday", "Wednesday", "Friday"],
        start_date="2004-01-28",
        website="https://www.palottery.com",
    ),
    "ohio_classic_lotto": LotteryGame(
        key="ohio_classic_lotto",
        name="Classic Lotto (OH)",
        main_numbers=6,
        main_max=49,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="2007-01-22",
        website="https://www.ohiolottery.com",
    ),
    "michigan_lotto47": LotteryGame(
        key="michigan_lotto47",
        name="Lotto 47 (MI)",
        main_numbers=6,
        main_max=47,
        draw_days=["Wednesday", "Saturday"],
        start_date="2005-05-15",
        website="https://www.michiganlottery.com",
    ),
    "new_jersey_pick6": LotteryGame(
        key="new_jersey_pick6",
        name="Pick-6 (NJ)",
        main_numbers=6,
        main_max=46,
        draw_days=["Monday", "Thursday"],
        start_date="1980-05-09",
        website="https://www.njlottery.com",
    ),
    "virginia_lotto": LotteryGame(
        key="virginia_lotto",
        name="Virginia Lotto (VA)",
        main_numbers=6,
        main_max=38,
        draw_days=["Wednesday", "Saturday"],
        start_date="1988-09-13",
        website="https://www.valottery.com",
    ),
    "massachusetts_megabucks": LotteryGame(
        key="massachusetts_megabucks",
        name="Mass Megabucks (MA)",
        main_numbers=6,
        main_max=44,
        draw_days=["Wednesday", "Saturday"],
        start_date="1982-09-15",
        website="https://www.masslottery.com",
    ),
    "washington_lotto": LotteryGame(
        key="washington_lotto",
        name="Lotto (WA)",
        main_numbers=6,
        main_max=49,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="1984-07-14",
        website="https://www.walottery.com",
    ),
    "colorado_lotto": LotteryGame(
        key="colorado_lotto",
        name="Colorado Lotto+ (CO)",
        main_numbers=6,
        main_max=40,
        draw_days=["Wednesday", "Saturday"],
        start_date="1989-01-24",
        website="https://www.coloradolottery.com",
    ),
    "arizona_the_pick": LotteryGame(
        key="arizona_the_pick",
        name="The Pick (AZ)",
        main_numbers=6,
        main_max=44,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="1984-10-01",
        website="https://www.arizonalottery.com",
    ),
    "indiana_hoosier_lotto": LotteryGame(
        key="indiana_hoosier_lotto",
        name="Hoosier Lotto (IN)",
        main_numbers=6,
        main_max=46,
        draw_days=["Wednesday", "Saturday"],
        start_date="1989-10-28",
        website="https://www.hoosierlottery.com",
    ),
    "missouri_lotto": LotteryGame(
        key="missouri_lotto",
        name="Missouri Lotto (MO)",
        main_numbers=6,
        main_max=44,
        draw_days=["Wednesday", "Saturday"],
        start_date="1986-11-12",
        website="https://www.molottery.com",
    ),
    "wisconsin_megabucks": LotteryGame(
        key="wisconsin_megabucks",
        name="Megabucks (WI)",
        main_numbers=6,
        main_max=49,
        draw_days=["Wednesday", "Saturday"],
        start_date="1992-04-04",
        website="https://www.wilottery.com",
    ),
    "maryland_multi_match": LotteryGame(
        key="maryland_multi_match",
        name="Multi-Match (MD)",
        main_numbers=6,
        main_max=43,
        draw_days=["Monday", "Thursday"],
        start_date="2006-01-31",
        website="https://www.mdlottery.com",
    ),
    "connecticut_lotto": LotteryGame(
        key="connecticut_lotto",
        name="Lotto! (CT)",
        main_numbers=6,
        main_max=44,
        draw_days=["Tuesday", "Friday"],
        start_date="1983-11-10",
        website="https://www.ctlottery.org",
    ),
    "oregon_megabucks": LotteryGame(
        key="oregon_megabucks",
        name="Megabucks (OR)",
        main_numbers=6,
        main_max=48,
        draw_days=["Monday", "Thursday"],
        start_date="1985-11-09",
        website="https://www.oregonlottery.org",
    ),
    "minnesota_gopher5": LotteryGame(
        key="minnesota_gopher5",
        name="Gopher 5 (MN)",
        main_numbers=5,
        main_max=47,
        draw_days=["Monday", "Wednesday", "Friday"],
        start_date="1991-04-01",
        website="https://www.mnlotto.com",
    ),
    "kentucky_lotto": LotteryGame(
        key="kentucky_lotto",
        name="Kentucky Lotto (KY)",
        main_numbers=6,
        main_max=43,
        draw_days=["Wednesday", "Saturday"],
        start_date="1989-04-12",
        website="https://www.kylottery.com",
    ),
    "tennessee_lotto": LotteryGame(
        key="tennessee_lotto",
        name="Tennessee Lotto (TN)",
        main_numbers=6,
        main_max=42,
        draw_days=["Wednesday", "Saturday"],
        start_date="2004-06-23",
        website="https://www.tnlottery.com",
    ),
    "louisiana_lotto": LotteryGame(
        key="louisiana_lotto",
        name="Louisiana Lotto (LA)",
        main_numbers=6,
        main_max=42,
        draw_days=["Wednesday", "Saturday"],
        start_date="1992-01-15",
        website="https://www.louisianalottery.com",
    ),
    "iowa_lotto": LotteryGame(
        key="iowa_lotto",
        name="Iowa Lotto (IA)",
        main_numbers=6,
        main_max=44,
        draw_days=["Wednesday", "Saturday"],
        start_date="1985-08-22",
        website="https://www.ialottery.com",
    ),
    "oklahoma_lotto": LotteryGame(
        key="oklahoma_lotto",
        name="Lotto (OK)",
        main_numbers=6,
        main_max=44,
        draw_days=["Wednesday", "Saturday"],
        start_date="2005-10-12",
        website="https://www.oklottery.com",
    ),
    "kansas_super_kansas_cash": LotteryGame(
        key="kansas_super_kansas_cash",
        name="Super Kansas Cash (KS)",
        main_numbers=5,
        main_max=32,
        bonus_numbers=1,
        bonus_max=25,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="2002-02-12",
        website="https://www.kslottery.com",
    ),
    "arkansas_natural_state": LotteryGame(
        key="arkansas_natural_state",
        name="Natural State Jackpot (AR)",
        main_numbers=5,
        main_max=39,
        draw_days=["Monday", "Wednesday", "Friday"],
        start_date="2009-08-31",
        website="https://www.myarkansaslottery.com",
    ),
    "new_mexico_roadrunner": LotteryGame(
        key="new_mexico_roadrunner",
        name="Roadrunner Cash (NM)",
        main_numbers=5,
        main_max=37,
        draw_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        start_date="1998-01-19",
        website="https://www.nmlottery.com",
    ),
    "nebraska_pick5": LotteryGame(
        key="nebraska_pick5",
        name="Pick 5 (NE)",
        main_numbers=5,
        main_max=38,
        draw_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        start_date="2004-09-08",
        website="https://www.nelottery.com",
    ),
    "idaho_weekly_grand": LotteryGame(
        key="idaho_weekly_grand",
        name="Weekly Grand (ID)",
        main_numbers=5,
        main_max=40,
        draw_days=["Monday", "Thursday"],
        start_date="2008-01-07",
        website="https://www.idaholottery.com",
    ),
    "montana_big_sky": LotteryGame(
        key="montana_big_sky",
        name="Big Sky Bonus (MT)",
        main_numbers=5,
        main_max=28,
        bonus_numbers=1,
        bonus_max=17,
        draw_days=["Monday", "Wednesday", "Saturday"],
        start_date="2001-05-07",
        website="https://www.montanlottery.com",
    ),
    "utah_no_lottery": LotteryGame(
        key="utah_no_lottery",
        name="No State Lottery (UT)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Utah does not have a state lottery"
    ),
    "nevada_no_lottery": LotteryGame(
        key="nevada_no_lottery",
        name="No State Lottery (NV)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Nevada does not have a state lottery"
    ),
    "alabama_no_lottery": LotteryGame(
        key="alabama_no_lottery",
        name="No State Lottery (AL)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Alabama does not have a state lottery"
    ),
    "mississippi_no_lottery": LotteryGame(
        key="mississippi_no_lottery",
        name="No State Lottery (MS)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Mississippi does not have a state lottery"
    ),
    "hawaii_no_lottery": LotteryGame(
        key="hawaii_no_lottery",
        name="No State Lottery (HI)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Hawaii does not have a state lottery"
    ),
    "alaska_no_lottery": LotteryGame(
        key="alaska_no_lottery",
        name="No State Lottery (AK)",
        main_numbers=0,
        main_max=0,
        draw_days=[],
        start_date="1900-01-01",
        website="",
        notes="Alaska does not have a state lottery"
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

    def create_data_manager(self, key: str):
        """Create a data manager for a lottery game."""
        game = self.get_game(key)
        if not game:
            raise ValueError(f"Unknown lottery: {key}")
        # Import here to avoid circular import
        from .data import LotteryDataManager
        # LOTTERY_CONFIGS is now pre-populated in data.py from universal system
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