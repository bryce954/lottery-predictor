# 🎲 Lottery Predictor - AI-Powered Universal Lottery Predictions

An intelligent lottery number prediction system using historical data analysis, statistical methods, and machine learning models. **Supports all US state lotteries and major international games.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Lotteries](https://img.shields.io/badge/Lotteries-48+-orange.svg)

## ⚠️ Important Disclaimer

> **This tool is for entertainment and educational purposes only.** Lottery draws are random events. No algorithm, AI, or statistical analysis can predict future lottery numbers with certainty. Past performance does not guarantee future results. Please play responsibly and within your means.

## Features

- 📊 **Historical Data Analysis** - Comprehensive analysis of years of lottery draw data
- 🔥 **Hot/Cold Number Tracking** - Identify frequently and rarely drawn numbers
- 📈 **Statistical Predictions** - Frequency-based, balanced, and pair-based strategies
- 🤖 **Machine Learning Models** - Random Forest, MLP Neural Network, XGBoost ensemble
- 🧠 **Deep Learning** - Neural network predictions (experimental)
- 🎯 **Ensemble Consensus** - Combines multiple methods for best predictions
- 📊 **Rich Visualizations** - Frequency charts, heatmaps, timeline analysis
- 🖥️ **Beautiful CLI** - Rich terminal interface with tables and progress bars
- 🌐 **Streamlit Web App** - Interactive web-based GUI (run `streamlit run streamlit_app.py`)
- 🌍 **Universal Support** - 48+ lotteries: all US states + international games

## Supported Lotteries

### Major Multi-State Games
| Lottery | Key | Main Numbers | Bonus Numbers | Draw Days |
|---------|-----|-------------|---------------|-----------|
| **Powerball** | `powerball` | 5 (1-69) | 1 (1-26) | Mon, Wed, Sat |
| **Mega Millions** | `mega_millions` | 5 (1-70) | 1 (1-25) | Tue, Fri |
| **Florida Lotto** | `florida_lotto` | 6 (1-53) | None | Wed, Sat |

### US State Lotteries (35+)
| Lottery | Key | Main Numbers | Bonus Numbers | Draw Days |
|---------|-----|-------------|---------------|-----------|
| SuperLotto Plus (CA) | `california_superlotto` | 5 (1-47) | 1 (1-27) | Wed, Sat |
| New York Lotto | `new_york_lotto` | 6 (1-59) | None | Wed, Sat |
| Lotto Texas | `texas_lotto` | 6 (1-54) | None | Mon, Wed, Sat |
| Fantasy 5 (GA) | `georgia_fantasy5` | 5 (1-42) | None | Daily |
| Illinois Lotto | `illinois_lotto` | 6 (1-52) | None | Mon, Thu, Sat |
| Match 6 (PA) | `pennsylvania_match6` | 6 (1-49) | None | Mon, Wed, Fri |
| Classic Lotto (OH) | `ohio_classic_lotto` | 6 (1-49) | None | Mon, Wed, Sat |
| Lotto 47 (MI) | `michigan_lotto47` | 6 (1-47) | None | Wed, Sat |
| Pick-6 (NJ) | `new_jersey_pick6` | 6 (1-46) | None | Mon, Thu |
| Virginia Lotto | `virginia_lotto` | 6 (1-38) | None | Wed, Sat |
| Mass Megabucks (MA) | `massachusetts_megabucks` | 6 (1-44) | None | Wed, Sat |
| Lotto (WA) | `washington_lotto` | 6 (1-49) | None | Mon, Wed, Sat |
| Colorado Lotto+ | `colorado_lotto` | 6 (1-40) | None | Wed, Sat |
| The Pick (AZ) | `arizona_the_pick` | 6 (1-44) | None | Mon, Wed, Sat |
| Hoosier Lotto (IN) | `indiana_hoosier_lotto` | 6 (1-46) | None | Wed, Sat |
| Missouri Lotto | `missouri_lotto` | 6 (1-44) | None | Wed, Sat |
| Megabucks (WI) | `wisconsin_megabucks` | 6 (1-49) | None | Wed, Sat |
| Multi-Match (MD) | `maryland_multi_match` | 6 (1-43) | None | Mon, Thu |
| Lotto! (CT) | `connecticut_lotto` | 6 (1-44) | None | Tue, Fri |
| Megabucks (OR) | `oregon_megabucks` | 6 (1-48) | None | Mon, Thu |
| Gopher 5 (MN) | `minnesota_gopher5` | 5 (1-47) | None | Mon, Wed, Fri |
| Kentucky Lotto | `kentucky_lotto` | 6 (1-43) | None | Wed, Sat |
| Tennessee Lotto | `tennessee_lotto` | 6 (1-42) | None | Wed, Sat |
| Louisiana Lotto | `louisiana_lotto` | 6 (1-42) | None | Wed, Sat |
| Iowa Lotto | `iowa_lotto` | 6 (1-44) | None | Wed, Sat |
| Lotto (OK) | `oklahoma_lotto` | 6 (1-44) | None | Wed, Sat |
| Super Kansas Cash | `kansas_super_kansas_cash` | 5 (1-32) | 1 (1-25) | Mon, Wed, Sat |
| Natural State Jackpot (AR) | `arkansas_natural_state` | 5 (1-39) | None | Mon, Wed, Fri |
| Roadrunner Cash (NM) | `new_mexico_roadrunner` | 5 (1-37) | None | Daily |
| Pick 5 (NE) | `nebraska_pick5` | 5 (1-38) | None | Daily |
| Weekly Grand (ID) | `idaho_weekly_grand` | 5 (1-40) | None | Mon, Thu |
| Big Sky Bonus (MT) | `montana_big_sky` | 5 (1-28) | 1 (1-17) | Mon, Wed, Sat |

*States without lotteries: AL, AK, HI, MS, NV, UT (listed in system for completeness)*

### International Games
| Lottery | Key | Main Numbers | Bonus Numbers | Draw Days |
|---------|-----|-------------|---------------|-----------|
| EuroMillions | `euromillions` | 5 (1-50) | 2 (1-12) | Tue, Fri |
| UK National Lottery | `uk_lotto` | 6 (1-59) | None | Wed, Sat |
| Lotto 6/49 (Canada) | `canada_lotto649` | 6 (1-49) | 1 (1-49) | Wed, Sat |
| Oz Lotto (Australia) | `australia_ozlotto` | 7 (1-47) | None | Tue |
| Powerball (Australia) | `australia_powerball` | 7 (1-35) | 1 (1-20) | Thu |
| SuperEnalotto (Italy) | `superenalotto` | 6 (1-90) | 1 (1-90) | Tue, Thu, Sat |
| Eurojackpot | `eurojackpot` | 5 (1-50) | 2 (1-12) | Tue, Fri |

**Use `python main.py info` to see all available lotteries with details.**

## Installation

```bash
# Clone the repository
git clone https://github.com/bryce954/lottery-predictor.git
cd lottery-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```bash
# Generate predictions for Powerball (default)
python main.py predict

# Generate predictions for any lottery
python main.py predict --lottery mega_millions
python main.py predict --lottery florida_lotto
python main.py predict --lottery california_superlotto
python main.py predict --lottery euromillions

# Use specific method
python main.py predict --method statistical --count 10
python main.py predict --method ml --count 5
python main.py predict --method ensemble --count 5

# Analyze historical data
python main.py analyze --lottery powerball --last-n 500

# Train ML models
python main.py train --lottery powerball

# Show recent draws
python main.py history --lottery florida_lotto --count 20

# Compare methods
python main.py compare --lottery powerball --methods statistical ml ensemble

# Generate visualizations
python main.py viz --lottery powerball

# Launch web GUI
streamlit run streamlit_app.py

# Launch desktop GUI (requires tkinter: sudo apt-get install python3-tk on Ubuntu/Debian)
python tkinter_app.py

# v1.4 Data Integration commands
python main.py update-data --lottery powerball
python main.py validate-data --lottery powerball
python main.py generate-workflow
```

## Commands

### `predict` - Generate Predictions
```bash
python main.py predict [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
                         Use 'info' command to see all 48+ options
  -m, --method TEXT      Method: statistical, frequency, cold, pairs, ml, time_series, ensemble (default: ensemble)
  -c, --count INTEGER    Number of predictions (default: 5)
  -n, --last-n INTEGER   Historical draws to analyze (default: 200)
  -t, --train            Retrain ML models before predicting
  -v, --visualize        Generate visualization charts
  -o, --output FILE      Save predictions to JSON file
```

### `analyze` - Historical Analysis
```bash
python main.py analyze [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
  -n, --last-n INTEGER   Draws to analyze (default: 200)
  --visualize/--no-visualize  Generate charts (default: True)
```

### `train` - Train ML Models
```bash
python main.py train [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
  --lookback INTEGER     Previous draws as features (default: 10)
```

### `history` - Recent Draws
```bash
python main.py history [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
  -c, --count INTEGER    Recent draws to show (default: 10)
```

### `compare` - Method Comparison
```bash
python main.py compare [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
  -m, --methods LIST     Methods to compare (default: all)
  -c, --count INTEGER    Predictions per method (default: 3)
```

### `viz` - Visualizations
```bash
python main.py viz [OPTIONS]

Options:
  -l, --lottery TEXT         Lottery type (default: powerball)
  -n, --last-n INTEGER       Draws to visualize (default: 200)
  -p, --predictions FILE     JSON file with predictions to visualize
```

### `backtest` - Backtesting Framework
```bash
python backtest.py [OPTIONS]

Options:
  -l, --lottery TEXT         Lottery type (default: powerball)
  -m, --methods TEXT         Methods to test (comma-separated, default: all)
  --train-window INT         Training window size (default: 200)
  --test-window INT          Test window size (default: 50)
  --step INT                 Walk-forward step size (default: 10)
  -p, --predictions INT      Predictions per iteration (default: 3)
  -o, --output FILE          Save results to JSON
```

### `info` - Lottery Information
```bash
python main.py info
```

### `update-data` - Update Lottery Data (v1.4)
```bash
python main.py update-data [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
  --all                  Update all lotteries
```

### `validate-data` - Validate Data & Detect Anomalies (v1.4)
```bash
python main.py validate-data [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
```

### `generate-workflow` - Generate GitHub Actions Workflow (v1.4)
```bash
python main.py generate-workflow
```

### `run-scheduler` - Run Scheduled Updater (v1.4)
```bash
python main.py run-scheduler [OPTIONS]

Options:
  --lotteries TEXT       Comma-separated lottery keys (default: powerball,mega_millions,florida_lotto)
  -i, --interval INTEGER Update interval in hours (default: 24)
```

### `portfolio` - Portfolio Optimizer (v1.5)
```bash
python main.py portfolio [OPTIONS]

Options:
  -l, --lottery TEXT       Lottery type (default: powerball)
  -b, --budget FLOAT       Budget in dollars (default: 20.0)
  --cost FLOAT             Cost per ticket (default: 2.0)
  -d, --diversity FLOAT    Diversity weight 0-1 (default: 0.5)
  -s, --strategy TEXT      Strategy: diverse, wheel (default: diverse)
  -n, --numbers TEXT       Comma-separated numbers for wheel strategy
  -o, --output FILE        Save portfolio to JSON file
```

### `syndicate` - Syndicate Manager (v1.5)
```bash
python main.py syndicate [OPTIONS]

Options:
  --name TEXT              Syndicate name (required)
  -l, --lottery TEXT       Lottery type (default: powerball)
  -b, --budget FLOAT       Total budget (default: 100.0)
  --cost FLOAT             Cost per ticket (default: 2.0)
  -m, --members TEXT       Members: name:email:share,name2:email2:share (required)
  -d, --draw-date TEXT     Draw date YYYY-MM-DD
  -o, --output FILE        Save syndicate to file
  -f, --format TEXT        Output format: json, csv (default: json)
```

### `notify` - Notification System (v1.5)
```bash
python main.py notify [OPTIONS]

Options:
  -l, --lottery TEXT       Lottery type (default: powerball)
  --method TEXT            Prediction method (default: ensemble)
  -c, --count INT          Number of predictions (default: 5)
  --email TEXT             Email recipient
  --discord TEXT           Discord webhook URL
  --pushbullet TEXT        Pushbullet token
  --smtp-server TEXT       SMTP server for email
  --smtp-port INT          SMTP port (default: 587)
  --smtp-user TEXT         SMTP username
  --smtp-pass TEXT         SMTP password
```

### `language` - Language Manager (v1.5)
```bash
python main.py language [OPTIONS]

Options:
  --lang TEXT              Language: en, es, fr, de, zh (default: en)
  --list                   List available languages
```

### `serve` - Start API Server (v2.0)
```bash
python main.py serve [OPTIONS]

Options:
  --host TEXT              Host to bind (default: 0.0.0.0)
  -p, --port INT           Port to bind (default: 8000)
  --reload                 Enable auto-reload
```

### `generate-deployment` - Generate Deployment Files (v2.0)
```bash
python main.py generate-deployment [OPTIONS]

Options:
  -t, --type TEXT          Type: dockerfile, docker-compose, k8s, all (default: all)
  -o, --output DIR         Output directory (default: .)
```

### `init-db` - Initialize Database (v2.0)
```bash
python main.py init-db [OPTIONS]

Options:
  --url TEXT               Database URL (default: sqlite:///lottery_predictor.db)
```

## Prediction Methods

### Statistical Methods
- **Frequency Weighted** - Favors historically frequent (hot) numbers
- **Cold Numbers** - Favors rarely drawn (due) numbers
- **Balanced** - Targets historical average sum with frequency weighting
- **Pair Based** - Seeds with frequent number pairs

### Machine Learning
- **Random Forest** - 200 trees, max depth 15
- **MLP Neural Network** - 3 hidden layers (256, 128, 64)
- **XGBoost** - 200 estimators, max depth 10
- **Ensemble** - Weighted voting from all models

### Deep Learning (Experimental)
- **TensorFlow/Keras** - Multi-layer perceptron with softmax output

## Project Structure

```
lottery-predictor/
├── main.py                 # Entry point
├── streamlit_app.py        # Streamlit web GUI
├── tkinter_app.py          # Tkinter desktop GUI
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── README.md              # This file
├── test_predictor.py      # Test suite
├── data/                  # Cached historical data
├── models/                # Trained ML models
├── visualizations/        # Generated charts
└── lottery_predictor/
    ├── __init__.py
    ├── data.py           # Data management & historical data
    ├── models.py         # Prediction models (statistical, ML, DL)
    ├── visualize.py      # Visualization generation
    ├── cli.py            # Command-line interface
    ├── universal.py      # Universal lottery configuration system
    ├── data_integration.py  # API integration, web scraping, validation (v1.4)
    ├── advanced.py       # Portfolio, syndicate, notifications, i18n (v1.5)
    └── cloud.py          # FastAPI, Docker, K8s, Database (v2.0)
```

## Example Output

```
🎯 Powerball Predictions
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ # ┃ Main Numbers                ┃ Bonus         ┃ Confidence ┃ Method       ┃
┃ 1 ┃  7   14   23   38   52      ┃  12           ┃ 78.5%      ┃ ensemble     ┃
┃ 2 ┃  3   19   28   41   61      ┃  24           ┃ 72.3%      ┃ ensemble     ┃
┃ 3 ┃  11  22  33   47   59      ┃  8            ┃ 69.1%      ┃ ensemble     ┃
┗━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛
```

## Visualizations

The tool generates several charts in the `visualizations/` directory:

- **Frequency Chart** - Bar chart of number frequencies
- **Hot/Cold Numbers** - Horizontal bars for hottest/coldest numbers
- **Sum Distribution** - Histogram with normal curve fit
- **Pair Heatmap** - Co-occurrence matrix of number pairs
- **Timeline Analysis** - Sum, even/odd, high/low trends over time
- **Prediction Analysis** - Analysis of generated predictions

## Data Sources

The tool generates realistic historical data based on actual lottery game rules and start dates:
- **Florida Lotto**: Since January 6, 1988
- **Powerball**: Since April 22, 1992
- **Mega Millions**: Since May 17, 2002
- ...and accurate start dates for all 48+ lotteries

For production use, you can implement real API fetching in `data.py` by adding official lottery API endpoints.

## Requirements

- Python 3.9+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- rich, typer
- streamlit, plotly (for web GUI)
- tensorflow (optional, for deep learning)

## Running the Test Suite

```bash
python test_predictor.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_predictor.py`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Responsible Gaming

If you or someone you know has a gambling problem, please contact:
- **National Problem Gambling Helpline**: 1-800-522-4700
- **Florida Council on Compulsive Gambling**: 1-888-ADMIT-IT

## 🗺️ Roadmap

### v1.1 - Universal Lottery Support ✅
- [x] Universal lottery configuration system
- [x] Support for any lottery worldwide (Mega Millions, EuroMillions, etc.)
- [x] All 50 US states + international games
- [x] Custom lottery creation via CLI and config file
- [x] Import/export lottery configurations

### v1.2 - Graphical Interface ✅
- [x] **Streamlit Web App** - Interactive web-based GUI
  - Real-time prediction generation
  - Interactive charts and visualizations
  - Lottery selection dropdown
  - Method comparison UI
  - History browser
- [x] **Desktop GUI** (Tkinter) - Native desktop application (`tkinter_app.py`)
  - Lottery selection with all 42+ games
  - Predict, Analyze, History, Compare, Visualize tabs
  - Threaded predictions & training (non-blocking UI)
  - Export to CSV/Excel/JSON
  - *Requires tkinter: `sudo apt-get install python3-tk` on Ubuntu/Debian*
- [ ] **Mobile-friendly** responsive design

### v1.3 - Enhanced AI/ML
- [x] Time-series forecasting (ARIMA, Prophet, ETS)
- [x] Transformer-based sequence models
- [x] Hyperparameter optimization (Optuna)
- [x] Model explainability (SHAP values)
- [x] Online learning / incremental training

### v1.4 - Data & Integration ✅
- [x] Real API integrations (official lottery APIs) - framework ready
- [x] Web scraping fallback for historical data - framework ready
- [x] Data validation & anomaly detection
- [x] Scheduled data updates (cron/GitHub Actions)
- [x] Export to CSV/Excel/JSON

### v1.5 - Advanced Features ✅
- [x] Backtesting framework with historical validation
- [x] Portfolio optimization (multiple tickets)
- [x] Syndicate/group play tools
- [x] Notification system (email, push, Discord)
- [x] Multi-language support

### v2.0 - Cloud & Scale ✅
- [x] REST API (FastAPI)
- [x] Docker containerization
- [x] Kubernetes deployment configs
- [x] Database backend (PostgreSQL/SQLite)
- [x] User accounts & prediction history
- [x] A/B testing for model versions

---

## 🤝 Contributing to Roadmap

Have ideas? Open an issue or discussion on GitHub! Priority is driven by community feedback.

---

**Remember**: The lottery is a game of chance. Play for fun, not as an investment strategy.