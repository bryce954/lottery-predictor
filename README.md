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
| The Pick (AZ) | `arizona_the_pick` | 6 (1-44) | None | Mon, Wed, Sat |
| Natural State Jackpot (AR) | `arkansas_natural_state` | 5 (1-39) | None | Mon, Wed, Fri |
| SuperLotto Plus (CA) | `california_superlotto` | 5 (1-47) | 1 (1-27) | Wed, Sat |
| Colorado Lotto+ | `colorado_lotto` | 6 (1-40) | None | Wed, Sat |
| Lotto! (CT) | `connecticut_lotto` | 6 (1-44) | None | Tue, Fri |
| Fantasy 5 (GA) | `georgia_fantasy5` | 5 (1-42) | None | Daily |
| Weekly Grand (ID) | `idaho_weekly_grand` | 5 (1-40) | None | Mon, Thu |
| Illinois Lotto | `illinois_lotto` | 6 (1-52) | None | Mon, Thu, Sat |
| Hoosier Lotto (IN) | `indiana_hoosier_lotto` | 6 (1-46) | None | Wed, Sat |
| Iowa Lotto | `iowa_lotto` | 6 (1-44) | None | Wed, Sat |
| Super Kansas Cash | `kansas_super_kansas_cash` | 5 (1-32) | 1 (1-25) | Mon, Wed, Sat |
| Kentucky Lotto | `kentucky_lotto` | 6 (1-43) | None | Wed, Sat |
| Louisiana Lotto | `louisiana_lotto` | 6 (1-42) | None | Wed, Sat |
| Mass Megabucks (MA) | `massachusetts_megabucks` | 6 (1-44) | None | Wed, Sat |
| Multi-Match (MD) | `maryland_multi_match` | 6 (1-43) | None | Mon, Thu |
| Megabucks (MI) | `michigan_lotto47` | 6 (1-47) | None | Wed, Sat |
| Gopher 5 (MN) | `minnesota_gopher5` | 5 (1-47) | None | Mon, Wed, Fri |
| Missouri Lotto | `missouri_lotto` | 6 (1-44) | None | Wed, Sat |
| Big Sky Bonus (MT) | `montana_big_sky` | 5 (1-28) | 1 (1-17) | Mon, Wed, Sat |
| Pick 5 (NE) | `nebraska_pick5` | 5 (1-38) | None | Daily |
| Roadrunner Cash (NM) | `new_mexico_roadrunner` | 5 (1-37) | None | Daily |
| Pick-6 (NJ) | `new_jersey_pick6` | 6 (1-46) | None | Mon, Thu |
| New York Lotto | `new_york_lotto` | 6 (1-59) | None | Wed, Sat |
| Classic Lotto (OH) | `ohio_classic_lotto` | 6 (1-49) | None | Mon, Wed, Sat |
| Lotto (OK) | `oklahoma_lotto` | 6 (1-44) | None | Wed, Sat |
| Megabucks (OR) | `oregon_megabucks` | 6 (1-48) | None | Mon, Thu |
| Match 6 (PA) | `pennsylvania_match6` | 6 (1-49) | None | Mon, Wed, Fri |
| Tennessee Lotto | `tennessee_lotto` | 6 (1-42) | None | Wed, Sat |
| Lotto Texas | `texas_lotto` | 6 (1-54) | None | Mon, Wed, Sat |
| Virginia Lotto | `virginia_lotto` | 6 (1-38) | None | Wed, Sat |
| Lotto (WA) | `washington_lotto` | 6 (1-49) | None | Mon, Wed, Sat |
| Megabucks (WI) | `wisconsin_megabucks` | 6 (1-49) | None | Wed, Sat |

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

# v1.3 Enhanced ML commands
python main.py optimize --lottery powerball
python main.py explain --lottery powerball
python main.py train-transformer --lottery powerball
python main.py online-update --lottery powerball

# v1.5 Advanced Features commands
python main.py portfolio --lottery powerball --budget 20
python main.py syndicate --name "My Group" --members "Alice:alice@email.com:0.5,Bob:bob@email.com:0.5"
python main.py notify --lottery powerball --email user@example.com
python main.py language --lang es

# v2.0 Cloud & Scale commands
python main.py serve --host 0.0.0.0 --port 8000
python main.py generate-deployment --type all
python main.py init-db

# NEW: Location auto-detection
python main.py predict --detect-state
python main.py predict --lottery auto
python main.py analyze --detect-state

# NEW: Full history analysis
python main.py predict --all-history
python main.py analyze --all-history
python main.py viz --all-history

# Launch web GUI
streamlit run streamlit_app.py

# Launch desktop GUI (requires tkinter: sudo apt-get install python3-tk on Ubuntu/Debian)
python tkinter_app.py
```

## 📋 Version / Interface Comparison Table

| Feature | CLI (`main.py`) | Streamlit Web GUI | Tkinter Desktop GUI | FastAPI REST API | Docker/K8s |
|---------|----------------|-------------------|---------------------|------------------|------------|
| **Installation** | `pip install -e .` | Same + `streamlit` | Same + `python3-tk` | Same + `fastapi`, `uvicorn` | Docker engine |
| **Run Command** | `python main.py predict` | `streamlit run streamlit_app.py` | `python tkinter_app.py` | `python main.py serve` | `docker compose up` |
| **Access** | Terminal | Browser (localhost:8501) | Native window | Browser (localhost:8000/docs) | Any container runtime |
| **Mobile Friendly** | ✅ SSH/Termux | ✅ Responsive web | ❌ Desktop only | ✅ REST clients | ✅ Cloud deploy |
| **Predictions** | ✅ All methods | ✅ All methods | ✅ All methods | ✅ All methods | ✅ All methods |
| **Analysis/Charts** | ✅ Static (matplotlib) | ✅ Interactive (Plotly) | ✅ Static (matplotlib) | ✅ JSON data | ✅ All |
| **Training Models** | ✅ Yes | ✅ Yes | ✅ Yes (threaded) | ✅ Yes | ✅ Yes |
| **Backtesting** | ✅ `backtest.py` | ✅ Tab | ✅ Tab | ❌ Not yet | ❌ Not yet |
| **Data Export** | ✅ CSV/JSON/Excel | ✅ CSV/JSON/Excel | ✅ CSV/JSON/Excel | ✅ JSON | ✅ All |
| **Data Updates (v1.4)** | ✅ Commands | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ Cron/GitHub Actions |
| **Portfolio/Syndicate (v1.5)** | ✅ Commands | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ All |
| **Notifications (v1.5)** | ✅ Commands | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ All |
| **Multi-language (v1.5)** | ✅ Commands | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ All |
| **User Accounts (v2.0)** | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ Database | ✅ PostgreSQL |
| **A/B Testing (v2.0)** | ❌ Not yet | ❌ Not yet | ❌ Not yet | ✅ Database | ✅ All |
| **Scheduled Updates** | ✅ `run-scheduler` | ❌ Not yet | ❌ Not yet | ✅ Background tasks | ✅ Cron/K8s CronJob |
| **Location Detection** | ✅ `--detect-state` | ✅ Button in sidebar | ✅ Button in settings | ❌ Not yet | ❌ Not yet |
| **Production Ready** | Development | Development | Development | ✅ With auth/DB | ✅ Production grade |

**Quick Recommendation:**
- **Developers/Automation** → CLI or REST API
- **Interactive Exploration** → Streamlit Web GUI (also works on mobile browser)
- **Desktop Users** → Tkinter Desktop GUI
- **Production/Team/Cloud** → Docker/Kubernetes with FastAPI

## Commands

### `predict` - Generate Predictions
```bash
python main.py predict [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
                         Use 'info' command to see all 48+ options
                         Use 'auto' or --detect-state for location-based detection
  -m, --method TEXT      Method: statistical, frequency, cold, pairs, ml, time_series, ensemble (default: ensemble)
  -c, --count INTEGER    Number of predictions (default: 5)
  -n, --last-n INTEGER   Historical draws to analyze (default: 200, ignored if --all-history)
  -t, --train            Retrain ML models before predicting
  -v, --visualize        Generate visualization charts
  -o, --output FILE      Save predictions to JSON file
      --all-history      Use all historical data from lottery start date
      --detect-state     Auto-detect lottery from your location (IP geolocation)
```

### `analyze` - Historical Analysis
```bash
python main.py analyze [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type (default: powerball)
                         Use 'auto' or --detect-state for location-based detection
  -n, --last-n INTEGER   Draws to analyze (default: 200, ignored if --all-history)
  --visualize/--no-visualize  Generate charts (default: True)
      --all-history      Use all historical data from lottery start date
      --detect-state     Auto-detect lottery from your location (IP geolocation)
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
  -n, --last-n INTEGER       Draws to visualize (default: 200, ignored if --all-history)
  -p, --predictions FILE     JSON file with predictions to visualize
      --all-history          Use all historical data from lottery start date
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

**Note:** This generates `.github/workflows/update_lottery_data.yml` locally. You must manually copy it to your GitHub repository (Settings → Actions → General → Allow all actions) due to GitHub OAuth scope limitations.

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

### Core (Required)
- Python 3.9+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- rich, typer

### Web GUI (Streamlit)
- streamlit>=1.28.0
- plotly>=5.17.0

### Desktop GUI (Tkinter)
- tkinter (system package: `sudo apt-get install python3-tk` on Ubuntu/Debian)

### Data Export
- openpyxl>=3.1.0

### v1.3 Enhanced ML (Optional)
- optuna>=3.4.0 (hyperparameter optimization)
- shap>=0.42.0 (model explainability)
- torch>=2.0.0 (Transformer models)

### v1.4 Data Integration
- aiohttp>=3.9.0 (async API clients)
- beautifulsoup4>=4.12.0 (web scraping)
- lxml>=4.9.0 (HTML parsing)
- schedule>=1.2.0 (scheduled tasks)
- brotli>=1.1.0 (compression)

### v1.5 Advanced Features
- (uses core + optional v1.3 deps)

### v2.0 Cloud & Scale
- fastapi>=0.104.0
- uvicorn>=0.24.0
- sqlalchemy>=2.0.0 (database)
- redis>=5.0.0 (caching)

### Development
- pytest>=7.4.0

### Install All at Once
```bash
pip install -r requirements.txt
```

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
  - **Works on mobile browsers** (responsive design)
- [x] **Desktop GUI** (Tkinter) - Native desktop application (`tkinter_app.py`)
  - Lottery selection with all 42+ games
  - Predict, Analyze, History, Compare, Visualize tabs
  - Threaded predictions & training (non-blocking UI)
  - Export to CSV/JSON/Excel
  - *Requires tkinter: `sudo apt-get install python3-tk` on Ubuntu/Debian*
- [ ] **Mobile APK** - Not available (Python project, not native mobile)
  - **Use Streamlit Web GUI on mobile browser** at `http://your-ip:8501`

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

### v1.6 - UX Enhancements ✅
- [x] **Location Auto-Detection** - IP geolocation to auto-select state lottery
  - CLI: `--detect-state` / `--auto` / `--lottery auto`
  - Streamlit: "📍 Detect My Location" button in sidebar
  - Tkinter: "📍 Detect My Location" button in settings panel
  - Maps 50 US states to their specific lottery games + multi-state games
- [x] **Full History Analysis** - Analyze all draws from lottery start date
  - CLI: `--all-history` / `--from-start` flag for predict, analyze, viz commands
  - Uses all available historical data (e.g., 5400+ draws for Powerball since 1992)
  - Shows date range of analyzed data
- [x] **Alphabetical State Sorting** - US state lotteries sorted A-Z in docs
- [x] **Official Website Links** - Direct links to lottery official sites in all interfaces

### v1.7 - Validation & Analytics 📋
- [ ] **Results Verification** - Auto-fetch actual draw results & compare against past predictions
  - Scheduled job to fetch winning numbers after each draw
  - Accuracy tracking per method (statistical, ML, ensemble, etc.)
  - Hit rate reports: matches per tier (jackpot, 2nd prize, 3rd prize, etc.)
- [ ] **Odds Calculator** - Exact odds, expected value, prize tiers for each game
  - Show probability for each prize tier
  - Expected value calculation (EV = Σ prize × probability)
  - Break-even analysis
- [ ] **Pattern Analysis** - Advanced statistical pattern detection
  - Consecutive numbers frequency
  - Odd/even split distribution
  - High/low number distribution
  - Decade analysis (1-10, 11-20, etc.)
  - Sum distribution with normal curve fit
- [ ] **Statistical Significance Testing** - Compare methods vs random baseline
  - Monte Carlo simulation for p-values
  - Confidence intervals for prediction accuracy

### v1.8 - Mobile & PWA 📱
- [ ] **Progressive Web App (PWA)** - Install Streamlit as native mobile app
  - Service worker for offline capability
  - App manifest for home screen installation
  - Push notifications for draw reminders
- [ ] **Mobile-Optimized UI** - Touch-friendly responsive design
  - Swipe gestures for tabs
  - Larger touch targets
  - Bottom navigation bar
- [ ] **Quick Pick Generator** - True random number generator for manual entry
  - Hardware entropy source (where available)
  - Multiple sets generation
  - Export to wallet/passbook

### v2.0 - Cloud & Scale ✅
- [x] REST API (FastAPI)
- [x] Docker containerization
- [x] Kubernetes deployment configs
- [x] Database backend (PostgreSQL/SQLite)
- [x] User accounts & prediction history
- [x] A/B testing for model versions

### v2.1 - Real Data & Production 🔌
- [ ] **Real API Integrations** - Replace simulated data with official sources
  - State lottery official APIs (where available)
  - Third-party aggregators (LotteryPost, etc.)
  - Web scraping as fallback with rate limiting
- [ ] **Historical Results Archive** - Browse past winning numbers
  - Search by date range, lottery, number pattern
  - Export results to CSV/Excel
- [ ] **Draw Calendar** - Upcoming draw dates/times with countdown
  - Timezone-aware scheduling
  - `.ics` export for calendar apps
  - Next draw countdown timer

### v2.2 - User Experience & Social 👥
- [ ] **User Profiles & Preferences** - Save favorites, settings, notification config
  - Favorite lotteries & methods
  - Default prediction count & history depth
  - Notification channels (email, Discord, Pushbullet)
- [ ] **Syndicate Web UI** - Manage groups, shares, payouts in Streamlit/Tkinter
  - Create/join syndicates
  - Track contributions & distributions
  - Automated payout calculations
- [ ] **Social Features** - Share predictions, community pools, leaderboards
  - Anonymous prediction sharing
  - Accuracy leaderboards
  - Community prediction pools

### v3.0 - Intelligence & Automation 🤖
- [ ] **Model Performance Dashboard** - Track accuracy over time, detect drift
  - Per-method performance charts
  - Model retraining triggers
  - A/B testing results visualization
- [ ] **Budget Tracker** - Track spending vs. winnings over time
  - Per-lottery spending reports
  - ROI calculations
  - Tax estimator (federal/state)
- [ ] **Multi-Draw Predictions** - Predict for multiple upcoming draws
  - Rolling prediction windows
  - Confidence decay modeling
- [ ] **Voice Assistant Integration** - "Hey Siri/Google, give me Powerball numbers"
  - Shortcuts/Intent support
  - Read aloud predictions

### v3.1 - Enhanced History & Analytics 📊
- [ ] **Per-Lottery Full History Toggle** - Option to use all historical data for each lottery
  - Already implemented via `--all-history` flag (works for all 48+ lotteries)
  - Add UI toggles in Streamlit & Tkinter for easy access
  - Show data quality metrics (completeness, date range) per lottery
- [ ] **Historical Data Health Dashboard** - Monitor data freshness & gaps
  - Auto-detect missing draws
  - Alert on data anomalies
  - Backfill scheduling
- [ ] **Cross-Lottery Historical Correlation** - Find patterns across similar games
  - Compare Powerball vs Mega Millions trends
  - State vs multi-state correlation analysis

---

## 🤝 Contributing to Roadmap

Have ideas? Open an issue or discussion on GitHub! Priority is driven by community feedback.

---

**Remember**: The lottery is a game of chance. Play for fun, not as an investment strategy.