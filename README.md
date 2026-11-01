# 🎲 Lottery Predictor - AI-Powered Florida Lotto & Powerball

An intelligent lottery number prediction system using historical data analysis, statistical methods, and machine learning models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## ⚠️ Important Disclaimer

> **This tool is for entertainment and educational purposes only.** Lottery draws are random events. No algorithm, AI, or statistical analysis can predict future lottery numbers with certainty. Past performance does not guarantee future results. Please play responsibly and within your means.

## Features

- 📊 **Historical Data Analysis** - Comprehensive analysis of years of lottery draw data
- 🔥 **Hot/Cold Number Tracking** - Identify frequently and rarely drawn numbers
- 📈 **Statistical Predictions** - Frequency-based, balanced, and pair-based strategies
- 🤖 **Machine Learning Models** - Random Forest, Gradient Boosting, MLP, XGBoost ensemble
- 🧠 **Deep Learning** - Neural network predictions (experimental)
- 🎯 **Ensemble Consensus** - Combines multiple methods for best predictions
- 📊 **Rich Visualizations** - Frequency charts, heatmaps, timeline analysis
- 🖥️ **Beautiful CLI** - Rich terminal interface with tables and progress bars

## Supported Lotteries

| Lottery | Main Numbers | Bonus Numbers | Draw Days |
|---------|-------------|---------------|-----------|
| **Florida Lotto** | 6 (1-53) | None | Wed, Sat |
| **Powerball** | 5 (1-69) | 1 (1-26) | Mon, Wed, Sat |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lottery-predictor.git
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

# Generate predictions for Florida Lotto
python main.py predict --lottery florida_lotto

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
```

## Commands

### `predict` - Generate Predictions
```bash
python main.py predict [OPTIONS]

Options:
  -l, --lottery TEXT     Lottery type: florida_lotto or powerball (default: powerball)
  -m, --method TEXT      Method: statistical, ml, ensemble (default: ensemble)
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

### `info` - Lottery Information
```bash
python main.py info
```

## Prediction Methods

### Statistical Methods
- **Frequency Weighted** - Favors historically frequent (hot) numbers
- **Cold Numbers** - Favors rarely drawn (due) numbers
- **Balanced** - Targets historical average sum with frequency weighting
- **Pair Based** - Seeds with frequent number pairs

### Machine Learning
- **Random Forest** - 200 trees, max depth 15
- **Gradient Boosting** - 150 estimators, max depth 8
- **MLP Neural Network** - 3 hidden layers (256, 128, 64)
- **XGBoost** - 200 estimators, max depth 10
- **Ensemble** - Weighted voting from all models

### Deep Learning (Experimental)
- **TensorFlow/Keras** - Multi-layer perceptron with softmax output

## Project Structure

```
lottery-predictor/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── README.md              # This file
├── data/                  # Cached historical data
├── models/                # Trained ML models
├── visualizations/        # Generated charts
└── lottery_predictor/
    ├── __init__.py
    ├── data.py           # Data management & historical data
    ├── models.py         # Prediction models (statistical, ML, DL)
    ├── visualize.py      # Visualization generation
    └── cli.py            # Command-line interface
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

For production use, you can implement real API fetching in `data.py` by adding official lottery API endpoints.

## Requirements

- Python 3.9+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- rich, typer
- tensorflow (optional, for deep learning)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Responsible Gaming

If you or someone you know has a gambling problem, please contact:
- **National Problem Gambling Helpline**: 1-800-522-4700
- **Florida Council on Compulsive Gambling**: 1-888-ADMIT-IT

---

**Remember**: The lottery is a game of chance. Play for fun, not as an investment strategy.