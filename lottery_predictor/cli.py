"""
Main CLI Entry Point for Lottery Predictor
AI-powered universal lottery number prediction for all US states and international games.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from typing import Optional, List
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .data import LotteryDataManager, get_data_manager, LOTTERY_CONFIGS
from .models import (
    StatisticalPredictor,
    MLModelPredictor,
    EnsemblePredictor,
    TimeSeriesPredictor,
    TransformerPredictor,
    OnlineLearningPredictor,
    HyperparameterOptimizer,
    ModelExplainer,
    PredictionResult,
    get_predictor
)
from sklearn.preprocessing import StandardScaler
from .visualize import LotteryVisualizer, visualize_predictions
from .universal import get_universal_manager, list_lotteries

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

app = typer.Typer(
    name="lottery-predictor",
    help="🎲 AI-Powered Universal Lottery Number Predictor (All US States + International)",
    add_completion=False,
)
console = Console()

# Get all available lotteries from universal system
_universal_manager = get_universal_manager()
LOTTERY_TYPES = sorted([game.key for game in _universal_manager.list_games() if game.main_numbers > 0])


def print_header(title: str):
    """Print a styled header."""
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def print_predictions(predictions: List[PredictionResult], lottery_type: str):
    """Print predictions in a nice table."""
    config = LOTTERY_CONFIGS[lottery_type]

    table = Table(title=f"🎯 {config['name']} Predictions", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Main Numbers", style="green", width=30)
    table.add_column("Bonus", style="yellow", width=15)
    table.add_column("Confidence", style="blue", width=12)
    table.add_column("Method", style="cyan", width=20)

    for i, pred in enumerate(predictions, 1):
        main_str = "  ".join(f"[bold]{n:2d}[/bold]" for n in pred.main_numbers)
        bonus_str = "  ".join(f"[bold]{n:2d}[/bold]" for n in pred.bonus_numbers) if pred.bonus_numbers else "N/A"
        conf_str = f"{pred.confidence:.1%}"

        table.add_row(str(i), main_str, bonus_str, conf_str, pred.method)

    console.print(table)


def print_disclaimer():
    """Print important disclaimer."""
    console.print(Panel(
        "[bold red]⚠️  IMPORTANT DISCLAIMER[/bold red]\n\n"
        "This tool is for [bold]entertainment and educational purposes only[/bold]. "
        "Lottery draws are random events. No algorithm or AI can predict "
        "future lottery numbers with certainty. Past performance does not "
        "guarantee future results. Please play responsibly and within your means.",
        title="Disclaimer",
        border_style="red"
    ))


@app.command()
def predict(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type: florida_lotto or powerball"),
    method: str = typer.Option("ensemble", "--method", "-m", help="Prediction method: statistical, frequency, cold, pairs, ml, time_series, transformer, online_learning, ensemble"),
    count: int = typer.Option(5, "--count", "-c", help="Number of predictions to generate"),
    last_n: int = typer.Option(200, "--last-n", "-n", help="Number of historical draws to analyze"),
    train: bool = typer.Option(False, "--train", "-t", help="Retrain ML models before predicting"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Generate visualization charts"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save predictions to JSON file"),
):
    """Generate lottery number predictions using AI and statistical analysis."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'. Choose from: {', '.join(LOTTERY_TYPES)}[/red]")
        raise typer.Exit(1)

    print_header(f"🎲 {LOTTERY_CONFIGS[lottery]['name']} Predictor")
    print_disclaimer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading historical data...", total=None)
        dm = get_data_manager(lottery)
        df = dm.load_data()
        progress.update(task, description=f"Loaded {len(df)} historical draws")

        # Get predictor
        progress.update(task, description="Initializing predictor...")
        predictor = get_predictor(lottery, method)

        # Train if requested
        if train and hasattr(predictor, 'ml'):
            progress.update(task, description="Training ML models...")
            predictor.ml.train()

        # Generate predictions
        progress.update(task, description="Generating predictions...")

        if method == "statistical":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_balanced(last_n=last_n, num_predictions=count)
        elif method == "frequency":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_frequency_based(last_n=last_n, num_predictions=count)
        elif method == "cold":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_cold_numbers(last_n=last_n, num_predictions=count)
        elif method == "pairs":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_pair_based(last_n=last_n, num_predictions=count)
        elif method == "ml":
            pred = MLModelPredictor(lottery)
            if not pred.is_trained:
                pred.train()
            predictions = pred.predict(num_predictions=count)
        elif method == "time_series":
            pred = TimeSeriesPredictor(lottery)
            predictions = pred.predict(num_predictions=count, last_n=last_n)
        elif method == "transformer":
            pred = TransformerPredictor(lottery)
            if pred.model is None:
                progress.update(task, description="Training transformer model...")
                pred.train()
            predictions = pred.predict(num_predictions=count)
        elif method == "online_learning":
            pred = OnlineLearningPredictor(lottery)
            predictions = pred.predict(num_predictions=count)
        else:  # ensemble
            pred = EnsemblePredictor(lottery)
            predictions = pred.get_consensus_prediction(num_predictions=count)

        progress.update(task, description="Done!", completed=True)

    # Display results
    print_predictions(predictions, lottery)

    # Visualize if requested
    if visualize:
        console.print("\n[cyan]Generating visualizations...[/cyan]")
        viz = LotteryVisualizer(lottery)
        viz.generate_all_visualizations(last_n)
        visualize_predictions(predictions, lottery)
        console.print(f"[green]Visualizations saved to visualizations/[/green]")

    # Save to file if requested
    if output:
        data = {
            "lottery": lottery,
            "method": method,
            "predictions": [
                {
                    "main_numbers": p.main_numbers,
                    "bonus_numbers": p.bonus_numbers,
                    "confidence": p.confidence,
                    "method": p.method,
                    "metadata": p.metadata
                }
                for p in predictions
            ]
        }
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Predictions saved to {output}[/green]")


@app.command()
def analyze(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    last_n: int = typer.Option(200, "--last-n", "-n", help="Number of draws to analyze"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate charts"),
):
    """Analyze historical lottery data and show statistics."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    print_header(f"📊 {LOTTERY_CONFIGS[lottery]['name']} Analysis")

    dm = get_data_manager(lottery)
    df = dm.load_data()
    df = df.tail(last_n)

    # Basic stats
    console.print(f"\n[bold]Total Draws Analyzed:[/bold] {len(df)}")
    console.print(f"[bold]Date Range:[/bold] {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Frequency table
    freq = dm.get_number_frequency(last_n)
    hot, cold = dm.get_hot_cold_numbers(last_n)

    table = Table(title="🔥 Hot & Cold Numbers", show_header=True)
    table.add_column("Rank", width=6)
    table.add_column("Hot Numbers", style="red")
    table.add_column("Freq", width=6)
    table.add_column("Cold Numbers", style="blue")
    table.add_column("Freq", width=6)

    for i in range(10):
        table.add_row(
            str(i+1),
            str(hot[i]),
            str(freq[hot[i]]),
            str(cold[i]),
            str(freq[cold[i]])
        )
    console.print(table)

    # Sum statistics
    stats = dm.get_sum_statistics(last_n)
    console.print(f"\n[bold]Sum Statistics:[/bold]")
    console.print(f"  Mean: {stats['mean']:.1f} | Median: {stats['median']:.1f} | Std: {stats['std']:.1f}")
    console.print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")

    # Pair analysis
    pairs = dm.get_pair_frequency(last_n)
    top_pairs = list(pairs.items())[:10]

    pair_table = Table(title="🔗 Top 10 Number Pairs", show_header=True)
    pair_table.add_column("Pair", style="cyan")
    pair_table.add_column("Count", style="green")

    for (n1, n2), count in top_pairs:
        pair_table.add_row(f"{n1:2d} - {n2:2d}", str(count))
    console.print(pair_table)

    if visualize:
        console.print("\n[cyan]Generating visualizations...[/cyan]")
        viz = LotteryVisualizer(lottery)
        files = viz.generate_all_visualizations(last_n)
        for f in files:
            console.print(f"  [green]✓[/green] {f}")


@app.command()
def train(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    lookback: int = typer.Option(10, "--lookback", help="Number of previous draws to use as features"),
):
    """Train ML models on historical data."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    print_header(f"🤖 Training ML Models for {LOTTERY_CONFIGS[lottery]['name']}")

    predictor = MLModelPredictor(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training models...", total=None)
        results = predictor.train(lookback=lookback)
        progress.update(task, description="Training complete!")

    console.print("\n[bold]Training Results:[/bold]")
    table = Table(show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Train Accuracy", style="green")
    table.add_column("Test Accuracy", style="yellow")

    for model, metrics in results.items():
        table.add_row(model, f"{metrics['train_acc']:.4f}", f"{metrics['test_acc']:.4f}")

    console.print(table)
    console.print(f"\n[green]Models saved to models/{lottery}_models.joblib[/green]")


@app.command()
def history(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    count: int = typer.Option(10, "--count", "-c", help="Number of recent draws to show"),
):
    """Show recent lottery draw history."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    dm = get_data_manager(lottery)
    df = dm.load_data()
    recent = dm.get_recent_draws(count)

    print_header(f"📜 Recent {LOTTERY_CONFIGS[lottery]['name']} Draws")

    table = Table(show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("Main Numbers", style="green")
    table.add_column("Bonus", style="yellow")
    table.add_column("Jackpot", style="magenta")

    for _, row in recent.iterrows():
        main_str = "  ".join(f"{n:2d}" for n in row['main_numbers'])
        bonus_str = "  ".join(f"{n:2d}" for n in row['bonus_numbers']) if row['bonus_numbers'] else "N/A"
        jackpot_str = f"${row['jackpot']:,}"

        # Handle both string and datetime dates
        date_val = row['date']
        if isinstance(date_val, str):
            date_str = date_val
        else:
            date_str = date_val.strftime('%Y-%m-%d')

        table.add_row(
            date_str,
            main_str,
            bonus_str,
            jackpot_str
        )

    console.print(table)


@app.command()
def compare(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    methods: List[str] = typer.Option(["statistical", "ml", "ensemble"], "--methods", "-m", help="Methods to compare"),
    count: int = typer.Option(3, "--count", "-c", help="Predictions per method"),
):
    """Compare predictions from different methods."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    print_header(f"⚔️ Method Comparison - {LOTTERY_CONFIGS[lottery]['name']}")
    print_disclaimer()

    all_predictions = {}

    for method in methods:
        console.print(f"\n[cyan]Running {method}...[/cyan]")
        predictor = get_predictor(lottery, method)

        if method == "statistical":
            pred = StatisticalPredictor(lottery)
            preds = pred.predict_balanced(num_predictions=count)
        elif method == "frequency":
            pred = StatisticalPredictor(lottery)
            preds = pred.predict_frequency_based(num_predictions=count)
        elif method == "cold":
            pred = StatisticalPredictor(lottery)
            preds = pred.predict_cold_numbers(num_predictions=count)
        elif method == "pairs":
            pred = StatisticalPredictor(lottery)
            preds = pred.predict_pair_based(num_predictions=count)
        elif method == "ml":
            pred = MLModelPredictor(lottery)
            if not pred.is_trained:
                pred.train()
            preds = pred.predict(num_predictions=count)
        elif method == "time_series":
            pred = TimeSeriesPredictor(lottery)
            preds = pred.predict(num_predictions=count)
        elif method == "transformer":
            pred = TransformerPredictor(lottery)
            if pred.model is None:
                pred.train()
            preds = pred.predict(num_predictions=count)
        elif method == "online_learning":
            pred = OnlineLearningPredictor(lottery)
            preds = pred.predict(num_predictions=count)
        else:  # ensemble
            pred = EnsemblePredictor(lottery)
            preds = pred.get_consensus_prediction(num_predictions=count)

        all_predictions[method] = preds
        print_predictions(preds, lottery)

    # Summary comparison
    console.print("\n[bold]Method Comparison Summary:[/bold]")
    summary = Table(show_header=True)
    summary.add_column("Method", style="cyan")
    summary.add_column("Avg Confidence", style="green")
    summary.add_column("Unique Predictions", style="yellow")

    for method, preds in all_predictions.items():
        avg_conf = np.mean([p.confidence for p in preds])
        unique = len(set(tuple(p.main_numbers) for p in preds))
        summary.add_row(method, f"{avg_conf:.1%}", str(unique))

    console.print(summary)


@app.command()
def viz(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    last_n: int = typer.Option(200, "--last-n", "-n", help="Number of draws to visualize"),
    predictions_file: Optional[str] = typer.Option(None, "--predictions", "-p", help="JSON file with predictions to visualize"),
):
    """Generate visualization charts."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    print_header(f"📈 Generating Visualizations for {LOTTERY_CONFIGS[lottery]['name']}")

    viz = LotteryVisualizer(lottery)
    files = viz.generate_all_visualizations(last_n)

    if predictions_file and Path(predictions_file).exists():
        with open(predictions_file) as f:
            data = json.load(f)

        predictions = [
            PredictionResult(
                main_numbers=p['main_numbers'],
                bonus_numbers=p['bonus_numbers'],
                confidence=p['confidence'],
                method=p['method'],
                metadata=p.get('metadata', {})
            )
            for p in data['predictions']
        ]
        viz.plot_prediction_analysis(predictions)
        files.append(str(Path("visualizations") / f"{lottery}_prediction_analysis.png"))

    console.print("\n[green]Generated visualizations:[/green]")
    for f in files:
        console.print(f"  ✓ {f}")


@app.command()
def optimize(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    model: str = typer.Option("random_forest", "--model", help="Model to optimize: random_forest, xgboost, mlp"),
    trials: int = typer.Option(50, "--trials", "-t", help="Number of optimization trials"),
):
    """Optimize hyperparameters using Optuna (v1.3)."""
    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    if not HAS_OPTUNA:
        console.print("[red]Optuna not installed. Install with: pip install optuna[/red]")
        raise typer.Exit(1)

    print_header(f"🔧 Hyperparameter Optimization - {LOTTERY_CONFIGS[lottery]['name']}")

    optimizer = HyperparameterOptimizer(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Optimizing {model}...", total=None)

        if model == "random_forest":
            best_params = optimizer.optimize_random_forest(n_trials=trials)
        elif model == "xgboost":
            best_params = optimizer.optimize_xgboost(n_trials=trials)
        elif model == "mlp":
            best_params = optimizer.optimize_mlp(n_trials=trials)
        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Optimization complete!")

    console.print("\n[bold]Best Parameters:[/bold]")
    table = Table(show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for param, value in best_params.items():
        table.add_row(param, str(value))
    console.print(table)

    # Save best params
    import json
    output = f"{lottery}_{model}_best_params.json"
    with open(output, 'w') as f:
        json.dump(best_params, f, indent=2)
    console.print(f"\n[green]Best parameters saved to {output}[/green]")


@app.command()
def explain(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    model: str = typer.Option("random_forest", "--model", help="Model to explain: random_forest, mlp, xgboost"),
    samples: int = typer.Option(100, "--samples", "-s", help="Number of samples for explanation"),
):
    """Explain model predictions using SHAP (v1.3)."""
    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    if not HAS_SHAP:
        console.print("[red]SHAP not installed. Install with: pip install shap[/red]")
        raise typer.Exit(1)

    print_header(f"🔍 Model Explainability - {LOTTERY_CONFIGS[lottery]['name']}")

    explainer = ModelExplainer(lottery)
    dm = get_data_manager(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing data...", total=None)

        X, y = dm.prepare_ml_features(lookback=10)
        if len(X) < samples:
            samples = len(X)
        X_sample = X[-samples:]
        X_scaled = StandardScaler().fit_transform(X_sample)

        progress.update(task, description=f"Training {model}...")

        # Train model
        if model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            clf.fit(X_scaled, y[-samples:].argmax(axis=1))
        elif model == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42, early_stopping=True)
            clf.fit(X_scaled, y[-samples:].argmax(axis=1))
        elif model == "xgboost":
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0)
            clf.fit(X_scaled, y[-samples:].argmax(axis=1))
        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Generating SHAP explanations...")

        # Get SHAP explanations
        result = explainer.explain_any_model(clf, X_scaled, model_type='tree')

        progress.update(task, description="Done!", completed=True)

    if result:
        console.print("\n[bold]Top 20 Important Features:[/bold]")
        table = Table(show_header=True)
        table.add_column("Rank", width=6)
        table.add_column("Feature", style="cyan")
        table.add_column("SHAP Importance", style="green")

        for i, (feat, imp) in enumerate(result['top_features'], 1):
            table.add_row(str(i), feat, f"{imp:.6f}")
        console.print(table)

        console.print(f"\n[green]SHAP analysis complete for {model}[/green]")
    else:
        console.print("[yellow]Could not generate SHAP explanations[/yellow]")


@app.command()
def train_transformer(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Training epochs"),
    seq_len: int = typer.Option(10, "--seq-len", help="Sequence length"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
):
    """Train Transformer model (v1.3)."""
    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    if not HAS_TORCH:
        console.print("[red]PyTorch not installed. Install with: pip install torch[/red]")
        raise typer.Exit(1)

    print_header(f"🤖 Training Transformer - {LOTTERY_CONFIGS[lottery]['name']}")

    predictor = TransformerPredictor(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training transformer...", total=None)
        predictor.train(epochs=epochs, batch_size=batch_size, seq_len=seq_len)
        progress.update(task, description="Training complete!")

    console.print("[green]Transformer model trained![/green]")


@app.command()
def online_update(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
):
    """Update online learning models with latest data (v1.3)."""
    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    print_header(f"🔄 Online Learning Update - {LOTTERY_CONFIGS[lottery]['name']}")

    predictor = OnlineLearningPredictor(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Updating online models...", total=None)
        predictor.initialize()
        # Get recent draws as "new" data
        dm = get_data_manager(lottery)
        recent = dm.get_recent_draws(20)
        new_draws = recent.to_dict('records')
        predictor.update(new_draws)
        progress.update(task, description="Update complete!")

    console.print("[green]Online models updated![/green]")

    # Test prediction
    preds = predictor.predict(3)
    print_predictions(preds, lottery)


@app.command()
def export(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv, json, excel"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    last_n: int = typer.Option(None, "--last-n", "-n", help="Number of draws to export (default: all)"),
    include_predictions: bool = typer.Option(False, "--include-predictions", help="Include predictions in export"),
    method: str = typer.Option("ensemble", "--method", "-m", help="Prediction method if including predictions"),
    count: int = typer.Option(10, "--count", "-c", help="Number of predictions to include"),
):
    """Export historical data and/or predictions to CSV, JSON, or Excel."""

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'. Choose from: {', '.join(LOTTERY_TYPES)}[/red]")
        raise typer.Exit(1)

    if format not in ("csv", "json", "excel"):
        console.print(f"[red]Error: Unknown format '{format}'. Choose from: csv, json, excel[/red]")
        raise typer.Exit(1)

    if format == "excel" and not HAS_OPENPYXL:
        console.print("[red]Error: Excel export requires openpyxl. Install with: pip install openpyxl[/red]")
        raise typer.Exit(1)

    print_header(f"📤 Exporting {LOTTERY_CONFIGS[lottery]['name']} Data")

    dm = get_data_manager(lottery)
    df = dm.load_data()

    if last_n:
        df = df.tail(last_n)

    config = LOTTERY_CONFIGS[lottery]

    # Prepare data for export
    export_data = []
    for _, row in df.iterrows():
        record = {
            'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
            'draw_number': row['draw_number'],
        }
        for i, num in enumerate(row['main_numbers']):
            record[f'main_{i+1}'] = num
        for i, num in enumerate(row['bonus_numbers']):
            record[f'bonus_{i+1}'] = num
        record['jackpot'] = row['jackpot']
        export_data.append(record)

    # Add predictions if requested
    if include_predictions:
        console.print(f"[cyan]Generating {count} predictions using {method} method...[/cyan]")
        predictor = get_predictor(lottery, method)
        if method == "statistical":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_balanced(last_n=last_n or 200, num_predictions=count)
        elif method == "frequency":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_frequency_based(num_predictions=count)
        elif method == "cold":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_cold_numbers(num_predictions=count)
        elif method == "pairs":
            pred = StatisticalPredictor(lottery)
            predictions = pred.predict_pair_based(num_predictions=count)
        elif method == "ml":
            pred = MLModelPredictor(lottery)
            if not pred.is_trained:
                pred.train()
            predictions = pred.predict(num_predictions=count)
        elif method == "time_series":
            pred = TimeSeriesPredictor(lottery)
            predictions = pred.predict(num_predictions=count, last_n=last_n or 200)
        elif method == "transformer":
            pred = TransformerPredictor(lottery)
            if pred.model is None:
                pred.train()
            predictions = pred.predict(num_predictions=count)
        elif method == "online_learning":
            pred = OnlineLearningPredictor(lottery)
            predictions = pred.predict(num_predictions=count)
        else:  # ensemble
            pred = EnsemblePredictor(lottery)
            predictions = pred.get_consensus_prediction(num_predictions=count)

        pred_data = []
        for i, p in enumerate(predictions, 1):
            pred_record = {
                'prediction_number': i,
                'method': p.method,
                'confidence': f"{p.confidence:.1%}",
            }
            for j, num in enumerate(p.main_numbers):
                pred_record[f'predicted_main_{j+1}'] = num
            for j, num in enumerate(p.bonus_numbers):
                pred_record[f'predicted_bonus_{j+1}'] = num
            pred_data.append(pred_record)

    # Determine output filename
    if not output:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        suffix = "_with_predictions" if include_predictions else ""
        ext = {"csv": "csv", "json": "json", "excel": "xlsx"}[format]
        output = f"{lottery}_export{suffix}_{timestamp}.{ext}"

    # Export
    if format == "csv":
        pd.DataFrame(export_data).to_csv(output, index=False)
        if include_predictions:
            pred_df = pd.DataFrame(pred_data)
            pred_output = output.replace('.csv', '_predictions.csv')
            pred_df.to_csv(pred_output, index=False)
            console.print(f"[green]Predictions exported to {pred_output}[/green]")
    elif format == "json":
        data = {
            'lottery': lottery,
            'lottery_name': config['name'],
            'exported_draws': len(export_data),
            'draws': export_data,
        }
        if include_predictions:
            data['predictions'] = pred_data
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == "excel":
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(export_data).to_excel(writer, sheet_name='Historical Data', index=False)
            if include_predictions:
                pd.DataFrame(pred_data).to_excel(writer, sheet_name='Predictions', index=False)

    console.print(f"[green]✓ Exported {len(export_data)} draws to {output}[/green]")
    if include_predictions:
        console.print(f"[green]✓ Included {count} predictions[/green]")


@app.command()
def info():
    """Show information about supported lotteries."""
    print_header("🎲 Supported Lotteries")

    for key, config in LOTTERY_CONFIGS.items():
        table = Table(title=config['name'], show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Main Numbers", f"{config['main_numbers']} (1-{config['main_max']})")
        if config['bonus_numbers'] > 0:
            table.add_row("Bonus Numbers", f"{config['bonus_numbers']} (1-{config['bonus_max']})")
        table.add_row("Draw Days", ", ".join(config['draw_days']))
        table.add_row("First Draw", config['start_date'])

        console.print(table)
        console.print()


if __name__ == "__main__":
    app()