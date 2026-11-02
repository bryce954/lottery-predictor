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
import asyncio
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
from .data_integration import (
    DataIntegrator, DataValidator, ScheduledUpdater,
    update_lottery_data, update_all_lotteries,
    save_github_workflow, run_scheduled_updater
)
from .advanced import (
    PortfolioOptimizer, SyndicateManager, Syndicate, SyndicateMember,
    NotificationSystem, I18nManager
)
from .cloud import (
    create_app, generate_dockerfile, generate_docker_compose, generate_kubernetes
)

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


@app.command()
def update_data(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type to update"),
    all_lotteries: bool = typer.Option(False, "--all", help="Update all lotteries"),
):
    """Update lottery data from real APIs and web scraping (v1.4)."""

    print_header("🔄 Updating Lottery Data (v1.4)")
    print_disclaimer()

    if all_lotteries:
        console.print("[cyan]Updating all lotteries...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching data from all sources...", total=None)
            results = asyncio.run(update_all_lotteries())
            progress.update(task, description="Done!", completed=True)

        for key, result in results.items():
            console.print(f"\n[bold]{LOTTERY_CONFIGS[key]['name']}:[/bold]")
            console.print(f"  Fetched: {result['total_fetched']} | Valid: {result['valid']} | Saved: {result['saved']}")
            if result['invalid'] > 0:
                console.print(f"  [red]Invalid: {result['invalid']}[/red]")
            if result['anomalies']:
                console.print(f"  [yellow]Anomalies: {len(result['anomalies'])}[/yellow]")
                for a in result['anomalies'][:3]:
                    console.print(f"    - {a['type']}: {a['details']}")
    else:
        if lottery not in LOTTERY_TYPES:
            console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Updating {LOTTERY_CONFIGS[lottery]['name']}...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching data from sources...", total=None)
            result = asyncio.run(update_lottery_data(lottery))
            progress.update(task, description="Done!", completed=True)

        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Total Fetched: {result['total_fetched']}")
        console.print(f"  Valid: {result['valid']}")
        console.print(f"  Saved: {result['saved']}")
        console.print(f"  Invalid: {result['invalid']}")

        if result['errors']:
            console.print(f"\n[yellow]Errors (first 10):[/yellow]")
            for err in result['errors'][:10]:
                console.print(f"  - {err}")

        if result['anomalies']:
            console.print(f"\n[yellow]Anomalies Detected:[/yellow]")
            for a in result['anomalies']:
                console.print(f"  - {a['type']}: {a['details']}")


@app.command()
def validate_data(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type to validate"),
):
    """Validate lottery data and detect anomalies (v1.4)."""

    print_header("🔍 Data Validation & Anomaly Detection (v1.4)")

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    dm = get_data_manager(lottery)
    df = dm.load_data()

    if len(df) == 0:
        console.print("[red]No data to validate[/red]")
        raise typer.Exit(1)

    validator = DataValidator(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Validating data...", total=None)
        anomalies = validator.detect_anomalies(df)
        progress.update(task, description="Done!", completed=True)

    console.print(f"\n[bold]Dataset:[/bold] {len(df)} draws from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    if anomalies:
        console.print(f"\n[yellow]Anomalies Found ({len(anomalies)}):[/yellow]")
        table = Table(show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="red")
        table.add_column("Details", style="yellow")
        for a in anomalies:
            table.add_row(a['type'], str(a['count']), a['details'])
        console.print(table)
    else:
        console.print("\n[green]✓ No anomalies detected[/green]")

    # Also validate each draw
    invalid_count = 0
    for _, row in df.iterrows():
        draw = row.to_dict()
        draw['date'] = draw['date'].strftime('%Y-%m-%d') if hasattr(draw['date'], 'strftime') else str(draw['date'])
        is_valid, errors = validator.validate_draw(draw)
        if not is_valid:
            invalid_count += 1
            if invalid_count <= 5:
                console.print(f"[red]Invalid draw {draw.get('draw_number', '?')}: {errors}[/red]")

    if invalid_count > 0:
        console.print(f"\n[red]Total invalid draws: {invalid_count}[/red]")
    else:
        console.print("\n[green]✓ All draws pass validation[/green]")


@app.command()
def generate_workflow():
    """Generate GitHub Actions workflow for scheduled data updates (v1.4)."""
    print_header("⚙️ Generating GitHub Actions Workflow (v1.4)")

    path = save_github_workflow()
    console.print(f"\n[green]✓ Workflow saved to {path}[/green]")
    console.print("\nThis workflow will:")
    console.print("  - Run daily at 6 AM UTC")
    console.print("  - Update data for major lotteries")
    console.print("  - Commit and push changes if data changed")
    console.print("\nTo enable: Push to GitHub and the workflow will run automatically.")


@app.command()
def run_scheduler(
    lotteries: str = typer.Option("powerball,mega_millions,florida_lotto", "--lotteries", help="Comma-separated lottery keys"),
    interval: int = typer.Option(24, "--interval", "-i", help="Update interval in hours"),
):
    """Run scheduled data updater (blocking, v1.4)."""
    print_header("⏰ Scheduled Data Updater (v1.4)")

    keys = [k.strip() for k in lotteries.split(',')]
    for k in keys:
        if k not in LOTTERY_TYPES:
            console.print(f"[red]Error: Unknown lottery type '{k}'[/red]")
            raise typer.Exit(1)

    console.print(f"[cyan]Scheduling updates for: {', '.join(keys)} every {interval} hours[/cyan]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")

    run_scheduled_updater(keys, interval)


@app.command()
def portfolio(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    budget: float = typer.Option(20.0, "--budget", "-b", help="Budget in dollars"),
    ticket_cost: float = typer.Option(2.0, "--cost", help="Cost per ticket"),
    diversity: float = typer.Option(0.5, "--diversity", "-d", help="Diversity weight (0-1)"),
    strategy: str = typer.Option("diverse", "--strategy", "-s", help="Strategy: diverse, wheel"),
    numbers: str = typer.Option(None, "--numbers", "-n", help="Comma-separated numbers for wheel strategy"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save portfolio to JSON file"),
):
    """Generate optimized ticket portfolio (v1.5)."""
    print_header(f"💼 Portfolio Optimizer - {LOTTERY_CONFIGS[lottery]['name']} (v1.5)")
    print_disclaimer()

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    optimizer = PortfolioOptimizer(lottery)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating portfolio...", total=None)

        if strategy == "wheel":
            if not numbers:
                console.print("[red]Error: --numbers required for wheel strategy[/red]")
                raise typer.Exit(1)
            num_list = [int(n.strip()) for n in numbers.split(',')]
            portfolio = optimizer.optimize_wheel(num_list, ticket_cost)
        else:
            portfolio = optimizer.generate_diverse_portfolio(
                budget=budget,
                ticket_cost=ticket_cost,
                diversity_weight=diversity
            )

        progress.update(task, description="Done!", completed=True)

    # Display portfolio
    console.print(f"\n[bold]Portfolio Summary:[/bold]")
    console.print(f"  Strategy: {portfolio.strategy}")
    console.print(f"  Tickets: {len(portfolio.tickets)}")
    console.print(f"  Total Cost: ${portfolio.total_cost:.2f}")
    console.print(f"  Budget: ${portfolio.budget:.2f}")

    table = Table(title="🎫 Tickets", show_header=True)
    table.add_column("#", width=4)
    table.add_column("Main Numbers", style="green")
    table.add_column("Bonus", style="yellow")
    table.add_column("Method", style="cyan")
    table.add_column("Confidence", style="blue")

    for i, ticket in enumerate(portfolio.tickets, 1):
        main_str = "  ".join(f"{n:2d}" for n in ticket.main_numbers)
        bonus_str = "  ".join(f"{n:2d}" for n in ticket.bonus_numbers) if ticket.bonus_numbers else "N/A"
        table.add_row(str(i), main_str, bonus_str, ticket.method, f"{ticket.confidence:.1%}")

    console.print(table)

    if output:
        data = {
            "lottery": portfolio.lottery_type,
            "strategy": portfolio.strategy,
            "budget": portfolio.budget,
            "total_cost": portfolio.total_cost,
            "tickets": [
                {
                    "main_numbers": t.main_numbers,
                    "bonus_numbers": t.bonus_numbers,
                    "method": t.method,
                    "confidence": t.confidence,
                    "cost": t.cost
                }
                for t in portfolio.tickets
            ]
        }
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        console.print(f"\n[green]Portfolio saved to {output}[/green]")


@app.command()
def syndicate(
    name: str = typer.Option(..., "--name", help="Syndicate name"),
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    budget: float = typer.Option(100.0, "--budget", "-b", help="Total budget"),
    ticket_cost: float = typer.Option(2.0, "--cost", help="Cost per ticket"),
    members: str = typer.Option(..., "--members", "-m", help="Members: name:email:share,name2:email2:share"),
    draw_date: Optional[str] = typer.Option(None, "--draw-date", "-d", help="Draw date (YYYY-MM-DD)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save syndicate to JSON file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv"),
):
    """Create and manage lottery syndicate (v1.5)."""
    print_header(f"👥 Syndicate Manager - {name} (v1.5)")
    print_disclaimer()

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    # Parse members
    member_list = []
    for member_str in members.split(','):
        parts = member_str.strip().split(':')
        if len(parts) >= 2:
            member_name = parts[0]
            email = parts[1] if parts[1] else None
            share = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        else:
            member_name = parts[0]
            email = None
            share = 0.0

        member_list.append(SyndicateMember(
            name=member_name,
            email=email,
            share_percentage=share
        ))

    manager = SyndicateManager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating syndicate...", total=None)
        syndicate = manager.create_syndicate(name, lottery, member_list, budget, ticket_cost)
        if draw_date:
            syndicate.draw_date = draw_date
        progress.update(task, description="Done!", completed=True)

    console.print(f"\n[bold]Syndicate Created:[/bold]")
    console.print(f"  Name: {syndicate.name}")
    console.print(f"  Lottery: {LOTTERY_CONFIGS[lottery]['name']}")
    console.print(f"  Total Budget: ${syndicate.total_cost:.2f}")
    console.print(f"  Tickets: {len(syndicate.tickets)}")

    if draw_date:
        console.print(f"  Draw Date: {draw_date}")

    # Members table
    member_table = Table(title="👥 Members", show_header=True)
    member_table.add_column("Name", style="cyan")
    member_table.add_column("Email", style="green")
    member_table.add_column("Share %", style="yellow")
    member_table.add_column("Contribution", style="blue")

    for m in syndicate.members:
        member_table.add_row(
            m.name,
            m.email or "N/A",
            f"{m.share_percentage:.1f}%",
            f"${m.contribution:.2f}"
        )
    console.print(member_table)

    # Tickets table
    ticket_table = Table(title="🎫 Tickets", show_header=True)
    ticket_table.add_column("#", width=4)
    ticket_table.add_column("Main Numbers", style="green")
    ticket_table.add_column("Bonus", style="yellow")
    ticket_table.add_column("Method", style="cyan")

    for i, ticket in enumerate(syndicate.tickets, 1):
        main_str = "  ".join(f"{n:2d}" for n in ticket.main_numbers)
        bonus_str = "  ".join(f"{n:2d}" for n in ticket.bonus_numbers) if ticket.bonus_numbers else "N/A"
        ticket_table.add_row(str(i), main_str, bonus_str, ticket.method)

    console.print(ticket_table)

    if output:
        content = manager.export_syndicate(name, format)
        with open(output, 'w') as f:
            f.write(content)
        console.print(f"\n[green]Syndicate saved to {output}[/green]")


@app.command()
def notify(
    lottery: str = typer.Option("powerball", "--lottery", "-l", help="Lottery type"),
    method: str = typer.Option("ensemble", "--method", help="Prediction method"),
    count: int = typer.Option(5, "--count", "-c", help="Number of predictions"),
    email: Optional[str] = typer.Option(None, "--email", help="Email recipient"),
    discord: Optional[str] = typer.Option(None, "--discord", help="Discord webhook URL"),
    pushbullet: Optional[str] = typer.Option(None, "--pushbullet", help="Pushbullet token"),
    smtp_server: Optional[str] = typer.Option(None, "--smtp-server", help="SMTP server for email"),
    smtp_port: int = typer.Option(587, "--smtp-port", help="SMTP port"),
    smtp_user: Optional[str] = typer.Option(None, "--smtp-user", help="SMTP username"),
    smtp_pass: Optional[str] = typer.Option(None, "--smtp-pass", help="SMTP password"),
):
    """Send prediction notifications (v1.5)."""
    print_header(f"📢 Notification System - {LOTTERY_CONFIGS[lottery]['name']} (v1.5)")

    if lottery not in LOTTERY_TYPES:
        console.print(f"[red]Error: Unknown lottery type '{lottery}'[/red]")
        raise typer.Exit(1)

    # Get predictions
    predictor = get_predictor(lottery, method)
    if method == "statistical":
        pred = StatisticalPredictor(lottery)
        predictions = pred.predict_balanced(num_predictions=count)
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
    else:
        pred = EnsemblePredictor(lottery)
        predictions = pred.get_consensus_prediction(num_predictions=count)

    notification = NotificationSystem()

    recipients = []
    if email:
        if not smtp_server or not smtp_user or not smtp_pass:
            console.print("[red]Error: SMTP settings required for email[/red]")
            raise typer.Exit(1)
        notification.configure_email(smtp_server, smtp_port, smtp_user, smtp_pass)
        recipients.append(email)

    if discord:
        notification.configure_discord(discord)
        recipients.append(f"discord:{discord}")

    if pushbullet:
        notification.configure_pushbullet(pushbullet)
        recipients.append(f"pushbullet:{pushbullet}")

    if not recipients:
        console.print("[yellow]No recipients specified. Use --email, --discord, or --pushbullet[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Sending notifications...", total=None)
        results = notification.notify_predictions(lottery, predictions, recipients)
        progress.update(task, description="Done!", completed=True)

    console.print("\n[bold]Notification Results:[/bold]")
    for recipient, success in results.items():
        status = "[green]✓ Sent[/green]" if success else "[red]✗ Failed[/red]"
        console.print(f"  {recipient}: {status}")


@app.command()
def language(
    lang: str = typer.Option("en", "--lang", help="Language code: en, es, fr, de, zh"),
    list_langs: bool = typer.Option(False, "--list", help="List available languages"),
):
    """Manage multi-language support (v1.5)."""
    print_header("🌐 Language Manager (v1.5)")

    i18n = I18nManager()

    if list_langs:
        console.print("\n[bold]Available Languages:[/bold]")
        for code in I18nManager.get_available_languages():
            current = " (current)" if code == i18n.language else ""
            console.print(f"  {code}{current}")
        return

    if lang not in I18nManager.get_available_languages():
        console.print(f"[red]Error: Language '{lang}' not available[/red]")
        console.print(f"Available: {', '.join(I18nManager.get_available_languages())}")
        raise typer.Exit(1)

    i18n.set_language(lang)
    console.print(f"\n[green]Language set to: {lang}[/green]")

    # Show sample translations
    console.print("\n[bold]Sample Translations:[/bold]")
    sample_keys = ["predictions", "main_numbers", "confidence", "disclaimer", "hot_numbers", "cold_numbers"]
    table = Table(show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Translation", style="green")
    for key in sample_keys:
        table.add_row(key, i18n.get(key)[:80] + "..." if len(i18n.get(key)) > 80 else i18n.get(key))
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start FastAPI REST API server (v2.0)."""
    print_header("🚀 Starting Lottery Predictor API Server (v2.0)")

    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Starting server on {host}:{port}[/cyan]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")

    uvicorn.run("lottery_predictor.cloud:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def generate_deployment(
    type: str = typer.Option("all", "--type", "-t", help="Type: dockerfile, docker-compose, k8s, all"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
):
    """Generate deployment files (v2.0)."""
    print_header("📦 Generating Deployment Files (v2.0)")

    if type in ("dockerfile", "all"):
        path = generate_dockerfile(f"{output_dir}/Dockerfile")
        console.print(f"[green]✓ Dockerfile generated: {path}[/green]")

    if type in ("docker-compose", "all"):
        path = generate_docker_compose(f"{output_dir}/docker-compose.yml")
        console.print(f"[green]✓ docker-compose.yml generated: {path}[/green]")

    if type in ("k8s", "all"):
        path = generate_kubernetes(f"{output_dir}/k8s")
        console.print(f"[green]✓ Kubernetes files generated: {path}[/green]")

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Review generated files")
    console.print("  2. For Docker: docker-compose up --build")
    console.print("  3. For Kubernetes: kubectl apply -f k8s/")


@app.command()
def init_db(
    database_url: str = typer.Option("sqlite:///lottery_predictor.db", "--url", help="Database URL"),
):
    """Initialize database for user accounts and history (v2.0)."""
    print_header("🗄️ Initializing Database (v2.0)")

    try:
        from .cloud import init_database
        SessionLocal = init_database(database_url)
        console.print(f"[green]✓ Database initialized at {database_url}[/green]")
        console.print("\nTables created:")
        console.print("  - users (accounts)")
        console.print("  - prediction_history")
        console.print("  - ab_tests")
        console.print("  - ab_test_results")
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Install with: pip install sqlalchemy")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()