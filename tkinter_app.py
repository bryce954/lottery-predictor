#!/usr/bin/env python3
"""
Tkinter Desktop GUI for Lottery Predictor
Native desktop application for lottery number prediction.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lottery_predictor.data import get_data_manager, LOTTERY_CONFIGS
from lottery_predictor.models import (
    StatisticalPredictor,
    MLModelPredictor,
    EnsemblePredictor,
    get_predictor,
    PredictionResult
)
from lottery_predictor.universal import get_universal_manager
from lottery_predictor.visualize import LotteryVisualizer


class LotteryPredictorApp:
    """Main Tkinter application for Lottery Predictor."""

    def __init__(self, root):
        self.root = root
        self.root.title("🎲 Lottery Predictor - Desktop")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)

        # State
        self.current_lottery = "powerball"
        self.predictions = []
        self.analysis_data = None
        self.universal_manager = get_universal_manager()
        self.valid_lotteries = [g for g in self.universal_manager.list_games() if g.main_numbers > 0]

        # Thread-safe UI updates
        self.update_queue = []

        # Build UI
        self._setup_styles()
        self._build_ui()
        self._load_lottery_info()

        # Periodic check for thread updates
        self.root.after(100, self._process_queue)

    def _setup_styles(self):
        """Configure custom styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), foreground='#1f77b4')
        style.configure('Subtitle.TLabel', font=('Helvetica', 11), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), foreground='#2c3e50')
        style.configure('Info.TLabel', font=('Helvetica', 10), foreground='#555')
        style.configure('Prediction.TLabel', font=('Consolas', 14, 'bold'))
        style.configure('Disclaimer.TLabel', font=('Helvetica', 9), foreground='#856404')
        style.configure('Success.TLabel', font=('Helvetica', 10, 'bold'), foreground='green')
        style.configure('Warning.TLabel', font=('Helvetica', 10, 'bold'), foreground='red')

        style.configure('Action.TButton', font=('Helvetica', 10, 'bold'))
        style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'), padding=[15, 5])

    def _build_ui(self):
        """Build the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="🎲 Lottery Predictor", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="AI-Powered Lottery Number Prediction", style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(10, 0))

        # Disclaimer
        disclaimer_frame = ttk.Frame(main_frame)
        disclaimer_frame.pack(fill=tk.X, pady=(0, 10))
        disclaimer_text = ("⚠️ DISCLAIMER: This tool is for entertainment and educational purposes only. "
                          "Lottery draws are random events. No algorithm or AI can predict future lottery numbers with certainty. "
                          "Past performance does not guarantee future results. Please play responsibly.")
        ttk.Label(disclaimer_frame, text=disclaimer_text, style='Disclaimer.TLabel', wraplength=950).pack()

        # Settings panel
        self._build_settings_panel(main_frame)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Build tabs
        self._build_predict_tab()
        self._build_analyze_tab()
        self._build_history_tab()
        self._build_compare_tab()
        self._build_visualize_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

    def _build_settings_panel(self, parent):
        """Build the settings panel at the top."""
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Lottery selection
        ttk.Label(settings_frame, text="🎰 Lottery:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.lottery_var = tk.StringVar(value="powerball")
        self.lottery_combo = ttk.Combobox(settings_frame, textvariable=self.lottery_var,
                                           state="readonly", width=35)
        self.lottery_combo['values'] = [f"{g.name} ({g.key})" for g in self.valid_lotteries]
        self.lottery_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.lottery_combo.bind('<<ComboboxSelected>>', self._on_lottery_change)

        # Method selection
        ttk.Label(settings_frame, text="🔮 Method:").grid(row=0, column=2, padx=15, pady=5, sticky=tk.W)
        self.method_var = tk.StringVar(value="ensemble")
        method_combo = ttk.Combobox(settings_frame, textvariable=self.method_var,
                                     state="readonly", width=20)
        method_combo['values'] = [
            "ensemble", "statistical", "frequency", "cold", "pairs", "ml"
        ]
        method_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        # Number of predictions
        ttk.Label(settings_frame, text="Count:").grid(row=0, column=4, padx=15, pady=5, sticky=tk.W)
        self.count_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=1, to=20, textvariable=self.count_var, width=5).grid(row=0, column=5, padx=5, pady=5)

        # Historical draws
        ttk.Label(settings_frame, text="Last N Draws:").grid(row=0, column=6, padx=15, pady=5, sticky=tk.W)
        self.last_n_var = tk.IntVar(value=200)
        ttk.Spinbox(settings_frame, from_=50, to=2000, increment=50, textvariable=self.last_n_var, width=7).grid(row=0, column=7, padx=5, pady=5)

        # Lottery info display
        self.info_var = tk.StringVar()
        ttk.Label(settings_frame, textvariable=self.info_var, style='Info.TLabel').grid(row=1, column=0, columnspan=8, pady=(5, 0), sticky=tk.W)

    def _build_predict_tab(self):
        """Build the Predict tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="🎯 Predict")

        # Action buttons
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="🎲 Generate Predictions", command=self._generate_predictions,
                   style='Action.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="💾 Save to JSON", command=self._save_predictions_json).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📤 Export (CSV/Excel)", command=self._export_predictions).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🤖 Train ML Models", command=self._train_models).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.predict_progress = ttk.Progressbar(action_frame, mode='indeterminate', length=200)
        self.predict_progress.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        # Predictions display
        pred_frame = ttk.LabelFrame(frame, text="🎯 Generated Predictions", padding=10)
        pred_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for predictions
        columns = ('#', 'Main Numbers', 'Bonus', 'Confidence', 'Method')
        self.pred_tree = ttk.Treeview(pred_frame, columns=columns, show='headings', height=12)
        for col in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=120, anchor=tk.CENTER)
        self.pred_tree.column('#', width=40)
        self.pred_tree.column('Main Numbers', width=250)
        self.pred_tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        pred_scroll = ttk.Scrollbar(pred_frame, orient=tk.VERTICAL, command=self.pred_tree.yview)
        self.pred_tree.configure(yscrollcommand=pred_scroll.set)
        pred_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Stats frame
        stats_frame = ttk.LabelFrame(frame, text="📊 Prediction Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        self.stats_vars = {}
        for i, (label, key) in enumerate([
            ("Avg Sum", "avg_sum"), ("Most Common", "most_common"),
            ("Even/Odd Ratio", "even_odd"), ("Avg Confidence", "avg_conf")
        ]):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=0, column=i*2, padx=5, sticky=tk.W)
            var = tk.StringVar(value="N/A")
            self.stats_vars[key] = var
            ttk.Label(stats_frame, textvariable=var, font=('Helvetica', 10, 'bold')).grid(row=0, column=i*2+1, padx=5, sticky=tk.W)

    def _build_analyze_tab(self):
        """Build the Analyze tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="📊 Analyze")

        # Controls
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Draws to Analyze:").pack(side=tk.LEFT, padx=5)
        self.analyze_n_var = tk.IntVar(value=200)
        ttk.Spinbox(ctrl_frame, from_=50, to=2000, increment=50, textvariable=self.analyze_n_var, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="🔄 Refresh", command=self._refresh_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="📈 Generate Charts", command=self._generate_analysis_charts).pack(side=tk.LEFT, padx=5)

        # Metrics
        metrics_frame = ttk.LabelFrame(frame, text="📈 Key Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.metric_vars = {}
        for i, (label, key) in enumerate([
            ("Total Draws", "total_draws"), ("Date Range", "date_range"),
            ("Avg Sum", "avg_sum_ana"), ("Sum Std Dev", "sum_std")
        ]):
            ttk.Label(metrics_frame, text=f"{label}:").grid(row=0, column=i*2, padx=10, sticky=tk.W)
            var = tk.StringVar(value="N/A")
            self.metric_vars[key] = var
            ttk.Label(metrics_frame, textvariable=var, font=('Helvetica', 10, 'bold')).grid(row=0, column=i*2+1, padx=10, sticky=tk.W)

        # Hot/Cold numbers
        hc_frame = ttk.Frame(frame)
        hc_frame.pack(fill=tk.BOTH, expand=True)

        # Hot numbers
        hot_frame = ttk.LabelFrame(hc_frame, text="🔥 Hot Numbers", padding=5)
        hot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        hot_cols = ('Rank', 'Number', 'Frequency')
        self.hot_tree = ttk.Treeview(hot_frame, columns=hot_cols, show='headings', height=15)
        for col in hot_cols:
            self.hot_tree.heading(col, text=col)
            self.hot_tree.column(col, width=80, anchor=tk.CENTER)
        self.hot_tree.pack(fill=tk.BOTH, expand=True)

        # Cold numbers
        cold_frame = ttk.LabelFrame(hc_frame, text="❄️ Cold Numbers", padding=5)
        cold_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.cold_tree = ttk.Treeview(cold_frame, columns=hot_cols, show='headings', height=15)
        for col in hot_cols:
            self.cold_tree.heading(col, text=col)
            self.cold_tree.column(col, width=80, anchor=tk.CENTER)
        self.cold_tree.pack(fill=tk.BOTH, expand=True)

        # Top pairs
        pairs_frame = ttk.LabelFrame(frame, text="🔗 Top Number Pairs", padding=5)
        pairs_frame.pack(fill=tk.X, pady=(10, 0))

        pair_cols = ('Rank', 'Pair', 'Count')
        self.pairs_tree = ttk.Treeview(pairs_frame, columns=pair_cols, show='headings', height=8)
        for col in pair_cols:
            self.pairs_tree.heading(col, text=col)
            self.pairs_tree.column(col, width=100, anchor=tk.CENTER)
        self.pairs_tree.pack(fill=tk.X)

    def _build_history_tab(self):
        """Build the History tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="📜 History")

        # Controls
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Recent Draws:").pack(side=tk.LEFT, padx=5)
        self.history_n_var = tk.IntVar(value=20)
        ttk.Spinbox(ctrl_frame, from_=5, to=100, textvariable=self.history_n_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="🔄 Refresh", command=self._refresh_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="📥 Download CSV", command=self._download_history_csv).pack(side=tk.LEFT, padx=5)

        # History tree
        hist_frame = ttk.LabelFrame(frame, text="📜 Recent Draw History", padding=5)
        hist_frame.pack(fill=tk.BOTH, expand=True)

        hist_cols = ('Date', 'Main Numbers', 'Bonus', 'Jackpot')
        self.history_tree = ttk.Treeview(hist_frame, columns=hist_cols, show='headings', height=20)
        for col in hist_cols:
            self.history_tree.heading(col, text=col)
        self.history_tree.column('Date', width=100, anchor=tk.CENTER)
        self.history_tree.column('Main Numbers', width=300, anchor=tk.CENTER)
        self.history_tree.column('Bonus', width=100, anchor=tk.CENTER)
        self.history_tree.column('Jackpot', width=150, anchor=tk.CENTER)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        hist_scroll = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=hist_scroll.set)
        hist_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_compare_tab(self):
        """Build the Compare tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="⚔️ Compare")

        # Controls
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Methods:").pack(side=tk.LEFT, padx=5)
        self.compare_methods = {}
        for method in ["statistical", "frequency", "cold", "pairs", "ml", "ensemble"]:
            var = tk.BooleanVar(value=method in ["statistical", "frequency", "cold", "ensemble"])
            self.compare_methods[method] = var
            ttk.Checkbutton(ctrl_frame, text=method.capitalize(), variable=var).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Predictions per Method:").pack(side=tk.LEFT, padx=15)
        self.compare_count_var = tk.IntVar(value=3)
        ttk.Spinbox(ctrl_frame, from_=1, to=10, textvariable=self.compare_count_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl_frame, text="⚔️ Compare", command=self._compare_methods, style='Action.TButton').pack(side=tk.LEFT, padx=15)

        # Results
        self.compare_frame = ttk.LabelFrame(frame, text="⚔️ Method Comparison Results", padding=10)
        self.compare_frame.pack(fill=tk.BOTH, expand=True)

        # Summary table
        sum_cols = ('Method', 'Avg Confidence', 'Unique Predictions', 'Predictions')
        self.compare_tree = ttk.Treeview(self.compare_frame, columns=sum_cols, show='headings', height=8)
        for col in sum_cols:
            self.compare_tree.heading(col, text=col)
            self.compare_tree.column(col, width=150, anchor=tk.CENTER)
        self.compare_tree.pack(fill=tk.X, pady=(0, 10))

        # Detailed predictions
        detail_cols = ('Method', '#', 'Main Numbers', 'Bonus', 'Confidence')
        self.compare_detail_tree = ttk.Treeview(self.compare_frame, columns=detail_cols, show='headings', height=12)
        for col in detail_cols:
            self.compare_detail_tree.heading(col, text=col)
            self.compare_detail_tree.column(col, width=150, anchor=tk.CENTER)
        self.compare_detail_tree.pack(fill=tk.BOTH, expand=True)

    def _build_visualize_tab(self):
        """Build the Visualize tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="📈 Visualize")

        # Controls
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Charts:").pack(side=tk.LEFT, padx=5)
        self.viz_options = {}
        for opt in ["Frequency", "Hot/Cold", "Sum Distribution", "Pair Heatmap", "Timeline"]:
            var = tk.BooleanVar(value=True)
            self.viz_options[opt] = var
            ttk.Checkbutton(ctrl_frame, text=opt, variable=var).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl_frame, text="🎨 Generate Visualizations", command=self._generate_visualizations,
                   style='Action.TButton').pack(side=tk.LEFT, padx=15)

        # Status
        self.viz_status_var = tk.StringVar(value="Click 'Generate Visualizations' to create charts")
        ttk.Label(ctrl_frame, textvariable=self.viz_status_var, style='Info.TLabel').pack(side=tk.LEFT, padx=15)

        # Preview area
        self.viz_frame = ttk.LabelFrame(frame, text="📊 Generated Charts (saved to visualizations/ folder)", padding=10)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)

        self.viz_listbox = tk.Listbox(self.viz_frame, font=('Consolas', 10))
        self.viz_listbox.pack(fill=tk.BOTH, expand=True)

    def _load_lottery_info(self):
        """Load and display lottery info."""
        game = self.universal_manager.get_game(self.current_lottery)
        if game:
            bonus = f" + {game.bonus_numbers} bonus (1-{game.bonus_max})" if game.bonus_numbers > 0 else ""
            self.info_var.set(
                f"{game.name} | Main: {game.main_numbers} numbers (1-{game.main_max}){bonus} | "
                f"Draws: {', '.join(game.draw_days)} | Since: {game.start_date}"
            )

    def _on_lottery_change(self, event=None):
        """Handle lottery selection change."""
        selection = self.lottery_var.get()
        for g in self.valid_lotteries:
            if f"{g.name} ({g.key})" == selection:
                self.current_lottery = g.key
                break
        self._load_lottery_info()
        self._clear_predictions()
        self.status_var.set(f"Selected: {selection}")

    def _clear_predictions(self):
        """Clear prediction display."""
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)
        for var in self.stats_vars.values():
            var.set("N/A")

    def _generate_predictions(self):
        """Generate predictions in background thread."""
        method = self.method_var.get()
        count = self.count_var.get()
        last_n = self.last_n_var.get()

        self.predict_progress.start()
        self.status_var.set(f"Generating {count} predictions using {method}...")

        def run_prediction():
            try:
                preds = self._run_prediction(method, count, last_n)
                self.update_queue.append(('predictions_done', preds))
            except Exception as e:
                self.update_queue.append(('error', str(e)))

        threading.Thread(target=run_prediction, daemon=True).start()

    def _run_prediction(self, method, count, last_n):
        """Run prediction logic (in background thread)."""
        if method == "statistical":
            pred = StatisticalPredictor(self.current_lottery)
            return pred.predict_balanced(last_n=last_n, num_predictions=count)
        elif method == "frequency":
            pred = StatisticalPredictor(self.current_lottery)
            return pred.predict_frequency_based(last_n=last_n, num_predictions=count)
        elif method == "cold":
            pred = StatisticalPredictor(self.current_lottery)
            return pred.predict_cold_numbers(last_n=last_n, num_predictions=count)
        elif method == "pairs":
            pred = StatisticalPredictor(self.current_lottery)
            return pred.predict_pair_based(last_n=last_n, num_predictions=count)
        elif method == "ml":
            pred = MLModelPredictor(self.current_lottery)
            if not pred.is_trained:
                pred.train()
            return pred.predict(num_predictions=count)
        else:  # ensemble
            pred = EnsemblePredictor(self.current_lottery)
            return pred.get_consensus_prediction(num_predictions=count)

    def _process_queue(self):
        """Process UI updates from background threads."""
        while self.update_queue:
            msg_type, data = self.update_queue.pop(0)

            if msg_type == 'predictions_done':
                self.predictions = data
                self._display_predictions(data)
                self.predict_progress.stop()
                self.status_var.set(f"Generated {len(data)} predictions")

            elif msg_type == 'analysis_done':
                self.analysis_data = data
                self._display_analysis(data)
                self.status_var.set("Analysis complete")

            elif msg_type == 'history_done':
                self._display_history(data)
                self.status_var.set("History loaded")

            elif msg_type == 'compare_done':
                self._display_compare(data)
                self.status_var.set("Comparison complete")

            elif msg_type == 'viz_done':
                self._display_visualizations(data)
                self.viz_status_var.set(f"Generated {len(data)} charts")
                self.status_var.set("Visualizations complete")

            elif msg_type == 'train_done':
                self.predict_progress.stop()
                if data.get('success'):
                    self.status_var.set("ML models trained successfully")
                    messagebox.showinfo("Training Complete", f"Models trained:\n{data['results']}")
                else:
                    self.status_var.set("Training failed")
                    messagebox.showerror("Training Failed", data['error'])

            elif msg_type == 'error':
                self.predict_progress.stop()
                self.status_var.set("Error occurred")
                messagebox.showerror("Error", data)

        self.root.after(100, self._process_queue)

    def _display_predictions(self, predictions):
        """Display predictions in the tree."""
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)

        for i, pred in enumerate(predictions, 1):
            main_str = "  ".join(f"{n:2d}" for n in pred.main_numbers)
            bonus_str = "  ".join(f"{n:2d}" for n in pred.bonus_numbers) if pred.bonus_numbers else "N/A"
            conf_str = f"{pred.confidence:.1%}"
            self.pred_tree.insert('', tk.END, values=(i, main_str, bonus_str, conf_str, pred.method))

        # Update stats
        if predictions:
            all_nums = [n for p in predictions for n in p.main_numbers]
            sums = [sum(p.main_numbers) for p in predictions]
            evens = sum(1 for n in all_nums if n % 2 == 0)
            avg_conf = np.mean([p.confidence for p in predictions])

            self.stats_vars['avg_sum'].set(f"{np.mean(sums):.1f}")
            self.stats_vars['most_common'].set(str(pd.Series(all_nums).mode()[0]) if all_nums else "N/A")
            self.stats_vars['even_odd'].set(f"{evens}/{len(all_nums)-evens}")
            self.stats_vars['avg_conf'].set(f"{avg_conf:.1%}")

    def _save_predictions_json(self):
        """Save predictions to JSON file."""
        if not self.predictions:
            messagebox.showwarning("No Predictions", "Generate predictions first!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.current_lottery}_predictions.json"
        )
        if filepath:
            data = {
                "lottery": self.current_lottery,
                "method": self.method_var.get(),
                "predictions": [
                    {
                        "main_numbers": p.main_numbers,
                        "bonus_numbers": p.bonus_numbers,
                        "confidence": p.confidence,
                        "method": p.method,
                        "metadata": p.metadata
                    }
                    for p in self.predictions
                ]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Predictions saved to {filepath}")

    def _export_predictions(self):
        """Export data using CLI export command."""
        import subprocess
        format_choice = messagebox.askyesno("Export Format", "Export as CSV? (No = JSON)")
        fmt = "csv" if format_choice else "json"

        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}"), ("All files", "*.*")],
            initialfile=f"{self.current_lottery}_export.{fmt}"
        )
        if filepath:
            try:
                cmd = [
                    sys.executable, "main.py", "export",
                    "--lottery", self.current_lottery,
                    "--format", fmt,
                    "--output", filepath,
                    "--last-n", str(self.last_n_var.get()),
                    "--include-predictions",
                    "--count", str(self.count_var.get())
                ]
                result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    messagebox.showinfo("Exported", f"Data exported to {filepath}")
                else:
                    messagebox.showerror("Export Failed", result.stderr)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _train_models(self):
        """Train ML models in background."""
        self.predict_progress.start()
        self.status_var.set("Training ML models...")

        def run_training():
            try:
                predictor = MLModelPredictor(self.current_lottery)
                results = predictor.train(lookback=10)
                self.update_queue.append(('train_done', {'success': True, 'results': results}))
            except Exception as e:
                self.update_queue.append(('train_done', {'success': False, 'error': str(e)}))

        threading.Thread(target=run_training, daemon=True).start()

    def _refresh_analysis(self):
        """Refresh analysis in background."""
        self.status_var.set("Loading analysis...")

        def run_analysis():
            try:
                last_n = self.analyze_n_var.get()
                dm = get_data_manager(self.current_lottery)
                df = dm.load_data()
                df = df.tail(last_n)

                freq = dm.get_number_frequency(last_n)
                hot, cold = dm.get_hot_cold_numbers(last_n)
                stats = dm.get_sum_statistics(last_n)
                pairs = dm.get_pair_frequency(last_n)
                top_pairs = list(pairs.items())[:15]

                self.update_queue.append(('analysis_done', {
                    'df': df, 'freq': freq, 'hot': hot, 'cold': cold,
                    'stats': stats, 'top_pairs': top_pairs
                }))
            except Exception as e:
                self.update_queue.append(('error', str(e)))

        threading.Thread(target=run_analysis, daemon=True).start()

    def _display_analysis(self, data):
        """Display analysis results."""
        df = data['df']
        freq = data['freq']
        hot = data['hot']
        cold = data['cold']
        stats = data['stats']
        top_pairs = data['top_pairs']

        # Metrics
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        self.metric_vars['total_draws'].set(str(len(df)))
        self.metric_vars['date_range'].set(date_range)
        self.metric_vars['avg_sum_ana'].set(f"{stats['mean']:.1f}")
        self.metric_vars['sum_std'].set(f"{stats['std']:.1f}")

        # Hot numbers
        for item in self.hot_tree.get_children():
            self.hot_tree.delete(item)
        for i, num in enumerate(hot, 1):
            self.hot_tree.insert('', tk.END, values=(i, num, freq[num]))

        # Cold numbers
        for item in self.cold_tree.get_children():
            self.cold_tree.delete(item)
        for i, num in enumerate(cold, 1):
            self.cold_tree.insert('', tk.END, values=(i, num, freq[num]))

        # Top pairs
        for item in self.pairs_tree.get_children():
            self.pairs_tree.delete(item)
        for i, ((n1, n2), count) in enumerate(top_pairs, 1):
            self.pairs_tree.insert('', tk.END, values=(i, f"{n1} - {n2}", count))

    def _generate_analysis_charts(self):
        """Generate analysis charts."""
        self.status_var.set("Generating charts...")
        try:
            viz = LotteryVisualizer(self.current_lottery)
            dm = get_data_manager(self.current_lottery)
            df = dm.load_data()
            df = df.tail(self.analyze_n_var.get())
            viz.dm = dm
            files = viz.generate_all_visualizations(self.analyze_n_var.get())
            messagebox.showinfo("Charts Generated", f"Charts saved to visualizations/:\n" + "\n".join(files))
            self.status_var.set("Charts generated")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error generating charts")

    def _refresh_history(self):
        """Refresh history in background."""
        self.status_var.set("Loading history...")

        def run_history():
            try:
                dm = get_data_manager(self.current_lottery)
                df = dm.load_data()
                recent = dm.get_recent_draws(self.history_n_var.get())
                self.update_queue.append(('history_done', recent))
            except Exception as e:
                self.update_queue.append(('error', str(e)))

        threading.Thread(target=run_history, daemon=True).start()

    def _display_history(self, recent):
        """Display history in tree."""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        for _, row in recent.iterrows():
            main_str = " - ".join(f"{n:2d}" for n in row['main_numbers'])
            bonus_str = " - ".join(f"{n:2d}" for n in row['bonus_numbers']) if row['bonus_numbers'] else "N/A"
            jackpot_str = f"${row['jackpot']:,}"
            date_val = row['date']
            date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)
            self.history_tree.insert('', tk.END, values=(date_str, main_str, bonus_str, jackpot_str))

    def _download_history_csv(self):
        """Download history as CSV."""
        try:
            dm = get_data_manager(self.current_lottery)
            df = dm.load_data()
            recent = dm.get_recent_draws(self.history_n_var.get())

            display_data = []
            for _, row in recent.iterrows():
                display_data.append({
                    'Date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'Main Numbers': ' - '.join(f"{n:2d}" for n in row['main_numbers']),
                    'Bonus': ' - '.join(f"{n:2d}" for n in row['bonus_numbers']) if row['bonus_numbers'] else 'N/A',
                    'Jackpot': f"${row['jackpot']:,}"
                })

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"{self.current_lottery}_history.csv"
            )
            if filepath:
                pd.DataFrame(display_data).to_csv(filepath, index=False)
                messagebox.showinfo("Downloaded", f"History saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _compare_methods(self):
        """Compare methods in background."""
        selected_methods = [m for m, v in self.compare_methods.items() if v.get()]
        if not selected_methods:
            messagebox.showwarning("No Methods", "Select at least one method to compare!")
            return

        count = self.compare_count_var.get()
        self.status_var.set("Comparing methods...")

        def run_compare():
            try:
                all_results = {}
                for method in selected_methods:
                    preds = self._run_prediction(method, count, self.last_n_var.get())
                    all_results[method] = preds
                self.update_queue.append(('compare_done', all_results))
            except Exception as e:
                self.update_queue.append(('error', str(e)))

        threading.Thread(target=run_compare, daemon=True).start()

    def _display_compare(self, all_results):
        """Display comparison results."""
        # Clear trees
        for item in self.compare_tree.get_children():
            self.compare_tree.delete(item)
        for item in self.compare_detail_tree.get_children():
            self.compare_detail_tree.delete(item)

        # Summary
        for method, preds in all_results.items():
            avg_conf = np.mean([p.confidence for p in preds])
            unique = len(set(tuple(p.main_numbers) for p in preds))
            self.compare_tree.insert('', tk.END, values=(method, f"{avg_conf:.1%}", unique, len(preds)))

        # Details
        for method, preds in all_results.items():
            for i, pred in enumerate(preds, 1):
                main_str = "  ".join(f"{n:2d}" for n in pred.main_numbers)
                bonus_str = "  ".join(f"{n:2d}" for n in pred.bonus_numbers) if pred.bonus_numbers else "N/A"
                conf_str = f"{pred.confidence:.1%}"
                self.compare_detail_tree.insert('', tk.END, values=(method, i, main_str, bonus_str, conf_str))

    def _generate_visualizations(self):
        """Generate visualizations in background."""
        selected = [opt for opt, var in self.viz_options.items() if var.get()]
        if not selected:
            messagebox.showwarning("No Charts", "Select at least one chart to generate!")
            return

        self.viz_status_var.set("Generating charts...")
        self.status_var.set("Generating visualizations...")

        def run_viz():
            try:
                viz = LotteryVisualizer(self.current_lottery)
                dm = get_data_manager(self.current_lottery)
                df = dm.load_data()
                df = df.tail(self.last_n_var.get())
                viz.dm = dm

                files = []
                import matplotlib
                matplotlib.use('Agg')

                if "Frequency" in selected:
                    files.append(viz.plot_number_frequency(self.last_n_var.get(), save=True))
                if "Hot/Cold" in selected:
                    files.append(viz.plot_hot_cold_numbers(self.last_n_var.get(), save=True))
                if "Sum Distribution" in selected:
                    files.append(viz.plot_sum_distribution(self.last_n_var.get(), save=True))
                if "Pair Heatmap" in selected:
                    files.append(viz.plot_pair_heatmap(self.last_n_var.get(), save=True))
                if "Timeline" in selected:
                    files.append(viz.plot_timeline_analysis(self.last_n_var.get(), save=True))

                self.update_queue.append(('viz_done', files))
            except Exception as e:
                self.update_queue.append(('error', str(e)))

        threading.Thread(target=run_viz, daemon=True).start()

    def _display_visualizations(self, files):
        """Display generated visualization files."""
        self.viz_listbox.delete(0, tk.END)
        for f in files:
            self.viz_listbox.insert(tk.END, f)


def main():
    """Main entry point for Tkinter app."""
    root = tk.Tk()
    app = LotteryPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()