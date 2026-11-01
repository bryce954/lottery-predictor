#!/usr/bin/env python3
"""
Streamlit Web GUI for Lottery Predictor
Interactive web-based graphical interface for lottery predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path

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
from lottery_predictor.universal import get_universal_manager, LotteryGame
from lottery_predictor.visualize import LotteryVisualizer

# Page config
st.set_page_config(
    page_title="🎲 Lottery Predictor",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'current_lottery' not in st.session_state:
    st.session_state.current_lottery = "powerball"
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Load universal manager
@st.cache_resource
def get_manager():
    return get_universal_manager()

manager = get_manager()

# Header
st.markdown('<div class="main-header">🎲 Lottery Predictor</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Lottery Number Prediction</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ <strong>IMPORTANT DISCLAIMER:</strong> This tool is for <strong>entertainment and educational purposes only</strong>.
Lottery draws are random events. No algorithm, AI, or statistical analysis can predict future lottery numbers with certainty.
Past performance does not guarantee future results. Please play responsibly and within your means.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Lottery selection
    games = manager.list_games()
    game_options = {f"{g.name} ({g.key})": g.key for g in games}
    selected_game_label = st.selectbox(
        "🎰 Select Lottery",
        options=list(game_options.keys()),
        index=list(game_options.values()).index(st.session_state.current_lottery)
    )
    selected_lottery = game_options[selected_game_label]
    st.session_state.current_lottery = selected_lottery

    game = manager.get_game(selected_lottery)

    # Show lottery info
    st.info(f"""
    **{game.name}**
    - Main: {game.main_numbers} numbers (1-{game.main_max})
    - Bonus: {game.bonus_numbers} numbers (1-{game.bonus_max})
    - Draws: {', '.join(game.draw_days)}
    - Since: {game.start_date}
    """)

    # Prediction settings
    st.subheader("🔮 Prediction Settings")
    method = st.selectbox(
        "Method",
        ["ensemble", "statistical", "frequency", "cold", "pairs", "ml"],
        format_func=lambda x: {
            "ensemble": "🎯 Ensemble (Recommended)",
            "statistical": "📊 Statistical (Balanced)",
            "frequency": "🔥 Frequency (Hot Numbers)",
            "cold": "❄️ Cold Numbers (Due)",
            "pairs": "🔗 Pair-Based",
            "ml": "🤖 Machine Learning"
        }[x]
    )

    num_predictions = st.slider("Number of Predictions", 1, 10, 5)
    last_n_draws = st.slider("Historical Draws to Analyze", 50, 1000, 200, step=50)

    # Advanced settings
    with st.expander("Advanced Settings"):
        retrain = st.checkbox("Retrain ML Models", value=False)
        lookback = st.slider("Lookback Window (ML)", 5, 20, 10)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict",
    "📊 Analyze",
    "📈 Visualize",
    "📜 History",
    "⚔️ Compare"
])

# Helper functions
@st.cache_data(ttl=300)
def load_lottery_data(lottery_key, last_n=None):
    dm = get_data_manager(lottery_key)
    df = dm.load_data()
    if last_n:
        df = df.tail(last_n)
    return dm, df

def generate_predictions(lottery_key, method, count, last_n, retrain=False, lookback=10):
    """Generate predictions using selected method."""
    with st.spinner(f"Generating {count} predictions using {method}..."):
        if method == "statistical":
            pred = StatisticalPredictor(lottery_key)
            preds = pred.predict_balanced(last_n=last_n, num_predictions=count)
        elif method == "frequency":
            pred = StatisticalPredictor(lottery_key)
            preds = pred.predict_frequency_based(last_n=last_n, num_predictions=count)
        elif method == "cold":
            pred = StatisticalPredictor(lottery_key)
            preds = pred.predict_cold_numbers(last_n=last_n, num_predictions=count)
        elif method == "pairs":
            pred = StatisticalPredictor(lottery_key)
            preds = pred.predict_pair_based(last_n=last_n, num_predictions=count)
        elif method == "ml":
            pred = MLModelPredictor(lottery_key)
            if not pred.is_trained or retrain:
                pred.train(lookback=lookback)
            preds = pred.predict(num_predictions=count)
        else:  # ensemble
            pred = EnsemblePredictor(lottery_key)
            preds = pred.get_consensus_prediction(num_predictions=count)
    return preds

def format_numbers(numbers):
    """Format numbers for display."""
    return "  ".join(f"**{n:2d}**" for n in numbers)

# Tab 1: Predict
with tab1:
    st.markdown('<div class="sub-header">🎯 Generate Predictions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🎲 Generate Predictions", type="primary", use_container_width=True):
            preds = generate_predictions(
                selected_lottery, method, num_predictions, last_n_draws, retrain, lookback
            )
            st.session_state.predictions = preds
            st.success(f"Generated {len(preds)} predictions!")

    with col2:
        if st.button("💾 Save to JSON", use_container_width=True):
            if st.session_state.predictions:
                data = {
                    "lottery": selected_lottery,
                    "method": method,
                    "predictions": [
                        {
                            "main_numbers": p.main_numbers,
                            "bonus_numbers": p.bonus_numbers,
                            "confidence": p.confidence,
                            "method": p.method,
                            "metadata": p.metadata
                        }
                        for p in st.session_state.predictions
                    ]
                }
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(data, indent=2),
                    file_name=f"{selected_lottery}_predictions.json",
                    mime="application/json"
                )
            else:
                st.warning("No predictions to save!")

    # Display predictions
    if st.session_state.predictions:
        st.markdown("### 🎯 Your Predictions")

        for i, pred in enumerate(st.session_state.predictions, 1):
            with st.container():
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>Prediction #{i}</h4>
                    <p style="font-size: 1.5rem; font-family: monospace; letter-spacing: 2px;">
                        {format_numbers(pred.main_numbers)}
                    </p>
                    {f'<p><strong>Bonus:</strong> {format_numbers(pred.bonus_numbers)}</p>' if pred.bonus_numbers else ''}
                    <p><strong>Confidence:</strong> {pred.confidence:.1%} | <strong>Method:</strong> {pred.method}</p>
                </div>
                """, unsafe_allow_html=True)

        # Quick stats
        st.markdown("### 📊 Prediction Statistics")
        cols = st.columns(4)

        all_nums = [n for p in st.session_state.predictions for n in p.main_numbers]
        with cols[0]:
            st.metric("Avg Sum", f"{np.mean([sum(p.main_numbers) for p in st.session_state.predictions]):.1f}")
        with cols[1]:
            st.metric("Most Common", f"{pd.Series(all_nums).mode()[0] if all_nums else 'N/A'}")
        with cols[2]:
            evens = sum(1 for n in all_nums if n % 2 == 0)
            st.metric("Even/Odd Ratio", f"{evens}/{len(all_nums)-evens}")
        with cols[3]:
            avg_conf = np.mean([p.confidence for p in st.session_state.predictions])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

# Tab 2: Analyze
with tab2:
    st.markdown('<div class="sub-header">📊 Historical Analysis</div>', unsafe_allow_html=True)

    dm, df = load_lottery_data(selected_lottery, last_n_draws)

    # Analysis controls
    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_last_n = st.slider("Draws to Analyze", 50, len(df), min(last_n_draws, len(df)), key="analyze_n")
    with col2:
        if st.button("🔄 Refresh Analysis", key="refresh_analyze"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        show_viz = st.checkbox("Show Charts", value=True)

    # Recalculate with selected draws
    dm, df = load_lottery_data(selected_lottery, analysis_last_n)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Draws", len(df))
    with col2:
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)
    with col3:
        sums = df['main_numbers'].apply(sum)
        st.metric("Avg Sum", f"{sums.mean():.1f}")
    with col4:
        st.metric("Sum Std Dev", f"{sums.std():.1f}")

    # Hot/Cold Numbers
    st.subheader("🔥 Hot & Cold Numbers")
    freq = dm.get_number_frequency(analysis_last_n)
    hot, cold = dm.get_hot_cold_numbers(analysis_last_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔥 Hottest Numbers**")
        hot_df = pd.DataFrame({
            'Number': hot,
            'Frequency': [freq[n] for n in hot]
        })
        st.dataframe(hot_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**❄️ Coldest Numbers**")
        cold_df = pd.DataFrame({
            'Number': cold,
            'Frequency': [freq[n] for n in cold]
        })
        st.dataframe(cold_df, use_container_width=True, hide_index=True)

    # Top Pairs
    st.subheader("🔗 Top Number Pairs")
    pairs = dm.get_pair_frequency(analysis_last_n)
    top_pairs = list(pairs.items())[:15]

    if top_pairs:
        pairs_df = pd.DataFrame({
            'Pair': [f"{a} - {b}" for (a, b), _ in top_pairs],
            'Count': [c for _, c in top_pairs]
        })
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    # Charts
    if show_viz:
        st.subheader("📈 Charts")

        # Frequency chart
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=list(freq.keys()),
            y=list(freq.values()),
            name='Frequency',
            marker_color=['red' if v > np.mean(list(freq.values())) else 'blue' for v in freq.values()]
        ))
        fig_freq.add_hline(y=np.mean(list(freq.values())), line_dash="dash", line_color="black",
                          annotation_text=f"Mean: {np.mean(list(freq.values())):.1f}")
        fig_freq.update_layout(
            title=f"Number Frequency (Last {analysis_last_n} Draws)",
            xaxis_title="Number",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        # Sum distribution
        fig_sum = go.Figure()
        fig_sum.add_trace(go.Histogram(
            x=sums,
            nbinsx=30,
            name='Sum Distribution',
            marker_color='green',
            opacity=0.7
        ))
        fig_sum.update_layout(
            title="Sum of Numbers Distribution",
            xaxis_title="Sum",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig_sum, use_container_width=True)

# Tab 3: Visualize
with tab3:
    st.markdown('<div class="sub-header">📈 Visualizations</div>', unsafe_allow_html=True)

    dm, df = load_lottery_data(selected_lottery, last_n_draws)
    viz = LotteryVisualizer(selected_lottery)
    viz.dm = dm

    viz_options = st.multiselect(
        "Select Charts to Generate",
        ["Frequency", "Hot/Cold", "Sum Distribution", "Pair Heatmap", "Timeline"],
        default=["Frequency", "Hot/Cold", "Sum Distribution", "Pair Heatmap", "Timeline"]
    )

    if st.button("🎨 Generate Visualizations", type="primary"):
        with st.spinner("Generating charts..."):
            # Use matplotlib visualizations
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            charts = []
            if "Frequency" in viz_options:
                charts.append(("Frequency", viz.plot_number_frequency(last_n_draws, save=False)))
            if "Hot/Cold" in viz_options:
                charts.append(("Hot/Cold", viz.plot_hot_cold_numbers(last_n_draws, save=False)))
            if "Sum Distribution" in viz_options:
                charts.append(("Sum Distribution", viz.plot_sum_distribution(last_n_draws, save=False)))
            if "Pair Heatmap" in viz_options:
                charts.append(("Pair Heatmap", viz.plot_pair_heatmap(last_n_draws, save=False)))
            if "Timeline" in viz_options:
                charts.append(("Timeline", viz.plot_timeline_analysis(last_n_draws, save=False)))

            # Also create interactive Plotly charts
            st.subheader("📊 Interactive Charts")

            # Frequency interactive
            freq = dm.get_number_frequency(last_n_draws)
            fig = px.bar(
                x=list(freq.keys()),
                y=list(freq.values()),
                title=f"Number Frequency (Last {last_n_draws} Draws)",
                labels={'x': 'Number', 'y': 'Frequency'},
                color=list(freq.values()),
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Hot/Cold interactive
            hot, cold = dm.get_hot_cold_numbers(last_n_draws)
            fig_hc = make_subplots(rows=1, cols=2, subplot_titles=('Hot Numbers', 'Cold Numbers'))

            fig_hc.add_trace(go.Bar(
                y=hot, x=[freq[n] for n in hot],
                orientation='h', name='Hot', marker_color='red',
                text=[freq[n] for n in hot], textposition='auto'
            ), row=1, col=1)

            fig_hc.add_trace(go.Bar(
                y=cold, x=[freq[n] for n in cold],
                orientation='h', name='Cold', marker_color='blue',
                text=[freq[n] for n in cold], textposition='auto'
            ), row=1, col=2)

            fig_hc.update_layout(height=500, showlegend=False, title="Hot vs Cold Numbers")
            fig_hc.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_hc, use_container_width=True)

            # Pair heatmap interactive
            pairs = dm.get_pair_frequency(last_n_draws)
            top_20 = dict(list(pairs.items())[:20])

            if top_20:
                max_num = game.main_max
                matrix = np.zeros((max_num, max_num))
                for (n1, n2), count in top_20.items():
                    matrix[n1-1, n2-1] = count
                    matrix[n2-1, n1-1] = count

                fig_heat = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=list(range(1, max_num+1)),
                    y=list(range(1, max_num+1)),
                    colorscale='YlOrRd',
                    showscale=True
                ))
                fig_heat.update_layout(
                    title="Top 20 Number Pairs Heatmap",
                    xaxis_title="Number",
                    yaxis_title="Number",
                    height=600
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            st.success("Visualizations generated!")

# Tab 4: History
with tab4:
    st.markdown('<div class="sub-header">📜 Recent Draw History</div>', unsafe_allow_html=True)

    dm, df = load_lottery_data(selected_lottery)

    num_history = st.slider("Number of Recent Draws", 5, 50, 15)
    recent = dm.get_recent_draws(num_history)

    # Format for display
    display_data = []
    for _, row in recent.iterrows():
        display_data.append({
            'Date': row['date'].strftime('%Y-%m-%d'),
            'Main Numbers': ' - '.join(f"{n:2d}" for n in row['main_numbers']),
            'Bonus': ' - '.join(f"{n:2d}" for n in row['bonus_numbers']) if row['bonus_numbers'] else 'N/A',
            'Jackpot': f"${row['jackpot']:,}"
        })

    history_df = pd.DataFrame(display_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    # Download history
    csv = history_df.to_csv(index=False)
    st.download_button(
        "📥 Download History as CSV",
        csv,
        file_name=f"{selected_lottery}_history.csv",
        mime="text/csv"
    )

# Tab 5: Compare
with tab5:
    st.markdown('<div class="sub-header">⚔️ Method Comparison</div>', unsafe_allow_html=True)

    compare_methods = st.multiselect(
        "Methods to Compare",
        ["statistical", "frequency", "cold", "pairs", "ml", "ensemble"],
        default=["statistical", "frequency", "cold", "ensemble"],
        format_func=lambda x: {
            "ensemble": "🎯 Ensemble",
            "statistical": "📊 Statistical",
            "frequency": "🔥 Frequency",
            "cold": "❄️ Cold",
            "pairs": "🔗 Pairs",
            "ml": "🤖 ML"
        }[x]
    )

    compare_count = st.slider("Predictions per Method", 1, 5, 2)

    if st.button("⚔️ Compare Methods", type="primary") and compare_methods:
        with st.spinner("Comparing methods..."):
            all_results = {}

            for m in compare_methods:
                preds = generate_predictions(
                    selected_lottery, m, compare_count, last_n_draws
                )
                all_results[m] = preds

            # Display comparison
            for method, preds in all_results.items():
                st.markdown(f"### {method.upper()}")

                for i, pred in enumerate(preds, 1):
                    bonus_str = ' - '.join(f"{n:2d}" for n in pred.bonus_numbers) if pred.bonus_numbers else 'N/A'
                    st.markdown(f"""
                    <div class="prediction-box">
                        <strong>#{i}:</strong> {' - '.join(f"{n:2d}" for n in pred.main_numbers)}
                        | Bonus: {bonus_str} | Confidence: {pred.confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)

            # Summary table
            st.markdown("### 📋 Summary")
            summary_data = []
            for method, preds in all_results.items():
                avg_conf = np.mean([p.confidence for p in preds])
                unique = len(set(tuple(p.main_numbers) for p in preds))
                summary_data.append({
                    'Method': method,
                    'Avg Confidence': f"{avg_conf:.1%}",
                    'Unique Predictions': unique,
                    'Predictions': compare_count
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Confidence comparison chart
            fig_comp = go.Figure()
            for method, preds in all_results.items():
                confs = [p.confidence for p in preds]
                fig_comp.add_trace(go.Box(
                    y=confs,
                    name=method,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
            fig_comp.update_layout(
                title="Confidence Distribution by Method",
                yaxis_title="Confidence",
                height=400
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🎲 Lottery Predictor | Built with Streamlit, Plotly, and ❤️</p>
    <p><strong>Play Responsibly</strong> | For entertainment purposes only</p>
</div>
""", unsafe_allow_html=True)