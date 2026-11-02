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

# PWA Meta Tags and Mobile Optimization
st.markdown("""
<!-- PWA Meta Tags -->
<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#00d4aa">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="LottoPredict">
<link rel="apple-touch-icon" href="icons/icon-192.png">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">

<!-- Service Worker Registration -->
<script>
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registered:', registration.scope);
                registration.addEventListener('updatefound', function() {
                    var newWorker = registration.installing;
                    newWorker.addEventListener('statechange', function() {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            // New content available, show update notification
                            if (confirm('New version available! Reload to update?')) {
                                window.location.reload();
                            }
                        }
                    });
                });
            })
            .catch(function(error) {
                console.log('ServiceWorker registration failed:', error);
            });
    });
}

// Detect standalone mode (PWA installed)
if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true) {
    document.body.classList.add('pwa-mode');
}
</script>
""", unsafe_allow_html=True)

# Custom CSS with Mobile Optimization
st.markdown("""
<style>
    /* Base styles */
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
        background-color: #856404;
        color: #ffffff;
        border: 1px solid #6d5303;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #ffffff;
        color: #212529;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #f8f9fa;
        color: #212529;
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

    /* ===== MOBILE OPTIMIZATION ===== */
    @media (max-width: 768px) {
        /* Larger touch targets */
        .stButton > button {
            height: 56px !important;
            font-size: 1.1rem !important;
            padding: 0 24px !important;
        }

        .stSelectbox > div > div {
            height: 56px !important;
            font-size: 1.1rem !important;
        }

        .stSlider > div > div > div > div {
            height: 48px !important;
        }

        .stNumberInput > div > div > input {
            height: 56px !important;
            font-size: 1.1rem !important;
        }

        /* Tabs - larger touch targets */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            height: 56px !important;
            min-width: 120px !important;
            font-size: 1rem !important;
            padding: 0 16px !important;
        }

        /* Sidebar - full width on mobile */
        .css-1d391kg, .css-1lcbmhc, .css-17lntkn {
            min-width: 100% !important;
        }

        /* Prediction boxes - larger text */
        .prediction-box {
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
        }

        .prediction-box p {
            font-size: 1.8rem !important;
            letter-spacing: 4px !important;
        }

        .prediction-box h4 {
            font-size: 1.3rem !important;
        }

        /* Metrics - larger touch targets */
        .stMetric {
            padding: 1rem !important;
        }

        .stMetric label {
            font-size: 1rem !important;
        }

        .stMetric div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }

        /* Headers */
        .main-header {
            font-size: 1.8rem !important;
        }

        .sub-header {
            font-size: 1.3rem !important;
        }

        /* Columns stack on mobile */
        .stColumns {
            flex-direction: column !important;
        }

        .stColumns > div {
            width: 100% !important;
            min-width: 100% !important;
        }

        /* Dataframes - horizontal scroll */
        .stDataFrame {
            overflow-x: auto !important;
        }

        /* Expander - larger touch target */
        .streamlit-expanderHeader {
            padding: 1rem !important;
            font-size: 1.1rem !important;
        }

        /* File uploader */
        .stFileUploader > div {
            padding: 2rem !important;
        }

        /* Radio/Checkbox - larger */
        .stRadio > div, .stCheckbox > div {
            padding: 0.5rem 0 !important;
        }

        .stRadio label, .stCheckbox label {
            font-size: 1.1rem !important;
            padding: 0.5rem !important;
        }
    }

    /* ===== BOTTOM NAVIGATION FOR PWA MODE ===== */
    @media (max-width: 768px) {
        .bottom-nav {
            display: none;
        }

        .pwa-mode .bottom-nav {
            display: flex !important;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #1a1a2e;
            border-top: 1px solid #333;
            z-index: 9999;
            justify-content: space-around;
            padding: 8px 0;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        }

        .bottom-nav a {
            color: #888;
            text-decoration: none;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 12px;
            min-width: 60px;
            transition: color 0.2s;
        }

        .bottom-nav a.active, .bottom-nav a:hover {
            color: #00d4aa;
        }

        .bottom-nav svg {
            width: 24px;
            height: 24px;
            margin-bottom: 4px;
        }

        .bottom-nav span {
            font-size: 0.7rem;
            font-weight: 500;
        }

        /* Add padding to bottom of page for nav */
        .pwa-mode .main .block-container {
            padding-bottom: 80px !important;
        }
    }

    /* ===== SWIPE GESTURE SUPPORT ===== */
    .swipe-container {
        touch-action: pan-y;
    }

    .swipe-container:hover {
        cursor: grab;
    }

    .swipe-container:active {
        cursor: grabbing;
    }

    /* ===== PWA MODE STYLES ===== */
    .pwa-mode .main-header {
        padding-top: 1rem;
    }

    .pwa-mode .stAppHeader {
        display: none !important;
    }

    /* ===== TOUCH FEEDBACK ===== */
    @media (hover: none) and (pointer: coarse) {
        .stButton > button:active,
        .stTabs [data-baseweb="tab"]:active,
        .stSelectbox > div:active {
            transform: scale(0.98);
            transition: transform 0.1s;
        }
    }

    /* ===== LANDSCAPE MOBILE ===== */
    @media (max-height: 500px) and (orientation: landscape) {
        .main-header {
            font-size: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px !important;
        }

        .stButton > button {
            height: 44px !important;
        }
    }

    /* ===== DARK MODE SUPPORT ===== */
    @media (prefers-color-scheme: dark) {
        .prediction-box {
            background-color: #2d2d3a;
            border-color: #444;
            color: #e0e0e0;
        }

        .metric-card {
            background-color: #2d2d3a;
            border-color: #444;
            color: #e0e0e0;
        }
    }
</style>

<!-- Bottom Navigation for PWA -->
<div class="bottom-nav" role="navigation" aria-label="Main navigation">
    <a href="/" class="nav-item" data-tab="predict" aria-label="Predictions">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
        <span>Predict</span>
    </a>
    <a href="/?tab=quickpick" class="nav-item" data-tab="quickpick" aria-label="Quick Pick">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 6v6l4 2"/>
        </svg>
        <span>Quick Pick</span>
    </a>
    <a href="/?tab=analyze" class="nav-item" data-tab="analyze" aria-label="Analysis">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 20V10M12 20V4M6 20v-6"/>
        </svg>
        <span>Analyze</span>
    </a>
    <a href="/?tab=settings" class="nav-item" data-tab="settings" aria-label="Settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        <span>Settings</span>
    </a>
</div>

<script>
// Bottom navigation active state
document.addEventListener('DOMContentLoaded', function() {
    const navItems = document.querySelectorAll('.nav-item');
    const currentPath = window.location.pathname + window.location.search;

    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (currentPath.includes(href.replace('/', '')) || (href === '/' && currentPath === '/')) {
            item.classList.add('active');
        }

        item.addEventListener('click', function(e) {
            navItems.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });
});

// Swipe gesture for tabs
let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
}, { passive: true });

document.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
}, { passive: true });

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) > swipeThreshold) {
        const tabs = document.querySelectorAll('.stTabs [data-baseweb="tab"]');
        const activeTab = document.querySelector('.stTabs [data-baseweb="tab"][aria-selected="true"]');

        if (activeTab && tabs.length > 1) {
            const activeIndex = Array.from(tabs).indexOf(activeTab);
            let newIndex;

            if (diff > 0) {
                // Swipe left - next tab
                newIndex = (activeIndex + 1) % tabs.length;
            } else {
                // Swipe right - previous tab
                newIndex = (activeIndex - 1 + tabs.length) % tabs.length;
            }

            tabs[newIndex].click();
        }
    }
}

// Prevent zoom on double tap for inputs
let lastTouchEnd = 0;
document.addEventListener('touchend', function(e) {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) {
        e.preventDefault();
    }
    lastTouchEnd = now;
}, { passive: false });
</script>
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

    # Lottery selection - filter out "no lottery" states
    games = [g for g in manager.list_games() if g.main_numbers > 0]
    game_options = {f"{g.name} ({g.key})": g.key for g in games}
    selected_game_label = st.selectbox(
        "🎰 Select Lottery",
        options=list(game_options.keys()),
        index=list(game_options.values()).index(st.session_state.current_lottery) if st.session_state.current_lottery in game_options.values() else 0
    )
    selected_lottery = game_options[selected_game_label]
    st.session_state.current_lottery = selected_lottery

    # Auto-detect location button
    if st.button("📍 Detect My Location", use_container_width=True):
        with st.spinner("Detecting your location..."):
            try:
                from lottery_predictor.universal import auto_select_lottery, detect_state_from_ip
                detected = auto_select_lottery()
                state = detect_state_from_ip()
                if detected and state:
                    # Find the index of the detected lottery
                    if detected in game_options.values():
                        detected_label = [k for k, v in game_options.items() if v == detected][0]
                        st.session_state.current_lottery = detected
                        st.success(f"Detected: {state} → {detected_label}")
                        st.rerun()
                else:
                    st.warning("Could not detect location. Using current selection.")
            except Exception as e:
                st.error(f"Location detection failed: {e}")

    game = manager.get_game(selected_lottery)

    # Show lottery info
    st.info(f"""
    **{game.name}**
    - Main: {game.main_numbers} numbers (1-{game.main_max})
    - Bonus: {game.bonus_numbers} numbers (1-{game.bonus_max})
    - Draws: {', '.join(game.draw_days)}
    - Since: {game.start_date}
    """)

    # Official website link
    if game.website:
        st.markdown(f"🔗 [Official Website]({game.website})", unsafe_allow_html=True)

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

    # Full history toggle (v2.4)
    use_all_history = st.checkbox("📚 Use All Historical Data (from lottery start date)", value=False, help="Analyze all available draws instead of recent N draws")

    if not use_all_history:
        last_n_draws = st.slider("Historical Draws to Analyze", 50, 1000, 200, step=50)
    else:
        last_n_draws = None
        st.info("Using all historical data from lottery start date")

    # Advanced settings
    with st.expander("Advanced Settings"):
        retrain = st.checkbox("Retrain ML Models", value=False)
        lookback = st.slider("Lookback Window (ML)", 5, 20, 10)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Predict",
    "📊 Analyze",
    "📈 Visualize",
    "📜 History",
    "⚔️ Compare",
    "👤 Profile & Social",
    "🏥 Data Health (v2.4)",
    "🔗 Cross-Lottery (v2.4)"
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
        if st.button("🎲 Generate Predictions", type="primary", width='stretch'):
            preds = generate_predictions(
                selected_lottery, method, num_predictions, last_n_draws, retrain, lookback
            )
            st.session_state.predictions = preds
            st.success(f"Generated {len(preds)} predictions!")

    with col2:
        if st.button("💾 Save to JSON", width='stretch'):
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

    # Full history toggle for analysis
    analyze_all_history = st.checkbox("📚 Use All Historical Data (from lottery start date)", value=(last_n_draws is None), key="analyze_all_history")

    if analyze_all_history:
        dm, df = load_lottery_data(selected_lottery, None)
        analysis_last_n = None
        st.info(f"Using all {len(df)} historical draws from {df['date'].min().strftime('%Y-%m-%d')}")
    else:
        dm, df = load_lottery_data(selected_lottery, last_n_draws)

    # Analysis controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if not analyze_all_history:
            analysis_last_n = st.slider("Draws to Analyze", 50, len(df), min(last_n_draws, len(df)), key="analyze_n")
        else:
            analysis_last_n = None
            st.write("Analyzing all draws")
    with col2:
        if st.button("🔄 Refresh Analysis", key="refresh_analyze"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        show_viz = st.checkbox("Show Charts", value=True)

    # Recalculate with selected draws
    if not analyze_all_history:
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
        st.dataframe(hot_df, width='stretch', hide_index=True)

    with col2:
        st.markdown("**❄️ Coldest Numbers**")
        cold_df = pd.DataFrame({
            'Number': cold,
            'Frequency': [freq[n] for n in cold]
        })
        st.dataframe(cold_df, width='stretch', hide_index=True)

    # Top Pairs
    st.subheader("🔗 Top Number Pairs")
    pairs = dm.get_pair_frequency(analysis_last_n)
    top_pairs = list(pairs.items())[:15]

    if top_pairs:
        pairs_df = pd.DataFrame({
            'Pair': [f"{a} - {b}" for (a, b), _ in top_pairs],
            'Count': [c for _, c in top_pairs]
        })
        st.dataframe(pairs_df, width='stretch', hide_index=True)

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
        st.plotly_chart(fig_freq, width='stretch')

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
        st.plotly_chart(fig_sum, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig_hc, width='stretch')

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
                st.plotly_chart(fig_heat, width='stretch')

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
    st.dataframe(history_df, width='stretch', hide_index=True)

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
            st.dataframe(summary_df, width='stretch', hide_index=True)

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
            st.plotly_chart(fig_comp, width='stretch')

# Tab 6: Profile & Social
with tab6:
    st.markdown('<div class="sub-header">👤 User Profile & Social</div>', unsafe_allow_html=True)

    # Import user profile manager
    try:
        from lottery_predictor.advanced import UserProfileManager, SocialManager
        user_manager = UserProfileManager()
        social_manager = SocialManager()
    except Exception as e:
        st.error(f"Could not load user features: {e}")
        user_manager = None
        social_manager = None

    if user_manager:
        # Initialize session state for user
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None

        # Load persisted current user
        if st.session_state.current_user is None and user_manager.current_user:
            st.session_state.current_user = user_manager.current_user

        # Login/Register section
        if st.session_state.current_user is None:
            st.subheader("🔐 Login / Create Profile")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Existing User**")
                usernames = user_manager.list_profiles()
                if usernames:
                    login_user = st.selectbox("Select User", usernames, key="login_user")
                    if st.button("Login", type="primary", use_container_width=True):
                        if user_manager.set_current_user(login_user):
                            st.session_state.current_user = login_user
                            st.success(f"Logged in as {login_user}")
                            st.rerun()
                        else:
                            st.error("Login failed")
                else:
                    st.info("No users found. Create a new profile.")

            with col2:
                st.markdown("**Create New Profile**")
                new_username = st.text_input("Username", key="new_username")
                new_email = st.text_input("Email (optional)", key="new_email")
                if st.button("Create Profile", type="secondary", use_container_width=True):
                    if new_username:
                        try:
                            profile = user_manager.create_profile(new_username, new_email if new_email else None)
                            user_manager.set_current_user(new_username)
                            st.session_state.current_user = new_username
                            st.success(f"Profile created for {new_username}")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.error("Please enter a username")

        else:
            # User is logged in - show profile
            profile = user_manager.get_current_profile()
            if profile:
                st.success(f"Logged in as **{profile.username}**")

                if st.button("🚪 Logout", use_container_width=True):
                    user_manager.current_user = None
                    user_manager._save_current_user()
                    st.session_state.current_user = None
                    st.rerun()

                # Profile tabs
                prof_tab1, prof_tab2, prof_tab3, prof_tab4, prof_tab5 = st.tabs([
                    "📋 Profile", "⭐ Favorites", "⚙️ Preferences", "📤 Share", "🏆 Leaderboard"
                ])

                with prof_tab1:
                    st.subheader("📋 Your Profile")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Username:** {profile.username}")
                        st.write(f"**Email:** {profile.email or 'Not set'}")
                        st.write(f"**Member since:** {profile.created_at[:10]}")
                        st.write(f"**Last login:** {profile.last_login[:10] if profile.last_login else 'Never'}")
                    with col2:
                        st.write(f"**Total Predictions:** {len(profile.prediction_history)}")
                        st.write(f"**Syndicates:** {len(profile.syndicate_memberships)}")
                        st.write(f"**Total Spent:** ${profile.total_spent:.2f}")
                        st.write(f"**Total Won:** ${profile.total_won:.2f}")
                        roi = user_manager.get_roi(profile.username)
                        if roi is not None:
                            st.write(f"**ROI:** {roi:+.2f}%")

                with prof_tab2:
                    st.subheader("⭐ Favorite Lotteries")

                    # Add favorite
                    games = [g for g in manager.list_games() if g.main_numbers > 0]
                    game_options = {f"{g.name} ({g.key})": g.key for g in games}

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fav_lottery = st.selectbox("Add Favorite", options=list(game_options.keys()), key="add_fav")
                    with col2:
                        if st.button("➕ Add", use_container_width=True):
                            selected_key = game_options[fav_lottery]
                            if user_manager.add_favorite_lottery(profile.username, selected_key):
                                st.success(f"Added {fav_lottery} to favorites!")
                                st.rerun()

                    # Show favorites
                    if profile.preferences.favorite_lotteries:
                        st.markdown("**Your Favorites:**")
                        for fav in profile.preferences.favorite_lotteries:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"⭐ {LOTTERY_CONFIGS[fav]['name']} ({fav})")
                            with col2:
                                if st.button("🗑️", key=f"remove_{fav}", use_container_width=True):
                                    if user_manager.remove_favorite_lottery(profile.username, fav):
                                        st.success(f"Removed {fav} from favorites")
                                        st.rerun()
                    else:
                        st.info("No favorites yet. Add some above!")

                with prof_tab3:
                    st.subheader("⚙️ Preferences")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Default lottery
                        default_lottery = st.selectbox(
                            "Default Lottery",
                            options=list(game_options.keys()),
                            index=list(game_options.values()).index(profile.preferences.default_lottery) if profile.preferences.default_lottery in game_options.values() else 0,
                            key="pref_default_lottery"
                        )

                        # Default method
                        methods = ["ensemble", "statistical", "frequency", "cold", "pairs", "ml", "time_series", "transformer", "online_learning"]
                        method_labels = {
                            "ensemble": "🎯 Ensemble",
                            "statistical": "📊 Statistical",
                            "frequency": "🔥 Frequency",
                            "cold": "❄️ Cold Numbers",
                            "pairs": "🔗 Pair-Based",
                            "ml": "🤖 Machine Learning",
                            "time_series": "📈 Time Series",
                            "transformer": "🔮 Transformer",
                            "online_learning": "🔄 Online Learning"
                        }
                        default_method = st.selectbox(
                            "Default Method",
                            options=methods,
                            format_func=lambda x: method_labels.get(x, x),
                            index=methods.index(profile.preferences.default_method) if profile.preferences.default_method in methods else 0,
                            key="pref_default_method"
                        )

                        # Default count
                        default_count = st.number_input(
                            "Default Prediction Count",
                            min_value=1, max_value=20,
                            value=profile.preferences.default_prediction_count,
                            key="pref_default_count"
                        )

                    with col2:
                        # Language
                        from lottery_predictor.advanced import I18nManager
                        langs = I18nManager.get_available_languages()
                        lang_labels = {"en": "English", "es": "Español", "fr": "Français", "de": "Deutsch", "zh": "中文"}
                        language = st.selectbox(
                            "Language",
                            options=langs,
                            format_func=lambda x: lang_labels.get(x, x),
                            index=langs.index(profile.preferences.language) if profile.preferences.language in langs else 0,
                            key="pref_language"
                        )

                        # Theme
                        theme = st.selectbox(
                            "Theme",
                            options=["system", "light", "dark"],
                            index=["system", "light", "dark"].index(profile.preferences.theme) if profile.preferences.theme in ["system", "light", "dark"] else 0,
                            key="pref_theme"
                        )

                        # Auto-detect
                        auto_detect = st.checkbox(
                            "Auto-detect lottery from location",
                            value=profile.preferences.auto_detect_state,
                            key="pref_auto_detect"
                        )

                        # All history
                        all_history = st.checkbox(
                            "Use all historical data by default",
                            value=profile.preferences.use_all_history,
                            key="pref_all_history"
                        )

                    if st.button("💾 Save Preferences", type="primary", use_container_width=True):
                        updates = {}
                        if game_options[default_lottery] != profile.preferences.default_lottery:
                            updates["default_lottery"] = game_options[default_lottery]
                        if default_method != profile.preferences.default_method:
                            updates["default_method"] = default_method
                        if default_count != profile.preferences.default_prediction_count:
                            updates["default_prediction_count"] = default_count
                        if language != profile.preferences.language:
                            updates["language"] = language
                        if theme != profile.preferences.theme:
                            updates["theme"] = theme
                        if auto_detect != profile.preferences.auto_detect_state:
                            updates["auto_detect_state"] = auto_detect
                        if all_history != profile.preferences.use_all_history:
                            updates["use_all_history"] = all_history

                        if updates:
                            if user_manager.update_preferences(profile.username, **updates):
                                st.success("Preferences saved!")
                                st.rerun()
                            else:
                                st.error("Failed to save preferences")
                        else:
                            st.info("No changes to save")

                with prof_tab4:
                    st.subheader("📤 Share Predictions")

                    share_lottery = st.selectbox(
                        "Lottery",
                        options=list(game_options.keys()),
                        index=list(game_options.values()).index(selected_lottery) if selected_lottery in game_options.values() else 0,
                        key="share_lottery"
                    )

                    share_method = st.selectbox(
                        "Method",
                        options=methods,
                        format_func=lambda x: method_labels.get(x, x),
                        key="share_method"
                    )

                    share_count = st.number_input("Number of Predictions", min_value=1, max_value=10, value=3, key="share_count")
                    share_anonymous = st.checkbox("Share Anonymously", key="share_anonymous")

                    if st.button("📤 Generate & Share", type="primary", use_container_width=True):
                        with st.spinner("Generating predictions..."):
                            try:
                                lottery_key = game_options[share_lottery]

                                # Generate predictions
                                from lottery_predictor.models import get_predictor, StatisticalPredictor, MLModelPredictor, TimeSeriesPredictor, EnsemblePredictor

                                if share_method == "statistical":
                                    pred = StatisticalPredictor(lottery_key)
                                    predictions = pred.predict_balanced(num_predictions=share_count)
                                elif share_method == "frequency":
                                    pred = StatisticalPredictor(lottery_key)
                                    predictions = pred.predict_frequency_based(num_predictions=share_count)
                                elif share_method == "cold":
                                    pred = StatisticalPredictor(lottery_key)
                                    predictions = pred.predict_cold_numbers(num_predictions=share_count)
                                elif share_method == "pairs":
                                    pred = StatisticalPredictor(lottery_key)
                                    predictions = pred.predict_pair_based(num_predictions=share_count)
                                elif share_method == "ml":
                                    pred = MLModelPredictor(lottery_key)
                                    if not pred.is_trained:
                                        pred.train()
                                    predictions = pred.predict(num_predictions=share_count)
                                elif share_method == "time_series":
                                    pred = TimeSeriesPredictor(lottery_key)
                                    predictions = pred.predict(num_predictions=share_count)
                                elif share_method == "transformer":
                                    from lottery_predictor.models import TransformerPredictor
                                    pred = TransformerPredictor(lottery_key)
                                    if pred.model is None:
                                        pred.train()
                                    predictions = pred.predict(num_predictions=share_count)
                                elif share_method == "online_learning":
                                    from lottery_predictor.models import OnlineLearningPredictor
                                    pred = OnlineLearningPredictor(lottery_key)
                                    predictions = pred.predict(num_predictions=share_count)
                                else:  # ensemble
                                    pred = EnsemblePredictor(lottery_key)
                                    predictions = pred.get_consensus_prediction(num_predictions=share_count)

                                # Convert to dict
                                pred_dicts = [
                                    {
                                        "main_numbers": [int(n) for n in p.main_numbers],
                                        "bonus_numbers": [int(n) for n in p.bonus_numbers] if p.bonus_numbers else [],
                                        "confidence": float(p.confidence),
                                        "method": p.method,
                                        "metadata": p.metadata
                                    }
                                    for p in predictions
                                ]

                                # Share
                                username = "Anonymous" if share_anonymous else profile.username
                                shared = social_manager.share_prediction(
                                    username=username,
                                    lottery=lottery_key,
                                    method=share_method,
                                    predictions=pred_dicts,
                                    is_anonymous=share_anonymous
                                )

                                st.success(f"Prediction shared! Share ID: {shared.id}")

                                # Show preview
                                for i, p in enumerate(pred_dicts, 1):
                                    bonus_str = ' - '.join(f"{n:2d}" for n in p["bonus_numbers"]) if p["bonus_numbers"] else 'N/A'
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <strong>#{i}:</strong> {' - '.join(f"{n:2d}" for n in p["main_numbers"])}
                                        | Bonus: {bonus_str} | Confidence: {p["confidence"]:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error sharing prediction: {e}")

                with prof_tab5:
                    st.subheader("🏆 Accuracy Leaderboard")

                    leaderboard_filter = st.selectbox(
                        "Filter by Lottery",
                        options=["All"] + list(game_options.values()),
                        format_func=lambda x: "All Lotteries" if x == "All" else LOTTERY_CONFIGS.get(x, {}).get('name', x),
                        key="lb_filter"
                    )

                    lb_data = social_manager.get_leaderboard(
                        lottery=None if leaderboard_filter == "All" else leaderboard_filter,
                        limit=20
                    )

                    if lb_data:
                        lb_df = pd.DataFrame(lb_data)
                        lb_df['Rank'] = range(1, len(lb_df) + 1)
                        lb_df = lb_df[['Rank', 'username', 'total_predictions', 'correct_predictions', 'accuracy']]
                        lb_df.columns = ['Rank', 'Username', 'Total Predictions', 'Correct', 'Accuracy %']
                        st.dataframe(lb_df, width='stretch', hide_index=True)
                    else:
                        st.info("No leaderboard data yet. Share predictions and verify them to compete!")

    # Community Pools section (visible to all)
    st.markdown("---")
    st.subheader("🤝 Community Prediction Pools")

    if social_manager:
        # Create pool form
        with st.expander("📝 Create New Pool"):
            if st.session_state.current_user:
                pool_name = st.text_input("Pool Name", key="pool_name")
                pool_lottery = st.selectbox(
                    "Lottery",
                    options=list(game_options.keys()),
                    key="pool_lottery"
                )
                pool_budget = st.number_input("Budget ($)", min_value=10.0, max_value=10000.0, value=100.0, step=10.0, key="pool_budget")
                pool_ticket_cost = st.number_input("Ticket Cost ($)", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="pool_ticket_cost")

                if st.button("Create Pool", type="primary", use_container_width=True):
                    if pool_name:
                        pool = social_manager.create_community_pool(
                            name=pool_name,
                            lottery=game_options[pool_lottery],
                            budget=pool_budget,
                            ticket_cost=pool_ticket_cost,
                            creator=st.session_state.current_user
                        )
                        st.success(f"Pool created! ID: {pool['id']}")
                        st.rerun()
                    else:
                        st.error("Please enter a pool name")
            else:
                st.warning("Please login to create a pool")

        # List pools
        pools = social_manager.list_pools(status="open")
        if pools:
            st.markdown("**Open Pools:**")
            for pool in pools:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.write(f"**{pool['name']}** (ID: {pool['id']})")
                        st.caption(f"Lottery: {LOTTERY_CONFIGS.get(pool['lottery'], {}).get('name', pool['lottery'])} | Budget: ${pool['budget']:.2f}")
                    with col2:
                        st.write(f"Participants: {len(pool['participants'])}")
                    with col3:
                        st.write(f"Created by: {pool['creator']}")
                    with col4:
                        if st.session_state.current_user:
                            contribution = st.number_input("Contribution", min_value=5.0, max_value=pool['budget'], value=10.0, step=5.0, key=f"contrib_{pool['id']}")
                            if st.button("Join", key=f"join_{pool['id']}", use_container_width=True):
                                if social_manager.join_pool(pool['id'], st.session_state.current_user, contribution):
                                    st.success(f"Joined {pool['name']}!")
                                    st.rerun()
                                else:
                                    st.error("Could not join pool")
                        else:
                            st.button("Login to Join", disabled=True, use_container_width=True)
        else:
            st.info("No open community pools. Create one above!")

# Tab 7: Data Health Dashboard (v2.4)
with tab7:
    st.markdown('<div class="sub-header">🏥 Historical Data Health Dashboard (v2.4)</div>', unsafe_allow_html=True)

    st.markdown("""
    Monitor data freshness, detect gaps in draw schedules, identify anomalies, and schedule backfills
    for missing historical data across all supported lotteries.
    """)

    from lottery_predictor.advanced import HistoricalDataHealthDashboard

    dashboard = HistoricalDataHealthDashboard()

    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        health_lottery = st.selectbox(
            "Select Lottery",
            options=["All Lotteries"] + [f"{g.name} ({g.key})" for g in manager.list_games() if g.main_numbers > 0],
            key="health_lottery"
        )
    with col2:
        auto_refresh = st.checkbox("Auto-refresh on load", value=True)
    with col3:
        if st.button("🔄 Analyze Health", type="primary", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    if health_lottery == "All Lotteries":
        with st.spinner("Analyzing health for all lotteries..."):
            health_data = dashboard.get_all_lotteries_health()

        if not health_data:
            st.warning("No health data available. Click 'Analyze Health' to run analysis.")
        else:
            # Summary metrics
            st.subheader("📊 Overall Health Summary")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Lotteries", len(health_data))
            with cols[1]:
                avg_completeness = np.mean([float(h['completeness_score'].rstrip('%'))/100 for h in health_data])
                st.metric("Avg Completeness", f"{avg_completeness:.1%}")
            with cols[2]:
                avg_freshness = np.mean([float(h['freshness_score'].rstrip('%'))/100 for h in health_data])
                st.metric("Avg Freshness", f"{avg_freshness:.1%}")
            with cols[3]:
                total_anomalies = sum(h['anomalies_count'] for h in health_data)
                st.metric("Total Anomalies", total_anomalies)

            # Health table
            st.subheader("📋 Detailed Health Report")
            health_df = pd.DataFrame(health_data)
            health_df['Lottery Name'] = health_df['lottery'].apply(lambda x: LOTTERY_CONFIGS.get(x, {}).get('name', x))
            display_cols = ['Lottery Name', 'total_draws', 'date_range', 'days_since_last_draw', 'expected_draws', 'missing_draws', 'completeness_score', 'freshness_score', 'gaps_count', 'anomalies_count']
            health_df_display = health_df[display_cols].copy()
            health_df_display.columns = ['Lottery', 'Total Draws', 'Date Range', 'Days Since Last', 'Expected', 'Missing', 'Completeness', 'Freshness', 'Gaps', 'Anomalies']
            st.dataframe(health_df_display, width='stretch', hide_index=True)

            # Download button
            csv = health_df_display.to_csv(index=False)
            st.download_button(
                "📥 Download Health Report (CSV)",
                csv,
                file_name="data_health_report.csv",
                mime="text/csv"
            )
    else:
        lottery_key = health_lottery.split("(")[-1].rstrip(")")
        with st.spinner(f"Analyzing health for {LOTTERY_CONFIGS[lottery_key]['name']}..."):
            record = dashboard.analyze_lottery_health(lottery_key)

        st.subheader(f"📋 Health Report: {LOTTERY_CONFIGS[lottery_key]['name']}")

        # Metrics
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Draws", record.total_draws)
        with cols[1]:
            st.metric("Days Since Last Draw", record.days_since_last_draw)
        with cols[2]:
            st.metric("Expected Draws", record.expected_draws)
        with cols[3]:
            st.metric("Missing Draws", record.missing_draws)
        with cols[4]:
            st.metric("Completeness", f"{record.completeness_score:.1%}")

        cols2 = st.columns(3)
        with cols2[0]:
            st.metric("Freshness", f"{record.freshness_score:.1%}")
        with cols2[1]:
            st.metric("Gaps Detected", len(record.gaps))
        with cols2[2]:
            st.metric("Anomalies", len(record.anomalies))

        st.write(f"**Date Range:** {record.date_range_start} to {record.date_range_end}")

        # Gaps
        if record.gaps:
            st.subheader("⚠️ Detected Gaps")
            gaps_df = pd.DataFrame(record.gaps)
            st.dataframe(gaps_df, width='stretch', hide_index=True)

            # Backfill button
            if st.button("📥 Schedule Backfill for Missing Data", type="secondary"):
                result = dashboard.schedule_backfill(lottery_key)
                st.success(f"Backfill scheduled: {result}")
        else:
            st.success("✅ No gaps detected in draw schedule")

        # Anomalies
        if record.anomalies:
            st.subheader("🔍 Detected Anomalies")
            anomalies_df = pd.DataFrame(record.anomalies)
            st.dataframe(anomalies_df, width='stretch', hide_index=True)
        else:
            st.success("✅ No anomalies detected")

# Tab 8: Cross-Lottery Correlation (v2.4)
with tab8:
    st.markdown('<div class="sub-header">🔗 Cross-Lottery Historical Correlation (v2.4)</div>', unsafe_allow_html=True)

    st.markdown("""
    Analyze historical correlations between different lottery games to discover patterns.
    Compare number frequencies, sum distributions, and odd/even patterns across lotteries.
    """)

    from lottery_predictor.advanced import CrossLotteryCorrelationAnalyzer

    analyzer = CrossLotteryCorrelationAnalyzer()

    # Mode selection
    mode = st.radio(
        "Analysis Mode",
        ["Compare Two Lotteries", "Find Correlated Lotteries"],
        horizontal=True
    )

    if mode == "Compare Two Lotteries":
        col1, col2 = st.columns(2)
        with col1:
            lotto_a = st.selectbox(
                "First Lottery",
                options=[f"{g.name} ({g.key})" for g in manager.list_games() if g.main_numbers > 0],
                index=0,
                key="corr_lotto_a"
            )
        with col2:
            lotto_b = st.selectbox(
                "Second Lottery",
                options=[f"{g.name} ({g.key})" for g in manager.list_games() if g.main_numbers > 0],
                index=1,
                key="corr_lotto_b"
            )

        lottery_a_key = lotto_a.split("(")[-1].rstrip(")")
        lottery_b_key = lotto_b.split("(")[-1].rstrip(")")

        if st.button("🔗 Analyze Correlations", type="primary", width='stretch'):
            if lottery_a_key == lottery_b_key:
                st.error("Please select two different lotteries")
            else:
                with st.spinner(f"Analyzing correlations between {LOTTERY_CONFIGS[lottery_a_key]['name']} and {LOTTERY_CONFIGS[lottery_b_key]['name']}..."):
                    results = analyzer.analyze_all_correlations(lottery_a_key, lottery_b_key)

                if results:
                    st.subheader(f"📊 Correlation Results: {LOTTERY_CONFIGS[lottery_a_key]['name']} vs {LOTTERY_CONFIGS[lottery_b_key]['name']}")

                    # Results table
                    corr_data = []
                    for r in results:
                        corr_data.append({
                            "Correlation Type": r.correlation_type.replace("_", " ").title(),
                            "Coefficient": f"{r.correlation_coefficient:.4f}",
                            "P-Value": f"{r.p_value:.4f}",
                            "Significant": "✅ Yes" if r.significant else "❌ No",
                            "Sample Size": r.sample_size,
                            "Details": str(r.details)
                        })

                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, width='stretch', hide_index=True)

                    # Interpretation
                    st.subheader("📖 Interpretation")
                    for r in results:
                        strength = "strong" if abs(r.correlation_coefficient) > 0.7 else "moderate" if abs(r.correlation_coefficient) > 0.3 else "weak"
                        direction = "positive" if r.correlation_coefficient > 0 else "negative"
                        sig_text = "statistically significant" if r.significant else "not statistically significant"
                        st.write(f"**{r.correlation_type.replace('_', ' ').title()}:** {strength.capitalize()} {direction} correlation ({sig_text}, p={r.p_value:.4f})")

                    # Visualization - correlation heatmap
                    st.subheader("📈 Correlation Visualization")
                    fig = go.Figure()
                    for r in results:
                        fig.add_trace(go.Bar(
                            name=r.correlation_type.replace("_", " ").title(),
                            x=[r.correlation_type.replace("_", " ").title()],
                            y=[r.correlation_coefficient],
                            text=[f"{r.correlation_coefficient:.3f}"],
                            textposition='auto'
                        ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.add_hline(y=0.3, line_dash="dot", line_color="green", annotation_text="Moderate")
                    fig.add_hline(y=-0.3, line_dash="dot", line_color="green")
                    fig.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="Strong")
                    fig.add_hline(y=-0.7, line_dash="dot", line_color="red")
                    fig.update_layout(
                        title=f"Correlation Coefficients: {LOTTERY_CONFIGS[lottery_a_key]['name']} vs {LOTTERY_CONFIGS[lottery_b_key]['name']}",
                        yaxis_title="Correlation Coefficient",
                        yaxis_range=[-1, 1],
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("No correlation results available. Try different lotteries.")

    else:  # Find Correlated Lotteries
        base_lotto = st.selectbox(
            "Base Lottery",
            options=[f"{g.name} ({g.key})" for g in manager.list_games() if g.main_numbers > 0],
            index=0,
            key="corr_base_lotto"
        )
        top_n = st.slider("Number of Top Correlations", 3, 10, 5)
        threshold = st.slider("Significance Threshold (p-value)", 0.01, 0.10, 0.05, 0.01)

        base_key = base_lotto.split("(")[-1].rstrip(")")

        if st.button("🔍 Find Correlated Lotteries", type="primary", width='stretch'):
            with st.spinner(f"Finding lotteries correlated with {LOTTERY_CONFIGS[base_key]['name']}..."):
                results = analyzer.find_correlated_lotteries(base_key, top_n)

            if results:
                st.subheader(f"🎯 Top {top_n} Lotteries Correlated with {LOTTERY_CONFIGS[base_key]['name']}")

                corr_data = []
                for i, r in enumerate(results, 1):
                    corr_data.append({
                        "Rank": i,
                        "Lottery": f"{r['name']} ({r['lottery']})",
                        "Avg Correlation": f"{r['avg_correlation']:.4f}",
                        "Min P-Value": f"{r['min_p_value']:.4f}",
                        "Significant": "✅ Yes" if r['significant'] else "❌ No",
                        "Details": ", ".join([f"{k}: {v['corr']:.3f}" for k, v in r['details'].items()])
                    })

                corr_df = pd.DataFrame(corr_data)
                st.dataframe(corr_df, width='stretch', hide_index=True)

                # Bar chart
                fig = px.bar(
                    corr_df,
                    x='Lottery',
                    y='Avg Correlation',
                    color='Significant',
                    title=f"Average Correlation with {LOTTERY_CONFIGS[base_key]['name']}",
                    color_discrete_map={"✅ Yes": "green", "❌ No": "red"}
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("No correlations found. Try a different base lottery.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🎲 Lottery Predictor | Built with Streamlit, Plotly, and ❤️</p>
    <p><strong>Play Responsibly</strong> | For entertainment purposes only</p>
</div>
""", unsafe_allow_html=True)