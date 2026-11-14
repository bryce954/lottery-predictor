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
    page_title="Lottery predictor",
    page_icon=":material/casino:",
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
// AGGRESSIVE Unicode arrow fix - removes ALL Material Icons ligature text
(function() {
  const replacements = {
    'keyboard_double_arrow_right': '▶',
    'keyboard_arrow_right': '▶',
    'expand_more': '▶',
    'chevron_right': '▶',
    'keyboard_arrow_left': '◀',
    'expand_less': '◀',
    'chevron_left': '◀',
    'keyboard_arrow_up': '▲',
    'keyboard_arrow_down': '▼',
    'arrow_forward': '▶',
    'arrow_back': '◀',
    'arrow_drop_down': '▼',
    'arrow_drop_up': '▲',
    'keyboard_arrow_down': '▼',
    'keyboard_arrow_up': '▲',
  };

  function fixTextNodes() {
    if (!document.body) return;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    let node;
    while (node = walker.nextNode()) {
      let text = node.textContent;
      let changed = false;
      for (const [search, replace] of Object.entries(replacements)) {
        if (text.includes(search)) {
          text = text.replace(new RegExp(search, 'g'), replace);
          changed = true;
        }
      }
      if (changed) {
        node.textContent = text;
      }
    }
  }

  function fixElementText() {
    // Also fix elements that might have ligature text as direct content
    document.querySelectorAll('*').forEach(el => {
      // Skip script/style/our debug elements
      if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE' || el.id === 'debug-overlay') return;

      // Check if element has only text content (no children)
      if (el.children.length === 0 && el.textContent) {
        let text = el.textContent;
        let changed = false;
        for (const [search, replace] of Object.entries(replacements)) {
          if (text.includes(search)) {
            text = text.replace(new RegExp(search, 'g'), replace);
            changed = true;
          }
        }
        if (changed) {
          el.textContent = text;
        }
      }
    });
  }

  function runFix() {
    fixTextNodes();
    fixElementText();
  }

  // Run immediately
  runFix();

  // Watch for dynamically added content
  const observer = new MutationObserver(runFix);
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });

  // Hook into Streamlit's render event
  document.addEventListener('streamlit:render', runFix);
  window.addEventListener('load', runFix);

  // Also run periodically as a safety net
  setInterval(runFix, 1000);
})();
</script>
""", unsafe_allow_html=True)

# Custom CSS with Mobile Optimization - using st.html() for Streamlit 1.66+
# Design System: LottoPredict - Modern Lottery Interface
# Colors: Deep indigo primary, warm gold accent, semantic status colors
# Typography: "Space Grotesk" for display, "DM Sans" for body, "JetBrains Mono" for numbers
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ============================================================
   NUCLEAR ICON FIX - Hide Material Icons, use Unicode instead
   This bypasses the font entirely - no ligatures, no loading issues
   ============================================================ */

  /* Hide ALL Material Icons font text - comprehensive selector */
  .material-icons,
  .material-icons-outlined,
  .material-symbols-outlined,
  [class*="material-icons"],
  [class*="material-symbols"],
  i[class*="material"],
  span[class*="material"] {
    font-size: 0 !important;
    display: inline-block !important;
    width: 1em !important;
    height: 1em !important;
    vertical-align: middle !important;
    overflow: hidden !important;
  }

  /* Expanders - use Unicode arrows via ::before */
  [data-testid="stExpander"] summary .material-icons::before,
  [data-testid="stExpander"] summary .material-icons-outlined::before,
  [data-testid="stExpander"] summary .material-symbols-outlined::before,
  [data-testid="stExpander"] summary [class*="material-icons"]::before {
    font-size: 0.875rem !important;
    font-family: system-ui, sans-serif !important;
    content: "▶" !important;
    display: inline-block !important;
    transition: transform 0.2s;
  }
  [data-testid="stExpander"][open] summary .material-icons::before,
  [data-testid="stExpander"][open] summary .material-icons-outlined::before,
  [data-testid="stExpander"][open] summary .material-symbols-outlined::before,
  [data-testid="stExpander"][open] summary [class*="material-icons"]::before {
    content: "▼" !important;
  }

  /* Native summary marker fallback */
  [data-testid="stExpander"] summary::marker {
    content: "▶" !important;
    font-family: system-ui !important;
    font-size: 0.75rem !important;
  }
  [data-testid="stExpander"][open] summary::marker {
    content: "▼" !important;
  }

  /* Selectbox / Multiselect - hide SVG, show Unicode */
  [data-testid="stSelectbox"] [data-baseweb="select"] svg,
  [data-testid="stMultiSelect"] [data-baseweb="select"] svg {
    display: none !important;
  }
  [data-testid="stSelectbox"] [data-baseweb="select"] .material-icons,
  [data-testid="stMultiSelect"] [data-baseweb="select"] .material-icons,
  [data-testid="stSelectbox"] [data-baseweb="select"] [class*="material-icons"],
  [data-testid="stMultiSelect"] [data-baseweb="select"] [class*="material-icons"] {
    font-size: 0 !important;
  }
  [data-testid="stSelectbox"] [data-baseweb="select"] .material-icons::before,
  [data-testid="stMultiSelect"] [data-baseweb="select"] .material-icons::before,
  [data-testid="stSelectbox"] [data-baseweb="select"] [class*="material-icons"]::before,
  [data-testid="stMultiSelect"] [data-baseweb="select"] [class*="material-icons"]::before {
    font-size: 0.7rem !important;
    font-family: system-ui, sans-serif !important;
    content: "▼" !important;
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
  }

  /* Sidebar collapse button */
  [data-testid="stSidebarCollapseButton"] .material-icons,
  [data-testid="stSidebarCollapseButton"] svg,
  [data-testid="stSidebarCollapseButton"] [class*="material-icons"] {
    display: none !important;
  }
  [data-testid="stSidebarCollapseButton"]::after {
    content: "◀" !important;
    font-family: system-ui !important;
    font-size: 1rem !important;
    display: inline-block;
  }

  /* Number input arrows */
  [data-testid="stNumberInput"] button .material-icons,
  [data-testid="stNumberInput"] button svg,
  [data-testid="stNumberInput"] button [class*="material-icons"] {
    display: none !important;
  }
  [data-testid="stNumberInput"] button:first-child::after {
    content: "▲" !important;
    font-family: system-ui !important;
    font-size: 0.6rem !important;
    display: inline-block;
  }
  [data-testid="stNumberInput"] button:last-child::after {
    content: "▼" !important;
    font-family: system-ui !important;
    font-size: 0.6rem !important;
    display: inline-block;
  }

  /* Generic material-icons spans - hide ligature text, show Unicode */
  span[class*="material-icons"] {
    font-size: 0 !important;
  }
  span[class*="material-icons"]::before {
    font-size: 1rem !important;
    font-family: system-ui, sans-serif !important;
  }

  /* Specific ligature replacements based on data-icon attribute */
  span[class*="material-icons"][data-icon="keyboard_double_arrow_right"]::before,
  span[class*="material-icons"][data-icon="keyboard_arrow_right"]::before,
  span[class*="material-icons"][data-icon="expand_more"]::before,
  span[class*="material-icons"][data-icon="chevron_right"]::before,
  span[class*="material-icons"][data-icon="arrow_forward"]::before,
  span[class*="material-icons"][data-icon="arrow_drop_down"]::before {
    content: "▶" !important;
  }
  span[class*="material-icons"][data-icon="keyboard_arrow_left"]::before,
  span[class*="material-icons"][data-icon="expand_less"]::before,
  span[class*="material-icons"][data-icon="chevron_left"]::before,
  span[class*="material-icons"][data-icon="arrow_back"]::before,
  span[class*="material-icons"][data-icon="arrow_drop_up"]::before {
    content: "◀" !important;
  }
  span[class*="material-icons"][data-icon="keyboard_arrow_up"]::before {
    content: "▲" !important;
  }
  span[class*="material-icons"][data-icon="keyboard_arrow_down"]::before {
    content: "▼" !important;
  }

  /* Fallback for when data-icon is not present - hide text, show bullet */
  .material-icons:not([data-icon]),
  .material-icons-outlined:not([data-icon]),
  .material-symbols-outlined:not([data-icon]),
  [class*="material-icons"]:not([data-icon]),
  [class*="material-symbols"]:not([data-icon]) {
    visibility: hidden;
  }
  .material-icons:not([data-icon])::before,
  .material-icons-outlined:not([data-icon])::before,
  .material-symbols-outlined:not([data-icon])::before,
  [class*="material-icons"]:not([data-icon])::before,
  [class*="material-symbols"]:not([data-icon])::before {
    visibility: visible;
    content: "•" !important;
    font-size: 1rem !important;
    font-family: system-ui, sans-serif !important;
    display: inline-block !important;
  }

  /* ULTRA-AGGRESSIVE: Hide any element containing ligature text patterns */
  /* This catches text nodes that might not have the material-icons class */
  [data-testid="stExpander"] summary,
  [data-testid="stSelectbox"],
  [data-testid="stMultiSelect"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stNumberInput"] button {
    /* Ensure no ligature text leaks through */
    font-variant-ligatures: none !important;
  }

/* ============================================================
   DESIGN TOKENS
   ============================================================ */

:root {
  /* Brand Colors */
  --color-primary: #2d3a8c;        /* Deep indigo */
  --color-primary-hover: #1e2a6b;
  --color-primary-light: #e8ebfa;
  --color-accent: #d4a82a;         /* Warm gold */
  --color-accent-hover: #c4961e;
  --color-accent-light: #fef7e0;

  /* Semantic Colors */
  --color-success: #059669;
  --color-success-light: #d1fae5;
  --color-warning: #d97706;
  --color-warning-light: #fef3c7;
  --color-error: #dc2626;
  --color-error-light: #fee2e2;
  --color-info: #2563eb;
  --color-info-light: #dbeafe;

  /* Neutral Colors (Light) */
  --bg-primary: #fafafa;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f5f5f5;
  --bg-elevated: #ffffff;
  --fg-primary: #111827;
  --fg-secondary: #4b5563;
  --fg-tertiary: #9ca3af;
  --fg-inverse: #ffffff;
  --border-light: #e5e7eb;
  --border-medium: #d1d5db;
  --border-focus: #2d3a8c;

  /* RGB versions for transparency */
  --bg-primary-rgb: 250, 250, 250;
  --bg-secondary-rgb: 255, 255, 255;
  --bg-tertiary-rgb: 245, 245, 245;
  --bg-elevated-rgb: 255, 255, 255;

  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgba(17, 24, 39, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(17, 24, 39, 0.1), 0 2px 4px -2px rgba(17, 24, 39, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(17, 24, 39, 0.1), 0 4px 6px -4px rgba(17, 24, 39, 0.1);
  --shadow-xl: 0 20px 25px -5px rgba(17, 24, 39, 0.1), 0 8px 10px -6px rgba(17, 24, 39, 0.1);

  /* Radius */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 24px;
  --radius-full: 9999px;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;

  /* Typography */
  --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-body: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;

  /* Transitions */
  --transition-fast: 120ms ease;
  --transition-normal: 200ms ease;
  --transition-slow: 300ms ease;
}

/* Dark Mode Tokens */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --color-primary: #4f5fd6;
    --color-primary-hover: #6b7ae0;
    --color-primary-light: #1e2a6b;
    --color-accent: #e8c54a;
    --color-accent-hover: #f0d16a;
    --color-accent-light: #3d3300;

    --color-success: #34d399;
    --color-success-light: #064e3b;
    --color-warning: #fbbf24;
    --color-warning-light: #422006;
    --color-error: #f87171;
    --color-error-light: #7f1d1d;
    --color-info: #60a5fa;
    --color-info-light: #1e3a5f;

    --bg-primary: #0f1115;
    --bg-secondary: #181a1f;
    --bg-tertiary: #1f2228;
    --bg-elevated: #1f2228;
    --fg-primary: #f3f4f6;
    --fg-secondary: #d1d5db;
    --fg-tertiary: #9ca3af;
    --fg-inverse: #111827;
    --border-light: #2d3139;
    --border-medium: #3d424c;
    --border-focus: #4f5fd6;

    /* RGB versions for transparency */
    --bg-primary-rgb: 15, 17, 21;
    --bg-secondary-rgb: 24, 26, 31;
    --bg-tertiary-rgb: 31, 34, 40;
    --bg-elevated-rgb: 31, 34, 40;

    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);

    color-scheme: dark;
  }
}

:root[data-theme="dark"] {
  --color-primary: #4f5fd6;
  --color-primary-hover: #6b7ae0;
  --color-primary-light: #1e2a6b;
  --color-accent: #e8c54a;
  --color-accent-hover: #f0d16a;
  --color-accent-light: #3d3300;

  --color-success: #34d399;
  --color-success-light: #064e3b;
  --color-warning: #fbbf24;
  --color-warning-light: #422006;
  --color-error: #f87171;
  --color-error-light: #7f1d1d;
  --color-info: #60a5fa;
  --color-info-light: #1e3a5f;

  --bg-primary: #0f1115;
  --bg-secondary: #181a1f;
  --bg-tertiary: #1f2228;
  --bg-elevated: #1f2228;
  --fg-primary: #f3f4f6;
  --fg-secondary: #d1d5db;
  --fg-tertiary: #9ca3af;
  --fg-inverse: #111827;
  --border-light: #2d3139;
  --border-medium: #3d424c;
  --border-focus: #4f5fd6;

  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);

  color-scheme: dark;
}

/* ============================================================
   BASE STYLES
   ============================================================ */
* {
  box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: var(--font-body) !important;
  background-color: var(--bg-primary) !important;
  color: var(--fg-primary) !important;
}

/* Main container - semi-transparent to show background particles */
.main .block-container {
  padding-top: var(--space-6) !important;
  padding-bottom: var(--space-8) !important;
  max-width: 1300px !important;
  background-color: rgba(var(--bg-primary-rgb), 0.95) !important;
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-xl);
}

/* Typography */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  font-family: var(--font-display) !important;
  font-weight: 600 !important;
  line-height: 1.2 !important;
  color: var(--fg-primary) !important;
}

.main-header {
  font-family: var(--font-display) !important;
  font-size: clamp(2rem, 5vw, 3rem) !important;
  font-weight: 700 !important;
  color: var(--color-primary) !important;
  text-align: center !important;
  margin-bottom: var(--space-4) !important;
  letter-spacing: -0.02em;
}

.main-header::after {
  content: "";
  display: block;
  width: 60px;
  height: 4px;
  margin: var(--space-3) auto 0;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: var(--radius-full);
}

.sub-header {
  font-family: var(--font-display) !important;
  font-size: clamp(1.25rem, 3vw, 1.75rem) !important;
  font-weight: 600 !important;
  color: var(--fg-primary) !important;
  margin-top: var(--space-8) !important;
  margin-bottom: var(--space-4) !important;
  padding-bottom: var(--space-2);
  border-bottom: 2px solid var(--border-light);
}

p, li, span, div {
  font-family: var(--font-body) !important;
  line-height: 1.6;
}

code, pre, .prediction-numbers {
  font-family: var(--font-mono) !important;
}

/* ============================================================
   COMPONENT: CARDS
   ============================================================ */
.card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--transition-normal), transform var(--transition-normal), border-color var(--transition-normal);
}

.card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-medium);
}

.card-interactive {
  cursor: pointer;
}

.card-interactive:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
  border-color: var(--color-primary);
}

.card-primary {
  border-color: var(--color-primary);
  background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--color-primary-light) 100%);
}

.card-accent {
  border-color: var(--color-accent);
  background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--color-accent-light) 100%);
}

/* ============================================================
   COMPONENT: PREDICTION BOX
   ============================================================ */
.prediction-box {
  background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--bg-tertiary) 100%);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-5) var(--space-6);
  margin: var(--space-3) 0;
  box-shadow:
    var(--shadow-sm),
    inset 0 1px 0 rgba(255,255,255,0.1);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.prediction-box::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(180deg, var(--color-primary), var(--color-accent));
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.prediction-box::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  width: 80px;
  height: 80px;
  background: radial-gradient(circle at top right, var(--color-accent-light) 0%, transparent 70%);
  border-radius: 0 0 0 80px;
  pointer-events: none;
  opacity: 0.3;
}

.prediction-box:hover {
  box-shadow:
    var(--shadow-lg),
    0 0 30px rgba(45, 58, 140, 0.15);
  border-color: var(--color-primary);
  transform: translateY(-2px);
}

.prediction-box:hover::before {
  opacity: 1;
}

.prediction-box.ticket-card {
  background: linear-gradient(135deg, #fff8e1 0%, #fff3c4 100%);
  border: 2px dashed var(--color-accent);
  box-shadow:
    0 4px 20px rgba(218, 165, 32, 0.15),
    inset 0 1px 0 rgba(255,255,255,0.2);
}

@media (prefers-color-scheme: dark) {
  .prediction-box.ticket-card {
    background: linear-gradient(135deg, #3d3300 0%, #2d2200 100%);
    border-color: var(--color-accent);
  }
}

.prediction-box h4 {
  font-family: var(--font-display) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  color: var(--fg-secondary) !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: var(--space-3) !important;
}

.prediction-numbers {
  font-size: 1.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  color: var(--fg-primary) !important;
  line-height: 1.4;
  word-break: break-all;
}

.prediction-bonus {
  font-size: 1.125rem !important;
  font-weight: 500 !important;
  color: var(--color-accent) !important;
  margin-top: var(--space-2) !important;
}

.prediction-meta {
  font-size: 0.8125rem !important;
  color: var(--fg-tertiary) !important;
  margin-top: var(--space-3) !important;
  display: flex;
  gap: var(--space-4);
  flex-wrap: wrap;
}

.prediction-meta strong {
  color: var(--fg-secondary);
}

/* Confidence badge */
.confidence-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-full);
  font-size: 0.75rem;
  font-weight: 600;
  font-family: var(--font-body);
}

.confidence-high { background: var(--color-success-light); color: var(--color-success); }
.confidence-medium { background: var(--color-warning-light); color: var(--color-warning); }
.confidence-low { background: var(--color-error-light); color: var(--color-error); }

/* ============================================================
   COMPONENT: DISCLAIMER
   ============================================================ */
.disclaimer {
  background: var(--color-warning-light);
  border: 1px solid var(--color-warning);
  border-radius: var(--radius-lg);
  padding: var(--space-5) var(--space-6);
  margin: var(--space-6) 0;
  color: #78350f !important;
  position: relative;
}

.disclaimer::before {
  content: "⚠️";
  margin-right: var(--space-3);
  font-size: 1.25rem;
}

.disclaimer strong {
  color: #92400e !important;
}

.disclaimer p {
  margin: 0 !important;
}

/* ============================================================
   COMPONENT: METRIC CARDS
   ============================================================ */
.metric-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  text-align: center;
  transition: all var(--transition-normal);
}

.metric-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--color-primary);
}

.metric-card .metric-label {
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-tertiary) !important;
  margin-bottom: var(--space-2) !important;
  font-family: var(--font-body) !important;
}

.metric-card .metric-value {
  font-family: var(--font-display) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--fg-primary) !important;
  line-height: 1.2;
}

.metric-card .metric-delta {
  font-size: 0.8125rem !important;
  margin-top: var(--space-2);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-1);
}

/* Override Streamlit metrics */
[data-testid="stMetric"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-5) !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stMetric"]:hover {
  box-shadow: var(--shadow-md) !important;
  border-color: var(--color-primary) !important;
}

[data-testid="stMetric"] label {
  font-family: var(--font-body) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em !important;
  color: var(--fg-tertiary) !important;
}

[data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--fg-primary) !important;
  line-height: 1.2 !important;
}

[data-testid="stMetricDelta"] {
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
}

/* ============================================================
   COMPONENT: BUTTONS
   ============================================================ */
[data-testid="stButton"] > button {
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-3) var(--space-6) !important;
  height: auto !important;
  min-height: 44px !important;
  transition: all var(--transition-fast) !important;
  border: none !important;
  cursor: pointer !important;
}

[data-testid="stButton"] > button[kind="primary"] {
  background: var(--color-primary) !important;
  color: var(--fg-inverse) !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
  background: var(--color-primary-hover) !important;
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stButton"] > button[kind="primary"]:active {
  transform: translateY(0) !important;
  background: var(--color-primary-hover) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
  background: var(--bg-tertiary) !important;
  color: var(--fg-primary) !important;
  border: 1px solid var(--border-medium) !important;
}

[data-testid="stButton"] > button[kind="secondary"]:hover {
  background: var(--border-medium) !important;
  border-color: var(--color-primary) !important;
}

[data-testid="stButton"] > button:disabled {
  opacity: 0.5 !important;
  cursor: not-allowed !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-3) var(--space-5) !important;
  background: var(--bg-tertiary) !important;
  color: var(--fg-secondary) !important;
  border: 1px solid var(--border-medium) !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stDownloadButton"] > button:hover {
  background: var(--color-primary) !important;
  color: var(--fg-inverse) !important;
  border-color: var(--color-primary) !important;
}

/* ============================================================
   COMPONENT: FORM ELEMENTS
   ============================================================ */
/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-medium) !important;
  border-radius: var(--radius-md) !important;
  min-height: 44px !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stSelectbox"] > div > div:hover {
  border-color: var(--color-primary) !important;
}

[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 3px var(--color-primary-light) !important;
}

[data-testid="stSelectbox"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
  height: 40px !important;
}

[data-testid="stSlider"] [role="slider"] {
  background: var(--color-primary) !important;
  width: 20px !important;
  height: 20px !important;
  border-radius: 50% !important;
  box-shadow: var(--shadow-md) !important;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast) !important;
}

[data-testid="stSlider"] [role="slider"]:hover {
  transform: scale(1.2) !important;
  box-shadow: var(--shadow-lg) !important;
}

[data-testid="stSlider"] [role="slider"]:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 4px var(--color-primary-light), var(--shadow-lg) !important;
}

[data-testid="stSlider"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* Number Input */
[data-testid="stNumberInput"] > div > div > input {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-medium) !important;
  border-radius: var(--radius-md) !important;
  min-height: 44px !important;
  padding: var(--space-2) var(--space-4) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.875rem !important;
  color: var(--fg-primary) !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stNumberInput"] > div > div > input:hover {
  border-color: var(--color-primary) !important;
}

[data-testid="stNumberInput"] > div > div > input:focus {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 3px var(--color-primary-light) !important;
  outline: none !important;
}

[data-testid="stNumberInput"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* Text Input */
[data-testid="stTextInput"] > div > div > input {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-medium) !important;
  border-radius: var(--radius-md) !important;
  min-height: 44px !important;
  padding: var(--space-2) var(--space-4) !important;
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  color: var(--fg-primary) !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stTextInput"] > div > div > input:hover {
  border-color: var(--color-primary) !important;
}

[data-testid="stTextInput"] > div > div > input:focus {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 3px var(--color-primary-light) !important;
  outline: none !important;
}

[data-testid="stTextInput"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] > div {
  padding: var(--space-1) 0 !important;
}

[data-testid="stCheckbox"] label {
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  color: var(--fg-primary) !important;
}

[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p {
  font-family: var(--font-body) !important;
}

[data-testid="stCheckbox"] input[type="checkbox"] {
  accent-color: var(--color-primary) !important;
  width: 20px !important;
  height: 20px !important;
}

/* Radio */
[data-testid="stRadio"] > div {
  padding: var(--space-2) 0 !important;
}

[data-testid="stRadio"] label {
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  color: var(--fg-primary) !important;
  padding: var(--space-2) var(--space-3) !important;
  border-radius: var(--radius-md) !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stRadio"] label:hover {
  background: var(--color-primary-light) !important;
}

[data-testid="stRadio"] input[type="radio"] {
  accent-color: var(--color-primary) !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-medium) !important;
  border-radius: var(--radius-md) !important;
  min-height: 44px !important;
  transition: all var(--transition-fast) !important;
}

[data-testid="stMultiSelect"] > div > div:hover {
  border-color: var(--color-primary) !important;
}

[data-testid="stMultiSelect"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] > div {
  background: var(--bg-elevated) !important;
  border: 2px dashed var(--border-medium) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-8) !important;
  transition: all var(--transition-normal) !important;
}

[data-testid="stFileUploader"] > div:hover {
  border-color: var(--color-primary) !important;
  background: var(--color-primary-light) !important;
}

[data-testid="stFileUploader"] label {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  margin-bottom: var(--space-2) !important;
}

/* ============================================================
   COMPONENT: TABS
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {
  gap: var(--space-2) !important;
  background: var(--bg-tertiary) !important;
  padding: var(--space-2) !important;
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--border-light) !important;
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  color: var(--fg-secondary) !important;
  background: transparent !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-3) var(--space-5) !important;
  height: auto !important;
  min-height: 44px !important;
  transition: all var(--transition-fast) !important;
  border: none !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--fg-primary) !important;
  background: var(--bg-elevated) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--fg-inverse) !important;
  background: var(--color-primary) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
  background: var(--color-primary-hover) !important;
}

/* ============================================================
   COMPONENT: SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
  min-width: 320px !important;
  max-width: 400px !important;
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  margin-bottom: var(--space-3) !important;
}

[data-testid="stSidebar"] [data-testid="stSubheader"],
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: var(--font-display) !important;
  font-size: 1.125rem !important;
  font-weight: 600 !important;
  color: var(--fg-primary) !important;
  margin-top: var(--space-6) !important;
  margin-bottom: var(--space-3) !important;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border-light);
}

[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] .stInfo,
[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stWarning,
[data-testid="stSidebar"] .stError {
  font-size: 0.8125rem !important;
  padding: var(--space-3) var(--space-4) !important;
  border-radius: var(--radius-md) !important;
  border: none !important;
}

[data-testid="stSidebar"] .stInfo {
  background: var(--color-info-light) !important;
  color: var(--color-info) !important;
}

[data-testid="stSidebar"] .stSuccess {
  background: var(--color-success-light) !important;
  color: var(--color-success) !important;
}

[data-testid="stSidebar"] .stWarning {
  background: var(--color-warning-light) !important;
  color: var(--color-warning) !important;
}

[data-testid="stSidebar"] .stError {
  background: var(--color-error-light) !important;
  color: var(--color-error) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important;
  line-height: 1.6;
}

/* ============================================================
   COMPONENT: EXPANDER
   ============================================================ */
[data-testid="stExpander"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stExpander"] summary {
  font-family: var(--font-body) !important;
  font-size: 0.9375rem !important;
  font-weight: 600 !important;
  color: var(--fg-primary) !important;
  padding: var(--space-4) var(--space-5) !important;
  background: var(--bg-tertiary) !important;
  border-bottom: 1px solid var(--border-light) !important;
  cursor: pointer !important;
  transition: background var(--transition-fast) !important;
}

[data-testid="stExpander"] summary:hover {
  background: var(--border-light) !important;
}

[data-testid="stExpander"] summary::marker {
  color: var(--fg-tertiary) !important;
}

[data-testid="stExpander"][open] summary {
  border-bottom-color: var(--border-light) !important;
}

[data-testid="stExpander"] > div:last-child {
  padding: var(--space-5) !important;
}

/* ============================================================
   COMPONENT: DATAFRAME
   ============================================================ */
[data-testid="stDataFrame"] {
  font-family: var(--font-body) !important;
  font-size: 0.8125rem !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  border: 1px solid var(--border-light) !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stDataFrame"] [data-testid="stTable"] {
  font-family: var(--font-body) !important;
}

[data-testid="stDataFrame"] th {
  font-family: var(--font-display) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--fg-tertiary) !important;
  background: var(--bg-tertiary) !important;
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 1px solid var(--border-light) !important;
}

[data-testid="stDataFrame"] td {
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 1px solid var(--border-light) !important;
  color: var(--fg-primary) !important;
}

[data-testid="stDataFrame"] tr:last-child td {
  border-bottom: none !important;
}

[data-testid="stDataFrame"] tr:hover td {
  background: var(--color-primary-light) !important;
}

/* ============================================================
   COMPONENT: ALERTS / NOTIFICATIONS
   ============================================================ */
.stAlert {
  border-radius: var(--radius-lg) !important;
  border: none !important;
  padding: var(--space-4) var(--space-5) !important;
  font-size: 0.875rem !important;
  box-shadow: var(--shadow-sm) !important;
}

.stInfo {
  background: var(--color-info-light) !important;
  color: var(--color-info) !important;
  border-left: 4px solid var(--color-info) !important;
}

.stSuccess {
  background: var(--color-success-light) !important;
  color: var(--color-success) !important;
  border-left: 4px solid var(--color-success) !important;
}

.stWarning {
  background: var(--color-warning-light) !important;
  color: var(--color-warning) !important;
  border-left: 4px solid var(--color-warning) !important;
}

.stError {
  background: var(--color-error-light) !important;
  color: var(--color-error) !important;
  border-left: 4px solid var(--color-error) !important;
}

/* ============================================================
   COMPONENT: PROGRESS / SPINNER
   ============================================================ */
.stProgress > div > div {
  background: var(--color-primary) !important;
  border-radius: var(--radius-full) !important;
}

.stSpinner > div {
  border-color: var(--color-primary) transparent transparent transparent !important;
}

/* ============================================================
   COMPONENT: COLUMNS
   ============================================================ */
.stColumns {
  gap: var(--space-4) !important;
}

.stColumns > div {
  min-width: 0 !important;
}

/* ============================================================
   ANIMATIONS
   ============================================================ */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
  from { opacity: 0; transform: translateX(-16px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.animate-fade-in {
  animation: fadeIn var(--transition-normal) ease forwards;
}

.animate-slide-in {
  animation: slideIn var(--transition-normal) ease forwards;
}

.animate-pulse {
  animation: pulse 2s ease-in-out infinite;
}

/* Staggered animation for prediction cards */
.prediction-box:nth-child(1) { animation-delay: 0ms; }
.prediction-box:nth-child(2) { animation-delay: 80ms; }
.prediction-box:nth-child(3) { animation-delay: 160ms; }
.prediction-box:nth-child(4) { animation-delay: 240ms; }
.prediction-box:nth-child(5) { animation-delay: 320ms; }
.prediction-box:nth-child(6) { animation-delay: 400ms; }
.prediction-box:nth-child(7) { animation-delay: 480ms; }
.prediction-box:nth-child(8) { animation-delay: 560ms; }
.prediction-box:nth-child(9) { animation-delay: 640ms; }
.prediction-box:nth-child(10) { animation-delay: 720ms; }

/* ============================================================
   RESPONSIVE: MOBILE (< 768px)
   ============================================================ */
@media (max-width: 768px) {
  .main .block-container {
    padding-top: var(--space-4) !important;
    padding-bottom: var(--space-6) !important;
    padding-left: var(--space-4) !important;
    padding-right: var(--space-4) !important;
  }

  .main-header {
    font-size: 2rem !important;
  }

  .sub-header {
    font-size: 1.25rem !important;
  }

  /* Larger touch targets */
  [data-testid="stButton"] > button {
    min-height: 52px !important;
    font-size: 1rem !important;
    padding: var(--space-3) var(--space-6) !important;
  }

  [data-testid="stSelectbox"] > div > div,
  [data-testid="stNumberInput"] > div > div > input,
  [data-testid="stTextInput"] > div > div > input,
  [data-testid="stMultiSelect"] > div > div {
    min-height: 52px !important;
    font-size: 1rem !important;
  }

  [data-testid="stSlider"] > div > div > div > div {
    height: 52px !important;
  }

  [data-testid="stSlider"] [role="slider"] {
    width: 28px !important;
    height: 28px !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
  }

  .stTabs [data-baseweb="tab"] {
    min-width: 120px !important;
    font-size: 0.875rem !important;
    padding: var(--space-3) var(--space-4) !important;
    min-height: 48px !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    min-width: 100% !important;
    max-width: 100% !important;
  }

  /* Prediction boxes */
  .prediction-box {
    padding: var(--space-5) !important;
  }

  .prediction-numbers {
    font-size: 1.5rem !important;
    letter-spacing: 0.08em !important;
  }

  .prediction-meta {
    flex-direction: column;
    gap: var(--space-2) !important;
  }

  /* Metrics */
  [data-testid="stMetric"] {
    padding: var(--space-4) !important;
  }

  [data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
  }

  /* Columns stack */
  .stColumns {
    flex-direction: column !important;
  }

  .stColumns > div {
    width: 100% !important;
    min-width: 100% !important;
  }

  /* Dataframe scroll */
  [data-testid="stDataFrame"] {
    overflow-x: auto !important;
  }

  /* Expander */
  [data-testid="stExpander"] summary {
    padding: var(--space-4) !important;
    font-size: 1rem !important;
  }

  /* Radio/Checkbox */
  [data-testid="stRadio"] label,
  [data-testid="stCheckbox"] label {
    font-size: 1rem !important;
    padding: var(--space-3) !important;
  }
}

/* ============================================================
   RESPONSIVE: LANDSCAPE MOBILE
   ============================================================ */
@media (max-height: 500px) and (orientation: landscape) {
  .main-header {
    font-size: 1.5rem !important;
    margin-bottom: var(--space-2) !important;
  }

  .stTabs [data-baseweb="tab"] {
    min-height: 40px !important;
    padding: var(--space-2) var(--space-4) !important;
  }

  [data-testid="stButton"] > button {
    min-height: 40px !important;
  }
}

/* ============================================================
   PWA MODE
   ============================================================ */
.pwa-mode .main-header {
  padding-top: var(--space-4);
}

.pwa-mode .stAppHeader {
  display: none !important;
}

/* Bottom nav for PWA */
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
    background: var(--bg-elevated) !important;
    border-top: 1px solid var(--border-light) !important;
    z-index: 9999;
    justify-content: space-around;
    padding: var(--space-2) 0;
    box-shadow: var(--shadow-xl) !important;
  }

  .bottom-nav a {
    color: var(--fg-tertiary) !important;
    text-decoration: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--space-2) var(--space-3);
    min-width: 64px;
    transition: color var(--transition-fast);
    border-radius: var(--radius-md);
  }

  .bottom-nav a.active,
  .bottom-nav a:hover {
    color: var(--color-primary) !important;
    background: var(--color-primary-light) !important;
  }

  .bottom-nav svg {
    width: 24px;
    height: 24px;
    margin-bottom: var(--space-1);
  }

  .bottom-nav span {
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
  }

  .pwa-mode .main .block-container {
    padding-bottom: 88px !important;
  }
}

/* ============================================================
   TOUCH FEEDBACK
   ============================================================ */
@media (hover: none) and (pointer: coarse) {
  [data-testid="stButton"] > button:active,
  .stTabs [data-baseweb="tab"]:active,
  [data-testid="stSelectbox"] > div:active,
  [data-testid="stCheckbox"]:active,
  [data-testid="stRadio"] label:active {
    transform: scale(0.98);
    transition: transform 80ms;
  }
}

/* ============================================================
   SCROLLBAR STYLING
   ============================================================ */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
  background: var(--border-medium);
  border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--fg-tertiary);
}

::-webkit-scrollbar-corner {
  background: var(--bg-tertiary);
}

/* ============================================================
   FOCUS VISIBLE (Accessibility)
   ============================================================ */
*:focus-visible {
  outline: 2px solid var(--color-primary) !important;
  outline-offset: 2px !important;
}

[data-testid="stButton"] > button:focus-visible,
[data-testid="stSelectbox"] > div > div:focus-visible,
[data-testid="stNumberInput"] > div > div > input:focus-visible,
[data-testid="stTextInput"] > div > div > input:focus-visible,
.stTabs [data-baseweb="tab"]:focus-visible {
  outline: 2px solid var(--color-primary) !important;
  outline-offset: 2px !important;
}

/* ============================================================
   REDUCED MOTION
   ============================================================ */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* ============================================================
   PRINT STYLES
   ============================================================ */
@media print {
  [data-testid="stSidebar"],
  [data-testid="stButton"],
  [data-testid="stDownloadButton"],
  .stTabs [data-baseweb="tab-list"],
  .bottom-nav {
    display: none !important;
  }

  .main .block-container {
    max-width: none !important;
    padding: 0 !important;
  }

  .prediction-box {
    break-inside: avoid;
    box-shadow: none;
    border: 1px solid #ccc;
  }
}


/* ============================================================
   VISUAL FLAIR - LOTTERY THEME
   ============================================================ */

/* Animated background particles */
@keyframes floatUp {
  0% { transform: translateY(100vh) scale(0); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(-10vh) scale(1); opacity: 0; }
}

@keyframes floatUpReverse {
  0% { transform: translateY(-10vh) scale(0); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(100vh) scale(1); opacity: 0; }
}

@keyframes sparkle {
  0%, 100% { opacity: 0; transform: scale(0) rotate(0deg); }
  50% { opacity: 1; transform: scale(1) rotate(180deg); }
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 20px rgba(212, 168, 42, 0.3), 0 0 40px rgba(45, 58, 140, 0.2); }
  50% { box-shadow: 0 0 40px rgba(212, 168, 42, 0.5), 0 0 80px rgba(45, 58, 140, 0.4); }
}

@keyframes jackpotCount {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}

@keyframes ballPop {
  0% { transform: scale(0) rotate(-180deg); opacity: 0; }
  50% { transform: scale(1.2) rotate(10deg); }
  100% { transform: scale(1) rotate(0deg); opacity: 1; }
}

@keyframes shimmerSweep {
  0% { background-position: -200% center; }
  100% { background-position: 200% center; }
}

/* Background particle container */
.lottery-bg {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  z-index: -1;
  overflow: hidden;
  opacity: 0.15;
}

.lottery-bg::before,
.lottery-bg::after {
  content: "";
  position: absolute;
  width: 200%;
  height: 200%;
  background-image: 
    radial-gradient(circle at 20% 30%, var(--color-primary) 2px, transparent 2px),
    radial-gradient(circle at 80% 70%, var(--color-accent) 2px, transparent 2px),
    radial-gradient(circle at 40% 80%, var(--color-primary) 1px, transparent 1px),
    radial-gradient(circle at 60% 20%, var(--color-accent) 1px, transparent 1px);
  background-size: 300px 300px;
  animation: floatUp 60s linear infinite;
}

.lottery-bg::after {
  animation: floatUpReverse 45s linear infinite;
  opacity: 0.5;
}

/* Sparkle particles */
.sparkle {
  position: fixed;
  width: 8px; height: 8px;
  pointer-events: none;
  z-index: 9998;
  background: var(--color-accent);
  clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
  animation: sparkle 1.5s ease-in-out infinite;
}

/* Main header enhanced */
.main-header {
  position: relative;
  z-index: 1;
}

.main-header::before {
  content: "";
  position: absolute;
  top: -20px; left: 50%;
  transform: translateX(-50%);
  width: 200px; height: 200px;
  background: radial-gradient(circle, var(--color-primary-light) 0%, transparent 70%);
  border-radius: 50%;
  z-index: -1;
  animation: pulseGlow 4s ease-in-out infinite;
}

/* Lottery ball styling for prediction numbers */
.lottery-ball {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 48px;
  height: 48px;
  border-radius: 50%;
  font-family: var(--font-display) !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
  color: white !important;
  box-shadow: 
    0 4px 12px rgba(0,0,0,0.3),
    inset 0 -2px 6px rgba(0,0,0,0.2),
    inset 0 2px 4px rgba(255,255,255,0.3);
  position: relative;
  overflow: hidden;
  animation: ballPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  margin: 2px 4px;
}

.lottery-ball::before {
  content: "";
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, transparent 50%);
  transform: rotate(30deg);
}

.lottery-ball::after {
  content: "";
  position: absolute;
  bottom: 2px; right: 2px;
  width: 12px; height: 12px;
  background: rgba(255,255,255,0.3);
  border-radius: 50%;
  filter: blur(1px);
}

/* Ball colors by number range */
.lottery-ball.ball-1 { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
.lottery-ball.ball-2 { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }
.lottery-ball.ball-3 { background: linear-gradient(135deg, #f39c12 0%, #d4ac0d 100%); }
.lottery-ball.ball-4 { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
.lottery-ball.ball-5 { background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); }
.lottery-ball.ball-6 { background: linear-gradient(135deg, #e67e22 0%, #d35400 100%); }
.lottery-ball.ball-7 { background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%); }
.lottery-ball.ball-8 { background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); }
.lottery-ball.ball-9 { background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%); }
.lottery-ball.ball-0 { background: linear-gradient(135deg, #607d8b 0%, #455a64 100%); }

/* Bonus ball - gold */
.lottery-ball.bonus {
  background: linear-gradient(135deg, #ffd700 0%, #daa520 100%) !important;
  box-shadow:
    0 4px 12px rgba(218, 165, 32, 0.5),
    inset 0 -2px 6px rgba(0,0,0,0.2),
    inset 0 2px 4px rgba(255,255,255,0.4),
    0 0 30px rgba(218, 165, 32, 0.3) !important;
  animation: ballPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

.lottery-ball.bonus:hover {
  animation: pulseGlow 1.5s ease-in-out infinite, ballPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

/* Prediction box enhanced */
.prediction-box {
  position: relative;
  z-index: 1;
}

.prediction-box::after {
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 100px; height: 100px;
  background: radial-gradient(circle at top right, var(--color-accent-light) 0%, transparent 70%);
  border-radius: 0 0 0 100%;
  pointer-events: none;
  opacity: 0.5;
}

/* Jackpot counter animation */
.jackpot-counter {
  font-family: var(--font-display) !important;
  font-size: 2.5rem !important;
  font-weight: 700 !important;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent), var(--color-primary));
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmerSweep 3s linear infinite, jackpotCount 2s ease-in-out infinite;
}

/* Confetti burst */
@keyframes confettiFall {
  0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
  100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
}

.confetti {
  position: fixed;
  width: 12px; height: 12px;
  pointer-events: none;
  z-index: 10000;
  animation: confettiFall 3s ease-out forwards;
}

.confetti:nth-child(odd) { border-radius: 50%; }
.confetti:nth-child(even) { clip-path: polygon(50% 0%, 0% 100%, 100% 100%); }

/* Shimmer on cards */
.prediction-box,
.metric-card,
.card {
  position: relative;
  overflow: hidden;
}

.prediction-box::before,
.metric-card::before,
.card::before {
  content: "";
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  animation: none;
  pointer-events: none;
  transition: left 0.6s ease-out, opacity 0.3s;
  opacity: 0;
}

.prediction-box:hover::before,
.metric-card:hover::before,
.card:hover::before {
  animation: shimmerSweep 0.8s ease-out forwards;
  opacity: 1;
}

/* Gold accent divider */
.gold-divider {
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--color-accent), transparent);
  border: none;
  margin: var(--space-6) 0;
}

/* Floating action hint */
.floating-hint {
  position: fixed;
  bottom: 100px; right: 24px;
  background: var(--color-primary);
  color: white;
  padding: var(--space-3) var(--space-5);
  border-radius: var(--radius-full);
  font-size: 0.875rem;
  font-weight: 500;
  box-shadow: var(--shadow-xl);
  z-index: 1000;
  animation: floatUp 3s ease-in-out infinite;
  pointer-events: none;
  opacity: 0.9;
}

/* Lottery ticket style card */
.ticket-card {
  background: linear-gradient(135deg, #fff8e1 0%, #fff3c4 100%);
  border: 2px dashed var(--color-accent);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  position: relative;
  overflow: hidden;
}

.ticket-card::before {
  content: "";
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background-image: 
    radial-gradient(circle, var(--color-accent) 1px, transparent 1px);
  background-size: 30px 30px;
  opacity: 0.05;
  pointer-events: none;
}

@media (prefers-color-scheme: dark) {
  .ticket-card {
    background: linear-gradient(135deg, #3d3300 0%, #2d2200 100%);
    border-color: var(--color-accent);
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .lottery-bg::before,
  .lottery-bg::after,
  .sparkle,
  .main-header::before,
  .lottery-ball,
  .jackpot-counter,
  .confetti,
  .prediction-box::before,
  .metric-card::before,
  .card::before,
  .floating-hint {
    animation: none !important;
  }
}
</style>
""")

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
st.title("Lottery predictor", icon=":material/casino:")
st.caption("AI-powered lottery number prediction")

# Disclaimer
st.error(
    "Lottery draws are random events. No algorithm, AI, or statistical analysis can predict future lottery "
    "numbers with certainty. Past performance does not guarantee future results. Please play responsibly and "
    "within your means.",
    title="Important disclaimer — entertainment and educational purposes only",
    icon=":material/warning:",
)

# Sidebar
with st.sidebar:
    st.header("Sections", icon=":material/menu:")
    selected_section = st.radio(
        "Navigate",
        ["Predict", "Analyze", "Visualize", "Compare", "History", "Profile", "Data Health", "Correlations"],
        key="main_navigation",
        label_visibility="collapsed",
    )
    st.divider()
    st.header("Settings", icon=":material/settings:")

    # Lottery selection - filter out "no lottery" states
    games = [g for g in manager.list_games() if g.main_numbers > 0]
    game_options = {f"{g.name} ({g.key})": g.key for g in games}
    selected_game_label = st.selectbox(
        "Select lottery",
        options=list(game_options.keys()),
        index=list(game_options.values()).index(st.session_state.current_lottery) if st.session_state.current_lottery in game_options.values() else 0
    )
    selected_lottery = game_options[selected_game_label]
    st.session_state.current_lottery = selected_lottery

    game = manager.get_game(selected_lottery)

    # Show lottery info in a bordered card
    with st.container(border=True):
        st.subheader(game.name, anchor=False)
        st.caption(f"Key: {game.key}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Main numbers", f"{game.main_numbers} (1-{game.main_max})")
        with c2:
            st.metric("Bonus numbers", f"{game.bonus_numbers} (1-{game.bonus_max})")
        st.caption(f"Draws: {', '.join(game.draw_days)}  ·  Since: {game.start_date}")

    # Official website link
    if game.website:
        st.link_button("Official website", game.website, icon=":material/launch:")

    # Animation / Visual Settings
    with st.expander("Visual settings", icon=":material/palette:", expanded=False):
        reduce_animations = st.checkbox("Reduce animations", value=False, help="Disable background particles, shimmer effects, and pulsing animations")
        if reduce_animations:
            st.markdown("""
            <style>
            .lottery-bg, .sparkle, .main-header::before, .floating-hint { display: none !important; }
            .prediction-box::before, .metric-card::before, .card::before { animation: none !important; }
            .lottery-ball.bonus { animation: ballPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both !important; }
            .jackpot-counter { animation: jackpotCount 2s ease-in-out infinite !important; }
            </style>
            """, unsafe_allow_html=True)
            st.caption("Animations disabled. Refresh to re-enable.")

    # Prediction settings
    st.subheader("Prediction settings", anchor=False)
    method = st.selectbox(
        "Method",
        ["ensemble", "statistical", "frequency", "cold", "pairs", "ml"],
        format_func=lambda x: {
            "ensemble": "Ensemble (recommended)",
            "statistical": "Statistical (balanced)",
            "frequency": "Frequency (hot numbers)",
            "cold": "Cold numbers (due)",
            "pairs": "Pair-based",
            "ml": "Machine learning",
        }[x]
    )

    num_predictions = st.slider("Number of predictions", 1, 10, 5)

    # Full history toggle (v2.4)
    use_all_history = st.checkbox("Use all historical data (from lottery start date)", value=False, help="Analyze all available draws instead of recent N draws")

    if not use_all_history:
        last_n_draws = st.slider("Historical draws to analyze", 50, 1000, 200, step=50)
    else:
        last_n_draws = None
        st.caption("Using all historical data from lottery start date")

    # Advanced settings
    with st.expander("Advanced settings", icon=":material/tune:"):
        retrain = st.checkbox("Retrain ML models", value=False)
        lookback = st.slider("Lookback window (ML)", 5, 20, 10)

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

def format_numbers_html(numbers, bonus=False):
    """Format numbers as lottery ball HTML."""
    ball_class = "bonus" if bonus else ""
    return "".join(
        f'<span class="lottery-ball {ball_class} ball-{{n % 10}}">{n:2d}</span>'
        for n in numbers
    )

def format_numbers(numbers):
    """Format numbers for display (markdown fallback)."""
    return "  ".join(f"**{n:2d}**" for n in numbers)


def _json_default(obj):
    """JSON serializer fallback for numpy and other non-standard types."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Predict page
if selected_section == "Predict":
    st.subheader("Generate predictions", icon=":material/analytics:", anchor=False)
    try:
        freshness_manager, _ = load_lottery_data(selected_lottery, 1)
        freshness = freshness_manager.refresh_status
        if freshness.get("verified"):
            st.caption(f"Verified results · {freshness.get('source')} · latest draw {freshness.get('latest_draw_date')}")
        else:
            st.warning(freshness.get("message", "Results could not be verified from an official source."))
    except Exception as exc:
        st.warning(f"Verified results are unavailable: {exc}")

    # Jackpot display for selected lottery
    lottery_info = LOTTERY_CONFIGS.get(selected_lottery, {})
    jackpot_amount = lottery_info.get('jackpot', 0)
    if jackpot_amount:
        st.metric("Estimated jackpot", f"${jackpot_amount:,}")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("Generate predictions", type="primary", width='stretch'):
            preds = generate_predictions(
                selected_lottery, method, num_predictions, last_n_draws, retrain, lookback
            )
            st.session_state.predictions = preds
            st.toast(f"Generated {len(preds)} predictions!", icon=":material/check_circle:")

    with col2:
        if st.button("Quick pick", width='stretch'):
            # Quick single prediction
            quick_pred = generate_predictions(selected_lottery, method, 1, last_n_draws, retrain, lookback)
            st.session_state.predictions = quick_pred
            st.toast("Quick pick generated!", icon=":material/auto_awesome:")

    with col3:
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
                "Download JSON",
                json.dumps(data, indent=2, default=_json_default),
                file_name=f"{selected_lottery}_predictions.json",
                mime="application/json",
                icon=":material/download:",
                width='stretch',
            )
        else:
            st.button("Download JSON", disabled=True, width='stretch')

    # Display predictions
    if st.session_state.predictions:
        st.subheader("Your predictions", anchor=False)

        for i, pred in enumerate(st.session_state.predictions, 1):
            with st.container(border=True):
                main_str = "  ".join(f"{n:2d}" for n in pred.main_numbers)
                bonus_str = "  ".join(f"{n:2d}" for n in pred.bonus_numbers) if pred.bonus_numbers else ""
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Prediction #{i}**  `{main_str}`")
                    if bonus_str:
                        st.caption(f"Bonus: `{bonus_str}`")
                with c2:
                    st.badge(f"{pred.confidence:.0%}", icon=":material/trending_up:", color="blue")
                    st.caption(pred.method)

        # Quick stats
        st.subheader("Prediction statistics", anchor=False)
        cols = st.columns(4)

        all_nums = [n for p in st.session_state.predictions for n in p.main_numbers]
        with cols[0]:
            st.metric("Avg sum", f"{np.mean([sum(p.main_numbers) for p in st.session_state.predictions]):.1f}")
        with cols[1]:
            st.metric("Most common", f"{pd.Series(all_nums).mode()[0] if all_nums else 'N/A'}")
        with cols[2]:
            evens = sum(1 for n in all_nums if n % 2 == 0)
            st.metric("Even/odd ratio", f"{evens}/{len(all_nums)-evens}")
        with cols[3]:
            avg_conf = np.mean([p.confidence for p in st.session_state.predictions])
            st.metric("Avg confidence", f"{avg_conf:.1%}")

# Analyze page
if selected_section == "Analyze":
    st.subheader("Historical analysis", icon=":material/bar_chart:", anchor=False)

    # Full history toggle for analysis
    analyze_all_history = st.checkbox("Use all historical data (from lottery start date)", value=(last_n_draws is None), key="analyze_all_history")

    if analyze_all_history:
        dm, df = load_lottery_data(selected_lottery, None)
        analysis_last_n = None
        st.caption(f"Using all {len(df)} historical draws from {df['date'].min().strftime('%Y-%m-%d')}")
    else:
        dm, df = load_lottery_data(selected_lottery, last_n_draws)

    # Analysis controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if not analyze_all_history:
            # Use sidebar value or default to 200
            default_draws = last_n_draws if last_n_draws is not None else 200
            analysis_last_n = st.slider("Draws to Analyze", 50, len(df), min(default_draws, len(df)), key="analyze_n")
        else:
            analysis_last_n = None
            st.caption("Analyzing all draws")
    with col2:
        if st.button("Refresh analysis", key="refresh_analyze"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        show_viz = st.checkbox("Show charts", value=True)

    # Recalculate with selected draws
    if not analyze_all_history:
        dm, df = load_lottery_data(selected_lottery, analysis_last_n)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total draws", len(df))
    with col2:
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        st.metric("Date range", date_range)
    with col3:
        sums = df['main_numbers'].apply(sum)
        st.metric("Avg sum", f"{sums.mean():.1f}")
    with col4:
        st.metric("Sum std dev", f"{sums.std():.1f}")

    # Hot/Cold Numbers
    st.subheader("Hot & cold numbers", icon=":material/trending_up:", anchor=False)
    freq = dm.get_number_frequency(analysis_last_n)
    hot, cold = dm.get_hot_cold_numbers(analysis_last_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Hottest numbers**")
        hot_df = pd.DataFrame({
            'Number': hot,
            'Frequency': [freq[n] for n in hot]
        })
        st.dataframe(hot_df, width='stretch', hide_index=True)

    with col2:
        st.markdown("**Coldest numbers**")
        cold_df = pd.DataFrame({
            'Number': cold,
            'Frequency': [freq[n] for n in cold]
        })
        st.dataframe(cold_df, width='stretch', hide_index=True)

    # Top Pairs
    st.subheader("Top number pairs", icon=":material/link:", anchor=False)
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
        st.subheader("Charts", icon=":material/bar_chart:", anchor=False)

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

# Visualize page
if selected_section == "Visualize":
    st.subheader("Visualizations", icon=":material/bar_chart:", anchor=False)

    dm, df = load_lottery_data(selected_lottery, last_n_draws)
    viz = LotteryVisualizer(selected_lottery)
    viz.dm = dm

    viz_options = st.multiselect(
        "Select charts to generate",
        ["Frequency", "Hot/Cold", "Sum Distribution", "Pair Heatmap", "Timeline"],
        default=["Frequency", "Hot/Cold", "Sum Distribution", "Pair Heatmap", "Timeline"]
    )

    if st.button("Generate visualizations", type="primary"):
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
            st.subheader("Interactive charts", icon=":material/insights:", anchor=False)

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

            fig_hc.update_layout(height=500, showlegend=False, title="Cold numbers")
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

            st.toast("Visualizations generated!", icon=":material/check_circle:")

# History page
if selected_section == "History":
    st.subheader("Recent draw history", icon=":material/history:", anchor=False)

    dm, df = load_lottery_data(selected_lottery)

    num_history = st.slider("Number of recent draws", 5, 50, 15)
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
        "Download history as CSV",
        csv,
        file_name=f"{selected_lottery}_history.csv",
        mime="text/csv"
    )

# Compare page
if selected_section == "Compare":
    st.subheader("Method comparison", icon=":material/compare_arrows:", anchor=False)

    compare_methods = st.multiselect(
        "Methods to compare",
        ["statistical", "frequency", "cold", "pairs", "ml", "ensemble"],
        default=["statistical", "frequency", "cold", "ensemble"],
        format_func=lambda x: {
            "ensemble": "Ensemble",
            "statistical": "Statistical",
            "frequency": "Frequency",
            "cold": "Cold",
            "pairs": "Pairs",
            "ml": "Machine learning"
        }[x]
    )

    compare_count = st.slider("Predictions per method", 1, 5, 2)

    if st.button("Compare methods", type="primary") and compare_methods:
        with st.spinner("Comparing methods..."):
            all_results = {}

            for m in compare_methods:
                preds = generate_predictions(
                    selected_lottery, m, compare_count, last_n_draws
                )
                all_results[m] = preds

            # Display comparison
            for method, preds in all_results.items():
                st.subheader(method.replace("_", " ").title(), anchor=False)

                for i, pred in enumerate(preds, 1):
                    with st.container(border=True):
                        main_str = "  ".join(f"{n:2d}" for n in pred.main_numbers)
                        bonus_str = "  ".join(f"{n:2d}" for n in pred.bonus_numbers) if pred.bonus_numbers else ""
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**#{i}**  `{main_str}`")
                            if bonus_str:
                                st.caption(f"Bonus: `{bonus_str}`")
                        with c2:
                            st.badge(f"{pred.confidence:.0%}", icon=":material/trending_up:", color="blue")

            # Summary table
            st.subheader("Summary", anchor=False)
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

# Profile page
if selected_section == "Profile":
    st.subheader("User profile & social", icon=":material/person:", anchor=False)

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
            st.subheader("Login / create profile")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Existing user**")
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
                    "📋 Profile", "⭐ Favorites", "Preferences", "📤 Share", "🏆 Leaderboard"
                ])

                with prof_tab1:
                    st.subheader("Your profile")
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
                    st.subheader("Favorite lotteries")

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
                                if st.button("Remove", key=f"remove_{fav}", use_container_width=True):
                                    if user_manager.remove_favorite_lottery(profile.username, fav):
                                        st.success(f"Removed {fav} from favorites")
                                        st.rerun()
                    else:
                        st.info("No favorites yet. Add some above!")

                with prof_tab3:
                    st.subheader("Preferences")

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
                            "ensemble": "Ensemble",
                            "statistical": "Statistical",
                            "frequency": "Frequency",
                            "cold": "Cold numbers",
                            "pairs": "Pair-based",
                            "ml": "Machine learning",
                            "time_series": "Time series",
                            "transformer": "Transformer",
                            "online_learning": "Online learning"
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

                    if st.button("Preferences", type="primary", use_container_width=True):
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
                    st.subheader("Share predictions")

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

                    if st.button("Generate & share", type="primary", use_container_width=True):
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
                                    main_balls = format_numbers_html(p["main_numbers"])
                                    bonus_balls = format_numbers_html(p["bonus_numbers"], bonus=True) if p["bonus_numbers"] else ""
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <strong>#{i}:</strong> <div class="prediction-numbers" style="display: inline;">{main_balls}</div>
                                        {f' | Bonus: <div class="prediction-numbers" style="display: inline;">{bonus_balls}</div>' if bonus_balls else ''}
                                        | Confidence: {p["confidence"]:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error sharing prediction: {e}")

                with prof_tab5:
                    st.subheader("Accuracy leaderboard")

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
    st.subheader("Community prediction pools")

    if social_manager:
        # Create pool form
        with st.expander("Create new pool"):
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

# Data Health page
if selected_section == "Data Health":
    st.subheader("Historical data health dashboard (v2.4)", icon=":material/health_and_safety:", anchor=False)

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
        if st.button("Analyze health", type="primary", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    if health_lottery == "All Lotteries":
        with st.spinner("Analyzing health for all lotteries..."):
            health_data = dashboard.get_all_lotteries_health()

        if not health_data:
            st.warning("No health data available. Click 'Analyze Health' to run analysis.")
        else:
            # Summary metrics
            st.subheader("Overall health summary")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total lotteries", len(health_data))
            with cols[1]:
                avg_completeness = np.mean([float(h['completeness_score'].rstrip('%'))/100 for h in health_data])
                st.metric("Avg completeness", f"{avg_completeness:.1%}")
            with cols[2]:
                avg_freshness = np.mean([float(h['freshness_score'].rstrip('%'))/100 for h in health_data])
                st.metric("Avg freshness", f"{avg_freshness:.1%}")
            with cols[3]:
                total_anomalies = sum(h['anomalies_count'] for h in health_data)
                st.metric("Total anomalies", total_anomalies)

            # Health table
            st.subheader("Detailed health report")
            health_df = pd.DataFrame(health_data)
            health_df['Lottery Name'] = health_df['lottery'].apply(lambda x: LOTTERY_CONFIGS.get(x, {}).get('name', x))
            display_cols = ['Lottery Name', 'total_draws', 'date_range', 'days_since_last_draw', 'expected_draws', 'missing_draws', 'completeness_score', 'freshness_score', 'gaps_count', 'anomalies_count']
            health_df_display = health_df[display_cols].copy()
            health_df_display.columns = ['Lottery', 'Total Draws', 'Date Range', 'Days Since Last', 'Expected', 'Missing', 'Completeness', 'Freshness', 'Gaps', 'Anomalies']
            st.dataframe(health_df_display, width='stretch', hide_index=True)

            # Download button
            csv = health_df_display.to_csv(index=False)
            st.download_button(
                "Download health report (CSV)",
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
            st.metric("Total draws", record.total_draws)
        with cols[1]:
            st.metric("Days since last draw", record.days_since_last_draw)
        with cols[2]:
            st.metric("Expected draws", record.expected_draws)
        with cols[3]:
            st.metric("Missing draws", record.missing_draws)
        with cols[4]:
            st.metric("Completeness", f"{record.completeness_score:.1%}")

        cols2 = st.columns(3)
        with cols2[0]:
            st.metric("Freshness", f"{record.freshness_score:.1%}")
        with cols2[1]:
            st.metric("Gaps detected", len(record.gaps))
        with cols2[2]:
            st.metric("Anomalies", len(record.anomalies))

        st.write(f"**Date Range:** {record.date_range_start} to {record.date_range_end}")

        # Gaps
        if record.gaps:
            st.subheader("Detected gaps")
            gaps_df = pd.DataFrame(record.gaps)
            st.dataframe(gaps_df, width='stretch', hide_index=True)

            # Backfill button
            if st.button("Schedule backfill for missing data", type="secondary"):
                result = dashboard.schedule_backfill(lottery_key)
                st.success(f"Backfill scheduled: {result}")
        else:
            st.success("✅ No gaps detected in draw schedule")

        # Anomalies
        if record.anomalies:
            st.subheader("Detected anomalies")
            anomalies_df = pd.DataFrame(record.anomalies)
            st.dataframe(anomalies_df, width='stretch', hide_index=True)
        else:
            st.success("✅ No anomalies detected")

# Correlations page
if selected_section == "Correlations":
    st.subheader("Cross-lottery historical correlation (v2.4)", icon=":material/compare:", anchor=False)

    st.markdown("""
    Analyze historical correlations between different lottery games to discover patterns.
    Compare number frequencies, sum distributions, and odd/even patterns across lotteries.
    """)

    from lottery_predictor.advanced import CrossLotteryCorrelationAnalyzer

    analyzer = CrossLotteryCorrelationAnalyzer()

    # Mode selection
    mode = st.radio(
        "Analysis mode",
        ["Compare two lotteries", "Find correlated lotteries"],
        horizontal=True
    )

    if mode == "Compare two lotteries":
        col1, col2 = st.columns(2)
        with col1:
            lotto_a = st.selectbox(
                "First lottery",
                options=[f"{g.name} ({g.key})" for g in manager.list_games() if g.main_numbers > 0],
                index=0,
                key="corr_lotto_a"
            )
        with col2:
            lotto_b = st.selectbox(
                "Second lottery",
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

        if st.button("Find correlated lotteries", type="primary", width='stretch'):
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