## Running Version Log

### 2026-11-14 - Verified Lottery Result Refresh
- **Added**: Refresh Powerball and Mega Millions from New York State Gaming Commission open data and Florida Lotto from the official Florida Lottery history PDF.
- **Added**: Five-minute refresh cache and in-app source/latest-draw freshness status.
- **Fixed**: Removed generated draw extensions and synthetic historical-data generation; updated local caches from official results.
- **Entry point**: `streamlit_app.py` (alternate Streamlit interface).
- **Environment variables / proxy ports**: None added.

### 2026-11-14 - Sidebar Navigation
- **Changed**: Moved all 11 web app sections into a persistent desktop sidebar; kept a horizontal scrollable navigation bar on narrow screens.
- **Preserved**: Existing section content, button actions, and active-section highlighting.
- **Active entry point**: `web/index.html` served at `http://127.0.0.1:8001/`.
- **Environment variables / proxy ports**: None added.
