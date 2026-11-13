#!/usr/bin/env python3
"""
Lottery Predictor Testbed
-------------------------
A self-contained test harness that verifies the Lottery Predictor works correctly.

It checks:
  * package imports
  * historical data loading for several lotteries
  * prediction methods (statistical, frequency, cold, pairs, ensemble)
  * CLI entry point (main.py) across commands: info, predict, analyze, history,
    viz, odds, quick_pick
  * odds calculation and quick-pick generation
  * JSON output file creation

Run:
    ./venv/bin/python testbed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (detect project root from this file's location)
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
VENV_PY = PROJECT / "venv" / "bin" / "python"
MAIN = PROJECT / "main.py"
RESULTS_DIR = PROJECT / "testbed_results"
RESULTS_DIR.mkdir(exist_ok=True)

if not VENV_PY.exists():
    sys.exit(f"❌ Venv python not found at {VENV_PY}")


def run(cmd: list[str], description: str, timeout: int = 120) -> tuple[bool, str]:
    """Run a command, return (success, output)."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT)
        )
        out = result.stdout + result.stderr
        if result.returncode == 0:
            print(f"✅ PASS: {description}")
            for line in out.strip().splitlines()[:8]:
                print(f"   {line}")
            if len(out.strip().splitlines()) > 8:
                print(f"   ... ({len(out.strip().splitlines())} total lines)")
            return True, out
        print(f"❌ FAIL: {description} (rc={result.returncode})")
        for line in out.strip().splitlines()[:8]:
            print(f"   {line}")
        return False, out
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {description} (>{timeout}s)")
        return False, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        print(f"💥 ERROR: {description} - {exc}")
        return False, str(exc)


# ---------------------------------------------------------------------------
# Direct API tests
# ---------------------------------------------------------------------------
def test_imports() -> bool:
    print("\n" + "=" * 60)
    print("TEST: package imports")
    print("=" * 60)
    code = (
        "import lottery_predictor\n"
        "from lottery_predictor.data import get_data_manager, LOTTERY_CONFIGS\n"
        "from lottery_predictor.models import StatisticalPredictor, EnsemblePredictor\n"
        "from lottery_predictor.validation import OddsCalculator\n"
        "print('imports OK')"
    )
    res = subprocess.run(
        [str(VENV_PY), "-c", code], capture_output=True, text=True, cwd=str(PROJECT)
    )
    ok = res.returncode == 0
    print((res.stdout or res.stderr).strip())
    print("✅ PASS: package imports" if ok else f"❌ FAIL: package imports (rc={res.returncode})")
    return ok


def test_data_loading() -> bool:
    print("\n" + "=" * 60)
    print("TEST: historical data loading (several lotteries)")
    print("=" * 60)
    code = (
        "from lottery_predictor.data import get_data_manager\n"
        "for key in ['powerball', 'mega_millions', 'florida_lotto', 'california_superlotto']:\n"
        "    dm = get_data_manager(key)\n"
        "    df = dm.load_data()\n"
        "    print(key, len(df), 'draws', df['date'].min(), '->', df['date'].max())\n"
    )
    res = subprocess.run(
        [str(VENV_PY), "-c", code], capture_output=True, text=True, cwd=str(PROJECT)
    )
    out = (res.stdout or "") + (res.stderr or "")
    print(out.strip())
    ok = res.returncode == 0 and all(
        k in out for k in ["powerball", "mega_millions", "florida_lotto", "california_superlotto"]
    )
    print("✅ PASS: data loading" if ok else "❌ FAIL: data loading")
    return ok


def test_predictions_api() -> bool:
    print("\n" + "=" * 60)
    print("TEST: prediction API (statistical, frequency, cold, pairs, ensemble)")
    print("=" * 60)
    code = (
        "from lottery_predictor.models import StatisticalPredictor, EnsemblePredictor\n"
        "for method, call in [\n"
        "    ('statistical', lambda p: p.predict_balanced(last_n=100, num_predictions=2)),\n"
        "    ('frequency', lambda p: p.predict_frequency_based(num_predictions=2)),\n"
        "    ('cold', lambda p: p.predict_cold_numbers(num_predictions=2)),\n"
        "    ('pairs', lambda p: p.predict_pair_based(num_predictions=2)),\n"
        "]:\n"
        "    pred = StatisticalPredictor('powerball')\n"
        "    ps = call(pred)\n"
        "    print(method, [(list(pr.main_numbers), list(pr.bonus_numbers), round(pr.confidence,2)) for pr in ps])\n"
        "ep = EnsemblePredictor('powerball')\n"
        "eps = ep.get_consensus_prediction(num_predictions=2)\n"
        "print('ensemble', [(list(pr.main_numbers), list(pr.bonus_numbers), round(pr.confidence,2)) for pr in eps])\n"
        "print('prediction API OK')\n"
    )
    res = subprocess.run(
        [str(VENV_PY), "-c", code], capture_output=True, text=True, cwd=str(PROJECT)
    )
    out = (res.stdout or "") + (res.stderr or "")
    print(out.strip())
    ok = res.returncode == 0 and "prediction API OK" in out
    print("✅ PASS: prediction API" if ok else "❌ FAIL: prediction API")
    return ok


def test_odds_and_quickpick() -> bool:
    print("\n" + "=" * 60)
    print("TEST: odds calculator + quick pick")
    print("=" * 60)
    code = (
        "from lottery_predictor.validation import OddsCalculator\n"
        "import secrets\n"
        "calc = OddsCalculator('powerball')\n"
        "r = calc.full_odds_analysis()\n"
        "print('jackpot odds:', r.jackpot_odds_str)\n"
        "print('expected value:', round(r.expected_value, 4))\n"
        "rng = secrets.SystemRandom()\n"
        "pick = sorted(rng.sample(range(1, 70), 5))\n"
        "bonus = sorted(rng.sample(range(1, 27), 1))\n"
        "print('quick pick:', pick, bonus)\n"
        "print('odds+quickpick OK')\n"
    )
    res = subprocess.run(
        [str(VENV_PY), "-c", code], capture_output=True, text=True, cwd=str(PROJECT)
    )
    out = (res.stdout or "") + (res.stderr or "")
    print(out.strip())
    ok = res.returncode == 0 and "odds+quickpick OK" in out
    print("✅ PASS: odds + quick pick" if ok else "❌ FAIL: odds + quick pick")
    return ok


# ---------------------------------------------------------------------------
# CLI tests (subprocess against main.py)
# ---------------------------------------------------------------------------
CLI_TESTS = [
    (["info"], "Show supported lotteries info"),
    (["predict", "--lottery", "powerball", "--method", "statistical", "--count", "2"], "CLI predict statistical (powerball)"),
    (["predict", "--lottery", "mega_millions", "--method", "frequency", "--count", "2"], "CLI predict frequency (mega_millions)"),
    (["predict", "--lottery", "florida_lotto", "--method", "cold", "--count", "2"], "CLI predict cold (florida_lotto)"),
    (["predict", "--lottery", "powerball", "--method", "pairs", "--count", "2"], "CLI predict pairs (powerball)"),
    (["predict", "--lottery", "powerball", "--method", "ensemble", "--count", "2"], "CLI predict ensemble (powerball)"),
    (["analyze", "--lottery", "powerball", "--last-n", "50", "--no-visualize"], "CLI analyze powerball (no viz)"),
    (["history", "--lottery", "powerball", "--count", "3"], "CLI powerball history"),
    (["viz", "--lottery", "powerball", "--last-n", "20"], "CLI viz powerball"),
    (["odds", "--lottery", "powerball"], "CLI odds powerball"),
    (["quick-pick", "--lottery", "powerball", "--count", "3"], "CLI quick pick powerball"),
]


def run_cli_tests() -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    base = [str(VENV_PY), str(MAIN)]
    for args, desc in CLI_TESTS:
        ok, _ = run(base + args, desc)
        results.append((desc, ok))
    return results


def run_cli_output_file() -> bool:
    out_path = PROJECT / "testbed_output.json"
    if out_path.exists():
        out_path.unlink()
    cmd = [str(VENV_PY), str(MAIN), "predict", "--lottery", "powerball",
           "--method", "statistical", "--count", "3", "--output", str(out_path)]
    ok, _ = run(cmd, "CLI save predictions to JSON file")
    if ok and out_path.exists():
        try:
            data = json.loads(out_path.read_text())
            assert data["lottery"] == "powerball"
            assert len(data["predictions"]) == 3
            print("   ✓ JSON contents valid")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"   ❌ JSON invalid: {exc}")
            return False
    print("   ❌ Output file not created")
    return False


def main() -> int:
    print("🧪 Lottery Predictor Testbed")
    print(f"Project: {PROJECT}")
    print(f"Venv:    {VENV_PY}")

    results: list[tuple[str, bool]] = []

    results.append(("package imports", test_imports()))
    results.append(("data loading", test_data_loading()))
    results.append(("prediction API", test_predictions_api()))
    results.append(("odds + quick pick", test_odds_and_quickpick()))

    for desc, ok in run_cli_tests():
        results.append((desc, ok))

    results.append(("CLI save predictions to JSON", run_cli_output_file()))

    # Summary
    print(f"\n{'='*60}")
    print("TESTBED SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for desc, ok in results:
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}: {desc}")
    print(f"\nTotal: {passed}/{total} tests passed")
    if passed == total:
        print("\n🎉 All testbed tests passed!")
        return 0
    print(f"\n⚠️  {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
