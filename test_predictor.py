#!/usr/bin/env python3
"""
Test suite for Lottery Predictor
Runs through all main commands to verify they work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, description, timeout=60):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"CMD:  {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/home/bryce/lottery-predictor"
        )

        if result.returncode == 0:
            print(f"✅ PASS: {description}")
            if result.stdout:
                # Show first few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[:10]:
                    print(f"   {line}")
                if len(lines) > 10:
                    print(f"   ... ({len(lines)} total lines)")
            return True
        else:
            print(f"❌ FAIL: {description}")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"   stdout: {result.stdout[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {description} (>{timeout}s)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Lottery Predictor Test Suite")
    print("=" * 60)

    # Change to project directory
    os.chdir("/home/bryce/lottery-predictor")

    # Activate venv in command
    venv_python = "/home/bryce/lottery-predictor/venv/bin/python"
    base_cmd = f"{venv_python} main.py"

    # Test lotteries (sample of available ones)
    test_lotteries = ["powerball", "mega_millions", "florida_lotto", "california_superlotto", "new_york_lotto", "euromillions"]

    tests = [
        # Basic commands
        (f"{base_cmd} info", "Show lottery info"),
    ]

    # Add prediction tests for each lottery
    for lottery in test_lotteries:
        tests.extend([
            (f"{base_cmd} predict --lottery {lottery} --method statistical --count 2", f"Statistical prediction ({lottery})"),
            (f"{base_cmd} predict --lottery {lottery} --method frequency --count 2", f"Frequency prediction ({lottery})"),
            (f"{base_cmd} predict --lottery {lottery} --method cold --count 2", f"Cold numbers prediction ({lottery})"),
            (f"{base_cmd} predict --lottery {lottery} --method pairs --count 2", f"Pair-based prediction ({lottery})"),
            (f"{base_cmd} predict --lottery {lottery} --method ensemble --count 2", f"Ensemble prediction ({lottery})"),
            (f"{base_cmd} analyze --lottery {lottery} --last-n 30 --no-visualize", f"Analyze {lottery} (no viz)"),
            (f"{base_cmd} history --lottery {lottery} --count 3", f"{lottery} history"),
            (f"{base_cmd} viz --lottery {lottery} --last-n 20", f"Generate visualizations ({lottery})"),
        ])

    # Compare commands
    tests.append((f"{base_cmd} compare --lottery powerball --methods statistical --methods frequency --methods cold --count 2", "Compare statistical methods"))

    # Output to file
    tests.append((f"{base_cmd} predict --lottery powerball --method statistical --count 3 --output test_output.json", "Save predictions to JSON"))

    results = []
    for cmd, desc in tests:
        success = run_cmd(cmd, desc, timeout=120)
        results.append((desc, success))

    # Verify output file was created
    if Path("test_output.json").exists():
        print(f"\n✅ PASS: Output file created")
        results.append(("Output file exists", True))
    else:
        print(f"\n❌ FAIL: Output file not created")
        results.append(("Output file exists", False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {desc}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())