#!/usr/bin/env python3
"""
Test runner for FactCheck AI Backend.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit            # Run only unit tests
    python run_tests.py --integration     # Run only integration tests
    python run_tests.py --e2e             # Run only e2e tests
    python run_tests.py --quick           # Run quick tests (unit + integration)
    python run_tests.py --coverage        # Run with coverage report
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest pytest-asyncio pytest-cov")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run FactCheck AI Backend tests")
    parser.add_argument("--unit", action="store_true",
                        help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                        help="Run only integration tests")
    parser.add_argument("--e2e", action="store_true",
                        help="Run only e2e tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests (unit + integration)")
    parser.add_argument("--coverage", action="store_true",
                        help="Run with coverage report")
    parser.add_argument("--verbose", "-v",
                        action="store_true", help="Verbose output")
    parser.add_argument("--no-cov", action="store_true",
                        help="Disable coverage collection")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add coverage if requested and not disabled
    if args.coverage and not args.no_cov:
        cmd.extend(["--cov=app", "--cov-report=term-missing",
                   "--cov-report=html:htmlcov"])
    elif not args.no_cov:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    # Add verbose flag
    if args.verbose:
        cmd.append("-v")

    # Determine which tests to run
    if args.unit:
        cmd.extend(["tests/unit/", "-m", "unit"])
        description = "Unit Tests"
    elif args.integration:
        cmd.extend(["tests/integration/", "-m", "integration"])
        description = "Integration Tests"
    elif args.e2e:
        cmd.extend(["tests/e2e/", "-m", "e2e"])
        description = "End-to-End Tests"
    elif args.quick:
        cmd.extend(["tests/unit/", "tests/integration/",
                   "-m", "unit or integration"])
        description = "Quick Tests (Unit + Integration)"
    else:
        # Run all tests
        description = "All Tests"

    # Run the tests
    success = run_command(cmd, description)

    if success:
        print(f"\n✅ {description} completed successfully!")
        return 0
    else:
        print(f"\n❌ {description} failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
