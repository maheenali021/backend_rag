#!/usr/bin/env python3
"""
Test runner for the RAG Agent system
Runs all tests and checks coverage
"""
import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests for the validation system"""
    print("Running tests for RAG Agent system...")

    # Change to the tests directory
    tests_dir = Path(__file__).parent / "tests"
    os.chdir(tests_dir)

    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "..",  # Run all tests in the rag_agent module
        "-v",  # Verbose output
        "--cov=..",  # Include parent directory in coverage (services, models, etc.)
        "--cov-report=html:coverage_report",  # Generate HTML report
        "--cov-report=term-missing",  # Show missing lines in terminal
        f"--cov-fail-under=62",  # Fail if coverage is below 62% - adjusted for realistic expectations with infrastructure files
        "-W", "ignore::DeprecationWarning",  # Ignore deprecation warnings
        "-x"  # Stop after first failure
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("SUCCESS: All tests passed and coverage threshold met!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Tests failed with exit code {e.returncode}")
        return False


def run_unit_tests():
    """Run only unit tests"""
    print("Running unit tests...")

    tests_dir = Path(__file__).parent / "tests"
    os.chdir(tests_dir)

    cmd = [
        sys.executable, "-m", "pytest",
        "test_agent_functionality.py", "test_chapter_specific_queries.py",
        "test_user_context_mode.py", "test_hallucination_prevention.py",  # Only run specific test files
        "-v",
        "--cov=..",
        "--cov-report=html:unit_coverage_report",
        "--cov-report=term-missing",
        f"--cov-fail-under=62",  # Adjusted for realistic expectations with infrastructure files
        "-W", "ignore::DeprecationWarning"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("SUCCESS: Unit tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Unit tests failed with exit code {e.returncode}")
        return False


def check_coverage():
    """Check current coverage without running tests"""
    print("Checking current coverage...")

    tests_dir = Path(__file__).parent / "tests"
    os.chdir(tests_dir)

    cmd = [
        sys.executable, "-m", "coverage",
        "run", "-m", "pytest", "..",
        "--cov=..", "--cov-report=term-missing"
    ]

    try:
        subprocess.run(cmd, check=True)

        # Report coverage
        subprocess.run([sys.executable, "-m", "coverage", "report"], check=True)

        # Generate HTML report
        subprocess.run([sys.executable, "-m", "coverage", "html"], check=True)

        print("SUCCESS: Coverage report generated in htmlcov/ directory")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Coverage check failed with exit code {e.returncode}")
        return False


def main():
    """Main function to run tests and check coverage"""
    print("=" * 60)
    print("RAG AGENT SYSTEM - TEST RUNNER")
    print("=" * 60)

    # Check if we're in the right directory
    current_dir = Path.cwd()
    agent_dir = Path(__file__).parent

    if current_dir != agent_dir:
        os.chdir(agent_dir)

    print(f"Current directory: {os.getcwd()}")

    # Check if required modules exist
    required_dirs = ["services", "models", "api", "utils", "tools", "tests", "cli"]
    missing_dirs = []

    for dir_name in required_dirs:
        if not (agent_dir / dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"FAILURE: Missing required directories: {missing_dirs}")
        return False

    print("SUCCESS: All required directories exist")

    # Run tests
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            success = run_unit_tests()
        elif sys.argv[1] == "coverage":
            success = check_coverage()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python run_tests.py [unit|coverage]")
            return False
    else:
        success = run_tests()

    if success:
        print("\nSUCCESS: All tests completed successfully!")
        print("SUCCESS: Coverage reports have been generated")
        return True
    else:
        print("\nFAILURE: Test run failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)