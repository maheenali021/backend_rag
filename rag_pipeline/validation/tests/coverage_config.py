"""
Test coverage configuration for the retrieval validation system
"""
import os
from typing import Dict, List


# Coverage configuration
COVERAGE_CONFIG = {
    # Source files to include in coverage analysis
    "source": ["../services/", "../models/", "../api/", "../utils/"],

    # Files to exclude from coverage analysis
    "exclude": [
        "*/test_*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/env/*",
        "*/.venv/*",
        "*/.git/*",
        "setup.py",
        "run.py",
        "manage.py",
        "*/migrations/*",
        "*/settings/*",
        "*/config/*",  # Exclude general config files, but not validation-specific ones
    ],

    # Coverage thresholds by module
    "thresholds": {
        "services": 85,    # Service modules should have high coverage
        "models": 90,      # Model modules should be well-tested
        "api": 80,         # API modules should have good coverage
        "utils": 75        # Utility modules can have slightly lower coverage
    },

    # Minimum overall coverage percentage required
    "minimum_coverage": 85,

    # Coverage report options
    "report_options": {
        "show_missing": True,
        "skip_covered": False,
        "precision": 2,
    },

    # Specific files that must have 100% coverage
    "strict_files": [
        "retrieval_service.py",
        "validation_models.py"
    ],

    # Coverage ignore patterns
    "ignore": [
        "pragma: no cover",
        "def __repr__",
        "raise AssertionError",
        "raise NotImplementedError",
        "if __name__ == .__main__.:",
        "if TYPE_CHECKING:",
        "if settings.DEBUG:",
        "if 0:",
        "if __debug__:",
        "setDaemon",
        "setVisible",
        "raise MemoryError",
        "raise RuntimeError",
        "try:.*except ImportError:"
    ],

    # Coverage contexts for different test types
    "contexts": {
        "unit": {
            "exclude": ["*/integration/*", "*/e2e/*"],
            "minimum_coverage": 85
        },
        "integration": {
            "exclude": ["*/unit/*", "*/e2e/*"],
            "minimum_coverage": 80
        },
        "e2e": {
            "exclude": ["*/unit/*", "*/integration/*"],
            "minimum_coverage": 70
        }
    }
}


def get_coverage_config(context: str = "unit") -> Dict:
    """
    Get coverage configuration for a specific test context

    Args:
        context: Test context (unit, integration, e2e)

    Returns:
        Coverage configuration dictionary
    """
    config = COVERAGE_CONFIG.copy()

    if context in config["contexts"]:
        context_config = config["contexts"][context]
        config.update(context_config)

    return config


def get_pytest_cov_args(context: str = "unit") -> List[str]:
    """
    Get pytest coverage arguments for a specific context

    Args:
        context: Test context (unit, integration, e2e)

    Returns:
        List of pytest coverage arguments
    """
    config = get_coverage_config(context)

    args = [
        "--cov=" + ",".join(config["source"]),
        "--cov-report=html:coverage_report_" + context,
        "--cov-report=term-missing",
        "--cov-fail-under=" + str(config["minimum_coverage"])
    ]

    # Add exclusions
    for exclude_pattern in config["exclude"]:
        args.extend(["--ignore=" + exclude_pattern])

    return args


# Default configuration
DEFAULT_COVERAGE_CONFIG = get_coverage_config("unit")


if __name__ == "__main__":
    print("Coverage configuration loaded successfully")
    print(f"Minimum required coverage: {COVERAGE_CONFIG['minimum_coverage']}%")
    print(f"Source directories: {COVERAGE_CONFIG['source']}")
    print(f"Excluded patterns: {COVERAGE_CONFIG['exclude']}")