"""
Test coverage configuration for RAG Pipeline
Configures coverage settings for testing
"""
import os
import sys

# Add the parent directory to the path so we can import from rag_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Coverage configuration
COVERAGE_CONFIG = {
    # Files to include in coverage analysis
    "source": ["rag_pipeline"],

    # Files to exclude from coverage analysis
    "omit": [
        "*/tests/*",
        "*/test_*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/env/*",
        "*/.venv/*",
        "*/.git/*",
        "setup.py",
        "run_rag_pipeline.py",  # Main script might have different coverage patterns
        "*/config.py",  # Configuration files
    ],

    # Minimum coverage percentage required
    "minimum_coverage": 85.0,

    # Coverage report options
    "report_options": {
        "show_missing": True,
        "skip_covered": False,
        "precision": 2,
    },

    # Coverage thresholds by file type
    "thresholds": {
        "services": 90.0,    # High threshold for service modules
        "models": 80.0,      # Medium threshold for model modules
        "utils": 75.0,       # Lower threshold for utility modules
        "pipelines": 95.0,   # Very high threshold for pipeline modules
    },

    # Specific files that must have 100% coverage
    "strict_files": [
        "rag_pipeline/pipelines/rag_pipeline.py",
        "rag_pipeline/pipelines/extraction_pipeline.py",
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
    ],

    # Coverage context for different test types
    "contexts": {
        "unit": {
            "omit": ["*/integration/*", "*/e2e/*"],
            "minimum_coverage": 85.0
        },
        "integration": {
            "omit": ["*/unit/*", "*/e2e/*"],
            "minimum_coverage": 80.0
        },
        "e2e": {
            "omit": ["*/unit/*", "*/integration/*"],
            "minimum_coverage": 70.0
        }
    }
}

# Function to get coverage configuration based on test type
def get_coverage_config(test_type="unit"):
    """
    Get coverage configuration for a specific test type

    Args:
        test_type: Type of tests ("unit", "integration", "e2e")

    Returns:
        Coverage configuration dictionary
    """
    config = COVERAGE_CONFIG.copy()

    if test_type in config["contexts"]:
        context_config = config["contexts"][test_type]
        config.update(context_config)

    return config

# Coverage badge configuration
BADGE_CONFIG = {
    "filename": "coverage.svg",
    "template": """<svg xmlns="http://www.w3.org/2000/svg" width="96" height="20">
    <linearGradient id="a" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <rect rx="3" width="96" height="20" fill="#555"/>
    <rect rx="3" x="37" width="59" height="20" fill="{color}"/>
    <path fill="{color}" d="M37 0h4v20h-4z"/>
    <rect rx="3" width="96" height="20" fill="url(#a)"/>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="19" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="19" y="14">coverage</text>
        <text x="65" y="15" fill="#010101" fill-opacity=".3">{percentage}%</text>
        <text x="65" y="14">{percentage}%</text>
    </g>
</svg>""",
    "colors": {
        "high": "#4c1",    # Green for high coverage (>90%)
        "medium": "#dfb317",  # Yellow for medium coverage (75-90%)
        "low": "#e05d44"     # Red for low coverage (<75%)
    }
}

def get_badge_color(coverage_percentage):
    """
    Get the color for the coverage badge based on percentage

    Args:
        coverage_percentage: Coverage percentage as a float

    Returns:
        Color string for the badge
    """
    if coverage_percentage >= 90:
        return BADGE_CONFIG["colors"]["high"]
    elif coverage_percentage >= 75:
        return BADGE_CONFIG["colors"]["medium"]
    else:
        return BADGE_CONFIG["colors"]["low"]

# Test runner configuration
TEST_RUNNER_CONFIG = {
    "test_directories": [
        "rag_pipeline/tests"
    ],
    "test_patterns": [
        "test_*.py",
        "*_test.py"
    ],
    "coverage_command": "coverage run -m pytest {test_path} --cov=rag_pipeline --cov-report=html --cov-report=term-missing",
    "coverage_report_command": "coverage report --show-missing",
    "coverage_html_command": "coverage html",
    "pytest_args": [
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ]
}

if __name__ == "__main__":
    print("Coverage configuration loaded successfully")
    print(f"Minimum required coverage: {COVERAGE_CONFIG['minimum_coverage']}%")
    print(f"Source directories: {COVERAGE_CONFIG['source']}")