#!/bin/bash

# Deployment script for RAG Pipeline
# This script sets up and runs the RAG pipeline for the AI Robotics textbook

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG Pipeline Deployment${NC}"

# Function to print status messages
print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in the correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "pipelines" ]; then
    print_error "Please run this script from the backend/rag_pipeline directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

print_status "Python3 found: $(python3 --version)"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv

    print_status "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
print_status "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found. Please create one based on .env.example"
    print_status "Copy the example file: cp .env.example .env"
    exit 1
fi

# Load environment variables
set +e  # Temporarily disable exit on error for env loading
source .env
set -e  # Re-enable exit on error

# Validate required environment variables
if [ -z "$COHERE_API_KEY" ] || [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
    print_error "Required environment variables are not set in .env file"
    print_status "Please set COHERE_API_KEY, QDRANT_URL, and QDRANT_API_KEY"
    exit 1
fi

print_status "Environment variables are set"

# Validate configuration
print_status "Validating configuration..."
python -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from rag_pipeline.utils.config_validator import ConfigValidator
validator = ConfigValidator()
is_valid, errors = validator.validate_config()

if not is_valid:
    print('Configuration validation failed:')
    for error in errors:
        print(f'  - {error}')
    sys.exit(1)
else:
    print('Configuration validation passed')
"

# Run the RAG pipeline
print_status "Running RAG pipeline..."
python run_rag_pipeline.py

# Check if pipeline ran successfully
if [ $? -eq 0 ]; then
    print_success "RAG pipeline completed successfully!"
else
    print_error "RAG pipeline failed!"
    exit 1
fi

# Optional: Run tests to verify deployment
read -p "Run tests to verify deployment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running tests..."

    # Find all test files
    TEST_FILES=$(find . -name "test_*.py" -path "*/tests/*")

    if [ -z "$TEST_FILES" ]; then
        print_error "No test files found"
    else
        for test_file in $TEST_FILES; do
            print_status "Running $test_file"
            python -m pytest "$test_file" -v
            if [ $? -ne 0 ]; then
                print_error "Tests failed in $test_file"
                exit 1
            fi
        done
        print_success "All tests passed!"
    fi
fi

# Create a deployment report
print_status "Creating deployment report..."
DEPLOYMENT_REPORT="deployment_report_$(date +%Y%m%d_%H%M%S).txt"
cat > "$DEPLOYMENT_REPORT" << EOF
RAG Pipeline Deployment Report
=============================

Deployment Date: $(date)
Python Version: $(python3 --version)
Virtual Environment: $VIRTUAL_ENV
Working Directory: $(pwd)

Environment Variables:
- Cohere API Key: $(if [ -n "$COHERE_API_KEY" ]; then echo "SET"; else echo "NOT SET"; fi)
- Qdrant URL: $(if [ -n "$QDRANT_URL" ]; then echo "$QDRANT_URL"; else echo "NOT SET"; fi)
- Qdrant API Key: $(if [ -n "$QDRANT_API_KEY" ]; then echo "SET"; else echo "NOT SET"; fi)
- Website URL: $WEBSITE_URL

Configuration Validation: PASSED

Pipeline Results:
- Status: COMPLETED
- Content Chunks Processed: (check pipeline output)
- Embeddings Generated: (check pipeline output)
- Storage Success: (check pipeline output)

Deployment completed successfully.
EOF

print_success "Deployment completed successfully!"
print_success "Deployment report saved to: $DEPLOYMENT_REPORT"

echo
echo -e "${GREEN}RAG Pipeline is now ready to use!${NC}"
echo "To perform semantic searches, use the SemanticSearch class in your application."