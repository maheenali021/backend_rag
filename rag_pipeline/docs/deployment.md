# Deployment Documentation for RAG Pipeline

## Overview
This document describes the deployment process for the AI Robotics textbook website and RAG pipeline.

## Website Deployment
The Docusaurus-based book website is deployed to GitHub Pages using a GitHub Actions workflow.

### GitHub Pages URL
The website is accessible at: https://maheenali021.github.io/hackathon-01-AI-textbook/

### Deployment Process
1. Changes to the `main` branch trigger the deployment workflow
2. The workflow runs in the `.github/workflows/deploy.yml` file
3. It builds the Docusaurus site in the `frontend/` directory
4. The built site is deployed to the `gh-pages` branch
5. GitHub Pages serves the site from the `gh-pages` branch

### Deployment Configuration
- **Base URL**: `/hackathon-01-AI-textbook/`
- **Deployment Branch**: `gh-pages`
- **Organization Name**: `maheenali021`
- **Project Name**: `hackathon-01-AI-textbook`

## RAG Pipeline Deployment
The RAG pipeline components are designed to run as standalone scripts that can be executed on demand or scheduled.

### Configuration
The pipeline uses environment variables for configuration, stored in a `.env` file:
- `COHERE_API_KEY`: API key for Cohere embedding service
- `QDRANT_URL`: URL for Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `WEBSITE_URL`: URL of the deployed book website (default: GitHub Pages URL)

### Execution
The RAG pipeline can be executed by running the main pipeline script once all dependencies are installed and configuration is set up.