#!/usr/bin/env python3
"""
Test script to verify the actual routes in the FastAPI application
"""
import sys
import os

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app

def test_routes():
    print("Testing actual routes in the FastAPI application...")
    print("="*50)

    # Print all routes
    print("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods:15} {route.path}")

    print("\nExpected behavior:")
    print("  - Main app endpoints should be at root level: /, /health")
    print("  - Agent endpoints should be prefixed with /api/v1: /api/v1/chat, /api/v1/session, etc.")
    print("  - No double prefix like /api/v1/api/v1 should exist")

    # Check for potential double prefixes
    double_prefix_routes = [route for route in app.routes
                           if hasattr(route, 'path') and '/api/v1/api/v1' in route.path.lower()]

    if double_prefix_routes:
        print(f"\nERROR: Found routes with double prefix '/api/v1/api/v1':")
        for route in double_prefix_routes:
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods:15} {route.path}")
    else:
        print(f"\nOK: No double prefix routes found")

    # Check for expected agent routes
    expected_routes = [
        '/api/v1/chat',
        '/api/v1/session',
        '/api/v1/session/{session_id}',
        '/api/v1/health',
        '/api/v1/debug/{session_id}',
        '/api/v1/validate-response'
    ]

    print(f"\nChecking for expected agent routes:")
    for expected_route in expected_routes:
        found = any(hasattr(route, 'path') and route.path == expected_route for route in app.routes)
        status = "OK" if found else "MISSING"
        print(f"  {status} {expected_route}")

if __name__ == "__main__":
    test_routes()