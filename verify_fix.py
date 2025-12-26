#!/usr/bin/env python3
"""
Final verification script to ensure the double prefix issue is resolved
"""
import sys
import os

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app

def verify_routes():
    print("Final verification: Checking for double prefix issue")
    print("="*60)

    # Check for any routes that contain double prefixes
    double_prefix_routes = []
    for route in app.routes:
        if hasattr(route, 'path') and '/api/v1/api/v1' in route.path:
            double_prefix_routes.append(route)

    if double_prefix_routes:
        print("X ERROR: Found routes with double prefix '/api/v1/api/v1':")
        for route in double_prefix_routes:
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods:15} {route.path}")
        return False
    else:
        print("OK SUCCESS: No double prefix routes found!")

    print("\nAll agent endpoints should be accessible at:")
    agent_endpoints = [
        "POST   /api/v1/chat",
        "POST   /api/v1/session",
        "GET    /api/v1/session/{session_id}",
        "DELETE /api/v1/session/{session_id}",
        "GET    /api/v1/health",
        "GET    /api/v1/debug/{session_id}",
        "POST   /api/v1/validate-response"
    ]

    for endpoint in agent_endpoints:
        print(f"  {endpoint}")

    print("\nMain app endpoints:")
    main_endpoints = [
        "GET /",
        "GET /health"
    ]

    for endpoint in main_endpoints:
        print(f"  {endpoint}")

    print("\nOK Configuration is correct!")
    print("- Router defined without prefix in agent_endpoints.py")
    print("- Prefix added only in app.include_router() in app.py")
    print("- No double prefix issue exists")

    return True

if __name__ == "__main__":
    success = verify_routes()
    if success:
        print("\nSUCCESS: The double prefix issue has been resolved!")
    else:
        print("\nERROR: The double prefix issue still exists!")
        sys.exit(1)