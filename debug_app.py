#!/usr/bin/env python3
"""
Debug script to check what the app actually contains
"""
import sys
import os
import asyncio

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

async def debug_app():
    print("Debugging the app configuration...")

    # Import the app
    from app import app

    print(f"App title: {app.title}")
    print(f"App description: {app.description}")

    # Find the root endpoint
    root_route = None
    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/':
            root_route = route
            break

    if root_route:
        print(f"Root endpoint found: {root_route.path}")
        print(f"Root endpoint function: {root_route.endpoint.__name__}")

        # Try to call the root function directly
        try:
            result = await root_route.endpoint()
            print(f"Root endpoint result: {result}")
        except Exception as e:
            print(f"Error calling root endpoint: {e}")
    else:
        print("No root endpoint found!")

    # Print all routes
    print("\nAll routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods:15} {route.path}")

if __name__ == "__main__":
    asyncio.run(debug_app())