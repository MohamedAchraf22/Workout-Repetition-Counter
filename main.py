"""
Main entry point for the Exercise Counter application.

This can run in two modes:

1. GUI MODE (default when run directly):
   python main.py
   - Launches the graphical UI for real-time exercise counting

2. API MODE (for FastAPI server):
   uvicorn main:app --reload
   - Starts a local FastAPI server with REST endpoints
   - Useful for programmatic access and integration

ARCHITECTURE:
- Core ML logic (detectors, pose processing) is reusable
- Services layer handles counting logic independently
- UI and API are separate interfaces on top of services
- All can coexist without coupling

The modular design means:
✓ Services can be called from UI, API, or CLI independently
✓ No changes to core logic when adding API
✓ Clean separation of concerns
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys

# Import API router
from api import router
from db.database import Base, engine
from db import models

# ================================================================
# FASTAPI APPLICATION SETUP
# ================================================================
# This creates the FastAPI app that can be run with uvicorn

app = FastAPI(
    title="Exercise Counter API",
    description="Real-time exercise repetition counting from video",
    version="1.0.0"
)

# Add CORS middleware (allows requests from different origins)
# This is useful for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include exercise counting routes
app.include_router(router)

# ================================================================
# ROOT ENDPOINT - API DOCUMENTATION
# ================================================================
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        JSON with API info and usage instructions
    """
    return {
        "name": "Exercise Counter API",
        "version": "1.0.0",
        "description": "Counts exercise repetitions from uploaded video files",
        "endpoints": {
            "POST /api/count": "Count exercise reps from video",
            "GET /api/health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)",
        },
        "usage": "See /docs for interactive API documentation",
        "supported_exercises": ["biceps", "jumpjack", "pushup"]
    }


# ================================================================
# GUI MODE - For backwards compatibility
# ================================================================

def run_gui():
    """Launch the graphical user interface."""
    from ui import ExerciseCounterUI
    gui_app = ExerciseCounterUI()
    gui_app.run()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    # Check if running with uvicorn (API mode) or direct execution (GUI mode)
    
    # If this script is executed directly (not via uvicorn), run GUI
    # To run API mode: uvicorn main:app --reload
    
    if "uvicorn" not in sys.modules:
        # GUI mode
        print("=" * 70)
        print("EXERCISE COUNTER - GUI MODE")
        print("=" * 70)
        print()
        print("Starting graphical interface...")
        print()
        print("To run in API mode instead, use:")
        print("  uvicorn main:app --reload")
        print()
        print("API will be available at: http://localhost:8000")
        print("API docs at: http://localhost:8000/docs")
        print()
        run_gui()
    else:
        # API mode (when run via uvicorn)
        # FastAPI handles everything, this block won't execute
        pass


Base.metadata.create_all(bind=engine)