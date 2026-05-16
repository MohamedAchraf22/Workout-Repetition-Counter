"""
API Response schemas for exercise counting endpoints.

These Pydantic models define the structure of API responses.
They ensure type safety and automatic OpenAPI documentation.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class ExerciseCountResponse(BaseModel):
    """
    Response model for exercise counting results.
    
    Attributes:
        success: Whether processing completed successfully
        exercise: Type of exercise that was counted
        count: Total number of reps/repetitions detected
        message: Human-readable status message
        debug_info: Optional debug information (empty in production)
    """
    success: bool
    exercise: str
    count: int
    message: str
    debug_info: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    
    Attributes:
        success: Always False for errors
        error: Error type/category
        message: Detailed error message
    """
    success: bool = False
    error: str
    message: str
