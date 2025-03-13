from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
import logging
from datetime import datetime
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recommendation_model import RecommendationModel

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="Real-time recommendation system API",
    version="1.0.0"
)

# Prometheus metrics
RECOMMENDATION_REQUESTS = Counter(
    'recommendation_requests_total',
    'Total number of recommendation requests'
)
RECOMMENDATION_LATENCY = Histogram(
    'recommendation_latency_seconds',
    'Latency of recommendation requests'
)

# Initialize model (in production, load from saved model)
MODEL = None

class UserPreference(BaseModel):
    user_id: int
    item_pool: List[int]
    num_recommendations: Optional[int] = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    latency_ms: float
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global MODEL
    try:
        # Set TensorFlow to use CPU only for this demo
        tf.config.set_visible_devices([], 'GPU')
        
        # Initialize the model with small dimensions for testing
        MODEL = RecommendationModel(num_users=100, num_items=100)
        
        # Compile the model with default optimizer
        MODEL.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Run a small forward pass to ensure the model is properly initialized
        test_user = tf.constant([[1]])
        test_item = tf.constant([[1]])
        _ = MODEL((test_user, test_item))
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(preferences: UserPreference):
    """Get personalized recommendations for a user."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    RECOMMENDATION_REQUESTS.inc()
    start_time = datetime.now()
    
    try:
        with RECOMMENDATION_LATENCY.time():
            # Get recommendations from model
            recommendations = MODEL.get_top_k_recommendations(
                user_id=preferences.user_id,
                item_pool=preferences.item_pool,
                k=preferences.num_recommendations
            )
            
            # Format recommendations
            formatted_recommendations = [
                {"item_id": int(item_id), "score": float(score)}
                for item_id, score in recommendations
            ]
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return RecommendationResponse(
                user_id=preferences.user_id,
                recommendations=formatted_recommendations,
                latency_ms=latency_ms,
                timestamp=end_time.isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 