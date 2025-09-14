from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Simple imports - use minimal service without config dependencies
from models import ItemRequest, ItemResponse
from services_minimal import ItemMetadataService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Item Metadata Extractor API",
    description="Extract category and attributes for items using AI",
    version="1.0.0"
)

# Add CORS middleware with simple settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No global service - create fresh for each request to avoid startup issues
@app.post("/extract-metadata", response_model=ItemResponse)
async def extract_metadata(request: ItemRequest):
    """Extract metadata for an item"""
    try:
        logger.info(f"Processing item: {request.item}")
        
        # Create service instance for this request
        service = ItemMetadataService()
        result = service.extract_metadata(request.item)
        
        logger.info(f"Success for '{request.item}': {result.item_type}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing '{request.item}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Item Metadata Extractor API"}

@app.get("/")
async def root():
    return {
        "message": "Item Metadata Extractor API", 
        "docs": "/docs",
        "version": "1.0.0"
    }

# Test endpoint to verify setup
@app.get("/test")
async def test():
    """Quick test endpoint"""
    try:
        service = ItemMetadataService()
        result = service.extract_metadata("iPhone")
        return {"test": "passed", "sample_result": result}
    except Exception as e:
        return {"test": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is required")
        exit(1)
    
    print("Starting simple FastAPI server...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main_simple:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )