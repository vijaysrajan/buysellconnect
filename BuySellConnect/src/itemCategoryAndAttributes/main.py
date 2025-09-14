from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import List
import os

# We'll import these from other files in the next steps
from models import ItemRequest, ItemResponse
from services import ItemMetadataService
from config import get_settings

# For now, define models here (we'll move these to models.py later)
class ItemRequest(BaseModel):
    item: str = Field(..., description="Name of the item to analyze", min_length=1, max_length=200)

class ItemResponse(BaseModel):
    item_type: str = Field(description="Typically this the same as item but sometimes if the item is too specific like Mahindra Thar, the item type is SUV")
    category: str = Field(description="What is the category of this input item? Example if Item is Mahindra Thar, this could be car.")
    super_category: str = Field(description="What is the category of this input item? Example if Item is Fridge, category is Household Appliance.")
    attributes: List[str] = Field(description="Typical list of attributes for the input Item. This could be built by looking at the typical specification for this item.")

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

# Initialize FastAPI app
app = FastAPI(
    title="Item Metadata Extractor API",
    description="Extract category and attributes for items using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the initialized chain
# This will be set up during startup
langchain_service = None

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the LangChain service when the app starts"""
#     global langchain_service
    
#     logger.info("Starting up Item Metadata Extractor API...")
    
#     try:
#         # Check for OpenAI API key
#         openai_key = os.getenv("OPENAI_API_KEY")
#         if not openai_key:
#             raise ValueError("OPENAI_API_KEY environment variable is required")
        
#         # Initialize the LangChain service (we'll create this class in services.py)
#         # For now, this is a placeholder
#         logger.info("Initializing LangChain service...")
#         langchain_service = ItemMetadataService(openai_key)
#         logger.info("LangChain service initialized successfully")
        
#         logger.info("API startup completed successfully")
        
#     except Exception as e:
#         logger.error(f"Failed to start up API: {str(e)}")
#         raise e


from services import create_metadata_service

@app.on_event("startup")
async def startup_event():
    """Initialize the LangChain service when the app starts"""
    global langchain_service
    
    logger.info("Starting up Item Metadata Extractor API...")
    
    try:
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        print(os.getenv("OPENAI_MODEL_NAME"))

        print("_________________________")
        print(f"OpenAI Key: {openai_key}")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Create config dictionary
        config = {
            "openai_api_key": openai_key,
            "model_name": os.getenv("OPENAI_MODEL_NAME", "random_model"), #"gpt-4o-mini"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "timeout_seconds": int(os.getenv("TIMEOUT_SECONDS", "30"))
        }
        
        logger.info("Initializing LangChain service...")
        langchain_service = ItemMetadataService(config)
        logger.info("LangChain service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to start up API: {str(e)}")
        raise e



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down"""
    logger.info("Shutting down Item Metadata Extractor API...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Item Metadata Extractor API",
        "version": "1.0.0"
    }

# Main API endpoint
@app.post(
    "/extract-metadata",
    response_model=ItemResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def extract_item_metadata(request: ItemRequest):
    """
    Extract metadata for an item including category, type, and attributes.
    """
    
    try:
        logger.info(f"Processing request for item: {request.item}")
        logger.info(f"Global service status: {langchain_service is not None}")
        
        # If global service is None, create a new one for this request
        if langchain_service is None:
            logger.warning("Global service is None, creating new instance")
            try:
                temp_service = ItemMetadataService()
                result = temp_service.extract_metadata(request.item)
                logger.info(f"Successfully processed item with temp service: {request.item}")
                return result
            except Exception as temp_error:
                logger.error(f"Failed to create temp service: {temp_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Service initialization failed: {str(temp_error)}"
                )
        
        # Use the global service
        result = langchain_service.extract_metadata(request.item)
        logger.info(f"Successfully processed item: {request.item}")
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error for item '{request.item}': {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        logger.error(f"Error processing item '{request.item}': {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# Additional endpoint for batch processing (optional)
@app.post("/extract-metadata-batch")
async def extract_metadata_batch(items: List[str]):
    """
    Extract metadata for multiple items in a single request.
    Note: This might be expensive with OpenAI API costs.
    """
    if len(items) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Batch size limited to 10 items"
        )
    
    results = []
    for item in items:
        try:
            request = ItemRequest(item=item)
            result = await extract_item_metadata(request)
            results.append({"item": item, "metadata": result, "success": True})
        except Exception as e:
            results.append({"item": item, "error": str(e), "success": False})
    
    return {"results": results}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Remove in production
        log_level="info"
    )