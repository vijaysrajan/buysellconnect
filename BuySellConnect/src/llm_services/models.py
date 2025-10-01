from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional
import re

class MessageRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., description="User's message to process")
    session_id: str = Field(..., description="Session identifier for chat history")

class MessageResponse(BaseModel):
    response: str = Field(..., description="AI response to the message")
    similar_messages: List[dict] = Field(default_factory=list, description="Similar messages from history")
    needs_clarification: str = Field(default="false", description="Clarification status: 'false', 'true', or 'new_thread'")

    @field_validator('needs_clarification')
    @classmethod
    def validate_needs_clarification(cls, v):
        """Ensure needs_clarification has valid values"""
        valid_values = {"false", "true", "new_thread"}
        if v not in valid_values:
            raise ValueError(f"needs_clarification must be one of: {valid_values}")
        return v

class ItemRequest(BaseModel):
    """Request model for item metadata extraction"""
    item: str = Field(
        ..., 
        description="Name of the item to analyze for metadata extraction",
        min_length=1,
        max_length=200,
        example="Mahindra Thar"
    )
    
    @validator('item')
    def validate_item_name(cls, v):
        """Validate and clean the item name"""
        if not v or not v.strip():
            raise ValueError("Item name cannot be empty")
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', v.strip())
        
        # Basic validation - no special characters that might cause issues
        if not re.match(r'^[a-zA-Z0-9\s\-_.()]+$', cleaned):
            raise ValueError("Item name contains invalid characters")
            
        return cleaned

class ItemResponse(BaseModel):
    """Response model for item metadata extraction - matches the ItemDetails from notebook"""
    item_type: str = Field(
        description="Typically this is the same as item but sometimes if the item is too specific like 'Mahindra Thar', the item type becomes 'SUV'",
        example="SUV"
    )
    category: str = Field(
        description="What is the category of this input item? Example: if Item is 'Mahindra Thar', category could be 'Car'",
        example="Car"
    )
    super_category: str = Field(
        description="What is the super category of this input item? Example: if Item is 'Fridge', super_category is 'Household Appliance'",
        example="Automobile"
    )
    attributes: List[str] = Field(
        description="Typical list of attributes for the input Item. Built by looking at typical specifications for this item. Could be up to 20 attributes.",
        example=["Engine Type", "Engine Displacement", "Power Output", "Torque", "Transmission Type"]
    )
    
    @validator('attributes')
    def validate_attributes(cls, v):
        """Ensure attributes list is not empty and has reasonable content"""
        if not v:
            raise ValueError("Attributes list cannot be empty")
        
        # Remove any empty or whitespace-only attributes
        cleaned_attributes = [attr.strip() for attr in v if attr and attr.strip()]
        
        if not cleaned_attributes:
            raise ValueError("No valid attributes provided")
            
        # Limit to reasonable number of attributes
        if len(cleaned_attributes) > 30:
            raise ValueError("Too many attributes (max 30)")
            
        return cleaned_attributes

class BatchItemRequest(BaseModel):
    """Request model for batch processing multiple items"""
    items: List[str] = Field(
        ...,
        description="List of item names to process",
        max_items=10,  # Limit batch size for API cost control
        example=["Mahindra Thar", "iPhone 15", "Samsung Refrigerator"]
    )
    
    @validator('items')
    def validate_items_list(cls, v):
        """Validate the list of items"""
        if not v:
            raise ValueError("Items list cannot be empty")
            
        if len(v) > 10:
            raise ValueError("Batch size limited to 10 items")
            
        # Validate each item using ItemRequest validation
        validated_items = []
        for item in v:
            try:
                validated_item = ItemRequest(item=item).item
                validated_items.append(validated_item)
            except Exception as e:
                raise ValueError(f"Invalid item '{item}': {str(e)}")
                
        return validated_items

class BatchItemResponse(BaseModel):
    """Response model for batch processing"""
    results: List[dict] = Field(
        description="List of results for each processed item",
        example=[
            {
                "item": "Mahindra Thar",
                "metadata": {
                    "item_type": "SUV",
                    "category": "Car",
                    "super_category": "Automobile",
                    "attributes": ["Engine Type", "Power Output"]
                },
                "success": True
            }
        ]
    )

class ErrorResponse(BaseModel):
    """Standard error response model"""
    detail: str = Field(
        description="Error message describing what went wrong",
        example="Invalid item name provided"
    )
    error_type: str = Field(
        description="Type of error that occurred",
        example="validation_error"
    )
    item: Optional[str] = Field(
        default=None,
        description="The item name that caused the error (if applicable)",
        example="Mahindra Thar"
    )

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(
        description="Health status of the service",
        example="healthy"
    )
    service: str = Field(
        description="Name of the service",
        example="Item Metadata Extractor API"
    )
    version: str = Field(
        description="Version of the API",
        example="1.0.0"
    )

class APIInfoResponse(BaseModel):
    """Root endpoint response model"""
    message: str = Field(
        description="Welcome message",
        example="Item Metadata Extractor API"
    )
    version: str = Field(
        description="API version",
        example="1.0.0"
    )
    docs: str = Field(
        description="URL for API documentation",
        example="/docs"
    )
    health: str = Field(
        description="URL for health check",
        example="/health"
    )

# Configuration model for the LangChain service
class ServiceConfig(BaseModel):
    """Configuration model for the metadata extraction service"""
    openai_api_key: str = Field(
        description="OpenAI API key for ChatGPT access"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for extraction"
    )
    temperature: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Temperature setting for the AI model"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of retries for API calls"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for API calls in seconds"
    )