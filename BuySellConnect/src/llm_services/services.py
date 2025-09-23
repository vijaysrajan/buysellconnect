import logging
from typing import Dict, Any, Optional
import os
from pydantic import ValidationError

# LangChain imports - using updated import paths
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback to older import for compatibility
    from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser

# Import our models
from models import ItemResponse, ServiceConfig

from config import get_langchain_config

# Configure logging
logger = logging.getLogger(__name__)

from models import ServiceConfig

class ItemMetadataService:
    def __init__(self, config):
        if isinstance(config, str):
            # If config is just the API key string, create a config dict
            config = {
                "openai_api_key": config,
                "model_name": "random model", #"gpt-4o-mini",
                "temperature": 0,
                "max_retries": 3,
                "timeout_seconds": 30
            }
        self.config = ServiceConfig(**config)
        
        # Initialize the LangChain components
        self._setup_parser()
        self._setup_llm()
        self._setup_prompt()
        self._setup_chain()
        
        logger.info(f"ItemMetadataService initialized with model: {self.config.model_name}")
    
    def _load_config_from_env(self) -> ServiceConfig:
        """
        DEPRECATED: Use config.py instead
        Load configuration from environment variables
        """
        logger.warning("_load_config_from_env is deprecated. Use config.py instead.")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return ServiceConfig(
            openai_api_key=openai_key,
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30"))
        )
    
    def _setup_parser(self):
        """Set up the Pydantic output parser - same as your notebook"""
        self.parser = PydanticOutputParser(pydantic_object=ItemResponse)
        logger.debug("PydanticOutputParser initialized")
    
    def _setup_llm(self):
        """Set up the ChatOpenAI LLM - same as your notebook's chatgpt variable"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                openai_api_key=self.config.openai_api_key,
                max_retries=self.config.max_retries,
                request_timeout=self.config.timeout_seconds
            )
            logger.debug(f"ChatOpenAI initialized with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise
    
    def _setup_prompt(self):
        """Set up the prompt template - exactly from your notebook"""
        # This is the exact prompt text from your notebook
        prompt_txt = """
            You are the data cataloger and metadata specialist who takes an item name and
            lists the item type, category and lists all attributes for the item. The attributes would
            typically be what what one sees in a typical specification for the item.
            If you do not know anything about the item, just say you do not know.

            Format Instructions:
            {format_instructions}

            Item:
            {item}
            """
        
        self.prompt = PromptTemplate(
            template=prompt_txt,
            input_variables=["item"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        logger.debug("Prompt template initialized")
    
    def _setup_chain(self):
        """Set up the LangChain chain - exactly from your notebook"""
        # This matches your notebook: chain = (prompt | chatgpt | parser)
        self.chain = (
            self.prompt |
            self.llm |
            self.parser
        )
        logger.debug("LangChain pipeline initialized")
    
    def extract_metadata(self, item_name: str) -> ItemResponse:
        """
        Extract metadata for a given item name using the LangChain pipeline.
        
        Args:
            item_name: Name of the item to analyze
            
        Returns:
            ItemResponse: Structured metadata including item_type, category, super_category, and attributes
            
        Raises:
            ValueError: If the item name is invalid or empty
            Exception: If the AI service fails or returns invalid data
        """
        if not item_name or not item_name.strip():
            raise ValueError("Item name cannot be empty")
        
        item_name = item_name.strip()
        logger.info(f"Extracting metadata for item: {item_name}")
        
        try:
            # This matches your notebook: responses = chain.invoke({"item": item})
            # Note: Your notebook used {item} but it should be {"item": item}
            result = self.chain.invoke({"item": item_name})
            
            logger.info(f"Successfully extracted metadata for: {item_name}")
            logger.debug(f"Result: {result}")
            
            return result
            
        except ValidationError as ve:
            logger.error(f"Validation error for item '{item_name}': {str(ve)}")
            raise ValueError(f"Invalid response format from AI service: {str(ve)}")
            
        except Exception as e:
            logger.error(f"Error processing item '{item_name}': {str(e)}")
            
            # Check if it's an OpenAI API error
            if "openai" in str(e).lower() or "api" in str(e).lower():
                raise Exception(f"OpenAI API error: {str(e)}")
            
            # Generic error
            raise Exception(f"Failed to extract metadata: {str(e)}")
    
    def extract_metadata_batch(self, item_names: list) -> list:
        """
        Extract metadata for multiple items.
        
        Args:
            item_names: List of item names to process
            
        Returns:
            List of dictionaries with results for each item
        """
        results = []
        
        for item_name in item_names:
            try:
                metadata = self.extract_metadata(item_name)
                results.append({
                    "item": item_name,
                    "metadata": metadata.dict(),  # Convert Pydantic model to dict
                    "success": True
                })
            except Exception as e:
                logger.error(f"Failed to process item '{item_name}': {str(e)}")
                results.append({
                    "item": item_name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check by testing the service with a simple item.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test with a simple, well-known item
            test_result = self.extract_metadata("Apple iPhone")
            
            return {
                "status": "healthy",
                "llm_model": self.config.model_name,
                "parser_ready": True,
                "test_successful": True,
                "last_test_item": "Apple iPhone"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "llm_model": self.config.model_name,
                "parser_ready": True,
                "test_successful": False,
                "error": str(e)
            }

# Utility function to create service instance
def create_metadata_service(openai_api_key: Optional[str] = None) -> ItemMetadataService:
    """
    Factory function to create ItemMetadataService instance.
    
    Args:
        openai_api_key: Optional API key. If not provided, will use environment variable.
        
    Returns:
        Configured ItemMetadataService instance
    """
    if openai_api_key:
        config = ServiceConfig(openai_api_key=openai_api_key)
        return ItemMetadataService(config)
    else:
        return ItemMetadataService()

# For testing/debugging - matches your notebook testing pattern
if __name__ == "__main__":
    # This allows you to test the service directly
    import json
    
    try:
        service = create_metadata_service()
        
        # Test with the same item from your notebook
        test_item = "Mahindra Thar"
        result = service.extract_metadata(test_item)
        
        print(f"Item: {test_item}")
        print("Result:")
        print(json.dumps(result.dict(), indent=4))
        
    except Exception as e:
        print(f"Error: {str(e)}")