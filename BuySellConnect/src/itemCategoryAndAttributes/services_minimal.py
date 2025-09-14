import logging
import os
from typing import Dict, Any

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Only import the response model, not config
from models import ItemResponse

logger = logging.getLogger(__name__)

class ItemMetadataService:
    """
    Minimal service class for extracting item metadata.
    Uses environment variables directly - no config system.
    """
    
    def __init__(self):
        """Initialize with direct environment variable access"""
        # Get configuration directly from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        
        # Set up the LangChain components
        self._setup_parser()
        self._setup_llm()
        self._setup_prompt()
        self._setup_chain()
        
        logger.info(f"ItemMetadataService initialized with model: {self.model_name}")
    
    def _setup_parser(self):
        """Set up the Pydantic output parser"""
        self.parser = PydanticOutputParser(pydantic_object=ItemResponse)
    
    def _setup_llm(self):
        """Set up the ChatOpenAI LLM"""
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            max_retries=3,
            request_timeout=30
        )
    
    def _setup_prompt(self):
        """Set up the prompt template - exactly from your notebook"""
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
    
    def _setup_chain(self):
        """Set up the LangChain chain"""
        self.chain = (
            self.prompt |
            self.llm |
            self.parser
        )
    
    def extract_metadata(self, item_name: str) -> ItemResponse:
        """
        Extract metadata for a given item name.
        
        Args:
            item_name: Name of the item to analyze
            
        Returns:
            ItemResponse: Structured metadata
        """
        if not item_name or not item_name.strip():
            raise ValueError("Item name cannot be empty")
        
        item_name = item_name.strip()
        logger.info(f"Extracting metadata for item: {item_name}")
        
        try:
            result = self.chain.invoke({"item": item_name})
            logger.info(f"Successfully extracted metadata for: {item_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing item '{item_name}': {str(e)}")
            raise Exception(f"Failed to extract metadata: {str(e)}")

# For testing
if __name__ == "__main__":
    import json
    
    try:
        service = ItemMetadataService()
        result = service.extract_metadata("Mahindra Thar")
        print("Success!")
        print(json.dumps(result.dict(), indent=4))
    except Exception as e:
        print(f"Error: {e}")