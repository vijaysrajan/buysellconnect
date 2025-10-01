# Standard library imports
import logging
import os
from typing import Any, Dict, List, Optional

# Third-party imports
from pydantic import ValidationError

# LangChain imports
from langchain_core.chat_history import BaseChatMessageHistory 
# For ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# For SQLChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
#from langchain.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnableLambda


# LangChain imports - using updated import paths
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback to older import for compatibility
    from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser

# Import our models
from models import ItemResponse, ServiceConfig, MessageRequest, MessageResponse
from config import get_langchain_config, get_settings

class ChatHistoryService:
    def __init__(self):
        self.settings = get_settings()
        self._setup_llm_chain()

    def _setup_llm_chain(self):
        """Set up the LLM chain with conversation history"""
        # Check if OpenAI API key is available
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key is required for ChatHistoryService. Please set OPENAI_API_KEY environment variable.")

        # Initialize the ChatGPT model
        self.chatgpt = ChatOpenAI(
            model_name=self.settings.openai_model_name or "gpt-4o-mini",
            temperature=0.7,
            openai_api_key=self.settings.openai_api_key
        )

        # Create prompt template with history placeholder for main conversation
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Act as a helpful AI Assistant"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{human_input}"),
        ])

        # Create prompt template for message similarity judgment
        self.similarity_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a message similarity judge. Given a new user message and a list of previous messages from the conversation history, determine which (if any) previous messages are contextually similar or related to the new message. If the new message is a statement without intent and could apply to the subject of more than one message in history, make sure that you return "similar_count:multiple". If the sentence begins with "it" or "they" followed by must or should then again you need to ask for clarification from the history if it exists.

Rules:
1. If EXACTLY ONE previous message is similar/related to the new message, return: "similar_count:1" followed by that message
2. If MORE THAN ONE previous message is similar/related to the new message, return: "similar_count:multiple" followed by all similar messages, each on a new line
3. If NO previous messages are similar/related to the new message, return: "similar_count:none"

Consider messages similar if they:
- Ask about the same topic or subject
- Are follow-up questions to the same conversation thread
- Reference the same entities, concepts, or problems
- Are variations of the same question

Do not give any response more than 1 sentence of no more than 15 words.

Previous messages from history:
{previous_messages}

New user message: {new_message}"""),
            ("human", "Analyze the similarity and provide your judgment:")
        ])

        # Create basic LLM chain with memory buffer window
        self.llm_chain = (
            RunnablePassthrough.assign(history=lambda x: self._memory_buffer_window(x["history"], -1))
            | self.prompt_template
            | self.chatgpt
        )

        # Create similarity judgment chain
        self.similarity_chain = self.similarity_prompt | self.chatgpt

        # Create conversation chain with message history
        self.conv_chain = RunnableWithMessageHistory(
            self.llm_chain,
            self._get_session_history_db,
            input_messages_key="human_input",
            history_messages_key="history",
        )

    def _get_session_history_db(self, session_id: str):
        """Used to retrieve conversation history from database based on session ID"""
        return SQLChatMessageHistory(session_id, self.settings.sql_database_url or "sqlite:///memory.db")

    def _memory_buffer_window(self, messages, k=2):
        """Create a memory buffer window function to return the last K conversations"""
        if not messages:
            return []
        if k < 0:
            return messages  # Return whole history when k is negative
        return messages[-(k+1):]

    def _get_history(self, session_id: str) -> SQLChatMessageHistory:
        """Get or create chat history for a session"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.settings.sql_database_url
        )

    def _judge_message_similarity(self, new_message: str, previous_messages: List[str]) -> Dict:
        """Use LLM to judge similarity between new message and previous messages"""
        if not previous_messages:
            return {"similar_count": "none", "similar_messages": []}

        # Format previous messages for the prompt
        formatted_messages = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(previous_messages)])

        try:
            # Use the similarity chain to get LLM judgment
            response = self.similarity_chain.invoke({
                "previous_messages": formatted_messages,
                "new_message": new_message
            })

            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the LLM response
            if "similar_count:1" in response_text:
                # Extract the similar message
                lines = response_text.split('\n')
                similar_msg = None
                for line in lines:
                    if line.strip() and not line.startswith("similar_count:"):
                        similar_msg = line.strip()
                        break

                return {
                    "similar_count": "1",
                    "similar_messages": [similar_msg] if similar_msg else []
                }

            elif "similar_count:multiple" in response_text:
                # Extract all similar messages
                lines = response_text.split('\n')
                similar_msgs = []
                for line in lines:
                    if line.strip() and not line.startswith("similar_count:"):
                        similar_msgs.append(line.strip())

                return {
                    "similar_count": "multiple",
                    "similar_messages": similar_msgs
                }

            else:  # similar_count:none
                return {"similar_count": "none", "similar_messages": []}

        except Exception as e:
            logger.error(f"Error in LLM similarity judgment: {str(e)}")
            return {"similar_count": "none", "similar_messages": []}

    async def process_message(self, request: MessageRequest) -> MessageResponse:
        """Process a new message using LLM as judge for similarity detection"""
        try:
            history = self._get_history(request.session_id)

            # Get previous human messages from history
            previous_messages = history.messages
            human_messages = [msg.content for msg in previous_messages if msg.type == "human"]

            # Use LLM to judge similarity
            similarity_result = self._judge_message_similarity(request.message, human_messages)

            # Process based on similarity judgment
            if similarity_result["similar_count"] == "1":
                # Exactly one similar message found - no clarification needed
                needs_clarification = "false"
                similar_messages = [{"message": msg} for msg in similarity_result["similar_messages"]]

            elif similarity_result["similar_count"] == "multiple":
                # Multiple similar messages - clarification needed
                needs_clarification = "true"
                similar_messages = [{"message": msg} for msg in similarity_result["similar_messages"]]

                clarification_msg = "I found multiple similar messages in our history. Are you referring to any of these?\n"
                for idx, msg in enumerate(similarity_result["similar_messages"], 1):
                    clarification_msg += f"{idx}. {msg}\n"

                return MessageResponse(
                    response=clarification_msg,
                    similar_messages=similar_messages,
                    needs_clarification="true"
                )

            else:  # similarity_result["similar_count"] == "none"
                # No similar messages - new thread
                needs_clarification = "new_thread"
                similar_messages = []

            # Process message using LLM chain with conversation history
            try:
                # Use the conversation chain to get AI response with history context
                response = self.conv_chain.invoke(
                    {"human_input": request.message},
                    config={'configurable': {'session_id': request.session_id}}
                )

                # Extract the content from the response
                ai_response = response.content if hasattr(response, 'content') else str(response)

                return MessageResponse(
                    response=ai_response,
                    similar_messages=similar_messages,
                    needs_clarification=needs_clarification
                )

            except Exception as llm_error:
                logger.error(f"LLM processing error: {str(llm_error)}")
                # Fallback response - still add to history for consistency
                ai_response = f"I encountered an issue processing your message. Please try again. Error: {str(llm_error)}"

                history.add_user_message(request.message)
                history.add_ai_message(ai_response)

                return MessageResponse(
                    response=ai_response,
                    similar_messages=similar_messages,
                    needs_clarification=needs_clarification
                )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def chat_with_llm(self, prompt: str, session_id: str):
        """
        Utility function to chat with LLM and stream results live back to the user.

        Args:
            prompt: User input prompt
            session_id: Session ID for conversation history

        Yields:
            Streamed response chunks from the LLM
        """
        try:
            for chunk in self.conv_chain.stream(
                {"human_input": prompt},
                {'configurable': {'session_id': session_id}}
            ):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            yield f"Error: {str(e)}"

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