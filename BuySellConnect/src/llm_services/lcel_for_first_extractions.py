from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Pydantic models for structured outputs
class UserIntent(BaseModel):
    intent: str = Field(description="User intent: buyer, seller, connector, rentor, lessor, etc.")
    confidence: float = Field(description="Confidence score 0-1")

class ItemFocus(BaseModel):
    item: str = Field(description="The main item the user is talking about")
    context: str = Field(description="Additional context about the item")

class ItemDetails(BaseModel):
    category: str = Field(description="Item category (e.g., electronics, furniture, vehicle)")
    subcategory: str = Field(description="Item subcategory if applicable")
    description: str = Field(description="Detailed description of the item")

class ItemAttributes(BaseModel):
    attributes: List[Dict[str, str]] = Field(description="List of item attributes with keys and values")
    missing_attributes: List[str] = Field(description="List of attributes that need to be collected")

class MakeYearModelCheck(BaseModel):
    applies: bool = Field(description="Whether make, year, and model apply to this item")
    reasoning: str = Field(description="Explanation of why it does or doesn't apply")

# Mock database/cache functions (replace with actual implementation)
def check_cache_db(item_category: str) -> Optional[Dict]:
    """Check if item details exist in cache/database"""
    # Mock implementation - replace with actual database lookup
    cache = {
        "smartphone": {"category": "electronics", "subcategory": "mobile_device"},
        "car": {"category": "vehicle", "subcategory": "automobile"},
        "laptop": {"category": "electronics", "subcategory": "computer"}
    }
    return cache.get(item_category.lower())

def store_in_db(item_details: Dict) -> bool:
    """Store item details in database"""
    # Mock implementation - replace with actual database storage
    print(f"Storing in DB: {item_details}")
    return True

# Step 1: Check if related to old posting
old_posting_prompt = ChatPromptTemplate.from_template(
    """Analyze the user statement and determine if it's related to an old/previous posting or listing.
    
    User statement: {user_statement}
    
    Respond with only 'yes' or 'no'."""
)

old_posting_chain = old_posting_prompt | llm | StrOutputParser()

# Step 2: Get user intent
intent_prompt = ChatPromptTemplate.from_template(
    """Analyze the user statement to determine their intent. Are they a:
    - buyer (looking to purchase/acquire)
    - seller (looking to sell/offer) 
    - connector (introducing/connecting people)
    - rentor (looking to rent something)
    - lessor (looking to rent out something)
    - other (specify)
    
    User statement: {user_statement}
    
    {format_instructions}"""
)

intent_parser = JsonOutputParser(pydantic_object=UserIntent)
intent_prompt = intent_prompt.partial(format_instructions=intent_parser.get_format_instructions())
intent_chain = intent_prompt | llm | intent_parser

# Step 3: Extract item in focus
item_focus_prompt = ChatPromptTemplate.from_template(
    """Extract the main item or object that the user is talking about from their statement.
    
    User statement: {user_statement}
    User intent: {intent}
    
    {format_instructions}"""
)

item_focus_parser = JsonOutputParser(pydantic_object=ItemFocus)
item_focus_prompt = item_focus_prompt.partial(format_instructions=item_focus_parser.get_format_instructions())
item_focus_chain = item_focus_prompt | llm | item_focus_parser

# Step 4: Extract item category and details with cache/DB logic
category_prompt = ChatPromptTemplate.from_template(
    """Extract detailed category and description information for this item.
    
    Item: {item}
    Context: {context}
    User statement: {user_statement}
    
    {format_instructions}"""
)

category_parser = JsonOutputParser(pydantic_object=ItemDetails)
category_prompt = category_prompt.partial(format_instructions=category_parser.get_format_instructions())
category_llm_chain = category_prompt | llm | category_parser

def extract_with_cache(data: Dict) -> Dict:
    """Extract item details using cache/DB first, then LLM if not found"""
    item = data["item_focus"]["item"]
    
    # Check cache/DB first
    cached_details = check_cache_db(item)
    if cached_details:
        print(f"Found in cache: {cached_details}")
        return {**data, "item_details": cached_details, "from_cache": True}
    
    # Use LLM if not in cache
    llm_result = category_llm_chain.invoke(data)
    
    # Store in DB for future use
    store_in_db(llm_result)
    
    return {**data, "item_details": llm_result, "from_cache": False}

# Step 5: Get item attributes
attributes_prompt = ChatPromptTemplate.from_template(
    """Based on the item and its category, identify at least 3 important attributes that should be collected from the user.
    For the given user statement, extract any attributes that are already mentioned and identify which ones are missing.
    
    Item: {item}
    Category: {category}
    User statement: {user_statement}
    
    Common attributes might include: condition, size, color, brand, price, location, age, model, etc.
    
    {format_instructions}"""
)

attributes_parser = JsonOutputParser(pydantic_object=ItemAttributes)
attributes_prompt = attributes_prompt.partial(format_instructions=attributes_parser.get_format_instructions())
attributes_chain = attributes_prompt | llm | attributes_parser

# Step 6: Check if make, year, model apply
make_year_model_prompt = ChatPromptTemplate.from_template(
    """Determine if the attributes "make", "year", and "model" are applicable to this item category.
    These typically apply to vehicles, electronics, appliances, tools, etc. but not to generic items like clothing, food, books, etc.
    
    Item: {item}
    Category: {category}
    
    {format_instructions}"""
)

make_year_model_parser = JsonOutputParser(pydantic_object=MakeYearModelCheck)
make_year_model_prompt = make_year_model_prompt.partial(format_instructions=make_year_model_parser.get_format_instructions())
make_year_model_chain = make_year_model_prompt | llm | make_year_model_parser

# Helper function to handle old posting case
def handle_old_posting(data: Dict) -> Dict:
    return {
        "status": "old_posting_detected",
        "user_statement": data["user_statement"],
        "message": "This appears to be related to an old posting. Please handle accordingly."
    }

# Helper function to format final output
def format_final_output(data: Dict) -> Dict:
    item_details = data.get("item_details", {})
    
    return {
        "user_statement": data["user_statement"],
        "intent": data["intent"],
        "item_focus": data["item_focus"],
        "item_details": item_details,
        "from_cache": data.get("from_cache", False),
        "item_attributes": data["item_attributes"],
        "make_year_model_applies": data["make_year_model_check"],
        "status": "completed"
    }

# Main LCEL Pipeline
def create_item_processing_pipeline():
    """Create the complete LCEL pipeline following the flowchart"""
    
    # Branch logic for old posting check
    old_posting_branch = RunnableBranch(
        # If old posting detected, handle separately
        (lambda x: x.get("is_old_posting", "").lower().strip() == "yes", 
         RunnableLambda(handle_old_posting)),
        
        # Otherwise, continue with main flow
        {
            "user_statement": RunnablePassthrough(),
            "intent": intent_chain,
        }
        | RunnablePassthrough.assign(
            item_focus=item_focus_chain
        )
        | RunnablePassthrough.assign(
            **RunnableLambda(extract_with_cache).invoke
        )
        | RunnablePassthrough.assign(
            item_attributes=attributes_chain.invoke
        )
        | RunnablePassthrough.assign(
            make_year_model_check=make_year_model_chain.invoke
        )
        | RunnableLambda(format_final_output)
    )
    
    # Complete pipeline
    pipeline = (
        {
            "user_statement": RunnablePassthrough(),
            "is_old_posting": old_posting_chain,
        }
        | old_posting_branch
    )
    
    return pipeline

# Usage example
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = create_item_processing_pipeline()
    
    # Test cases
    test_statements = [
        "I want to sell my 2020 Honda Civic in excellent condition",
        "Looking for a used laptop under $500",
        "Need help with my previous listing from last week",
        "Have a vintage guitar, need to find the right buyer"
    ]
    
    for statement in test_statements:
        print(f"\n{'='*50}")
        print(f"Processing: {statement}")
        print(f"{'='*50}")
        
        try:
            result = pipeline.invoke(statement)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error processing statement: {e}")

# Additional utility functions
def get_missing_attributes_questions(attributes_data: Dict) -> List[str]:
    """Generate questions to collect missing attributes"""
    missing = attributes_data.get("missing_attributes", [])
    questions = []
    
    for attr in missing:
        questions.append(f"What is the {attr} of your item?")
    
    return questions

def continue_conversation(pipeline_result: Dict, user_response: str) -> Dict:
    """Continue the conversation to collect missing attributes"""
    # This would be expanded to handle follow-up questions
    # and update the item attributes based on user responses
    pass