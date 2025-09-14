import os
from typing import Optional
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
import logging
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic BaseSettings for automatic env var loading and validation.
    """
    
    # Pydantic V2 model configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        env_prefix="",
        #extra="forbid"  # This prevents the extra_forbidden error
    )
    
    # API Configuration
    app_name: str = Field(default="Item Metadata Extractor API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload on changes (development only)")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model_name: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Model temperature")
    openai_max_retries: int = Field(default=3, ge=1, le=10, description="Max API retries")
    openai_timeout_seconds: int = Field(default=30, ge=5, le=300, description="API timeout in seconds")
    
    # CORS Configuration
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: list = Field(default=["*"], description="Allowed HTTP methods")
    cors_allow_headers: list = Field(default=["*"], description="Allowed HTTP headers")
    
    # API Limits
    max_batch_size: int = Field(default=10, ge=1, le=50, description="Maximum items in batch request")
    max_item_name_length: int = Field(default=200, ge=1, le=1000, description="Maximum item name length")
    max_attributes_count: int = Field(default=30, ge=1, le=100, description="Maximum attributes per item")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format"""
        if not v:
            raise ValueError("OpenAI API key is required")
        if not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v):
        """Ensure CORS origins is a list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

class DevelopmentSettings(Settings):
    """Development-specific settings"""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"

class ProductionSettings(Settings):
    """Production-specific settings"""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: list = Field(default=[], description="Specific origins for production")

class TestSettings(Settings):
    """Test-specific settings"""
    debug: bool = True
    openai_api_key: str = Field(default="sk-test-key", description="Test API key")
    log_level: str = "DEBUG"

def get_settings() -> Settings:
    """
    Factory function to get settings based on environment.
    
    Returns:
        Settings: Configuration object based on APP_ENV environment variable
    """
    env = os.getenv("APP_ENV", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()

def setup_logging(settings: Settings) -> None:
    """
    Set up logging configuration based on settings.
    
    Args:
        settings: Application settings object
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
        force=True  # Override any existing logging config
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    if settings.debug:
        logging.getLogger("langchain").setLevel(logging.DEBUG)
    else:
        logging.getLogger("langchain").setLevel(logging.WARNING)

# Global settings instance
_settings: Optional[Settings] = None

def get_cached_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).
    
    Returns:
        Settings: Cached settings object
    """
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings

# Configuration for LangChain service specifically
def get_langchain_config() -> dict:
    """
    Get configuration specific to LangChain service.
    
    Returns:
        dict: Configuration dictionary for ItemMetadataService
    """
    settings = get_cached_settings()
    
    return {
        "openai_api_key": settings.openai_api_key,
        "model_name": settings.openai_model_name,
        "temperature": settings.openai_temperature,
        "max_retries": settings.openai_max_retries,
        "timeout_seconds": settings.openai_timeout_seconds
    }

# For development convenience
if __name__ == "__main__":
    # Test configuration loading
    import json
    
    try:
        settings = get_settings()
        print("Configuration loaded successfully:")
        print(json.dumps(settings.model_dump(), indent=2, default=str))
        
        # Test LangChain config
        langchain_config = get_langchain_config()
        print("\nLangChain configuration:")
        print(json.dumps(langchain_config, indent=2, default=str))
        
    except Exception as e:
        print(f"Configuration error: {e}")