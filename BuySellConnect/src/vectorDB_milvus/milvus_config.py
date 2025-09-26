import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pymilvus import connections
import json

@dataclass
class MilvusConfig:
    """Configuration class for Milvus connections"""
    host: str = "localhost"
    port: str = "19530"
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False
    alias: str = "default"
    
    # Cloud-specific settings
    cloud_provider: Optional[str] = None  # 'zilliz', 'aws', 'gcp', 'azure'
    api_key: Optional[str] = None
    cluster_endpoint: Optional[str] = None

    #ENV to set
    # MILVUS_HOST
    # MILVUS_PORT
    # MILVUS_USERNAME
    # MILVUS_PASSWORD
    # MILVUS_SECURE
    # MILVUS_ALIAS
    # MILVUS_CLOUD_PROVIDER
    # MILVUS_API_KEY
    # MILVUS_CLUSTER_ENDPOINT


    
    @classmethod
    def from_env(cls, env_prefix: str = "MILVUS_"):
        """Create config from environment variables"""
        print(f"username = {os.getenv(f"{env_prefix}USERNAME"), password = {os.getenv(f"{env_prefix}PASSWORD")}")
        return cls(
            host=os.getenv(f"{env_prefix}HOST", "localhost"),
            port=os.getenv(f"{env_prefix}PORT", "19530"),
            username=os.getenv(f"{env_prefix}USERNAME"),
            password=os.getenv(f"{env_prefix}PASSWORD"),
            secure=os.getenv(f"{env_prefix}SECURE", "false").lower() == "true",
            alias=os.getenv(f"{env_prefix}ALIAS", "default"),
            cloud_provider=os.getenv(f"{env_prefix}CLOUD_PROVIDER"),
            api_key=os.getenv(f"{env_prefix}API_KEY"),
            cluster_endpoint=os.getenv(f"{env_prefix}CLUSTER_ENDPOINT")
        )
    
    @classmethod
    def from_file(cls, config_file: str):
        """Create config from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_connection_params(self) -> Dict[str, Any]:
        """Convert to pymilvus connection parameters"""
        params = {
            "alias": self.alias,
            "host": self.host,
            "port": self.port
        }
        
        if self.username:
            params["user"] = self.username
        if self.password:
            params["password"] = self.password
        if self.secure:
            params["secure"] = True
        
        return params

class MilvusConnectionManager:
    """Manages multiple Milvus connections"""
    
    def __init__(self):
        self.configs = {}
        self.active_connections = set()
    
    def add_config(self, name: str, config: MilvusConfig):
        """Add a named configuration"""
        self.configs[name] = config
    
    def connect(self, config_name: str = "default") -> str:
        """Connect using a named configuration"""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        config = self.configs[config_name]
        connection_params = config.to_connection_params()
        
        try:
            connections.connect(**connection_params)
            self.active_connections.add(config.alias)
            print(f"Connected to Milvus ({config_name}): {config.host}:{config.port}")
            return config.alias
        except Exception as e:
            print(f"Failed to connect to {config_name}: {e}")
            raise
    
    def disconnect(self, alias: str = "default"):
        """Disconnect from Milvus"""
        try:
            connections.disconnect(alias)
            self.active_connections.discard(alias)
            print(f"Disconnected from {alias}")
        except Exception as e:
            print(f"Error disconnecting from {alias}: {e}")
    
    def disconnect_all(self):
        """Disconnect from all active connections"""
        for alias in list(self.active_connections):
            self.disconnect(alias)

# Predefined configurations for different environments
class MilvusConfigs:
    """Predefined configurations for common setups"""
    
    @staticmethod
    def local_docker():
        """Local Docker setup"""
        return MilvusConfig(
            host="localhost",
            port="19530",
            alias="local"
        )
    
    @staticmethod
    def zilliz_cloud(cluster_endpoint: str, api_key: str):
        """Zilliz Cloud configuration"""
        return MilvusConfig(
            host=cluster_endpoint,
            port="443",
            username="db_admin",  # Default for Zilliz Cloud
            password=api_key,
            secure=True,
            cloud_provider="zilliz",
            cluster_endpoint=cluster_endpoint,
            api_key=api_key,
            alias="zilliz"
        )
    
    @staticmethod
    def aws_managed(host: str, port: str = "19530", username: str = None, password: str = None):
        """AWS managed Milvus"""
        return MilvusConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            secure=True,
            cloud_provider="aws",
            alias="aws"
        )
    
    @staticmethod
    def custom_cloud(host: str, port: str, username: str, password: str, provider: str):
        """Custom cloud configuration"""
        return MilvusConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            secure=True,
            cloud_provider=provider,
            alias=f"{provider}_cloud"
        )

# Example usage
if __name__ == "__main__":
    # Create connection manager
    manager = MilvusConnectionManager()
    
    # Add local configuration
    local_config = MilvusConfigs.local_docker()
    manager.add_config("local", local_config)
    
    # Add environment-based config
    env_config = MilvusConfig.from_env("MILVUS_PROD_")
    manager.add_config("production", env_config)
    
    # Connect to local
    try:
        manager.connect("local")
        print("Local connection successful!")
    except Exception as e:
        print(f"Local connection failed: {e}")
    
    # Example environment variables setup
    print("\n=== Environment Variables Setup Example ===")
    print("For production, set these environment variables:")
    print("MILVUS_PROD_HOST=your-cloud-host.com")
    print("MILVUS_PROD_PORT=443")
    print("MILVUS_PROD_USERNAME=your-username")
    print("MILVUS_PROD_PASSWORD=your-password")
    print("MILVUS_PROD_SECURE=true")
    print("MILVUS_PROD_CLOUD_PROVIDER=zilliz")