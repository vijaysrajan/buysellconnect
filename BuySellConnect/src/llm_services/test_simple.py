#!/usr/bin/env python3
"""
Simple test to check what's working and what's not
"""
import os

print("=== Dependency Check ===")
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError as e:
    print(f"❌ FastAPI missing: {e}")

try:
    import pydantic
    print("✅ Pydantic installed")
except ImportError as e:
    print(f"❌ Pydantic missing: {e}")

try:
    import langchain
    print("✅ LangChain installed")
except ImportError as e:
    print(f"❌ LangChain missing: {e}")

try:
    from langchain_openai import ChatOpenAI
    print("✅ LangChain OpenAI installed")
except ImportError as e:
    print(f"❌ LangChain OpenAI missing: {e}")

print("\n=== File Check ===")
files_needed = ["main.py", "models.py", "services.py", "config.py"]
for file in files_needed:
    if os.path.exists(file):
        print(f"✅ {file} exists")
    else:
        print(f"❌ {file} missing")

print("\n=== Environment Check ===")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    if api_key.startswith("sk-"):
        print(f"✅ OPENAI_API_KEY found (starts with: sk-...{api_key[-4:]})")
    else:
        print(f"❌ OPENAI_API_KEY format invalid: {api_key[:10]}...")
else:
    print("❌ OPENAI_API_KEY not found")

print("\n=== Import Test ===")
try:
    from config import get_cached_settings
    settings = get_cached_settings()
    print("✅ config.py imports successfully")
    print(f"✅ App name: {settings.app_name}")
except Exception as e:
    print(f"❌ config.py import failed: {e}")

try:
    from models import ItemRequest, ItemResponse
    print("✅ models.py imports successfully")
except Exception as e:
    print(f"❌ models.py import failed: {e}")

try:
    from services import ItemMetadataService
    print("✅ services.py imports successfully")
except Exception as e:
    print(f"❌ services.py import failed: {e}")

print("\n=== Quick Service Test ===")
try:
    from services import ItemMetadataService
    service = ItemMetadataService()
    result = service.extract_metadata("iPhone")
    print("✅ Service works!")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Service failed: {e}")
    import traceback
    print(f"Full error: {traceback.format_exc()}")