import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility # type: ignore
import json

class MilvusVectorDB:
    def __init__(self, host="localhost", port="19530", alias="default"):
        """Initialize Milvus connection"""
        self.host = host
        self.port = port
        self.alias = alias
        self.connect()
    
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port
            )
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise
    
    #defining a collection with partitioning support using the fields as suggested by Claude
    #https://claude.ai/share/46c80e28-a729-4c20-a100-a2784549bce0
    # "user_id": int64,     # Reference to RDBMS
    # "thread_id": int64,   # Reference to RDBMS
    # "message_id": int64,  # Reference to RDBMS
    # "vector_embedding_msg": float_vector(dimension=768),  # Embedding
    # "category": varchar,   # For filtering
    # "location": varchar,   # For geo-filtering
    # "created_at": int64   # Unix timestamp

    def create_collection_with_partitions(self, collection_name, dim=128):
        """Create a collection with schema for partitioning"""
        
        # Define fields
        fields = [ FieldSchema(name="message_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="thread_id", dtype=DataType.INT64, auto_id=False),
            FieldSchema(name="user_id", dtype=DataType.INT64, auto_id=False),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),  # For exact matching
            FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=100),  # For sub-partitions
            FieldSchema(name="item_or_service", dtype=DataType.VARCHAR, max_length=100),  # For sub-sub-partitions
            FieldSchema(name="vector_embedding_msg", dtype=DataType.FLOAT_VECTOR, dim=dim), # Embedding vector
            FieldSchema(name="json_attributes", dtype=DataType.JSON),  # JSON string for additional attributes
            FieldSchema(name="metadata", dtype=DataType.JSON),  # Additional metadata
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),  # Additional location field
            FieldSchema(name="created_at", dtype=DataType.INT64, auto_id=False),   # Unix timestamp
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=100),  # active or passive
        ]
        
        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="Vector collection with partitioning support",
            enable_dynamic_field=True
        )
        
        # Create collection
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists. Dropping...")
            utility.drop_collection(collection_name)
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )
        
        print(f"Created collection: {collection_name}")
        return collection
    
    def create_partitions(self, collection, partition_names):
        """Create partitions in the collection"""
        for partition_name in partition_names:
            if not collection.has_partition(partition_name):
                collection.create_partition(partition_name)
                print(f"Created partition: {partition_name}")
            else:
                print(f"Partition {partition_name} already exists")
    
    def create_index(self, collection):
        """Create vector index for efficient search"""
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        collection.create_index(
            field_name="vector_embedding_msg",
            index_params=index_params
        )
        print("vector_embedding_msg index created")

    def create_scalar_indexes(self, collection):
        """Create scalar indexes for filtering"""
        
        # Create index on item_or_service field
        collection.create_index(
            field_name="item_or_service",
            index_params={
                "index_type": "TRIE"  # For VARCHAR fields
            }
        )
        print("Index created on item_or_service field")

        # Create index on category field
        collection.create_index(
            field_name="category",
            index_params={
                "index_type": "TRIE"
            }
        )
        print("Index created on category field")

        # Create index on subcategory field
        collection.create_index(
            field_name="subcategory",
            index_params={
                "index_type": "TRIE"
            }
        )
        print("Index created on subcategory field")
    
    def insert_data(self, collection, data, partition_name=None):
        """Insert data into collection with optional partition"""
        try:
            if partition_name:
                collection.insert(data, partition_name=partition_name)
                print(f"Inserted {len(data[0])} records into partition {partition_name}")
            else:
                collection.insert(data)
                print(f"Inserted {len(data[0])} records into collection")
        except Exception as e:
            print(f"Insert failed: {e}")
            raise
    
    def search_with_filter(self, collection, query_vectors, category_filter=None, 
                          subcategory_filter=None, top_k=10, partition_names=None):
        """Search with exact match filtering + nearest neighbor"""
        
        # Build filter expression
        expr_conditions = []
        if category_filter:
            expr_conditions.append(f'category == "{category_filter}"')
        if subcategory_filter:
            expr_conditions.append(f'subcategory == "{subcategory_filter}"')
        
        expr = " and ".join(expr_conditions) if expr_conditions else None
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Load collection for search
        collection.load()
        
        try:
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                partition_names=partition_names,
                output_fields=["id", "category", "subcategory", "metadata"]
            )
            return results
        except Exception as e:
            print(f"Search failed: {e}")
            raise
    
    def delete_data(self, collection, ids, partition_name=None):
        """Delete data by IDs"""
        expr = f"id in {ids}"
        if partition_name:
            collection.delete(expr, partition_name=partition_name)
        else:
            collection.delete(expr)
        print(f"Deleted records with IDs: {ids}")
    
    def get_collection_stats(self, collection_name):
        """Get collection statistics"""
        collection = Collection(collection_name)
        collection.load()
        return {
            "num_entities": collection.num_entities,
            "partitions": [p.name for p in collection.partitions]
        }

# Example usage and testing
# 
import numpy as np
import time
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pymilvus import Collection

class MilvusInteractiveTest:
    def __init__(self, db_instance):
        self.db = db_instance
        self.collections = ["buyers", "sellers", "renters", "lessors", "connectors", "service_seekers", "service_providers"]
        
        # Initialize sentence transformer for embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        self.vector_embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        print(f"Embedding model loaded. Vector dimension: {self.vector_embedding_dim}")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Convert text to vector embedding"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random vector if embedding fails
            return np.random.random(self.vector_embedding_dim).tolist()
    
    def check_collection_status(self, collection_name):
        """Safely check collection status without using has_index()"""
        try:
            collection = Collection(collection_name)
            
            # Check if collection exists and get basic info
            print(f"📊 Collection '{collection_name}' info:")
            print(f"  - Schema: {len(collection.schema.fields)} fields")
            print(f"  - Description: {collection.description}")
            
            # Try to load (this will fail if no indexes exist)
            try:
                collection.load()
                print(f"  - Status: ✅ Ready (loaded successfully)")
                return True
            except Exception as load_error:
                print(f"  - Status: ⚠️  Load warning: {load_error}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking collection '{collection_name}': {e}")
            return False
    
    def check_collections_exist(self):
        """Check if collections exist and create them if needed"""
        from pymilvus import utility
        
        print("Checking if collections exist...")
        missing_collections = []
        
        for collection_name in self.collections:
            if not utility.has_collection(collection_name):
                missing_collections.append(collection_name)
        
        if missing_collections:
            print(f"⚠️  Missing collections: {missing_collections}")
            create_choice = input("Would you like to create missing collections? (y/N): ").strip().lower()
            if create_choice == 'y':
                self.setup_collections()
            else:
                print("Some operations may fail without proper collections.")
        else:
            print("✅ All collections exist!")
        """Check if collections exist and create them if needed"""
        from pymilvus import utility
        
        print("Checking if collections exist...")
        missing_collections = []
        
        for collection_name in self.collections:
            if not utility.has_collection(collection_name):
                missing_collections.append(collection_name)
        
        if missing_collections:
            print(f"⚠️  Missing collections: {missing_collections}")
            create_choice = input("Would you like to create missing collections? (y/N): ").strip().lower()
            if create_choice == 'y':
                self.setup_collections()
            else:
                print("Some operations may fail without proper collections.")
        else:
            print("✅ All collections exist!")
    
    def setup_collections(self):
        """Create all collections with proper schema"""
        print("Setting up collections...")
        
        for coll_name in self.collections:
            collection = self.db.create_collection_with_partitions(coll_name, dim=self.vector_embedding_dim)
            
            # Create partitions (categories)
            partition_names = ["electronics", "clothing", "books", "services", "real_estate", "automotive"]
            self.db.create_partitions(collection, partition_names)
            
            # Create indexes
            self.db.create_index(collection)
            self.db.create_scalar_indexes(collection)
            
        print("All collections created successfully!")
    
    def get_user_choice(self, prompt: str, choices: List[str]) -> str:
        """Get user choice from a list of options"""
        while True:
            print(f"\n{prompt}")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            
            try:
                choice_num = int(input("Enter your choice (number): "))
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def get_user_input_for_insert(self) -> Dict[str, Any]:
        """Get user input for all fields needed for insertion"""
        data = {}
        
        print("\n--- Enter data for insertion ---")
        
        # Get basic fields
        data['message_id'] = input("Message ID: ").strip()
        data['thread_id'] = input("Thread ID: ").strip()  
        data['user_id'] = input("User ID: ").strip()
        data['category'] = input("Category: ").strip()
        data['subcategory'] = input("Subcategory: ").strip()
        data['item_or_service'] = input("Item or Service: ").strip()
        
        # Get sentence for embedding
        sentence = input("Enter sentence to convert to vector: ").strip()
        if sentence:
            print("Generating embedding...")
            data['vector_embedding_msg'] = self.get_text_embedding(sentence)
            print("Embedding generated successfully!")
        else:
            print("No sentence provided, using random vector")
            data['vector_embedding_msg'] = np.random.random(self.vector_embedding_dim).tolist()
        
        # Get JSON attributes
        json_attr_input = input("JSON attributes (enter valid JSON or press enter for empty): ").strip()
        try:
            data['json_attributes'] = json.loads(json_attr_input) if json_attr_input else {}
        except json.JSONDecodeError:
            print("Invalid JSON format, using empty dict")
            data['json_attributes'] = {}
        
        # Get metadata
        metadata_input = input("Metadata (enter valid JSON or press enter for empty): ").strip()
        try:
            data['metadata'] = json.loads(metadata_input) if metadata_input else {}
        except json.JSONDecodeError:
            print("Invalid JSON format, using empty dict")
            data['metadata'] = {}
        
        data['location'] = input("Location: ").strip()
        
        # Auto-fill created_at and status
        data['created_at'] = int(time.time())  # Unix epoch timestamp
        data['status'] = "active"
        
        print(f"\nAuto-filled fields:")
        print(f"Created at: {data['created_at']} (Unix timestamp)")
        print(f"Status: {data['status']}")
        
        return data
    
    def insert_data_interactive(self):
        """Interactive data insertion"""
        # Choose collection
        collection_name = self.get_user_choice("Select collection to insert data:", self.collections)
        
        # Get collection instance
        collection = Collection(collection_name)
        
        # Load collection into memory
        try:
            collection.load()
            print(f"✅ Collection '{collection_name}' loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning loading collection: {e}")
        
        # Load collection into memory if not already loaded
        # if not collection.has_index("vector_embedding_msg"):
        #     print(f"Warning: Collection '{collection_name}' may not be properly indexed")
        
        print("before load")
        collection.load()
        print("after load")
        
        # Get data from user
        data = self.get_user_input_for_insert()
        
        # Choose partition (category)
        available_partitions = ["electronics", "clothing", "books", "services", "real_estate", "automotive"]
        partition_name = self.get_user_choice("Select partition (category):", available_partitions)
        print(f"Selected partition: {partition_name}")
        
        try:
            # Prepare data for insertion (single record)
            # Note: Adjust field order based on your actual schema
            entities = [
                {
                    "message_id": int(data['message_id']),
                    "thread_id": int(data['thread_id']),
                    "user_id": int(data['user_id']),
                    "category": data['category'],
                    "subcategory": data['subcategory'],
                    "item_or_service": data['item_or_service'],
                    "vector_embedding_msg": data['vector_embedding_msg'],
                    "json_attributes": json.dumps(data['json_attributes']),
                    "metadata": json.dumps(data['metadata']),
                    "location": data['location'],
                    "created_at": data['created_at'],
                    "status": data['status']
                }
            ]

            print(f"\nInserting data into collection '{collection_name}', partition '{partition_name}'...")
            print(f"Data to insert: {entities}")
            
            # Insert data using Collection.insert or your db method
            try:
                # Try using the db method first (if it works with Collection object)
                mr = collection.insert(entities, partition_name=partition_name)
                collection.flush()  # Ensure data is written
                print(f"\n✅ Data inserted successfully! Insert result: {mr}")
            except Exception as insert_error:
                print(f"Collection.insert failed, trying db method: {insert_error}")
                # Fallback to your original db method
                insert_data = [
                    [1],  # ID (auto-generated)
                    [data['message_id']],
                    [data['thread_id']],
                    [data['user_id']],
                    [data['category']],
                    [data['subcategory']],
                    [data['item_or_service']],
                    [data['vector_embedding_msg']],
                    [json.dumps(data['json_attributes'])],
                    [json.dumps(data['metadata'])],
                    [data['location']],
                    [data['created_at']],
                    [data['status']]
                ]
                self.db.insert_data(collection, insert_data, partition_name=partition_name)
            
            print(f"✅ Data inserted into collection '{collection_name}', partition '{partition_name}'!")
            
        except Exception as e:
            print(f"❌ Error inserting data: {e}")
            print("Please check your collection schema and ensure all fields match.")
    
    def search_data_interactive(self):
        """Interactive data search"""
        # Choose collection
        collection_name = self.get_user_choice("Select collection to search:", self.collections)
        
        # Get collection instance
        collection = Collection(collection_name)
        
        print("\n--- Search Parameters ---")
        
        # Get search query
        search_text = input("Enter text to search for: ").strip()
        if not search_text:
            print("No search text provided, using random query vector")
            query_vector = [np.random.random(self.vector_embedding_dim).tolist()]
        else:
            print("Generating query embedding...")
            query_vector = [self.get_text_embedding(search_text)]
            print("Query embedding generated!")
        
        # Get optional filters
        category_filter = input("Category filter (press enter to skip): ").strip() or None
        subcategory_filter = input("Subcategory filter (press enter to skip): ").strip() or None
        
        # Get top_k
        try:
            top_k = int(input("Number of results to return (default 5): ").strip() or "5")
        except ValueError:
            top_k = 5
        
        try:
            # Perform search
            print(f"\nSearching in collection '{collection_name}'...")
            
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build filter expression if filters are provided
            filter_expr = ""
            if category_filter:
                filter_expr += f'category == "{category_filter}"'
            if subcategory_filter:
                if filter_expr:
                    filter_expr += f' && subcategory == "{subcategory_filter}"'
                else:
                    filter_expr += f'subcategory == "{subcategory_filter}"'
            
            # Perform search using Collection.search or fallback to db method
            try:
                results = collection.search(
                    data=query_vector,
                    anns_field="vector_embedding_msg",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr if filter_expr else None,
                    output_fields=["category", "subcategory", "item_or_service", "location", "status", "message_id"]
                )
            except Exception as search_error:
                print(f"Collection.search failed, trying db method: {search_error}")
                # Fallback to your original db method
                results = self.db.search_with_filter(
                    collection, 
                    query_vector,
                    category_filter=category_filter,
                    subcategory_filter=subcategory_filter,
                    top_k=top_k
                )
            
            print(f"\n🔍 Search Results (Top {top_k}):")
            print("-" * 80)
            
            if results and len(results) > 0 and len(results[0]) > 0:
                for i, hit in enumerate(results[0], 1):
                    print(f"\nResult {i}:")
                    print(f"  ID: {hit.id}")
                    print(f"  Distance: {hit.distance:.4f}")
                    print(f"  Category: {hit.entity.get('category', 'N/A')}")
                    print(f"  Subcategory: {hit.entity.get('subcategory', 'N/A')}")
                    print(f"  Item/Service: {hit.entity.get('item_or_service', 'N/A')}")
                    print(f"  Location: {hit.entity.get('location', 'N/A')}")
                    print(f"  Status: {hit.entity.get('status', 'N/A')}")
                    print("-" * 40)
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"❌ Error searching: {e}")
    
    def show_collection_stats(self):
        """Show statistics for all collections"""
        print("\n📊 Collection Statistics:")
        print("=" * 50)
        
        for collection_name in self.collections:
            try:
                # Try to get collection and load it
                collection = Collection(collection_name)
                collection.load()
                
                # Get basic stats
                num_entities = collection.num_entities
                print(f"{collection_name}: {num_entities} entities")
                
                # Try to get additional stats if available
                try:
                    stats = self.db.get_collection_stats(collection_name)
                    print(f"  Additional stats: {stats}")
                except:
                    pass  # If db method doesn't work, just show entity count
                    
            except Exception as e:
                print(f"{collection_name}: Error - {e}")
    
    def run_interactive_session(self):
        """Main interactive session"""
        print("🚀 Milvus Interactive Test Session")
        print("=" * 50)
        
        # Check if collections exist
        self.check_collections_exist()
        
        # Option to check collection status
        status_choice = input("Do you want to check collection status? (y/N): ").strip().lower()
        if status_choice == 'y':
            for collection_name in self.collections:
                self.check_collection_status(collection_name)
                print()
        
        # Setup collections
        setup_choice = input("Do you want to setup/reset collections? (y/N): ").strip().lower()
        if setup_choice == 'y':
            self.setup_collections()
        
        while True:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            
            choices = [
                "Insert Data",
                "Search Data", 
                "Show Collection Statistics",
                "Exit"
            ]
            
            choice = self.get_user_choice("What would you like to do?", choices)
            
            if choice == "Insert Data":
                self.insert_data_interactive()
            elif choice == "Search Data":
                self.search_data_interactive()
            elif choice == "Show Collection Statistics":
                self.show_collection_stats()
            elif choice == "Exit":
                print("👋 Goodbye!")
                break
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Initialize connection
    db = MilvusVectorDB()
    
    # Start interactive session
    try:
        interactive_test = MilvusInteractiveTest(db)
        interactive_test.run_interactive_session()
    except ImportError:
        print("❌ Error: sentence-transformers not installed.")
        print("Please install it using: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Error initializing: {e}")