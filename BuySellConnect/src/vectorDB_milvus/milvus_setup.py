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
    
    def create_collection_with_partitions(self, collection_name, dim=128):
        """Create a collection with schema for partitioning"""
        
        # Define fields
        fields = [ FieldSchema(name="message_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="thread_id", dtype=DataType.INT64, auto_id=False),
            FieldSchema(name="user_id", dtype=DataType.INT64, auto_id=False),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),  # For exact matching
            FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=100),   # For sub-sub-partitions
            FieldSchema(name="item_or_service", dtype=DataType.VARCHAR, max_length=100), 
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
