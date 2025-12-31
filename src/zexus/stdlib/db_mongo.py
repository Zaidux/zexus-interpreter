"""MongoDB database driver for Zexus.
Requires pymongo: pip install pymongo
"""

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    from bson.objectid import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: pymongo not installed. MongoDB support unavailable.")
    print("Install with: pip install pymongo")

from typing import Any, List, Dict, Optional


class MongoDBConnection:
    """MongoDB database connection."""
    
    def __init__(self, host: str = 'localhost', port: int = 27017,
                 database: str = 'test', username: Optional[str] = None,
                 password: Optional[str] = None):
        """Create MongoDB connection.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Optional username
            password: Optional password
        """
        self.host = host
        self.port = port
        self.database_name = database
        self.username = username
        self.password = password
        self.client = None
        self.db = None
    
    def connect(self) -> bool:
        """Open connection to database."""
        if not MONGO_AVAILABLE:
            print("MongoDB driver not available (pymongo not installed)")
            return False
        
        try:
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
            else:
                uri = f"mongodb://{self.host}:{self.port}/"
            
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[self.database_name]
            return True
        except Exception as e:
            print(f"MongoDB connect error: {e}")
            return False
    
    def collection(self, name: str):
        """Get a collection."""
        if self.db is None:
            print("MongoDB error: Not connected to database")
            return None
        return self.db[name]
    
    def insert_one(self, collection: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert a single document."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return None
            result = coll.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            print(f"MongoDB insert_one error: {e}")
            return None
    
    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Insert multiple documents."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return None
            result = coll.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
        except PyMongoError as e:
            print(f"MongoDB insert_many error: {e}")
            return None
    
    def find(self, collection: str, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return []
            
            query = query or {}
            results = list(coll.find(query))
            
            # Convert ObjectId to string for Zexus
            for doc in results:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
            
            return results
        except PyMongoError as e:
            print(f"MongoDB find error: {e}")
            return []
    
    def find_one(self, collection: str, query: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find a single document matching query."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return None
            
            query = query or {}
            doc = coll.find_one(query)
            
            if doc and '_id' in doc and isinstance(doc['_id'], ObjectId):
                doc['_id'] = str(doc['_id'])
            
            return doc
        except PyMongoError as e:
            print(f"MongoDB find_one error: {e}")
            return None
    
    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update a single document."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return 0
            result = coll.update_one(query, update)
            return result.modified_count
        except PyMongoError as e:
            print(f"MongoDB update_one error: {e}")
            return 0
    
    def update_many(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update multiple documents."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return 0
            result = coll.update_many(query, update)
            return result.modified_count
        except PyMongoError as e:
            print(f"MongoDB update_many error: {e}")
            return 0
    
    def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete a single document."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return 0
            result = coll.delete_one(query)
            return result.deleted_count
        except PyMongoError as e:
            print(f"MongoDB delete_one error: {e}")
            return 0
    
    def delete_many(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete multiple documents."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return 0
            result = coll.delete_many(query)
            return result.deleted_count
        except PyMongoError as e:
            print(f"MongoDB delete_many error: {e}")
            return 0
    
    def count(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching query."""
        try:
            coll = self.collection(collection)
            if coll is None:
                return 0
            query = query or {}
            return coll.count_documents(query)
        except PyMongoError as e:
            print(f"MongoDB count error: {e}")
            return 0
    
    def close(self) -> bool:
        """Close database connection."""
        try:
            if self.client:
                self.client.close()
            return True
        except Exception as e:
            print(f"MongoDB close error: {e}")
            return False
