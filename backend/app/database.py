"""
MongoDB database connection
"""

from pymongo import MongoClient
from pymongo import IndexModel, ASCENDING, DESCENDING
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    client = None
    db = None
    
    @classmethod
    def connect(cls):
        """Connect to MongoDB Atlas"""
        try:
            logger.info(f"🔄 Connecting to MongoDB: {settings.MONGODB_URL[:30]}...")
            
            cls.client = MongoClient(settings.MONGODB_URL)
            cls.db = cls.client[settings.MONGODB_DB_NAME]
            
            # Test connection
            cls.client.admin.command('ping')
            
            # Create indexes
            cls._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB Atlas: {settings.MONGODB_DB_NAME}")
            return cls.db
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    @classmethod
    def close(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info("✅ MongoDB connection closed")
    
    @classmethod
    def _create_indexes(cls):
        """Create database indexes"""
        try:
            # Users collection indexes
            users_collection = cls.db.users
            users_collection.create_indexes([
                IndexModel([("username", ASCENDING)], unique=True),
                IndexModel([("email", ASCENDING)], unique=True),
                IndexModel([("google_id", ASCENDING)]),
            ])
            
            # Predictions collection indexes
            predictions_collection = cls.db.predictions
            predictions_collection.create_indexes([
                IndexModel([("patient_id", ASCENDING)]),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
            ])
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls.db is None:
            raise Exception("Database not connected. Call connect() first.")
        return cls.db

# For FastAPI dependency
def get_database():
    """Dependency for FastAPI to get database"""
    return Database.get_db()
