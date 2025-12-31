"""PostgreSQL database driver for Zexus.
Requires psycopg2: pip install psycopg2-binary
"""

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("Warning: psycopg2 not installed. PostgreSQL support unavailable.")
    print("Install with: pip install psycopg2-binary")

from typing import Any, List, Dict, Optional, Tuple


class PostgreSQLConnection:
    """PostgreSQL database connection."""
    
    def __init__(self, host: str = 'localhost', port: int = 5432, 
                 database: str = 'postgres', user: str = 'postgres', 
                 password: str = ''):
        """Create PostgreSQL connection.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None
        self.in_transaction: bool = False  # Track explicit transaction state
    
    def connect(self) -> bool:
        """Open connection to database."""
        if not POSTGRES_AVAILABLE:
            print("PostgreSQL driver not available (psycopg2 not installed)")
            return False
        
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"PostgreSQL connect error: {e}")
            return False
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> bool:
        """Execute a query (INSERT, UPDATE, DELETE, CREATE, etc)."""
        try:
            if not self.cursor or not self.conn:
                print("PostgreSQL error: Not connected to database")
                return False
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Only auto-commit if we're not in an explicit transaction
            if not self.in_transaction:
                self.conn.commit()
            return True
        except Exception as e:
            print(f"PostgreSQL execute error: {e}")
            if not self.in_transaction:
                self.conn.rollback() if self.conn else None
            return False
    
    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        try:
            if not self.cursor or not self.conn:
                print("PostgreSQL error: Not connected to database")
                return []
            
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            # Convert rows to dictionaries
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        
        except Exception as e:
            print(f"PostgreSQL query error: {e}")
            return []
    
    def query_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return first result."""
        results = self.query(sql, params)
        return results[0] if results else None
    
    def last_insert_id(self) -> int:
        """Get the last inserted row ID (requires RETURNING id clause)."""
        return self.cursor.lastrowid if self.cursor and hasattr(self.cursor, 'lastrowid') else 0
    
    def affected_rows(self) -> int:
        """Get number of affected rows from last query."""
        return self.cursor.rowcount if self.cursor else 0
    
    def begin_transaction(self) -> bool:
        """Begin a transaction."""
        try:
            if self.conn:
                # PostgreSQL auto-starts transactions, just ensure we're not in autocommit
                self.conn.autocommit = False
                self.in_transaction = True
                return True
            return False
        except Exception as e:
            print(f"PostgreSQL begin transaction error: {e}")
            return False
    
    def commit(self) -> bool:
        """Commit current transaction."""
        try:
            if self.conn:
                self.conn.commit()
                self.in_transaction = False
                return True
            return False
        except Exception as e:
            print(f"PostgreSQL commit error: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        try:
            if self.conn:
                self.conn.rollback()
                self.in_transaction = False
                return True
            return False
        except Exception as e:
            print(f"PostgreSQL rollback error: {e}")
            return False
    
    def close(self) -> bool:
        """Close database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            return True
        except Exception as e:
            print(f"PostgreSQL close error: {e}")
            return False
