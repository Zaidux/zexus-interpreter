"""SQLite database driver for Zexus."""

import sqlite3
from typing import Any, List, Dict, Optional, Tuple


class SQLiteConnection:
    """SQLite database connection."""
    
    def __init__(self, database: str):
        """Create SQLite connection.
        
        Args:
            database: Path to SQLite database file, or ':memory:' for in-memory DB
        """
        self.database = database
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.in_transaction: bool = False  # Track explicit transaction state
    
    def connect(self) -> bool:
        """Open connection to database."""
        try:
            self.conn = sqlite3.connect(self.database)
            # Disable automatic transaction management - we'll handle it manually
            self.conn.isolation_level = None  
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"SQLite connect error: {e}")
            return False
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> bool:
        """Execute a query (INSERT, UPDATE, DELETE, CREATE, etc)."""
        try:
            if not self.cursor or not self.conn:
                print("SQLite error: Not connected to database")
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
            print(f"SQLite execute error: {e}")
            if not self.in_transaction:
                self.conn.rollback() if self.conn else None
            return False
    
    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        try:
            if not self.cursor or not self.conn:
                print("SQLite error: Not connected to database")
                return []
            
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            # Convert rows to dictionaries
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        
        except Exception as e:
            print(f"SQLite query error: {e}")
            return []
    
    def query_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return first result."""
        results = self.query(sql, params)
        return results[0] if results else None
    
    def last_insert_id(self) -> int:
        """Get the last inserted row ID."""
        return self.cursor.lastrowid if self.cursor else 0
    
    def affected_rows(self) -> int:
        """Get number of affected rows from last query."""
        return self.cursor.rowcount if self.cursor else 0
    
    def begin_transaction(self) -> bool:
        """Begin a transaction."""
        try:
            # SQLite doesn't support nested transactions
            # If we're already in a transaction, just return True
            if self.in_transaction:
                return True
            self.conn.execute("BEGIN TRANSACTION")
            self.in_transaction = True
            return True
        except Exception as e:
            print(f"SQLite begin transaction error: {e}")
            return False
    
    def commit(self) -> bool:
        """Commit current transaction."""
        try:
            self.conn.commit()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"SQLite commit error: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        try:
            self.conn.rollback()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"SQLite rollback error: {e}")
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
            print(f"SQLite close error: {e}")
            return False
