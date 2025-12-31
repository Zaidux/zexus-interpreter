"""MySQL database driver for Zexus.
Requires mysql-connector-python: pip install mysql-connector-python
"""

try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("Warning: mysql-connector-python not installed. MySQL support unavailable.")
    print("Install with: pip install mysql-connector-python")

from typing import Any, List, Dict, Optional, Tuple


class MySQLConnection:
    """MySQL database connection."""
    
    def __init__(self, host: str = 'localhost', port: int = 3306,
                 database: str = 'mysql', user: str = 'root',
                 password: str = ''):
        """Create MySQL connection.
        
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
        if not MYSQL_AVAILABLE:
            print("MySQL driver not available (mysql-connector-python not installed)")
            return False
        
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor(dictionary=True)
            return True
        except Error as e:
            print(f"MySQL connect error: {e}")
            return False
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> bool:
        """Execute a query (INSERT, UPDATE, DELETE, CREATE, etc)."""
        try:
            if not self.cursor or not self.conn:
                print("MySQL error: Not connected to database")
                return False
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Only auto-commit if we're not in an explicit transaction
            if not self.in_transaction:
                self.conn.commit()
            return True
        except Error as e:
            print(f"MySQL execute error: {e}")
            if not self.in_transaction:
                self.conn.rollback() if self.conn else None
            return False
    
    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        try:
            if not self.cursor or not self.conn:
                print("MySQL error: Not connected to database")
                return []
            
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            # Fetch all rows as dictionaries
            rows = self.cursor.fetchall()
            return rows if rows else []
        
        except Error as e:
            print(f"MySQL query error: {e}")
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
            if self.conn:
                self.conn.start_transaction()
                self.in_transaction = True
                return True
            return False
        except Error as e:
            print(f"MySQL begin transaction error: {e}")
            return False
    
    def commit(self) -> bool:
        """Commit current transaction."""
        try:
            if self.conn:
                self.conn.commit()
                self.in_transaction = False
                return True
            return False
        except Error as e:
            print(f"MySQL commit error: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        try:
            if self.conn:
                self.conn.rollback()
                self.in_transaction = False
                return True
            return False
        except Error as e:
            print(f"MySQL rollback error: {e}")
            return False
    
    def close(self) -> bool:
        """Close database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            return True
        except Error as e:
            print(f"MySQL close error: {e}")
            return False
