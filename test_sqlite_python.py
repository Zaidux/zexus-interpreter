#!/usr/bin/env python3
"""Test SQLite directly in Python to verify it works"""

import sqlite3

# Create in-memory database
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# Create table
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
conn.commit()
print("✅ Table created")

# Insert data
cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")
conn.commit()
print("✅ Data inserted")

# Query data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
print(f"✅ Found {len(rows)} rows: {rows}")

conn.close()
print("✅ Test complete - SQLite works in Python")
