"""
Pytest configuration for Zexus tests.
"""
import sys
import os

# Add src to Python path so tests can import zexus modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
