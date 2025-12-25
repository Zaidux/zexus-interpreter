#!/usr/bin/env python3
# check_parser_config.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_parser_config():
    print("=== Parser Configuration ===")
    
    try:
        from zexus.compiler.parser import parser
        print(f"✅ Compiler parser: {type(parser)}")
        
        # Check if we can configure it
        if hasattr(parser, 'tolerant'):
            print(f"   Tolerant mode: {parser.tolerant}")
        if hasattr(parser, 'debug'):
            print(f"   Debug mode: {parser.debug}")
            
    except Exception as e:
        print(f"❌ Cannot check parser config: {e}")
        
    # Check what parsing library is used
    try:
        import inspect
        from zexus.compiler.parser import parser
        print(f"   Parser module: {inspect.getmodule(parser)}")
    except:
        pass

if __name__ == "__main__":
    check_parser_config()
