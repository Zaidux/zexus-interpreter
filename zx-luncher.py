#!/usr/bin/env python3
# zx-launcher.py - Place this in your zexus-interpreter directory
import sys
import os
from main import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: zx <filename.zx>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    
    main(filename)