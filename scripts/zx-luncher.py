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

    # Resolve symlinks so we operate on the real target
    real_path = os.path.realpath(filename)

    if not os.path.isfile(real_path):
        print(f"Error: File '{filename}' not found or is not a regular file")
        sys.exit(1)

    # Restrict to .zx files
    if not real_path.endswith('.zx'):
        print("Error: Only .zx files are accepted")
        sys.exit(1)
    
    main(real_path)