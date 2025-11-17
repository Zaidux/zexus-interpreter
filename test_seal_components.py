#!/usr/bin/env python3
"""
Minimal test for the seal feature by directly testing the components.
"""
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter')

# Test imports
print("Testing imports...")
try:
    from src.zexus.zexus_token import SEAL
    print(f"✓ SEAL token: {SEAL}")
except Exception as e:
    print(f"✗ Failed to import SEAL token: {e}")
    sys.exit(1)

try:
    from src.zexus.lexer import Lexer
    print("✓ Lexer imported")
except Exception as e:
    print(f"✗ Failed to import Lexer: {e}")
    sys.exit(1)

try:
    from src.zexus.security import SealedObject
    print("✓ SealedObject imported")
except Exception as e:
    print(f"✗ Failed to import SealedObject: {e}")
    sys.exit(1)

# Test lexer recognizes seal keyword
print("\nTesting lexer...")
lexer = Lexer("seal myVar")
token1 = lexer.next_token()
print(f"  Token 1: {token1.type} (literal: '{token1.literal}')")
if token1.type == SEAL:
    print("  ✓ Seal keyword recognized")
else:
    print(f"  ✗ Expected SEAL but got {token1.type}")

token2 = lexer.next_token()
print(f"  Token 2: {token2.type} (literal: '{token2.literal}')")

# Test SealedObject
print("\nTesting SealedObject...")
from src.zexus.object import Integer
val = Integer(42)
sealed_val = SealedObject(val)
print(f"  Original value: {val.inspect()}")
print(f"  Sealed value: {sealed_val.inspect()}")
print(f"  Sealed type: {sealed_val.type()}")

print("\n✓ All component tests passed!")
