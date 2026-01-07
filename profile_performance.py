#!/usr/bin/env python3
"""Profile Zexus interpreter to identify performance bottlenecks"""

import argparse
import cProfile
import pstats
import sys
from io import StringIO

# Add src to path
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.evaluator import evaluate
from zexus.parser import Parser
from zexus.lexer import Lexer
from zexus.object import Environment

# Test code - inline token with 1000 transfers
CODE = '''
contract TokenSilent {
    data total_supply = 0
    data balances = {}
    
    action mint(to, amount) {
        let to_balance = balances[to]
        if to_balance == null {
            to_balance = 0
        }
        balances[to] = to_balance + amount
        total_supply = total_supply + amount
        return { "success": true }
    }
    
    action transfer(from, to, amount) {
        let from_balance = balances[from]
        if from_balance == null {
            from_balance = 0
        }
        
        if from_balance < amount {
            return { "success": false }
        }
        
        balances[from] = from_balance - amount
        
        let to_balance = balances[to]
        if to_balance == null {
            to_balance = 0
        }
        balances[to] = to_balance + amount
        
        return { "success": true }
    }
}

let token = TokenSilent()
token.mint("0xTEST", 1000000)

let i = 0
let success = 0

while i < 1000 {
    let addr = "0xUSER_" + string(i)
    let res = token.transfer("0xTEST", addr, 10)
    
    if res["success"] {
        success = success + 1
    }
    
    i = i + 1
}
'''

def run_test():
    """Parse and execute the test code"""
    # Parse once, outside profiling
    lexer = Lexer(CODE)
    parser = Parser(lexer, "zexus")
    ast = parser.parse_program()
    env = Environment()
    # Profile only the execution
    return ast, env

def main():
    parser = argparse.ArgumentParser(description="Profile Zexus interpreter performance")
    parser.add_argument("--use-vm", action="store_true", help="Enable VM execution during profiling")
    args = parser.parse_args()

    mode = "VM" if args.use_vm else "interpreter"
    print(f"Profiling Zexus {mode} with 1000 token transfers...\n")

    # Parse first (not profiled)
    print("Parsing...")
    ast, env = run_test()
    print("Parsing complete, starting execution profiling...\n")

    # Profile only the execution
    profiler = cProfile.Profile()
    profiler.enable()
    evaluate(ast, env, use_vm=args.use_vm)
    profiler.disable()
    
    # Get stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    
    # Sort by cumulative time and print top 50 functions
    print("=" * 100)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 100)
    stats.sort_stats('cumulative')
    stats.print_stats(50)
    
    # Sort by internal time (time excluding subcalls)
    print("\n" + "=" * 100)
    print("TOP 50 FUNCTIONS BY INTERNAL TIME (excluding subcalls)")
    print("=" * 100)
    stats.sort_stats('time')
    stats.print_stats(50)
    
    # Print output
    print(stream.getvalue())


if __name__ == '__main__':
    main()
