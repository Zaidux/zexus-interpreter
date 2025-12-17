#!/bin/bash
# Test runner for Zexus keyword tests
# Usage: ./run_keyword_test.sh <keyword> <level>
# Example: ./run_keyword_test.sh let easy

set -e

KEYWORD=$1
LEVEL=$2

if [ -z "$KEYWORD" ] || [ -z "$LEVEL" ]; then
    echo "Usage: $0 <keyword> <level>"
    echo "Example: $0 let easy"
    echo ""
    echo "Available levels: easy, medium, complex, all"
    exit 1
fi

# Convert keyword to lowercase
KEYWORD_LOWER=$(echo "$KEYWORD" | tr '[:upper:]' '[:lower:]')

# Function to run a single test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    
    echo "======================================"
    echo "Running: $test_name"
    echo "======================================"
    
    if [ -f "$test_file" ]; then
        ./zx "$test_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ $test_name PASSED"
        else
            echo ""
            echo "❌ $test_name FAILED"
        fi
    else
        echo "❌ Test file not found: $test_file"
    fi
    
    echo ""
}

# Run tests based on level
case "$LEVEL" in
    easy)
        run_test "tests/keyword_tests/easy/test_${KEYWORD_LOWER}_easy.zx"
        ;;
    medium)
        run_test "tests/keyword_tests/medium/test_${KEYWORD_LOWER}_medium.zx"
        ;;
    complex)
        run_test "tests/keyword_tests/complex/test_${KEYWORD_LOWER}_complex.zx"
        ;;
    all)
        echo "Running ALL tests for $KEYWORD"
        echo ""
        run_test "tests/keyword_tests/easy/test_${KEYWORD_LOWER}_easy.zx"
        run_test "tests/keyword_tests/medium/test_${KEYWORD_LOWER}_medium.zx"
        run_test "tests/keyword_tests/complex/test_${KEYWORD_LOWER}_complex.zx"
        ;;
    *)
        echo "Invalid level: $LEVEL"
        echo "Available levels: easy, medium, complex, all"
        exit 1
        ;;
esac
