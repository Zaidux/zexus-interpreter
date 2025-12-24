#!/bin/bash
# find_affected_imports.sh
echo "🔍 FINDING FILES AFFECTED BY PARSER DIRECTORY MOVE"
echo "=================================================="
echo ""

# First, let's see what files import parser-related modules
echo "1. Searching for files that import parser modules..."
echo "---------------------------------------------------"

# Search for imports of parser, strategy_context, strategy_structural
patterns=(
    "from.*\.parser"
    "import.*parser"
    "from.*strategy_context"
    "from.*strategy_structural"
    "from.*strategy_recovery"
    "Parser.*import"
    "UltimateParser"
)

for pattern in "${patterns[@]}"; do
    echo ""
    echo "📌 Searching for: $pattern"
    echo "--------------------------------"
    grep -r "$pattern" . --include="*.py" 2>/dev/null | \
        grep -v "__pycache__" | \
        grep -v "compiler/parser.py" | \
        grep -v "parser/parser.py" | \
        head -20
done

echo ""
echo "2. Checking specific known files that might need updates:"
echo "---------------------------------------------------------"

# List of files likely to need updates
critical_files=(
    "cli/main.py"
    "hybrid_orchestrator.py"
    "__init__.py"
    "syntax_validator.py"
    "strategy_recovery.py"
    "compiler/compat_runtime.py"
    "compiler/__init__.py"
    "compiler/semantic.py"
    "vm/vm.py"
    "evaluator/core.py"
    "evaluator/statements.py"
    "module_cache.py"
    "environment_manager.py"
    "compare_interpreter_compiler.py"
)

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo ""
        echo "🔎 Checking: $file"
        # Check for parser imports
        if grep -q "parser" "$file"; then
            echo "   ⚠️  Contains 'parser' references:"
            grep -n "parser" "$file" | head -3 | sed 's/^/      /'
        fi
        # Check for specific parser classes
        if grep -q "Parser\|UltimateParser" "$file"; then
            echo "   ⚠️  Contains parser class references:"
            grep -n "Parser\|UltimateParser" "$file" | head -3 | sed 's/^/      /'
        fi
    fi
done

echo ""
echo "3. Analyzing import patterns in parser directory:"
echo "-------------------------------------------------"

# Look at what the parser files themselves export
echo "Parser directory structure:"
ls -la parser/

echo ""
echo "Parser/__init__.py contents:"
if [ -f "parser/__init__.py" ]; then
    cat parser/__init__.py
else
    echo "No __init__.py in parser directory"
fi

echo ""
echo "4. Finding import statements that need changing:"
echo "-----------------------------------------------"

# Most common patterns that need updating
echo "Common patterns that need updating:"
echo "  OLD: from .parser import Parser"
echo "  NEW: from .parser.parser import Parser"
echo ""
echo "  OLD: from . import parser"
echo "  NEW: from .parser import parser"
echo ""
echo "  OLD: from ..parser import ..."
echo "  NEW: from ..parser.parser import ... (or from ..parser.strategy_context import ...)"

echo ""
echo "5. Running actual import test to see what breaks:"
echo "-------------------------------------------------"

# Quick Python test to see what imports work
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')

print('Testing parser imports...')
try:
    from parser import Parser
    print('✓ from parser import Parser - WORKS')
except ImportError as e:
    print(f'✗ from parser import Parser - FAILS: {e}')

try:
    from parser.parser import Parser
    print('✓ from parser.parser import Parser - WORKS')
except ImportError as e:
    print(f'✗ from parser.parser import Parser - FAILS: {e}')

try:
    from parser.strategy_context import StrategyContext
    print('✓ from parser.strategy_context import StrategyContext - WORKS')
except ImportError as e:
    print(f'✗ from parser.strategy_context import StrategyContext - FAILS: {e}')

try:
    from parser.strategy_structural import StructuralStrategy
    print('✓ from parser.strategy_structural import StructuralStrategy - WORKS')
except ImportError as e:
    print(f'✗ from parser.strategy_structural import StructuralStrategy - FAILS: {e}')
"

echo ""
echo "6. Creating migration commands:"
echo "------------------------------"

# Generate sed commands for common replacements
echo "To update imports, you might need these sed commands:"
echo ""
echo "# Update imports in the same directory as parser/"
echo "sed -i 's/from \\.parser import/from \\.parser.parser import/g' *.py"
echo "sed -i 's/from parser import/from parser.parser import/g' *.py"
echo ""
echo "# Update imports from parent directory"
echo "sed -i 's/from \\.\\.parser import/from \\.\\.parser.parser import/g' *.py"
echo ""
echo "# Update specific strategy imports"
echo "sed -i 's/from \\.strategy_context import/from \\.parser.strategy_context import/g' *.py"
echo "sed -i 's/from \\.strategy_structural import/from \\.parser.strategy_structural import/g' *.py"
