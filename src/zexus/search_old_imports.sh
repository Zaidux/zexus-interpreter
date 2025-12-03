#!/bin/bash
# search_old_imports.sh
echo "🔍 Searching for files that need import updates..."
echo "================================================"

# Search patterns
patterns=(
    "from.*\.evaluator"
    "import.*evaluator"
    "from evaluator import"
    "eval_node"
    "evaluate\("
    "Environment.*from.*evaluator"
)

for pattern in "${patterns[@]}"; do
    echo ""
    echo "📌 Searching for: $pattern"
    echo "--------------------------------"
    grep -r "$pattern" . --include="*.py" 2>/dev/null | \
        grep -v "__pycache__" | \
        grep -v "evaluator/" | \
        grep -v "evaluator_original.py" | \
        head -20
done

# Specifically check known problematic files
echo ""
echo "📋 Checking specific known files:"
echo "================================="
files_to_check=(
    "cli/main.py"
    "hybrid_orchestrator.py"
    "__main__.py"
    "compare_interpreter_compiler.py"
    "syntax_validator.py"
    "strategy_recovery.py"
    "compiler/compat_runtime.py"
    "vm/vm.py"
    "vm/jit.py"
    "parser/parser.py"
    "embedding/__init__.py"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo ""
        echo "🔎 $file:"
        if grep -q "evaluator" "$file"; then
            echo "   ⚠️  Contains 'evaluator' references:"
            grep -n "evaluator" "$file" | head -5
        else
            echo "   ✅ No evaluator references found"
        fi
    fi
done

# Check for eval_node function calls
echo ""
echo "🔍 Searching for eval_node function calls (should be replaced with evaluate):"
echo "============================================================================"
grep -r "eval_node" . --include="*.py" 2>/dev/null | \
    grep -v "__pycache__" | \
    grep -v "evaluator/" | \
    grep -v "evaluator_original.py"
