# Implementation Guide - Step-by-Step Technical Instructions

## How to Use This Guide

Each feature has:
1. **Overview** - What it does
2. **Files to Modify** - Which source files need changes
3. **Step-by-step Implementation** - Exact changes needed
4. **Testing Checklist** - How to verify it works
5. **Integration Points** - How it connects to other features

## Quick Reference: System Architecture

```
┌─────────────────────────────────────────┐
│  User Code (.zx files)                  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Lexer (src/zexus/lexer.py)             │
│  - Tokenization                         │
│  - Keyword recognition                  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Parser (src/zexus/parser.py)           │
│  - Syntax parsing                       │
│  - AST construction                     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  AST (src/zexus/zexus_ast.py)           │
│  - Abstract syntax trees                │
│  - Statement/Expression nodes           │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Evaluator (src/zexus/evaluator/)       │
│  - Execution engine                     │
│  - Variable binding                     │
│  - Function evaluation                  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Runtime (src/zexus/)                   │
│  - Object model                         │
│  - Security module                      │
│  - Built-in functions                   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Output / Results                       │
└─────────────────────────────────────────┘
```

## Feature: `const` - Immutable Variables

### Overview
Declare immutable variables that cannot be reassigned after initialization.

### Files to Modify
1. `src/zexus/zexus_token.py` - Add CONST token
2. `src/zexus/lexer.py` - Recognize 'const' keyword
3. `src/zexus/zexus_ast.py` - Add ConstStatement AST node
4. `src/zexus/parser.py` - Parse const statements
5. `src/zexus/evaluator/evaluator.py` - Evaluate const statements
6. `src/zexus/environment_manager.py` - Track const variables

### Step-by-Step Implementation

#### Step 1: Add CONST Token
**File**: `src/zexus/zexus_token.py`

Add to the token constants:
```python
CONST = "CONST"
```

#### Step 2: Update Lexer
**File**: `src/zexus/lexer.py`

In the `lookup_ident()` method, add:
```python
"const": CONST
```

#### Step 3: Add AST Node
**File**: `src/zexus/zexus_ast.py`

Add new class:
```python
class ConstStatement(Statement):
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value
    
    def __repr__(self):
        return f"ConstStatement({self.identifier}, {self.value})"
```

#### Step 4: Update Parser
**File**: `src/zexus/parser.py`

In `parse_statement()` method, add:
```python
elif self.cur_token_is(CONST):
    return self.parse_const_statement()
```

Add new method:
```python
def parse_const_statement(self):
    token = self.cur_token
    self.next_token()
    
    if not isinstance(self.cur_token, Identifier):
        self.errors.append(f"Expected identifier after 'const', got {self.cur_token}")
        return None
    
    identifier = self.cur_token.value
    self.next_token()
    
    if not self.cur_token_is(ASSIGN):
        self.errors.append(f"Expected '=' in const declaration")
        return None
    
    self.next_token()
    value = self.parse_expression(LOWEST)
    
    if self.peek_token_is(SEMICOLON):
        self.next_token()
    
    return ConstStatement(identifier, value)
```

#### Step 5: Update Evaluator
**File**: `src/zexus/evaluator/evaluator.py`

Add method to handle ConstStatement:
```python
def eval_const_statement(self, const_stmt):
    value = self.eval(const_stmt.value)
    if isinstance(value, Error):
        return value
    
    # Mark as const in current environment
    self.current_env.set_const(const_stmt.identifier, value)
    return NULL
```

In `eval()` method, add:
```python
elif isinstance(node, ConstStatement):
    return self.eval_const_statement(node)
```

#### Step 6: Update Environment Manager
**File**: `src/zexus/environment_manager.py`

Add const tracking:
```python
class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.const_vars = set()  # Track const variables
        self.outer = outer
    
    def set_const(self, name, value):
        """Set a constant variable"""
        self.store[name] = value
        self.const_vars.add(name)
    
    def get(self, name):
        # Return existing implementation
        ...
    
    def set(self, name, value):
        """Set a variable, checking if it's const"""
        if name in self.const_vars:
            raise ValueError(f"Cannot reassign const variable '{name}'")
        self.store[name] = value
```

### Testing Checklist

```zexus
// Test 1: Basic const declaration
const X = 10;
print(X);  // Should print: 10

// Test 2: Cannot reassign const
const Y = 20;
Y = 30;    // Should error: Cannot reassign const variable

// Test 3: Const with strings
const NAME = "Alice";
print(NAME);  // Should print: Alice

// Test 4: Const with objects
const CONFIG = {timeout: 5000, retries: 3};
print(CONFIG.timeout);  // Should print: 5000

// Test 5: Const combined with seal (bonus)
const seal user = {name: "Bob", age: 30};
// Both const (can't reassign) AND sealed (can't modify properties)
```

### Integration Points
- Works with all types (primitives, objects, functions)
- Can be combined with `seal` keyword
- Environment tracks const variables
- Type system can optimize based on const hint

---

## Feature: `elif` - Else-If Conditionals

### Overview
Standard `elif` conditional branching for cleaner multi-branch conditionals.

### Files to Modify
1. `src/zexus/zexus_token.py` - Add ELIF token
2. `src/zexus/lexer.py` - Recognize 'elif' keyword
3. `src/zexus/zexus_ast.py` - Extend IfExpression to support elif
4. `src/zexus/parser.py` - Parse elif clauses
5. `src/zexus/evaluator/evaluator.py` - Evaluate elif chains

### Step-by-Step Implementation

#### Step 1: Add ELIF Token
**File**: `src/zexus/zexus_token.py`

Add to token constants:
```python
ELIF = "ELIF"
```

#### Step 2: Update Lexer
**File**: `src/zexus/lexer.py`

In `lookup_ident()` method, add:
```python
"elif": ELIF
```

#### Step 3: Extend AST Node
**File**: `src/zexus/zexus_ast.py`

Modify existing `IfExpression` class:
```python
class IfExpression(Expression):
    def __init__(self, condition, consequence, elif_parts=None, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.elif_parts = elif_parts or []  # List of (condition, consequence) tuples
        self.alternative = alternative
    
    def __repr__(self):
        return f"IfExpression({self.condition}, {self.consequence}, {self.elif_parts}, {self.alternative})"
```

#### Step 4: Update Parser
**File**: `src/zexus/parser.py`

Modify `parse_if_expression()` method:
```python
def parse_if_expression(self):
    token = self.cur_token
    self.next_token()
    
    condition = self.parse_expression(LOWEST)
    
    if not self.cur_token_is(LBRACE):
        self.errors.append(f"Expected '{{' after if condition")
        return None
    
    consequence = self.parse_block_statement()
    
    # Parse elif clauses
    elif_parts = []
    while self.peek_token_is(ELIF):
        self.next_token()  # Move to elif
        self.next_token()  # Move to condition
        
        elif_condition = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(LBRACE):
            self.errors.append(f"Expected '{{' after elif condition")
            return None
        
        elif_consequence = self.parse_block_statement()
        elif_parts.append((elif_condition, elif_consequence))
    
    # Parse else clause
    alternative = None
    if self.peek_token_is(ELSE):
        self.next_token()
        
        if not self.cur_token_is(LBRACE):
            self.errors.append(f"Expected '{{' after else")
            return None
        
        alternative = self.parse_block_statement()
    
    return IfExpression(condition, consequence, elif_parts, alternative)
```

#### Step 5: Update Evaluator
**File**: `src/zexus/evaluator/evaluator.py`

Modify `eval_if_expression()` method:
```python
def eval_if_expression(self, if_expr):
    condition = self.eval(if_expr.condition)
    
    if self.is_truthy(condition):
        return self.eval(if_expr.consequence)
    
    # Check elif conditions
    for elif_condition, elif_consequence in if_expr.elif_parts:
        condition = self.eval(elif_condition)
        if self.is_truthy(condition):
            return self.eval(elif_consequence)
    
    # Check else clause
    if if_expr.alternative:
        return self.eval(if_expr.alternative)
    
    return NULL
```

### Testing Checklist

```zexus
// Test 1: Basic elif
let x = 15;
if x > 20 {
  print("large");
} elif x > 10 {
  print("medium");
} else {
  print("small");
}
// Should print: medium

// Test 2: Multiple elif
let grade = 85;
if grade >= 90 {
  print("A");
} elif grade >= 80 {
  print("B");
} elif grade >= 70 {
  print("C");
} elif grade >= 60 {
  print("D");
} else {
  print("F");
}
// Should print: B

// Test 3: Elif with no else
let code = 404;
if code == 200 {
  print("OK");
} elif code == 404 {
  print("Not Found");
} elif code == 500 {
  print("Server Error");
}
// Should print: Not Found

// Test 4: Nested elif
let a = 10;
let b = 20;
if a > b {
  print("a is larger");
} elif a == b {
  print("equal");
} elif a < b {
  if b > 30 {
    print("b is much larger");
  } else {
    print("b is slightly larger");
  }
}
// Should print: b is slightly larger
```

### Integration Points
- Replaces nested if-else patterns
- Works with expressions and statements
- Compatible with all condition types
- Can be optimized to jump tables

---

## Feature Implementation Order

For maximum impact with minimum dependencies:

### Phase 1 (Week 1): Convenience Foundation
1. ✅ `seal` - Already done
2. 🚀 `const` - Easy, no dependencies
3. 🚀 `elif` - Easy, no dependencies

### Phase 2 (Week 2-3): Developer Experience
4. `defer` - Small, improves ergonomics
5. `pattern` - Medium, enables better code

### Phase 3 (Week 4-6): Performance
6. `native` - Required for perf gains
7. `gc` - Works with native
8. `buffer` - Works with native

### Phase 4 (Week 7+): Advanced
9. Remaining features in order of complexity

---

## Testing Strategy

For each feature:
1. Unit tests in test file
2. Integration tests with existing features
3. Regression tests (ensure nothing breaks)
4. Performance tests (if applicable)
5. Documentation examples (must run)

## Common Pitfalls

1. **Lexer/Parser synchronization** - Keep them in sync
2. **AST node updates** - Update all callers when changing nodes
3. **Environment tracking** - Must track variable scope correctly
4. **Type system** - Update type checker if present
5. **Error messages** - Clear, actionable error reporting

---

**Next Steps**:
- Review CODE_EXAMPLES.md for working examples
- Check ROADMAP.md for timeline
- Start with `const` implementation (easiest)
