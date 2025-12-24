# Async/Await Implementation Plan

## Overview
Implement proper async/await support for Zexus language.

## Components

### 1. Async Actions
- Actions marked with `async` keyword
- Return Promise/Future objects

### 2. Await Expression  
- `await` keyword to wait for async results
- Can only be used inside async actions

### 3. Promise System
- Promise object to represent pending/resolved/rejected states
- Automatic resolution when async action completes

### 4. Event Loop (Simplified)
- For now: synchronous execution (await blocks)
- Future: actual async event loop

## Implementation Steps

1. ✅ Add ASYNC and AWAIT tokens (already exists)
2. Parse async action declarations
3. Parse await expressions  
4. Create Promise object type
5. Implement async action execution
6. Implement await evaluation

## Syntax

```zexus
async action fetch_data() -> string {
    return "data loaded"
}

async action main() {
    let result = await fetch_data()
    print(result)
}
```
