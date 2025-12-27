# src/zexus/strategy_structural.py
from ..zexus_token import *
from typing import List, Dict
from ..config import config as zexus_config

class StructuralAnalyzer:
    """Lightweight structural analyzer that splits token stream into top-level blocks.
    Special handling for try/catch to avoid merging statements inside try blocks.
    """

    def __init__(self):
        # blocks: id -> block_info
        self.blocks = {}

    def analyze(self, tokens: List):
        """Analyze tokens and produce a block map used by the context parser.

        block_info keys:
            - id: unique id
            - type/subtype: block type (e.g. 'try', 'let', 'print', 'block')
            - tokens: list of tokens that belong to the block
            - start_token: token object where block starts
            - start_index / end_index: indices in original token stream
            - parent: optional parent block id
        """
        self.blocks = {}
        i = 0
        block_id = 0
        n = len(tokens)

        # helper sets for stopping heuristics (mirrors context parser)
        stop_types = {SEMICOLON, RBRACE}
        
        # Modifier tokens that should be merged with the following statement
        modifier_tokens = {PUBLIC, PRIVATE, SEALED, ASYNC, NATIVE, INLINE, SECURE, PURE, VIEW, PAYABLE}
        
        # Statement starters (keywords that begin a new statement)
        # NOTE: SEND and RECEIVE removed - they can be used as function calls in expressions
        statement_starters = {
              LET, CONST, DATA, PRINT, FOR, IF, WHILE, RETURN, CONTINUE, BREAK, THROW, ACTION, FUNCTION, TRY, EXTERNAL,
              SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, SEAL, PERSISTENT, AUDIT,
              RESTRICT, SANDBOX, TRAIL, GC, BUFFER, SIMD,
              DEFER, PATTERN, ENUM, STREAM, WATCH,
              CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE,
              INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING,
              CHANNEL, ATOMIC,
              # Blockchain keywords
              LEDGER, STATE, REQUIRE, REVERT, LIMIT
          }

        while i < n:
            t = tokens[i]
            # skip EOF tokens
            if t.type == EOF:
                i += 1
                continue

            # Helper: skip tokens that are empty/whitespace-only literals when building blocks
            def _is_empty_token(tok):
                lit = getattr(tok, 'literal', None)
                return (lit == '' or lit is None) and tok.type != STRING and tok.type != IDENT

            # === FIXED: Enhanced USE statement detection ===
            if t.type == USE:
                start_idx = i
                use_tokens = [t]
                i += 1

                # Handle use { ... } from ... syntax
                if i < n and tokens[i].type == LBRACE:
                    # Collect until closing brace
                    brace_count = 1
                    use_tokens.append(tokens[i])
                    i += 1

                    while i < n and brace_count > 0:
                        use_tokens.append(tokens[i])
                        if tokens[i].type == LBRACE:
                            brace_count += 1
                        elif tokens[i].type == RBRACE:
                            brace_count -= 1
                        i += 1

                    # Look for 'from' and file path
                    # FIX: Stop if we hit a statement starter, semicolon, or EOF
                    while i < n and tokens[i].type not in stop_types and tokens[i].type not in statement_starters:
                        # FIX: Check for FROM token type OR identifier 'from'
                        is_from = (tokens[i].type == FROM) or (tokens[i].type == IDENT and tokens[i].literal == 'from')
                        
                        if is_from:
                            # Include 'from' and the following string
                            use_tokens.append(tokens[i])
                            i += 1
                            if i < n and tokens[i].type == STRING:
                                use_tokens.append(tokens[i])
                                i += 1
                            break
                        else:
                            use_tokens.append(tokens[i])
                            i += 1
                else:
                    # Simple use 'path' syntax
                    # FIX: Stop at statement starters to prevent greedy consumption
                    while i < n and tokens[i].type not in stop_types and tokens[i].type != EOF:
                        if tokens[i].type in statement_starters:
                            break
                        use_tokens.append(tokens[i])
                        i += 1

                # Create block for this use statement
                filtered_tokens = [tk for tk in use_tokens if not _is_empty_token(tk)]
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'statement',
                    'subtype': 'use_statement',
                    'tokens': filtered_tokens,
                    'start_token': tokens[start_idx],
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'parent': None
                }
                block_id += 1
                continue

            # Enhanced ENTITY statement detection
            elif t.type == ENTITY:
                start_idx = i
                entity_tokens = [t]
                i += 1

                # Collect entity name
                if i < n and tokens[i].type == IDENT:
                    entity_tokens.append(tokens[i])
                    i += 1

                # Collect until closing brace
                brace_count = 0
                while i < n:
                    # Check if we've found the opening brace
                    if tokens[i].type == LBRACE:
                        brace_count = 1
                        entity_tokens.append(tokens[i])
                        i += 1
                        break
                    entity_tokens.append(tokens[i])
                    i += 1

                # Now collect until matching closing brace
                while i < n and brace_count > 0:
                    entity_tokens.append(tokens[i])
                    if tokens[i].type == LBRACE:
                        brace_count += 1
                    elif tokens[i].type == RBRACE:
                        brace_count -= 1
                    i += 1

                # Create block
                filtered_tokens = [tk for tk in entity_tokens if not _is_empty_token(tk)]
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'statement',
                    'subtype': 'entity_statement',
                    'tokens': filtered_tokens,
                    'start_token': tokens[start_idx],
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'parent': None
                }
                block_id += 1
                continue
            
            # CONTRACT statement detection
            elif t.type == CONTRACT:
                start_idx = i
                contract_tokens = [t]
                i += 1

                # Collect contract name
                if i < n and tokens[i].type == IDENT:
                    contract_tokens.append(tokens[i])
                    i += 1

                # Collect until closing brace
                brace_count = 0
                while i < n:
                    if tokens[i].type == LBRACE:
                        brace_count = 1
                        contract_tokens.append(tokens[i])
                        i += 1
                        break
                    contract_tokens.append(tokens[i])
                    i += 1

                while i < n and brace_count > 0:
                    contract_tokens.append(tokens[i])
                    if tokens[i].type == LBRACE:
                        brace_count += 1
                    elif tokens[i].type == RBRACE:
                        brace_count -= 1
                    i += 1

                filtered_tokens = [tk for tk in contract_tokens if not _is_empty_token(tk)]
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'statement',
                    'subtype': 'contract_statement',
                    'tokens': filtered_tokens,
                    'start_token': tokens[start_idx],
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'parent': None
                }
                block_id += 1
                continue

            # Try-catch: collect the try block and catch block TOGETHER
            if t.type == TRY:
                start_idx = i
                # collect try token + following block tokens (brace-aware)
                try_block_tokens, next_idx = self._collect_brace_block(tokens, i + 1)
                
                # Check for catch block
                catch_tokens = []
                final_idx = next_idx
                
                if next_idx < n and tokens[next_idx].type == CATCH:
                    catch_token = tokens[next_idx]
                    
                    # Collect tokens between CATCH and LBRACE (e.g. (e))
                    pre_brace_tokens = []
                    curr = next_idx + 1
                    while curr < n and tokens[curr].type != LBRACE and tokens[curr].type != EOF:
                        pre_brace_tokens.append(tokens[curr])
                        curr += 1
                    
                    catch_block_tokens, after_catch_idx = self._collect_brace_block(tokens, curr)
                    catch_tokens = [catch_token] + pre_brace_tokens + catch_block_tokens
                    final_idx = after_catch_idx
                
                # Combine all tokens
                full_tokens = [t] + try_block_tokens + catch_tokens
                full_tokens = [tk for tk in full_tokens if not _is_empty_token(tk)]
                
                # Create the main try-catch block
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'statement',
                    'subtype': 'try_catch_statement',
                    'tokens': full_tokens,
                    'start_token': t,
                    'start_index': start_idx,
                    'end_index': final_idx - 1,
                    'parent': None
                }
                parent_id = block_id
                block_id += 1
                i = final_idx

                # Process inner statements of TRY block
                inner = try_block_tokens[1:-1] if try_block_tokens and len(try_block_tokens) >= 2 else []
                inner = [tk for tk in inner if not _is_empty_token(tk)]
                if inner:
                    if self._is_map_literal(inner):
                        # ... map literal handling ...
                        pass 
                    else:
                        stmts = self._split_into_statements(inner)
                        for stmt_tokens in stmts:
                            self.blocks[block_id] = {
                                'id': block_id,
                                'type': 'statement',
                                'subtype': stmt_tokens[0].type if stmt_tokens else 'unknown',
                                'tokens': [tk for tk in stmt_tokens if not _is_empty_token(tk)],
                                'start_token': (stmt_tokens[0] if stmt_tokens else try_block_tokens[0]),
                                'start_index': start_idx, # Approximate
                                'end_index': start_idx,   # Approximate
                                'parent': parent_id
                            }
                            block_id += 1

                # Process inner statements of CATCH block
                if catch_tokens:
                    # catch_tokens[0] is CATCH
                    # catch_tokens[1] might be (error) or {
                    # We need to find the brace block inside catch_tokens
                    catch_brace_tokens = []
                    for k, ctk in enumerate(catch_tokens):
                        if ctk.type == LBRACE:
                            catch_brace_tokens = catch_tokens[k:]
                            break
                    
                    inner_catch = catch_brace_tokens[1:-1] if catch_brace_tokens and len(catch_brace_tokens) >= 2 else []
                    inner_catch = [tk for tk in inner_catch if not _is_empty_token(tk)]
                    
                    if inner_catch:
                        stmts = self._split_into_statements(inner_catch)
                        for stmt_tokens in stmts:
                            self.blocks[block_id] = {
                                'id': block_id,
                                'type': 'statement',
                                'subtype': stmt_tokens[0].type if stmt_tokens else 'unknown',
                                'tokens': [tk for tk in stmt_tokens if not _is_empty_token(tk)],
                                'start_token': (stmt_tokens[0] if stmt_tokens else catch_tokens[0]),
                                'start_index': next_idx, # Approximate
                                'end_index': next_idx,   # Approximate
                                'parent': parent_id
                            }
                            block_id += 1
                continue

            # Brace-delimited top-level block
            if t.type == LBRACE:
                block_tokens, next_idx = self._collect_brace_block(tokens, i)
                this_block_id = block_id
                # filter empty tokens before storing
                filtered_block_tokens = [tk for tk in block_tokens if not _is_empty_token(tk)]
                self.blocks[this_block_id] = {
                    'id': this_block_id,
                    'type': 'block',
                    'subtype': 'brace_block',
                    'tokens': filtered_block_tokens,
                    'start_token': tokens[i],
                    'start_index': i,
                    'end_index': next_idx - 1,
                    'parent': None
                }
                block_id += 1

                # split inner tokens into child blocks unless it's a map literal
                inner = block_tokens[1:-1] if block_tokens and len(block_tokens) >= 2 else []
                inner = [tk for tk in inner if not _is_empty_token(tk)]
                if inner:
                    if self._is_map_literal(inner):
                        self.blocks[block_id] = {
                            'id': block_id,
                            'type': 'map_literal',
                            'subtype': 'map_literal',
                            'tokens': [tk for tk in block_tokens if not _is_empty_token(tk)],  # keep full braces
                            'start_token': block_tokens[0],
                            'start_index': i,
                            'end_index': next_idx - 1,
                            'parent': this_block_id
                        }
                        block_id += 1
                    else:
                        stmts = self._split_into_statements(inner)
                        for stmt_tokens in stmts:
                            self.blocks[block_id] = {
                                'id': block_id,
                                'type': 'statement',
                                'subtype': stmt_tokens[0].type if stmt_tokens else 'unknown',
                                'tokens': [tk for tk in stmt_tokens if not _is_empty_token(tk)],
                                'start_token': (stmt_tokens[0] if stmt_tokens else block_tokens[0]),
                                'start_index': i,
                                'end_index': i + len(stmt_tokens),
                                'parent': this_block_id
                            }
                            block_id += 1

                i = next_idx
                continue

            # Modifier tokens: merge with the following statement
            if t.type in modifier_tokens:
                start_idx = i
                modifier_list = []
                
                # Collect consecutive modifiers
                while i < n and tokens[i].type in modifier_tokens:
                    modifier_list.append(tokens[i])
                    i += 1
                
                # Skip EOF/whitespace
                while i < n and tokens[i].type == EOF:
                    i += 1
                
                # If followed by a statement starter, continue to statement parsing
                # by falling through to the elif below
                if i < n and tokens[i].type in statement_starters:
                    # Update t to point to the statement starter
                    t = tokens[i]
                    # Don't increment i - let the statement parsing handle it
                else:
                    # Modifiers without a following statement - this is an async expression!
                    # Collect the modifiers AND the following expression into one block
                    # Example: "async producer()" should be one block
                    
                    # Start collecting the expression that follows
                    j = i
                    expr_tokens = modifier_list[:]  # Include modifiers in the block
                    nesting = 0
                    started_expr = False
                    
                    # Collect tokens for the expression
                    while j < n:
                        tj = tokens[j]
                        
                        # Track nesting
                        if tj.type in {LPAREN, LBRACKET, LBRACE}:
                            nesting += 1
                            started_expr = True
                        elif tj.type in {RPAREN, RBRACKET, RBRACE}:
                            nesting -= 1
                        
                        expr_tokens.append(tj)
                        j += 1
                        
                        # Stop at semicolon when at nesting 0
                        if nesting == 0 and tj.type == SEMICOLON:
                            break
                        
                        # Stop after completing a simple expression at nesting 0
                        # (identifier with optional call, or after closing all parens)
                        if started_expr and nesting == 0:
                            break
                    
                    # Create block for async expression
                    self.blocks[block_id] = {
                        'id': block_id,
                        'type': 'statement',
                        'subtype': modifier_list[0].type,  # ASYNC
                        'tokens': expr_tokens,
                        'start_token': modifier_list[0],
                        'start_index': start_idx,
                        'end_index': j
                    }
                    block_id += 1
                    i = j
                    # Clear modifier_list so it doesn't affect next statement
                    del modifier_list
                    continue
            
            # Statement-like tokens: try to collect tokens up to a statement boundary
            # DUAL-MODE DEBUG: skip if debug( ) which is a function call, not statement
            if t.type in statement_starters and not (t.type == DEBUG and i + 1 < n and tokens[i + 1].type == LPAREN):
                # Check if we just processed modifiers
                if 'modifier_list' in locals() and start_idx < i:
                    # Start from modifier position, include modifiers in stmt_tokens
                    stmt_start_idx = start_idx
                    stmt_tokens = modifier_list + [t]
                    j = i + 1
                    del modifier_list  # Clear for next iteration
                else:
                    stmt_start_idx = i
                    stmt_tokens = [t]  # Start with the statement starter token
                    j = i + 1
                nesting = 0  # Track nesting level for (), [], {}
                found_brace_block = False  # Did we encounter a { ... } block?
                found_colon_block = False  # Did we encounter a : (tolerable syntax)?
                baseline_column = None  # Track indentation for colon-based blocks
                in_assignment = (t.type in {LET, CONST})  # Are we in an assignment RHS?

                while j < n:
                    tj = tokens[j]

                    # Check if this is a statement terminator at nesting 0 BEFORE updating nesting
                    if nesting == 0 and tj.type in stop_types and not found_colon_block:
                        break
                    
                    # CRITICAL FIX: Check if next token starts a new statement (assignment or function call)
                    if nesting == 0 and len(stmt_tokens) > 1:  # Only check if we've collected some tokens
                        # IDENT followed by ASSIGN is an assignment statement
                        if tj.type == IDENT and j + 1 < n and tokens[j + 1].type == ASSIGN:
                            break
                        # IDENT followed by LPAREN is a function call (already handled below, but listed for clarity)
                    
                    # Detect colon-based block (tolerable syntax for action/function/if/while etc.)
                    if tj.type == COLON and nesting == 0 and t.type in {ACTION, FUNCTION, IF, WHILE, FOR}:
                        found_colon_block = True
                        stmt_tokens.append(tj)
                        j += 1
                        # Record the baseline column for dedent detection
                        # This is the column of the first token AFTER the colon
                        if j < n:
                            baseline_column = tokens[j].column if hasattr(tokens[j], 'column') else 1
                        continue
                    
                    # Track nesting level BEFORE dedent check (so we don't break inside {...} or [...] or (...))
                    if tj.type in {LPAREN, LBRACE, LBRACKET}:
                        # Only mark as brace block if NOT already in colon block (to distinguish code blocks from data literals)
                        if tj.type == LBRACE and not found_colon_block:
                            found_brace_block = True
                        nesting += 1
                    elif tj.type in {RPAREN, RBRACE, RBRACKET}:
                        nesting -= 1
                    
                    # If we're in a colon block, collect until dedent
                    if found_colon_block and nesting == 0:
                        current_column = tj.column if hasattr(tj, 'column') else 1
                        # Stop if we hit a dedent (token BEFORE baseline column, indicating unindent)
                        # This works because baseline_column is the indented level (e.g., 6)
                        # and when we see column 2, that's < 6, so we stop
                        #print(f"    [DEDENT CHECK] token={tj.type} col={current_column} baseline={baseline_column} nesting={nesting}")
                        if current_column < baseline_column and tj.type in statement_starters:
                            #print(f"    [DEDENT BREAK] Breaking on dedent: {tj.type} at col {current_column}")
                            break

                    # Stop at new statement starters only if we're at nesting 0
                    # BUT: for LET/CONST, allow function expressions in the RHS
                    if nesting == 0 and tj.type in statement_starters and not found_colon_block:
                        # Exception: allow chained method calls
                        prev = tokens[j-1] if j > 0 else None
                        if not (prev and prev.type == DOT):
                            # For LET/CONST, allow FUNCTION, SANDBOX, SANITIZE as RHS (expressions)
                            # Also allow DEBUG when followed by ( for debug(x) function calls in assignments
                            # Also allow IF when followed by THEN (if-then-else expression)
                            allow_in_assignment = tj.type in {FUNCTION, SANDBOX, SANITIZE}
                            allow_debug_call = tj.type == DEBUG and j + 1 < n and tokens[j + 1].type == LPAREN
                            allow_if_then_else = False
                            if tj.type == IF:
                                # Look ahead for THEN to detect if-then-else expression
                                for k in range(j + 1, min(j + 20, n)):  # Look ahead up to 20 tokens
                                    if tokens[k].type == THEN:
                                        allow_if_then_else = True
                                        break
                                    elif tokens[k].type in {LBRACE, COLON}:
                                        # Found statement form indicators
                                        break
                            if not (in_assignment and (allow_in_assignment or allow_debug_call or allow_if_then_else)):
                                break
                    
                    # CRITICAL FIX: Also break on modifier tokens at nesting 0 when followed by statement keywords
                    # This prevents previous statements from consuming modifiers like "async action foo()"
                    # But ALLOWS "async foo()" expressions to stay together
                    if nesting == 0 and tj.type in modifier_tokens and not found_colon_block and len(stmt_tokens) > 0:
                        # Look ahead to see if modifier is followed by a statement keyword
                        next_idx = j + 1
                        while next_idx < n and tokens[next_idx].type in modifier_tokens:
                            next_idx += 1
                        if next_idx < n and tokens[next_idx].type in statement_starters:
                            # Modifier followed by statement keyword - break here
                            break
                        # ALSO break if this is an ASYNC modifier followed by IDENT+LPAREN (async expression)
                        # This prevents LET statements from consuming "async func()" on the next line
                        if tj.type == ASYNC and next_idx < n and tokens[next_idx].type == IDENT:
                            if next_idx + 1 < n and tokens[next_idx + 1].type == LPAREN:
                                # This is "async ident(" - an async expression
                                break
                        # Otherwise, continue collecting (async expression case)
                    
                    # FIX: Also break at expression statements (IDENT followed by LPAREN)  when we're at nesting 0
                    # and not in an assignment context
                    # EXCEPTION: Don't break if we're parsing ACTION/FUNCTION (their names are followed by LPAREN for parameters)
                    if nesting == 0 and not in_assignment and not found_colon_block and not found_brace_block and t.type not in {ACTION, FUNCTION}:
                        if tj.type == IDENT and j + 1 < n and tokens[j + 1].type == LPAREN:
                            # This looks like a function call starting a new expression statement
                            # Only break if we've already collected some tokens (not the first token)
                            if len(stmt_tokens) > 1:
                                break

                    # Always collect tokens
                    stmt_tokens.append(tj)
                    j += 1
                    
                    # MODIFIED: For RETURN and CONTINUE, stop after closing parens
                    # But PRINT can have multiple comma-separated arguments, so don't break for PRINT
                    if t.type in {RETURN, CONTINUE} and nesting == 0 and tj.type == RPAREN:
                        break
                    
                    # If we just closed a brace block and are back at nesting 0, stop
                    if found_brace_block and nesting == 0:
                        # CRITICAL FIX: For IF statements, check if followed by ELSE or ELIF
                        if t.type == IF:
                            # Look ahead for else/elif
                            if j < n and tokens[j].type in {ELSE, ELIF}:
                                # Found else/elif - continue collecting
                                found_brace_block = False
                                continue
                        
                        # REQUIRE tolerance block: the {...} is part of the statement, not separate
                        # Don't break yet - the brace block is the tolerance logic
                        if t.type == REQUIRE:
                            found_brace_block = False
                            continue
                        
                        break

                # Skip any trailing semicolons
                while j < n and tokens[j].type == SEMICOLON:
                    j += 1

                # Create block for the collected statement
                filtered_stmt_tokens = [tk for tk in stmt_tokens if not _is_empty_token(tk)]
                if filtered_stmt_tokens:  # Only create block if we have meaningful tokens
                    self.blocks[block_id] = {
                        'id': block_id,
                        'type': 'statement', 
                        'subtype': t.type,
                        'tokens': filtered_stmt_tokens,
                        'start_token': tokens[stmt_start_idx],
                        'start_index': stmt_start_idx,
                        'end_index': j,
                        'parent': None
                    }
                    block_id += 1
                i = j
                continue

            # Fallback: collect a run of tokens until a clear statement boundary
            # Respect nesting so that constructs inside parentheses/braces aren't split
            # FIX: Handle expression statements (function calls not assigned to variables)
            start_idx = i
            run_tokens = [t]
            j = i + 1
            nesting = 0
            
            # Check if this is a simple function call expression statement: ident(...) 
            is_function_call_start = (t.type == IDENT and j < n and tokens[j].type == LPAREN)
            
            while j < n:
                tj = tokens[j]
                # Update nesting for parentheses/brackets/braces
                if tj.type in {LPAREN, LBRACE, LBRACKET}:
                    nesting += 1
                elif tj.type in {RPAREN, RBRACE, RBRACKET}:
                    if nesting > 0:
                        nesting -= 1

                # Only consider these as boundaries when at top-level (nesting == 0)
                if nesting == 0:
                    # Check if current token (tj) starts a new statement
                    # CRITICAL FIX: IDENT followed by ASSIGN is an assignment statement
                    is_assignment_start = (tj.type == IDENT and j + 1 < n and tokens[j + 1].type == ASSIGN)
                    is_new_statement = (
                        tj.type in stop_types or 
                        tj.type in statement_starters or 
                        tj.type == LBRACE or 
                        tj.type == TRY or
                        is_assignment_start
                    )
                    if is_new_statement and j > start_idx:  # Only break if we've collected at least one token
                        break
                
                # FIX: If this is a function call and nesting just became 0 (closed all parens),
                # check if next token looks like start of new statement
                if is_function_call_start and nesting == 0 and j > start_idx + 1:
                    # We've closed the function call parens
                    # Check if next token starts a new statement (IDENT followed by LPAREN, or a statement keyword)
                    next_idx = j + 1
                    # Skip semicolons
                    while next_idx < n and tokens[next_idx].type == SEMICOLON:
                        next_idx += 1
                    if next_idx < n:
                        next_tok = tokens[next_idx]
                        # If next token is a statement starter OR an identifier followed by (, it's a new statement
                        if next_tok.type in statement_starters:
                            run_tokens.append(tj)
                            j += 1
                            break
                        elif next_tok.type == IDENT and next_idx + 1 < n and tokens[next_idx + 1].type == LPAREN:
                            # Next statement is also a function call
                            run_tokens.append(tj)
                            j += 1
                            break

                run_tokens.append(tj)
                j += 1
            
            # Skip trailing semicolons (they're statement terminators, not part of the statement)
            while j < n and tokens[j].type == SEMICOLON:
                j += 1
            
            filtered_run_tokens = [tk for tk in run_tokens if not _is_empty_token(tk)]
            if filtered_run_tokens:  # Only create block if we have meaningful tokens
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'statement',
                    'subtype': (filtered_run_tokens[0].type if filtered_run_tokens else (run_tokens[0].type if run_tokens else 'token_run')),
                    'tokens': filtered_run_tokens,
                    'start_token': (filtered_run_tokens[0] if filtered_run_tokens else (run_tokens[0] if run_tokens else t)),
                    'start_index': start_idx,
                    'end_index': j - 1,
                    'parent': None
                }
                block_id += 1
            i = j

        return self.blocks

    def _collect_brace_block(self, tokens: List, start_index: int):
        """Collect tokens comprising a brace-delimited block.
        start_index should point at the token immediately after the 'try' or at a LBRACE.
        Returns (collected_tokens_including_braces, next_index_after_block)
        """
        n = len(tokens)
        # find the opening brace if start_index points to something else
        i = start_index
        # if the next token is not a LBRACE, try to find it
        if i < n and tokens[i].type != LBRACE:
            # scan forward to first LBRACE or EOF
            while i < n and tokens[i].type != LBRACE and tokens[i].type != EOF:
                i += 1
            if i >= n or tokens[i].type != LBRACE:
                # no brace, return empty block
                return [], start_index

        # i points to LBRACE
        depth = 0
        collected = []
        while i < n:
            tok = tokens[i]
            collected.append(tok)
            if tok.type == LBRACE:
                depth += 1
            elif tok.type == RBRACE:
                depth -= 1
                if depth == 0:
                    return collected, i + 1
            i += 1

        # Reached EOF without closing brace - return what we have (tolerant)
        return collected, i

    def _split_into_statements(self, tokens: List):
        """Split a flat list of tokens into a list of statement token lists using statement boundaries."""
        results = []
        if not tokens:
            return results

        stop_types = {SEMICOLON, RBRACE}
        # NOTE: SEND and RECEIVE removed - they can be used as function calls in expressions
        statement_starters = {
              LET, CONST, DATA, PRINT, FOR, IF, WHILE, RETURN, CONTINUE, BREAK, THROW, ACTION, FUNCTION, TRY, EXTERNAL, 
              SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, SEAL, AUDIT,
              RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD,
              DEFER, PATTERN, ENUM, STREAM, WATCH,
              CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE,
              INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING,
              CHANNEL, ATOMIC, ASYNC  # Added ASYNC to recognize async expressions as statement boundaries
          }

        cur = []
        i = 0
        n = len(tokens)

        while i < n:
            t = tokens[i]

            # Enhanced use statement detection (with braces) in inner blocks
            if t.type == USE:
                if cur:  # Finish current statement
                    results.append(cur)
                    cur = []

                # Collect the entire use statement
                use_tokens = [t]
                i += 1
                brace_count = 0

                # FIX: Check for statement starters here too to be safe
                while i < n:
                    if brace_count == 0 and tokens[i].type in statement_starters:
                         break

                    use_tokens.append(tokens[i])
                    if tokens[i].type == LBRACE:
                        brace_count += 1
                    elif tokens[i].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            # Look for 'from' after closing brace
                            # FIX: Check FROM token type
                            if i + 1 < n and (tokens[i + 1].type == FROM or (tokens[i + 1].type == IDENT and tokens[i + 1].literal == 'from')):
                                use_tokens.append(tokens[i + 1])
                                i += 1
                                if i + 1 < n and tokens[i + 1].type == STRING:
                                    use_tokens.append(tokens[i + 1])
                                    i += 1
                            break
                    elif brace_count == 0 and tokens[i].type in stop_types:
                        break
                    i += 1

                results.append(use_tokens)
                i += 1
                continue

            # Entity/Contract statement detection (generic brace collector)
            if t.type == ENTITY or t.type == CONTRACT:
                if cur:
                    results.append(cur)
                    cur = []

                # Collect until closing brace
                entity_tokens = [t]
                i += 1
                brace_count = 0

                while i < n:
                    entity_tokens.append(tokens[i])
                    if tokens[i].type == LBRACE:
                        brace_count += 1
                    elif tokens[i].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    i += 1

                results.append(entity_tokens)
                i += 1
                continue

            # start of a statement
            if not cur:
                cur.append(t)
                i += 1
                continue

            # accumulate until boundary
            if t.type in stop_types:
                # end current statement (do not include terminator)
                results.append(cur)
                cur = []
                i += 1
                continue

            if t.type in statement_starters:
                # boundary: emit current and start new
                results.append(cur)
                cur = [t]
                i += 1
                continue

            # Assignment RHS vs function-call heuristic:
            # if current token is IDENT followed by LPAREN and we've seen ASSIGN in cur, treat as a boundary
            # ALSO: if current token is IDENT+LPAREN and the previous token was RPAREN (end of prev call), new statement
            if t.type == IDENT and i + 1 < n and tokens[i + 1].type == LPAREN:
                if any(st.type == ASSIGN for st in cur):
                    results.append(cur)
                    cur = []
                    continue
                # New heuristic: if previous token was RPAREN (completing a call), this is likely a new statement
                if cur and cur[-1].type == RPAREN:
                    results.append(cur)
                    cur = [t]
                    i += 1
                    continue

            cur.append(t)
            i += 1

        if cur:
            results.append(cur)
        return results

    def _is_map_literal(self, inner_tokens: List):
        """Detect simple map/object literal pattern: STRING/IDENT followed by COLON somewhere early."""
        if not inner_tokens:
            return False
        # look at the first few tokens: key(:)value pairs
        for i in range(min(len(inner_tokens)-1, 8)):
            if inner_tokens[i].type in (STRING, IDENT) and i+1 < len(inner_tokens) and inner_tokens[i+1].type == COLON:
                return True
        return False

    def print_structure(self):
        print("🔎 Structural Analyzer - Blocks:")
        for bid, info in self.blocks.items():
            start = info.get('start_index')
            end = info.get('end_index')
            ttype = info.get('type')
            subtype = info.get('subtype')
            token_literals = [t.literal for t in info.get('tokens', []) if getattr(t, 'literal', None)]
            print(f"  [{bid}] {ttype}/{subtype} @ {start}-{end}: {token_literals}")