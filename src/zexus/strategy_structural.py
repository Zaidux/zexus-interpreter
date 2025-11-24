# src/zexus/strategy_structural.py
from .zexus_token import *
from typing import List, Dict

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
        
        # Statement starters (keywords that begin a new statement)
        statement_starters = {
            LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, 
            SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, SEAL
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

            # Try-catch: collect the try block and catch block separately
            if t.type == TRY:
                start_idx = i
                # collect try token + following block tokens (brace-aware)
                try_block_tokens, next_idx = self._collect_brace_block(tokens, i + 1)
                # include the 'try' token as part of the block for context
                full_try_tokens = [t] + try_block_tokens
                # filter out empty tokens from the recorded token lists
                full_try_tokens = [tk for tk in full_try_tokens if not _is_empty_token(tk)]
                self.blocks[block_id] = {
                    'id': block_id,
                    'type': 'try_catch',
                    'subtype': 'try',
                    'tokens': full_try_tokens,
                    'start_token': t,
                    'start_index': start_idx,
                    'end_index': next_idx - 1,
                    'parent': None
                }
                current_try_id = block_id
                block_id += 1
                i = next_idx

                # If try block has inner statements, create child blocks for them
                inner = try_block_tokens[1:-1] if try_block_tokens and len(try_block_tokens) >= 2 else []
                inner = [tk for tk in inner if not _is_empty_token(tk)]
                if inner:
                    # If it's a map-like object, keep as single child map_literal
                    if self._is_map_literal(inner):
                        self.blocks[block_id] = {
                            'id': block_id,
                            'type': 'map_literal',
                            'subtype': 'map_literal',
                            'tokens': [tk for tk in try_block_tokens if not _is_empty_token(tk)],  # include braces
                            'start_token': try_block_tokens[0] if try_block_tokens else t,
                            'start_index': start_idx,
                            'end_index': next_idx - 1,
                            'parent': current_try_id
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
                                'start_token': (stmt_tokens[0] if stmt_tokens else try_block_tokens[0]),
                                'start_index': start_idx,
                                'end_index': start_idx + len(stmt_tokens),
                                'parent': current_try_id
                            }
                            block_id += 1

                # Look for catch token after try block
                if i < n and tokens[i].type == CATCH:
                    catch_token = tokens[i]
                    catch_block_tokens, after_catch_idx = self._collect_brace_block(tokens, i + 1)
                    full_catch_tokens = [catch_token] + catch_block_tokens
                    full_catch_tokens = [tk for tk in full_catch_tokens if not _is_empty_token(tk)]
                    self.blocks[block_id] = {
                        'id': block_id,
                        'type': 'try_catch',
                        'subtype': 'catch',
                        'tokens': full_catch_tokens,
                        'start_token': catch_token,
                        'start_index': i,
                        'end_index': after_catch_idx - 1,
                        'parent': None
                    }
                    current_catch_id = block_id
                    block_id += 1
                    i = after_catch_idx

                    # create child statements for catch block similarly
                    inner_catch = catch_block_tokens[1:-1] if catch_block_tokens and len(catch_block_tokens) >= 2 else []
                    inner_catch = [tk for tk in inner_catch if not _is_empty_token(tk)]
                    if inner_catch:
                        if self._is_map_literal(inner_catch):
                            self.blocks[block_id] = {
                                'id': block_id,
                                'type': 'map_literal',
                                'subtype': 'map_literal',
                                'tokens': [tk for tk in catch_block_tokens if not _is_empty_token(tk)],
                                'start_token': catch_block_tokens[0],
                                'start_index': i,
                                'end_index': after_catch_idx - 1,
                                'parent': current_catch_id
                            }
                            block_id += 1
                        else:
                            stmts = self._split_into_statements(inner_catch)
                            for stmt_tokens in stmts:
                                self.blocks[block_id] = {
                                    'id': block_id,
                                    'type': 'statement',
                                    'subtype': stmt_tokens[0].type if stmt_tokens else 'unknown',
                                    'tokens': [tk for tk in stmt_tokens if not _is_empty_token(tk)],
                                    'start_token': (stmt_tokens[0] if stmt_tokens else catch_block_tokens[0]),
                                    'start_index': i,
                                    'end_index': i + len(stmt_tokens),
                                    'parent': current_catch_id
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

            # Statement-like tokens: try to collect tokens up to a statement boundary
            if t.type in statement_starters:
                start_idx = i
                stmt_tokens = [t]  # Start with the statement starter token
                j = i + 1
                nesting = 0  # Track nesting level for (), [], {}

                while j < n:
                    tj = tokens[j]

                    # Track nesting level
                    if tj.type in {LPAREN, LBRACE, LBRACKET}:
                        nesting += 1
                    elif tj.type in {RPAREN, RBRACE, RBRACKET}:
                        nesting -= 1

                    # Only consider statement boundaries when not in nested structure
                    if nesting == 0:
                        # Stop at explicit terminators
                        if tj.type in stop_types:
                            stmt_tokens.append(tj)  # Include the terminator
                            break
                        # Stop at new statement starters only if we're done with current
                        if tj.type in statement_starters:
                            # Exception: allow chained method calls
                            prev = tokens[j-1] if j > 0 else None
                            if not (prev and prev.type == DOT):
                                break

                    # Always collect tokens while in nested structures
                    stmt_tokens.append(tj)
                    j += 1

                # Create block for the collected statement
                filtered_stmt_tokens = [tk for tk in stmt_tokens if not _is_empty_token(tk)]
                if filtered_stmt_tokens:  # Only create block if we have meaningful tokens
                    self.blocks[block_id] = {
                        'id': block_id,
                        'type': 'statement', 
                        'subtype': t.type,
                        'tokens': filtered_stmt_tokens,
                        'start_token': tokens[start_idx],
                        'start_index': start_idx,
                        'end_index': j,
                        'parent': None
                    }
                    block_id += 1
                i = j
                continue

            # Fallback: collect a run of tokens until a clear statement boundary
            start_idx = i
            run_tokens = [t]
            j = i + 1
            while j < n:
                tj = tokens[j]
                if tj.type in stop_types or tj.type in statement_starters or tj.type == LBRACE or tj.type == TRY:
                    break
                run_tokens.append(tj)
                j += 1
            filtered_run_tokens = [tk for tk in run_tokens if not _is_empty_token(tk)]
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
        statement_starters = {
            LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, 
            SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, SEAL
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
            if t.type == IDENT and i + 1 < n and tokens[i + 1].type == LPAREN:
                if any(st.type == ASSIGN for st in cur):
                    results.append(cur)
                    cur = []
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