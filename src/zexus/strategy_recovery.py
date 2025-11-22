# strategy_recovery.py
from .zexus_token import *
from .zexus_ast import (
    EntityStatement,
    UseStatement,
    ExportStatement,
    TryCatchStatement,
    BlockStatement,
    ExpressionStatement,
    Identifier,
    StringLiteral,
)

class ErrorRecoveryEngine:
    def __init__(self, structural_analyzer, context_parser):
        self.structural_analyzer = structural_analyzer
        self.context_parser = context_parser
        self.recovery_strategies = {
            'expected_catch': self._recover_expected_catch,
            'unexpected_token': self._recover_unexpected_token,
            'missing_parenthesis': self._recover_missing_parenthesis,
            'missing_brace': self._recover_missing_brace,
            'syntax_error': self._recover_generic_syntax
        }

    def create_recovery_plan(self, error, current_block_info, current_tokens, token_index):
        """Create intelligent recovery plan based on error type and context"""
        print(f"🛠️ [Recovery] Analyzing error: {error}")

        error_type = self._classify_error(error)
        current_context = self.context_parser.get_current_context()

        if error_type in self.recovery_strategies:
            recovery_plan = self.recovery_strategies[error_type](
                error, current_block_info, current_context, current_tokens, token_index
            )

            if recovery_plan['can_recover']:
                print(f"✅ [Recovery] {recovery_plan['message']}")
                return recovery_plan

        # Final fallback
        return self._create_fallback_recovery(error, current_block_info, current_context)

    def _classify_error(self, error):
        """Classify the error type for appropriate recovery strategy"""
        error_msg = str(error).lower()

        if "expected catch" in error_msg or "catch" in error_msg:
            return 'expected_catch'
        elif "unexpected token" in error_msg:
            return 'unexpected_token'
        elif "missing )" in error_msg or "parenthesis" in error_msg:
            return 'missing_parenthesis'
        elif "missing }" in error_msg or "brace" in error_msg:
            return 'missing_brace'
        else:
            return 'syntax_error'

    def _recover_expected_catch(self, error, block_info, context, tokens, token_index):
        """Recover from 'expected catch' errors using structural analysis"""
        print("🛠️ [Recovery] Handling 'expected catch' error")

        # Check if structural analysis shows a catch block exists
        current_block_id = block_info.get('id') if block_info else None
        catch_block = self._find_catch_block_in_structure(current_block_id)

        if catch_block:
            return {
                'can_recover': True,
                'recovered_statement': self._create_dummy_try_catch(),
                'next_action': 'skip_to_block',
                'target_block': catch_block['id'],
                'message': 'Structural analysis shows catch block exists later - skipping to catch'
            }

        # If no catch block found, create a synthetic one
        return {
            'can_recover': True,
            'recovered_statement': self._create_synthetic_catch_block(),
            'next_action': 'continue',
            'message': 'Created synthetic catch block for recovery'
        }

    def _recover_unexpected_token(self, error, block_info, context, tokens, token_index):
        """Recover from unexpected token errors"""
        print(f"🛠️ [Recovery] Handling unexpected token at index {token_index}")

        # Extract the problematic token from error message or use current token
        problematic_token = self._extract_problematic_token(error, tokens, token_index)

        if problematic_token:
            return {
                'can_recover': True,
                'recovered_statement': self._create_skip_statement(problematic_token),
                'next_action': 'skip_tokens',
                'skip_count': 1,
                'message': f'Skipped problematic token: {problematic_token}'
            }

        return {'can_recover': False}

    def _recover_missing_parenthesis(self, error, block_info, context, tokens, token_index):
        """Recover from missing parenthesis errors"""
        print("🛠️ [Recovery] Handling missing parenthesis")

        # Look ahead to find a likely closing parenthesis
        for i in range(token_index, min(token_index + 10, len(tokens))):
            if tokens[i].type == RPAREN:
                return {
                    'can_recover': True,
                    'recovered_statement': self._create_dummy_expression(),
                    'next_action': 'skip_to_token',
                    'target_token_index': i + 1,
                    'message': 'Found potential closing parenthesis ahead'
                }

        # Insert synthetic closing parenthesis
        return {
            'can_recover': True,
            'recovered_statement': self._create_dummy_expression(),
            'next_action': 'insert_token',
            'insert_token': Token(RPAREN, ')', line=tokens[token_index].line, column=tokens[token_index].column + 1),
            'message': 'Inserted synthetic closing parenthesis'
        }

    def _recover_missing_brace(self, error, block_info, context, tokens, token_index):
        """Recover from missing brace errors using structural analysis"""
        print("🛠️ [Recovery] Handling missing brace")

        # Use structural analysis to find matching brace
        if block_info:
            # Structural analyzer already found the block boundaries
            return {
                'can_recover': True,
                'recovered_statement': BlockStatement(),
                'next_action': 'skip_to_block_end',
                'target_block': block_info['id'],
                'message': 'Using structural analysis to find block end'
            }

        return {'can_recover': False}

    def _recover_generic_syntax(self, error, block_info, context, tokens, token_index):
        """Generic syntax error recovery"""
        print("🛠️ [Recovery] Handling generic syntax error")

        # Try to skip a few tokens and continue
        return {
            'can_recover': True,
            'recovered_statement': self._create_dummy_statement(),
            'next_action': 'skip_tokens',
            'skip_count': 3,  # Skip next 3 tokens
            'message': 'Skipping ahead to recover from syntax error'
        }

    def _find_catch_block_in_structure(self, current_block_id):
        """Find a catch block in the structural analysis"""
        blocks = self.structural_analyzer.blocks

        # Look for catch blocks in current block or parent blocks
        for block_id, block in blocks.items():
            if block.get('subtype') == 'try_catch':
                if block.get('catch_section'):
                    return block
            # Check nested blocks
            for nested in block.get('nested_blocks', []):
                if nested.get('subtype') == 'try_catch' and nested.get('catch_section'):
                    return nested

        return None

    def _extract_problematic_token(self, error, tokens, token_index):
        """Extract the problematic token from error message or token stream"""
        if token_index < len(tokens):
            return tokens[token_index]

        # Try to extract from error message
        error_str = str(error)
        if "'" in error_str:
            # Extract token literal from error message like "Unexpected token 'catch'"
            start = error_str.find("'") + 1
            end = error_str.find("'", start)
            if start > 0 and end > start:
                token_literal = error_str[start:end]
                # Find matching token in stream
                for token in tokens:
                    if token.literal == token_literal:
                        return token

        return None

    def _create_dummy_try_catch(self):
        """Create a placeholder try-catch statement for recovery"""
        return TryCatchStatement(
            try_block=BlockStatement(),
            error_variable=Identifier("error"),
            catch_block=BlockStatement()
        )

    def _create_synthetic_catch_block(self):
        """Create a synthetic catch block when none exists"""
        return TryCatchStatement(
            try_block=BlockStatement(),
            error_variable=Identifier("error"),
            catch_block=BlockStatement([ExpressionStatement(StringLiteral("Recovery catch block"))])
        )

    def _create_skip_statement(self, skipped_token):
        """Create a statement that represents skipping a token"""
        return ExpressionStatement(StringLiteral(f"Skipped: {skipped_token.literal}"))

    def _create_dummy_expression(self):
        """Create a placeholder expression"""
        return ExpressionStatement(Identifier("recovery_expression"))

    def _create_dummy_statement(self):
        """Create a placeholder statement"""
        return ExpressionStatement(StringLiteral("Recovery statement"))

    def _create_fallback_recovery(self, error, block_info, context):
        """Final fallback recovery strategy"""
        return {
            'can_recover': True,
            'recovered_statement': self._create_dummy_statement(),
            'next_action': 'skip_tokens', 
            'skip_count': 5,
            'message': 'Using fallback recovery - skipping ahead'
        }

    def apply_recovery_plan(self, recovery_plan, current_tokens, current_index):
        """Apply the recovery plan to adjust parsing state"""
        if not recovery_plan['can_recover']:
            return current_index

        action = recovery_plan.get('next_action', 'continue')

        if action == 'skip_tokens':
            skip_count = recovery_plan.get('skip_count', 1)
            return min(current_index + skip_count, len(current_tokens) - 1)

        elif action == 'skip_to_token':
            target_index = recovery_plan.get('target_token_index', current_index)
            return min(target_index, len(current_tokens) - 1)

        elif action == 'skip_to_block':
            # This would require coordination with the main parser
            print("⚠️ [Recovery] Block skipping requires parser integration")
            return current_index + 1

        elif action == 'insert_token':
            # This would modify the token stream
            print("⚠️ [Recovery] Token insertion requires stream modification")
            return current_index

        return current_index + 1  # Default: move to next token