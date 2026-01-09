"""Minimal lexer smoke tests covering keyword recognition."""

from zexus.lexer import Lexer
from zexus.zexus_token import ACTION, IDENT, EOF


def test_action_keyword_token():
	lexer = Lexer("action")
	token = lexer.next_token()
	assert token.type == ACTION
	assert token.literal == "action"
	assert lexer.next_token().type == EOF


def test_action_followed_by_identifier():
	lexer = Lexer("action simplest")
	first = lexer.next_token()
	second = lexer.next_token()

	assert first.type == ACTION
	assert first.literal == "action"
	assert second.type == IDENT
	assert second.literal == "simplest"
	assert lexer.next_token().type == EOF
