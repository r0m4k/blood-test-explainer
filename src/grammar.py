"""GBNF grammar for the extraction schema.

Grammar-constrained decoding makes the local model **physically unable** to emit anything but
a valid `{tests:[...], notes:[...]}` object in our exact schema. For a small model this is the
single biggest reliability lever: no parse failures, no stray prose, no hallucinated keys.
Passed to llama.cpp via `LlamaGrammar.from_string(...)`.
"""

from __future__ import annotations

EXTRACTION_GRAMMAR = r"""
root    ::= "{" ws "\"tests\"" ws ":" ws tests ws "," ws "\"notes\"" ws ":" ws notes ws "}"

tests   ::= "[" ws ( test ( ws "," ws test )* )? ws "]"
test    ::= "{" ws
            "\"marker\"" ws ":" ws string ws "," ws
            "\"value\"" ws ":" ws string ws "," ws
            "\"unit\"" ws ":" ws strornull ws "," ws
            "\"reference_range\"" ws ":" ws strornull ws "," ws
            "\"status\"" ws ":" ws status ws "," ws
            "\"source_text\"" ws ":" ws strornull ws "," ws
            "\"confidence\"" ws ":" ws number ws
            "}"

notes   ::= "[" ws ( string ( ws "," ws string )* )? ws "]"

status  ::= "\"low\"" | "\"normal\"" | "\"high\"" | "\"abnormal\"" | "\"unknown\""
strornull ::= string | "null"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" ["\\/bfnrt]
number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?
ws      ::= [ \t\n]*
"""


def extraction_grammar() -> str:
    return EXTRACTION_GRAMMAR
