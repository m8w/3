#!/usr/bin/env python3
"""Fully parenthesise HLSL expressions into strictly-binary form.

WHY THIS EXISTS
---------------
hlslparser-js 0.1.1 (pinned by milkdrop-preset-converter 0.1.2, the converter
this project uses) mistranslates any binary expression with three or more
operands at one nesting level. The top-level operator is replaced with `&&`
and both sides are wrapped in bool() casts:

    a * b * c      ->  float( (bool((a * b)) && bool(c)) )        WRONG
    a + b + c + d  ->  float( (bool((a + b)) && bool((c + d))) )  WRONG
    a * b + c * d  ->  float( (bool((a * b)) && bool((c * d))) )  WRONG
    (a * b) * c    ->  ((a * b) * c)                              correct

The arithmetic silently becomes boolean logic, so the shader compiles and
renders something - just not what was written. Explicitly parenthesising
every binary operation into two-operand form avoids the broken path.

Precedence-climbing parser over the HLSL subset these presets use:
    literals, identifiers, function calls, member/swizzle access, indexing,
    unary - + !, and the binary operators below. No statements, no control
    flow inside expressions (the ternary is supported but unused).

VERIFYING THE RESULT
--------------------
These shaders contain no boolean logic at all, so any `&&`, `||` or `bool(`
in the converted GLSL is proof the bug was hit. build_presets.py gates on
exactly that.
"""
from __future__ import annotations

import re
import sys

# (operators, left-associative) in increasing precedence
PRECEDENCE: list[tuple[str, ...]] = [
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<=", ">=", "<", ">"),
    ("+", "-"),
    ("*", "/", "%"),
]

TOKEN_RE = re.compile(r"""
      (?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%<>&^|!~?:.,()\[\]])
    | (?P<ws>\s+)
""", re.VERBOSE)


class Tok:
    __slots__ = ("kind", "val")

    def __init__(self, kind: str, val: str):
        self.kind, self.val = kind, val

    def __repr__(self):
        return f"{self.kind}:{self.val}"


def tokenize(s: str) -> list[Tok]:
    toks: list[Tok] = []
    i = 0
    while i < len(s):
        m = TOKEN_RE.match(s, i)
        if not m:
            raise ValueError(f"cannot tokenize at {i}: {s[i:i+30]!r}")
        i = m.end()
        if m.lastgroup == "ws":
            continue
        toks.append(Tok(m.lastgroup, m.group()))
    return toks


class Parser:
    def __init__(self, toks: list[Tok]):
        self.t = toks
        self.i = 0

    def peek(self) -> Tok | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self) -> Tok:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, val: str):
        tok = self.next()
        if tok.val != val:
            raise ValueError(f"expected {val!r}, got {tok.val!r}")

    # primary := number | ident | ident(args) | (expr)   with postfix . [ ]
    def parse_primary(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end of expression")

        if tok.val in ("-", "+", "!", "~"):
            self.next()
            return f"({tok.val}{self.parse_primary()})"

        if tok.val == "(":
            self.next()
            inner = self.parse_expr(0)
            self.expect(")")
            out = f"({inner})"
            return self.parse_postfix(out)

        if tok.kind == "num":
            self.next()
            return self.parse_postfix(tok.val)

        if tok.kind == "ident":
            self.next()
            nxt = self.peek()
            if nxt is not None and nxt.val == "(":
                self.next()
                args: list[str] = []
                if self.peek() is not None and self.peek().val != ")":
                    while True:
                        args.append(self.parse_expr(0))
                        nx = self.peek()
                        if nx is not None and nx.val == ",":
                            self.next()
                            continue
                        break
                self.expect(")")
                return self.parse_postfix(f"{tok.val}({', '.join(args)})")
            return self.parse_postfix(tok.val)

        raise ValueError(f"unexpected token {tok.val!r}")

    def parse_postfix(self, base: str) -> str:
        while True:
            tok = self.peek()
            if tok is None:
                return base
            if tok.val == ".":
                self.next()
                member = self.next()
                base = f"{base}.{member.val}"
                continue
            if tok.val == "[":
                self.next()
                idx = self.parse_expr(0)
                self.expect("]")
                base = f"{base}[{idx}]"
                continue
            return base

    def parse_expr(self, level: int) -> str:
        if level >= len(PRECEDENCE):
            return self.parse_primary()
        left = self.parse_expr(level + 1)
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op" or tok.val not in PRECEDENCE[level]:
                return left
            self.next()
            right = self.parse_expr(level + 1)
            # the whole point: emit strictly two-operand, explicitly grouped
            left = f"({left} {tok.val} {right})"


def reparen_expression(expr: str) -> str:
    """Return `expr` fully parenthesised. Raises ValueError if unparseable."""
    p = Parser(tokenize(expr))
    out = p.parse_expr(0)
    if p.i != len(p.t):
        raise ValueError(f"trailing tokens from {p.i}: {p.t[p.i:]}")
    return out


if __name__ == "__main__":
    for line in sys.argv[1:] or ["a*b*c", "a+b+c+d", "a*b + c*d", "dot(c, lw)*1.4"]:
        try:
            print(f"{line}  ->  {reparen_expression(line)}")
        except ValueError as e:
            print(f"{line}  ->  ERROR {e}")
