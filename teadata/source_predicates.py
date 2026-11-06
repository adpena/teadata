"""Compile a bounded query string into the same Arrow predicates as inline filters."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from functools import reduce
from io import StringIO
from operator import and_, eq, ge, gt, le, lt, ne, or_
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyarrow.dataset import Expression

_PARAMETER = "__teadata_parameter_"
_QUERY_BYTES = 8192
_SCALARS = (str, int, float, bool)
_COMPARE = {ast.Eq: eq, ast.NotEq: ne, ast.Lt: lt, ast.LtE: le, ast.Gt: gt, ast.GtE: ge}


def _parameter_size(value: object) -> int:
    if type(value) is str and len(value) <= _QUERY_BYTES:
        return len(value.encode("utf-8")) + 1
    if value is None or type(value) in (float, bool) or (type(value) is int and value.bit_length() <= 64):
        return 8
    raise ValueError("Query parameters require bounded scalar values")


def compile_predicate(expression: str, params: Mapping[str, object] | None) -> Expression:
    """Read comparisons, boolean operators, membership and explicit @parameters."""
    from pyarrow.dataset import Expression, field

    if (not isinstance(expression, str) or not 0 < len(expression) <= _QUERY_BYTES
            or len(expression.encode("utf-8")) > _QUERY_BYTES):
        raise ValueError("Query expression must be nonempty and at most 8192 UTF-8 bytes")
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("Query parameters must be a mapping")
    parameters = {} if params is None else params

    try:
        tokens = iter(tokenize.generate_tokens(StringIO(expression).readline))
        source = []
        for token in tokens:
            if token.type == tokenize.NAME and (not token.string.isascii() or token.string.startswith(_PARAMETER)):
                raise ValueError("Query field names must use canonical identifiers")
            if token.type == tokenize.OP and token.string == "@":
                name = next(tokens)
                if name.type != tokenize.NAME or not name.string.isascii() or name.start != token.end:
                    raise ValueError("Expected an explicit @parameter name")
                source.append((tokenize.NAME, _PARAMETER + name.string))
            else:
                source.append((token.type, token.string))
        tree = ast.parse(tokenize.untokenize(source).strip(), mode="eval")
    except (SyntaxError, StopIteration, tokenize.TokenError, RecursionError) as error:
        raise ValueError("Invalid source query syntax") from error
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("Query expression exceeds 512 syntax nodes")

    parameter_bytes = 0

    def operand(node: ast.expr):
        nonlocal parameter_bytes
        if isinstance(node, ast.Name):
            if not node.id.startswith(_PARAMETER):
                return field(node.id)
            name = node.id.removeprefix(_PARAMETER)
            if name not in parameters:
                raise ValueError(f"Missing query parameter {name!r}")
            value = parameters[name]
            if type(value) is list or type(value) is tuple:
                if len(value) > 512:
                    raise ValueError("Query membership parameters require at most 512 values")
                parameter_bytes += sum(_parameter_size(item) for item in value)
            else:
                parameter_bytes += _parameter_size(value)
            if parameter_bytes > _QUERY_BYTES:
                raise ValueError("Query parameter uses exceed 8192 bytes")
            return value
        if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.UnaryOp)):
            try:
                return ast.literal_eval(node)
            except ValueError:
                pass
        raise ValueError("Query operands must be fields, literal values or explicit parameters")

    def compare(left, operator, right) -> Expression:
        if isinstance(operator, (ast.Is, ast.IsNot)):
            if not isinstance(left, Expression) or right is not None:
                raise ValueError("Use 'field is None' or 'field is not None' to test nulls")
            return left.is_null() if isinstance(operator, ast.Is) else left.is_valid()
        if isinstance(operator, (ast.In, ast.NotIn)):
            if (not isinstance(left, Expression) or type(right) not in (list, tuple)
                    or len(right) > 512 or any(type(value) not in _SCALARS for value in right)):
                raise ValueError("Membership tests require a field and up to 512 scalar values")
            member = left.isin(right)
            return ~member if isinstance(operator, ast.NotIn) else member
        operation = _COMPARE.get(type(operator))
        if operation is None or not any(isinstance(value, Expression) for value in (left, right)) or any(
                type(value) not in _SCALARS and not isinstance(value, Expression) for value in (left, right)):
            raise ValueError("Comparisons require a field and another field or a scalar value")
        return operation(left, right)

    def predicate(node: ast.expr) -> Expression:
        if isinstance(node, ast.Compare):
            values = [operand(value) for value in (node.left, *node.comparators)]
            return reduce(and_, (compare(left, operator, right) for left, operator, right in
                                 zip(values[:-1], node.ops, values[1:], strict=True)))
        if isinstance(node, ast.BoolOp):
            return reduce(and_ if isinstance(node.op, ast.And) else or_, map(predicate, node.values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~predicate(node.operand)
        if isinstance(node, ast.Name) and not node.id.startswith(_PARAMETER):
            return field(node.id)
        raise ValueError("Unsupported source query. Use comparisons, membership and boolean operators")

    return predicate(tree.body)
