"""Compile a bounded query string into the same Arrow predicates as inline filters."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN
from functools import reduce
from io import StringIO
from operator import and_, eq, ge, gt, le, lt, ne, or_
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyarrow import Schema
    from pyarrow.dataset import Expression

_PARAMETER = "__teadata_parameter_"
_QUERY_BYTES = 8192
_SCALARS = (str, int, float, bool, Decimal)
_COMPARE = {ast.Eq: eq, ast.NotEq: ne, ast.Lt: lt, ast.LtE: le, ast.Gt: gt, ast.GtE: ge}


@dataclass(frozen=True, slots=True)
class _Field:
    name: str


@dataclass(frozen=True, slots=True)
class _FloatLiteral:
    spelling: str
    value: float


def _decimal(value: Decimal) -> Decimal:
    if value.is_finite():
        _, digits, exponent = value.as_tuple()
        if isinstance(exponent, int) and -76 <= exponent and len(digits) + max(exponent, 0) <= 76:
            return value
    raise ValueError("Query decimals must be finite and fit Arrow decimal precision and scale")


def _parameter_size(value: object) -> int:
    if type(value) is str and len(value) <= _QUERY_BYTES:
        return len(value.encode("utf-8")) + 1
    if value is None or type(value) in (float, bool) or (type(value) is int and value.bit_length() <= 64):
        return 8
    if type(value) is Decimal:
        return len(str(_decimal(value))) + 1
    raise ValueError("Query parameters require bounded scalar values")


def prepare_predicate(expression: str, params: Mapping[str, object] | None) -> Callable[[Schema], Expression]:
    """Validate a query now and bind numeric operands to the native schema at read time."""
    from pyarrow import array, types
    from pyarrow.dataset import field

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
        parsed = tokenize.untokenize(source).strip()
        tree = ast.parse(parsed, mode="eval")
    except (SyntaxError, StopIteration, tokenize.TokenError, RecursionError) as error:
        raise ValueError("Invalid source query syntax") from error
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("Query expression exceeds 512 syntax nodes")

    parameter_bytes = 0

    def operand(node: ast.expr):
        nonlocal parameter_bytes
        if isinstance(node, ast.Name):
            if not node.id.startswith(_PARAMETER):
                return _Field(node.id)
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
            return tuple(value) if type(value) is list or type(value) is tuple else value
        if isinstance(node, (ast.List, ast.Tuple)):
            return tuple(operand(item) for item in node.elts)
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            try:
                value = ast.literal_eval(node)
            except ValueError:
                pass
            else:
                if type(value) is float:
                    number = node.operand if isinstance(node, ast.UnaryOp) else node
                    spelling = ast.get_source_segment(parsed, number)
                    if spelling is None:
                        raise ValueError("Cannot read the exact query literal")
                    sign = "-" if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) else ""
                    return _FloatLiteral(sign + spelling.replace("_", ""), value)
                return value
        raise ValueError("Query operands must be fields, literal values or explicit parameters")

    def scalar(value, schema, column):
        if isinstance(value, _Field):
            return field(value.name)
        kind = schema.field(column.name).type
        if isinstance(value, _FloatLiteral):
            if types.is_decimal(kind):
                return _decimal(Decimal(value.spelling))
            return value.value
        if types.is_decimal(kind) and type(value) is float:
            raise ValueError("Use a Decimal parameter to compare with an exact decimal column")
        if type(value) is Decimal and types.is_floating(kind):
            return float(value)
        return value

    def compare(left, operator, right) -> Callable[[Schema], Expression]:
        if isinstance(operator, (ast.Is, ast.IsNot)):
            if not isinstance(left, _Field) or right is not None:
                raise ValueError("Use 'field is None' or 'field is not None' to test nulls")
            column = field(left.name)
            return lambda _: column.is_null() if isinstance(operator, ast.Is) else column.is_valid()
        if isinstance(operator, (ast.In, ast.NotIn)):
            if (not isinstance(left, _Field) or type(right) is not tuple or len(right) > 512
                    or any(type(value) not in (*_SCALARS, _FloatLiteral) for value in right)):
                raise ValueError("Membership tests require a field and up to 512 scalar values")

            def membership(schema: Schema) -> Expression:
                kind = schema.field(left.name).type
                values = [scalar(value, schema, left) for value in right]
                if types.is_decimal(kind):
                    if any(type(value) not in (int, Decimal) for value in values):
                        raise ValueError("Decimal membership requires exact numeric values")
                    context = Context(prec=152, Emin=-152, Emax=152, rounding=ROUND_HALF_EVEN,
                                      traps=[InvalidOperation])
                    unit = Decimal((0, (1,), -kind.scale))
                    exact = []
                    for value in values:
                        candidate = context.quantize(Decimal(value), unit)
                        if candidate == value and len(candidate.as_tuple().digits) <= kind.precision:
                            exact.append(candidate)
                    values = array(exact, type=kind)
                member = field(left.name).isin(values)
                return ~member if isinstance(operator, ast.NotIn) else member

            return membership
        operation = _COMPARE.get(type(operator))
        if operation is None or not any(isinstance(value, _Field) for value in (left, right)) or any(
                type(value) not in (*_SCALARS, _FloatLiteral, _Field) for value in (left, right)):
            raise ValueError("Comparisons require a field and another field or a scalar value")

        def comparison(schema: Schema) -> Expression:
            if isinstance(left, _Field) and isinstance(right, _Field):
                left_type, right_type = schema.field(left.name).type, schema.field(right.name).type
                if (types.is_decimal(left_type) and types.is_floating(right_type)
                        or types.is_floating(left_type) and types.is_decimal(right_type)):
                    raise ValueError("Comparing decimal and floating columns requires an explicit cast")
            return operation(scalar(left, schema, right), scalar(right, schema, left))

        return comparison

    def predicate(node: ast.expr) -> Callable[[Schema], Expression]:
        if isinstance(node, ast.Compare):
            values = [operand(value) for value in (node.left, *node.comparators)]
            terms = [compare(left, operator, right) for left, operator, right in
                     zip(values[:-1], node.ops, values[1:], strict=True)]
            return lambda schema: reduce(and_, (term(schema) for term in terms))
        if isinstance(node, ast.BoolOp):
            children = list(map(predicate, node.values))
            combine = and_ if isinstance(node.op, ast.And) else or_
            return lambda schema: reduce(combine, (child(schema) for child in children))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child = predicate(node.operand)
            return lambda schema: ~child(schema)
        if isinstance(node, ast.Name) and not node.id.startswith(_PARAMETER):
            return lambda _: field(node.id)
        raise ValueError("Unsupported source query. Use comparisons, membership and boolean operators")

    return predicate(tree.body)
