# Query System

The `Query` object wraps lists of `District` or `Campus` and supports fluent chaining via the `>>` operator.

## Basics

```python
# Start a query from a district TEA number, expand to campuses, then score
district_q = engine >> ("district", "101902")

top5 = (
    district_q
    >> ("campuses_in",)
    >> ("filter", lambda c: (c.enrollment or 0) > 1000)
    >> ("sort", lambda c: c.enrollment or 0, True)  # True -> descending
    >> ("take", 5)
    >> ("map", lambda c: (c.name, c.enrollment))
)

print(top5)  # list[(name, enrollment)]
```

## Common Operators

- `>> ("campuses_in",)` — expand the current district query into its district-operated campuses (excludes charters and private schools)
- `>> ("privates_within", district)` — private schools inside or on the reported district boundary (`covers`); geographic containment is separate from institutional governance, and border points can match multiple districts
- `>> ("filter", predicate)` or `>> (lambda x: ...)`
- `>> ("sort", key_fn, descending: bool=False)`
- `>> ("take", n)`
- `>> ("nearest_charter", coords, mile_limit, k)` — from repo-level query context
- `first()` — take first element
- Attribute fall-through: `Query.attr` proxies to `first().attr` for convenience.

!!! tip "Chaining"
    You can mix spatial and attribute filters easily:
    
    ```python
    engine >> ("within", d, "campuses") >> (lambda c: c.rating in {"A","B"})
    ```
