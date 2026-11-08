"""Exact, case-insensitive selection with bounded diagnostic suggestions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from difflib import SequenceMatcher, get_close_matches
from itertools import islice, zip_longest
import re
import unicodedata
from typing import Generic, TypeVar

T = TypeVar("T")

_SUGGESTION_LIMIT = 5
_HIGH_SCHOOL_SUFFIX = re.compile(r"\b(?:h s|hs)$")
_NUMBER_TOKEN = re.compile(r"[0-9]+")


def _suggest(value: str, options: Iterable[str]) -> tuple[str, ...]:
    return tuple(get_close_matches(value.casefold(), sorted(set(options)), n=_SUGGESTION_LIMIT))


def _name_tokens(name: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", name).replace("'", "").replace("’", "")
    words = " ".join("".join(char if char.isalnum() else " " for char in text).split())
    return tuple(_HIGH_SCHOOL_SUFFIX.sub("high school", words).split())


def _entity_suggestions(key: str, labels: Iterable[str]) -> tuple[str, ...]:
    if len(key) > 256:
        return ()
    tokens = _name_tokens(key)
    if not tokens:
        return ()
    unordered = sorted(tokens)
    normalized = " ".join(tokens)
    numerals = sorted(_NUMBER_TOKEN.findall(normalized))
    exact_form, reordered = [], []
    same_numbers: dict[str, list[str]] = {}
    other: dict[str, list[str]] = {}
    for label in sorted(labels):
        candidate = _name_tokens(label)
        if candidate == tokens:
            exact_form.append(label)
        elif sorted(candidate) == unordered:
            reordered.append(label)
        else:
            spelling = " ".join(candidate)
            group = same_numbers if numerals and sorted(_NUMBER_TOKEN.findall(spelling)) == numerals else other
            group.setdefault(spelling, []).append(label)
    ranked = exact_form + reordered
    for group in (same_numbers, other):
        remaining = _SUGGESTION_LIMIT - len(ranked)
        if remaining <= 0:
            break
        ranked.extend(islice(_rank_spellings(normalized, " ".join(unordered), group), remaining))
    return tuple(ranked[:_SUGGESTION_LIMIT])


def _rank_spellings(query: str, sorted_query: str, group: Mapping[str, list[str]]) -> Iterable[str]:
    by_tokens: dict[str, list[str]] = {}
    token_order = {}
    for spelling in group:
        ordered = " ".join(sorted(spelling.split()))
        token_order[spelling] = ordered
        by_tokens.setdefault(ordered, []).append(spelling)
    shortlisted = set(_suggest(query, group))
    for ordered in _suggest(sorted_query, by_tokens):
        shortlisted.update(by_tokens[ordered])

    def rank(spelling: str) -> tuple[float, str]:
        direct = SequenceMatcher(None, spelling, query).ratio()
        reordered = SequenceMatcher(None, token_order[spelling], sorted_query).ratio()
        return -max(direct, reordered), spelling

    for spelling in sorted(shortlisted, key=rank):
        yield from group[spelling]


def _preview(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit - 1] + "…"


class EntitySelectionError(ValueError):
    """An exact entity selector failed. Candidates are (stored name, ID) pairs."""

    def __init__(self, scope: str, selector: str, reason: str, candidates: tuple[tuple[str, str], ...], *,
                 contexts: Mapping[tuple[str, str], Mapping[str, object]] | None = None):
        self.scope = scope
        self.selector = selector
        self.reason = reason
        self.candidates = candidates
        self._contexts = {key: dict(fields) for key, fields in (contexts or {}).items()}
        if reason == "ambiguous_name":
            guidance = "Select a matching ID"
        else:
            guidance = "Did you mean" if candidates else "Check the name or supply the entity ID"
        labels = []
        for name, identifier in candidates[:_SUGGESTION_LIMIT]:
            fields = self._contexts.get((name, identifier), {})
            place = " · ".join(_preview(value, 32) for field in ("site_city", "administrative_county")
                               if isinstance(value := fields.get(field), str))
            detail = f"{place} · " if place else ""
            labels.append(f"{_preview(name, 64)} ({detail}{_preview(identifier, 36)})")
        details = ", ".join(labels)
        extra = len(candidates) - _SUGGESTION_LIMIT
        if extra > 0:
            details += f", and {extra} more"
        issue = "Multiple exact matches" if reason == "ambiguous_name" else "No exact match"
        display = repr(_preview(selector, 120))
        super().__init__(f"{issue} for {scope} name {display}. {guidance}"
                         + (f": {details}." if details else "."))

    def to_dict(self) -> dict[str, object]:
        """Return stable error data for interactive clients."""
        return {
            "scope": self.scope,
            "selector": self.selector,
            "reason": self.reason,
            "candidates": [{**self._contexts.get((name, identifier), {}), "name": name, "id": identifier}
                           for name, identifier in self.candidates],
        }


class _EntityNames(Generic[T]):
    def __init__(self, entries: Iterable[tuple[str, T]], identifier: Callable[[T], str]):
        self._entries: dict[str, list[tuple[str, T]]] = {}
        self._identifier = identifier
        for name, value in entries:
            if name.strip():
                self._entries.setdefault(name.strip().casefold(), []).append((name, value))

    def resolve(self, name: str, *, scope: str) -> T:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Select a nonblank {scope} name or use its ID.")
        key = name.strip().casefold()
        matches = self._entries.get(key)
        if matches and len(matches) == 1:
            return matches[0][1]
        if matches:
            pairs = sorted((label, self._identifier(value)) for label, value in matches)
            raise EntitySelectionError(scope, name, "ambiguous_name", tuple(pairs))
        groups = [
            sorted((label, self._identifier(value)) for label, value in self._entries[alternative])
            for alternative in _entity_suggestions(key, self._entries)
        ]
        suggestions = (pair for row in zip_longest(*groups) for pair in row if pair is not None)
        raise EntitySelectionError(scope, name, "unknown_name", tuple(islice(suggestions, _SUGGESTION_LIMIT)))


def _live_name(name: str, scope: str, entities: Iterable[T], *, label: Callable[[T], str],
               identifier: Callable[[T], str]) -> T:
    """Read live names, computing identifiers only when an error needs them."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Select a nonblank {scope} name or use its ID.")
    target = name.strip().casefold()
    matches = [item for item in entities if label(item).strip().casefold() == target]
    if len(matches) == 1:
        return matches[0]
    return _EntityNames(((label(item), item) for item in matches or entities), identifier).resolve(name, scope=scope)


def _field_suggestions(field: str, choices: Iterable[str]) -> tuple[str, ...]:
    by_fold: dict[str, list[str]] = {}
    for choice in sorted(set(choices)):
        by_fold.setdefault(choice.casefold(), []).append(choice)
    return tuple(label for key in _suggest(field, by_fold) for label in by_fold[key])[:5]


def _folded_field_names(fields: tuple[str, ...], names: Iterable[str]) -> Mapping[str, tuple[str, ...]]:
    """Return exact normalized candidates, preserving casefold collisions."""
    by_fold: dict[str, list[str]] = {field.casefold(): [] for field in fields}
    for name in names:
        if (matches := by_fold.get(name.casefold())) is not None:
            matches.append(name)
    return {field: tuple(sorted(by_fold[field.casefold()])) for field in fields}
