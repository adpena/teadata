"""Ordered publisher HTML dictionary observations, without scientific inference."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any


@dataclass(frozen=True)
class TaprDictionaryOccurrence:
    ordinal: int
    code: str
    publisher_type: str | None
    publisher_description: str | None
    source_sha256: str
    source_line: int
    source_column: int
    source_html: str
    publisher_length: str | None = None

    def column_record(self) -> dict[str, Any]:
        """Project into the existing column-schema occurrence contract.

        Coordinates locate the outer tooltip start tag: one-based line and
        zero-based character column in the explicitly decoded source. The raw
        artifact checksum, rather than this decoded slice, owns byte identity.
        """
        record = {name: getattr(self, name) for name in (
            'ordinal', 'code', 'publisher_type', 'publisher_description',
            'source_sha256', 'source_line', 'source_column')}
        if self.publisher_length is not None:
            record['publisher_length'] = self.publisher_length
        return record


def _rendered(parts: list[str]) -> str:
    # HTML presentation whitespace only; punctuation/case/identifiers stay exact.
    return ' '.join(''.join(parts).split())


class _DictionaryParser(HTMLParser):
    def __init__(self, text: str, source_sha256: str):
        super().__init__(convert_charrefs=True)
        self.text, self.source_sha256 = text, source_sha256
        self.lines = [0, *(i + 1 for i, char in enumerate(text) if char == '\n')]
        self.occurrences: list[TaprDictionaryOccurrence] = []
        self.span_depth = 0
        self.div_depth = 0
        self.inner_depth: int | None = None
        self.seen_inner = False
        self.capture: tuple[str, int, list[str]] | None = None
        self.pending: str | None = None
        self.values: dict[str, str | None] = {}
        self.display: list[str] = []
        self.loose: list[str] = []
        self.start = (0, 0)

    def source_offset(self, position: tuple[int, int]) -> int:
        return self.lines[position[0] - 1] + position[1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        class_values = [value or '' for name, value in attrs if name == 'class']
        relevant = {'tooltip', 'tooltiptext', 'dd_hdr', 'dd_desc'}
        if len(class_values) > 1 and any(relevant & set(value.split()) for value in class_values):
            raise ValueError('Ambiguous dictionary class attributes')
        classes = (dict(attrs).get('class') or '').split()
        if tag == 'span' and 'tooltip' in classes:
            if self.span_depth:
                raise ValueError('Nested dictionary tooltip')
            self.span_depth = 1
            self.div_depth, self.inner_depth, self.seen_inner = 0, None, False
            self.capture, self.pending = None, None
            self.values, self.display, self.loose = {}, [], []
            self.start = self.getpos()
            return
        if not self.span_depth:
            return
        if tag == 'span':
            self.span_depth += 1
            if 'tooltiptext' in classes:
                if self.seen_inner:
                    raise ValueError('Repeated dictionary tooltip body')
                self.inner_depth, self.seen_inner = self.span_depth, True
        if tag == 'div':
            self.handle_data(' ')
            self.div_depth += 1
            kinds = [kind for kind in ('dd_hdr', 'dd_desc') if kind in classes]
            if kinds:
                if len(kinds) != 1 or self.inner_depth is None or self.capture is not None:
                    raise ValueError('Ambiguous dictionary field structure')
                if kinds[0] == 'dd_hdr' and self.pending is not None:
                    raise ValueError('Dictionary heading lacks its value')
                if kinds[0] == 'dd_desc' and self.pending is None:
                    raise ValueError('Dictionary value lacks its heading')
                self.capture = kinds[0], self.div_depth, []
        if tag in {'br', 'p'}:
            self.handle_data(' ')

    def handle_endtag(self, tag: str) -> None:
        if not self.span_depth:
            return
        if tag in {'br', 'p'}:  # Publisher also emits the literal </br> spelling.
            self.handle_data(' ')
            return
        if tag == 'div':
            if self.div_depth < 1:
                raise ValueError('Unbalanced dictionary field')
            if self.capture is not None and self.capture[1] == self.div_depth:
                kind, _, parts = self.capture
                value = _rendered(parts)
                self.capture = None
                if kind == 'dd_hdr':
                    if value not in {'Name:', 'Type:', 'Length:', 'Description:'}:
                        raise ValueError('Unsupported dictionary heading')
                    self.pending = value[:-1]
                else:
                    if self.pending is None or self.pending in self.values:
                        raise ValueError('Repeated dictionary field')
                    self.values[self.pending] = value or None
                    self.pending = None
            self.div_depth -= 1
            self.handle_data(' ')
        if tag == 'span':
            if self.inner_depth == self.span_depth:
                self.inner_depth = None
            self.span_depth -= 1
            if not self.span_depth:
                self.finish_occurrence()

    def handle_data(self, data: str) -> None:
        if not self.span_depth:
            return
        if self.capture is not None:
            self.capture[2].append(data)
        elif self.inner_depth is not None:
            self.loose.append(data)
        elif not self.seen_inner:
            self.display.append(data)
        elif data.strip():
            raise ValueError('Unexpected text after dictionary tooltip body')

    def finish_occurrence(self) -> None:
        if not self.seen_inner or self.capture is not None or self.pending is not None or self.div_depth:
            raise ValueError('Incomplete dictionary field structure')
        code = _rendered(self.display)
        if not code:
            raise ValueError('Dictionary occurrence lacks its displayed code')
        if self.values:
            if self.values.get('Name') != code:
                raise ValueError('Dictionary displayed code and Name differ')
            if _rendered(self.loose):
                raise ValueError('Unexpected text outside dictionary fields')
        elif _rendered(self.loose) != code:
            raise ValueError('Code-only dictionary body differs from displayed code')
        end = self.text.find('>', self.source_offset(self.getpos()))
        if end < 0:
            raise ValueError('Truncated dictionary tooltip')
        self.occurrences.append(TaprDictionaryOccurrence(
            ordinal=len(self.occurrences) + 1, code=code,
            publisher_type=self.values.get('Type'),
            publisher_description=self.values.get('Description'),
            source_sha256=self.source_sha256,
            source_line=self.start[0], source_column=self.start[1],
            source_html=self.text[self.source_offset(self.start):end + 1],
            publisher_length=self.values.get('Length'),
        ))


def extract_tapr_dictionary(raw: bytes, *, encoding: str) -> tuple[TaprDictionaryOccurrence, ...]:
    """Extract exact ordered complete/partial observations from retained bytes.

    Optional publisher length stays literal metadata, never a numeric value
    bound or scientific unit. Encoding is explicit; decoding errors and
    ambiguous/truncated tooltip structure fail. Rendered text normalizes
    presentation whitespace only.
    Original decoded HTML slices and raw SHA remain available. Duplicate codes
    and conflicting definitions remain separate observations; column_schema
    owns whether those observations can document reviewed source fields.
    """
    if not isinstance(raw, bytes) or not isinstance(encoding, str) or not encoding:
        raise ValueError('Dictionary requires retained bytes and an explicit encoding')
    parser = _DictionaryParser(raw.decode(encoding), sha256(raw).hexdigest())
    parser.feed(parser.text)
    parser.close()
    if parser.span_depth:
        raise ValueError('Truncated dictionary tooltip')
    if not parser.occurrences:
        raise ValueError('No publisher dictionary occurrences found')
    return tuple(parser.occurrences)
