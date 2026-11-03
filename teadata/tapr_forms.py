"""Observe retained TAPR forms without submitting controls or executing scripts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import urljoin, urlsplit

Attributes = tuple[tuple[str, str | None], ...]
DEFAULT_FORM_NODES = 20_000
_WINDOW_OPEN = re.compile(r"\bwindow\.open\(\s*(['\"])([^'\"\\]+)\1\s*[,)]")


def _attributes(items: list[tuple[str, str | None]]) -> Attributes:
    names = [name for name, _ in items]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous duplicate form attribute")
    return tuple(items)


def _text(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


def _resolve(base_url: str | None, value: str | None) -> str | None:
    if value is None or base_url is None:
        return None
    resolved = urljoin(base_url, value)
    return resolved if urlsplit(resolved).scheme in {"http", "https"} else None


@dataclass(frozen=True)
class FormSnapshot:
    """Detached source-order facts; these are not successful submission controls."""

    attributes: Attributes
    position: tuple[int, int]
    controls: tuple[dict[str, Any], ...]
    labels: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    contexts: tuple[dict[str, Any], ...]
    closed: bool
    base_url: str | None

    def to_dict(self) -> dict[str, Any]:
        attributes = dict(self.attributes)
        return {"attributes": list(self.attributes), "position": list(self.position),
                "method": attributes.get("method"), "action": attributes.get("action"),
                "resolved_action": _resolve(self.base_url, attributes.get("action")),
                "controls": deepcopy(list(self.controls)), "labels": deepcopy(list(self.labels)),
                "links": deepcopy(list(self.links)), "contexts": deepcopy(list(self.contexts)),
                "closed": self.closed}


@dataclass
class _Form:
    attributes: Attributes
    position: tuple[int, int]
    controls: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)


class _Forms(HTMLParser):
    def __init__(self, base_url: str | None, max_nodes: int):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.forms: list[FormSnapshot] = []
        self.form: _Form | None = None
        self.control: dict[str, Any] | None = None
        self.control_text: list[str] = []
        self.option: dict[str, Any] | None = None
        self.option_text: list[str] = []
        self.label: dict[str, Any] | None = None
        self.label_text: list[str] = []
        self.ignored: list[str] = []
        self.context: list[int] = []
        self.nodes, self.max_nodes = 0, max_nodes

    def take_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError("Form projection exceeds max-nodes")

    def finish_label(self) -> None:
        if self.label is not None and self.form is not None:
            self.label["text"] = _text(self.label_text)
            self.form.labels.append(self.label)
        self.label, self.label_text = None, []

    def finish_option(self) -> None:
        if self.option is not None and self.control is not None:
            self.option["text"] = _text(self.option_text)
            self.control["options"].append(self.option)
        self.option, self.option_text = None, []

    def finish_control(self) -> None:
        self.finish_option()
        if self.control is not None:
            self.control["text"] = ("".join(self.control_text) if self.control["tag"] == "textarea"
                                    else _text(self.control_text))
            if (self.control["tag"] == "select" and self.context and self.form is not None
                    and self.form.contexts[self.context[-1]]["tag"] == "optgroup"):
                self.context.pop()
        self.control, self.control_text = None, []

    def finish_form(self, *, closed: bool) -> None:
        if self.form is None:
            raise ValueError("Unmatched form end tag")
        self.finish_label()
        self.finish_control()
        form = self.form
        self.forms.append(FormSnapshot(form.attributes, form.position, tuple(form.controls),
                         tuple(form.labels), tuple(form.links), tuple(form.contexts), closed, self.base_url))
        self.form = None
        self.context = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.ignored.append(tag)
            return
        if self.ignored:
            return
        if tag == "form":
            if self.form is not None:
                raise ValueError("Nested forms require source review")
            self.take_node()
            self.form = _Form(_attributes(attrs), self.getpos())
        if self.form is None:
            return
        if tag in {"fieldset", "legend", "optgroup"}:
            if tag == "optgroup":
                if self.control is None or self.control["tag"] != "select":
                    raise ValueError("Option group requires a select control")
                self.finish_option()
                if self.context and self.form.contexts[self.context[-1]]["tag"] == "optgroup":
                    self.context.pop()  # The preceding optgroup end tag is optional.
            elif self.control is not None and self.control["tag"] == "select":
                raise ValueError("Unsupported fieldset/legend context inside select")
            self.take_node()
            self.form.contexts.append({"tag": tag, "attributes": list(_attributes(attrs)),
                "position": list(self.getpos()), "parent": self.context[-1] if self.context else None})
            self.context.append(len(self.form.contexts) - 1)
        if tag in {"input", "select", "textarea", "button"}:
            if self.control is not None:
                raise ValueError("Nested form controls require source review")
            self.take_node()
            attributes = _attributes(attrs)
            row: dict[str, Any] = {"tag": tag, "attributes": list(attributes),
                                   "position": list(self.getpos()),
                                   "context_id": self.context[-1] if self.context else None}
            self.form.controls.append(row)
            if tag != "input":
                self.control = row
                if tag == "select":
                    row["options"] = []
            for name, value in attributes:
                if name == "onclick" and value:
                    for match in _WINDOW_OPEN.finditer(value):
                        self.take_node()
                        target = match[2]
                        self.form.links.append({"kind": "window.open_literal", "target": target,
                            "resolved_url": _resolve(self.base_url, target), "position": list(self.getpos())})
        elif tag == "option" and self.control is not None and self.control["tag"] == "select":
            self.finish_option()
            self.take_node()
            self.option = {"attributes": list(_attributes(attrs)), "position": list(self.getpos()),
                           "context_id": self.context[-1] if self.context else None}
        elif tag == "label":
            # Publisher pages sometimes open another <label> instead of closing
            # the preceding one; row/cell boundaries also end observed labels.
            self.finish_label()
            self.take_node()
            self.label = {"attributes": list(_attributes(attrs)), "position": list(self.getpos())}
        elif tag == "a":
            attributes = _attributes(attrs)
            target = dict(attributes).get("href")
            if target is not None:
                self.take_node()
                self.form.links.append({"kind": "anchor", "target": target,
                    "resolved_url": _resolve(self.base_url, target), "position": list(self.getpos())})

    def handle_endtag(self, tag: str) -> None:
        if self.ignored:
            if tag == self.ignored[-1]:
                self.ignored.pop()
            return
        if self.form is None:
            return
        if tag == "form":
            self.finish_form(closed=True)
        elif tag in {"fieldset", "legend", "optgroup"}:
            if not self.context or self.form.contexts[self.context[-1]]["tag"] != tag:
                raise ValueError("Ambiguous form-control context nesting")
            self.context.pop()
        elif tag in {"label", "td", "tr"}:
            self.finish_label()
        elif tag == "option":
            self.finish_option()
        elif self.control is not None and tag == self.control["tag"]:
            self.finish_control()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"form", "select", "textarea", "button", "option", "fieldset", "optgroup", "legend"}:
            raise ValueError("Self-closing non-void form structure requires source review")
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.ignored:
            return
        if self.label is not None:
            self.label_text.append(data)
        if self.option is not None:
            self.option_text.append(data)
        elif self.control is not None:
            self.control_text.append(data)


def extract_tapr_forms(data: bytes, *, encoding: str, base_url: str | None = None,
                       max_nodes: int = DEFAULT_FORM_NODES) -> tuple[FormSnapshot, ...]:
    """Keep observed values, duplicates and order; never infer defaults or methods.

    Missing end tags are reported as ``closed=False``. Controls retain literal
    attributes; textarea text is exact, display-label whitespace is normalized.
    Disabled/checked/select semantics and JavaScript are not executed.
    Only anchor URLs and literal window.open first arguments are resolved.
    """
    if not isinstance(data, bytes):
        raise TypeError("Retained HTML bytes required")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("Form max-nodes must be a positive integer")
    if base_url is not None and urlsplit(base_url).scheme not in {"http", "https"}:
        raise ValueError("Observed base URL must be HTTP or HTTPS")
    parser = _Forms(base_url, max_nodes)
    parser.feed(data.decode(encoding, errors="strict"))
    parser.close()
    if parser.form is not None:
        parser.finish_form(closed=False)
    return tuple(parser.forms)
