"""Citations from installed software metadata and an explicitly selected release."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import getaddresses
from importlib.metadata import metadata
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .release import DataRelease

_TITLE = "TEA Data"
_TEX_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%",
    "&": r"\&", "_": r"\_", "#": r"\#", "$": r"\$",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
})


def _bibtex_entry(key: str, **fields: str) -> str:
    def field_value(name: str, value: str) -> str:
        escaped = value.translate(_TEX_ESCAPES)
        return "{" + escaped + "}" if name == "title" else escaped

    lines = [f"  {name} = {{{field_value(name, value)}}}," for name, value in fields.items()]
    return "\n".join([f"@misc{{{key},", *lines, "}"])


@dataclass(frozen=True)
class Citation:
    """Software citation with an optional separate data bundle citation."""

    authors: tuple[str, ...]
    software_version: str
    repository_url: str
    build_id: str | None = None
    manifest_url: str | None = None

    def __post_init__(self) -> None:
        if self.build_id is not None and re.fullmatch(r"[0-9a-f]{64}", self.build_id) is None:
            raise ValueError("A data citation requires a SHA-256 build ID")

    @property
    def text(self) -> str:
        authors = ", ".join(self.authors)
        software = (
            f"{authors}. {_TITLE} (version {self.software_version}) [Computer software].\n"
            f"{self.repository_url}"
        )
        if self.build_id is None:
            return software
        data = f"{authors}. {_TITLE} [Data release].\nBuild ID: {self.build_id}"
        if self.manifest_url:
            data += f"\n{self.manifest_url}"
        return f"{software}\n\n{data}"

    @property
    def bibtex(self) -> str:
        authors = " and ".join(self.authors)
        version_key = re.sub(r"[^A-Za-z0-9_-]", "_", self.software_version)
        software = _bibtex_entry(
            f"teadata_{version_key}", author=authors, title=_TITLE,
            howpublished="Computer software", note=f"Version {self.software_version}",
            url=self.repository_url,
        )
        if self.build_id is None:
            return software
        data = _bibtex_entry(
            f"teadata_data_{self.build_id}", author=authors, title=_TITLE,
            howpublished="Data release", note=f"Build ID: {self.build_id}",
            **({"url": self.manifest_url} if self.manifest_url else {}),
        )
        return f"{software}\n\n{data}"


def citation(release: DataRelease | None = None) -> Citation:
    """Cite this installation and, when supplied, a resolved data release.

    Formatting uses the release's existing catalog identity. It does not
    open artifacts or resolve a default release. Only the repository's
    versioned public manifest URLs appear in the rendered citation.
    """
    project = metadata("teadata")
    authors = tuple(name for name, _ in getaddresses(project.get_all("Author-email") or []) if name)
    repository_url = next(
        url.strip() for label, _, url in (value.partition(",") for value in project.get_all("Project-URL") or [])
        if label == "Homepage"
    )
    manifest_url = None
    # Named manifests are adjacent, portable ASCII basenames of at most 255 bytes.
    if release is not None and re.fullmatch(
        re.escape(repository_url) + r"/releases/download/v[0-9]+\.[0-9]+\.[0-9]+/"
        r"(?:release\.json|[A-Za-z0-9][A-Za-z0-9_-]{0,241}\.release\.json)",
        release.source,
    ):
        manifest_url = release.source
    return Citation(
        authors, project["Version"], repository_url,
        release.build_id if release is not None else None, manifest_url,
    )
