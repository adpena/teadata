"""Internal, lazy field planning for observations attached to an entity build."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .dataset_metadata import ArtifactRef, DataPeriod, EditionKey, FieldBinding
from .manifest import ContractError

if TYPE_CHECKING:
    from .manifest import DatasetCatalog
    from .release import DataRelease


@dataclass(frozen=True, slots=True)
class _ObservationRequest:
    edition: EditionKey
    field: str
    period: DataPeriod
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.period, DataPeriod) or self.period.kind == "unknown":
            raise ContractError("Observation requests require an exact measurement period")


@dataclass(frozen=True, slots=True)
class _ObservationBinding:
    request: _ObservationRequest
    manifest_sha256: str
    binding: FieldBinding


@dataclass(frozen=True, slots=True)
class _ObservationProjection:
    manifest_sha256: str
    artifact: ArtifactRef
    columns: tuple[str, ...]

    @property
    def grain_columns(self) -> tuple[str, ...]:
        return tuple(self.artifact.contract["key_columns"])


@dataclass(frozen=True, slots=True)
class _ObservationPlan:
    entity_build_id: str
    bindings: tuple[_ObservationBinding, ...]
    projections: tuple[_ObservationProjection, ...]


class _ObservationModel:
    """Bind exact releases without reading a native leaf or altering entity state."""

    def __init__(self, entities: DatasetCatalog, sources: Iterable[DataRelease]):
        self.entity_build_id = entities.build_id
        self._sources: dict[str, DataRelease] = {}
        self._manifests_by_edition: dict[EditionKey, list[str]] = {}
        for release in sources:
            digest = release.manifest_sha256
            if digest in self._sources:
                raise ContractError("Observation sources must have distinct manifests")
            self._sources[digest] = release
            for entry in release.catalog.editions(include_metadata=False):
                key = EditionKey.from_dict({name: value for name, value in entry.items() if name != "edition_id"})
                self._manifests_by_edition.setdefault(key, []).append(digest)

    def plan(self, requests: Iterable[_ObservationRequest]) -> _ObservationPlan:
        selected: dict[_ObservationRequest, str] = {}
        grouped: dict[str, list[_ObservationRequest]] = {}
        for request in requests:
            if request in selected:
                continue
            choices = self._manifests_by_edition.get(request.edition, ())
            digest = request.manifest_sha256
            if digest is None:
                if len(choices) != 1:
                    raise ContractError("Observation edition is unavailable or ambiguous; select an exact manifest")
                digest = choices[0]
            elif digest not in choices:
                raise ContractError("Observation edition is unavailable in the pinned manifest")
            grouped.setdefault(digest, []).append(request)
            selected[request] = digest

        bindings: dict[_ObservationRequest, _ObservationBinding] = {}
        projections: list[_ObservationProjection] = []
        for digest, group in grouped.items():
            leaves, primary = self._sources[digest].catalog._resolve_field_projections(
                (item.edition, item.field, item.period) for item in group
            )
            bindings.update((item, _ObservationBinding(item, digest, primary[item.edition, item.field])) for item in group)
            projections.extend(_ObservationProjection(digest, leaf, columns) for leaf, columns in leaves)
        return _ObservationPlan(self.entity_build_id, tuple(bindings[item] for item in selected), tuple(projections))
