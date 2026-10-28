"""One immutable dataset catalog shared by a snapshot and all its sidecars."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import os
import sqlite3
from typing import Any, Mapping

from .dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from .provenance import ProvenanceStore
from .values import Measurement, ValuePolicy, aggregate, measure


CATALOG_SCHEMA = 1
CATALOG_TABLE = "teadata_catalog"


class ContractError(RuntimeError):
    """An artifact or source does not satisfy the dataset contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def schema_fingerprint(columns: Any) -> str:
    """Identify an ordered source schema independently of row values."""
    return sha256(
        canonical_json([str(column) for column in columns]).encode()
    ).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def processor_identity() -> str:
    """Hash processing source content without machine paths or modification times."""
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        if any(
            part in {"data", ".cache", "__pycache__"}
            for part in path.relative_to(root).parts
        ):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    lockfile = root.parent / "uv.lock"
    if lockfile.exists():
        digest.update(b"uv.lock\0" + lockfile.read_bytes())
    return digest.hexdigest()


class DatasetCatalog:
    """Validated, read-only metadata describing sources actually used in a build."""

    def __init__(self, document: Mapping[str, Any]):
        document = json.loads(canonical_json(document))
        if document.get("schema_version") != CATALOG_SCHEMA:
            raise ContractError(
                "Unsupported dataset catalog schema; rebuild the artifact"
            )
        identity = document.get("build_id")
        content = {k: v for k, v in document.items() if k != "build_id"}
        if identity != sha256(canonical_json(content).encode()).hexdigest():
            raise ContractError("Dataset catalog identity mismatch")
        sources = document.get("datasets")
        if not isinstance(sources, dict) or not sources:
            raise ContractError("A data release requires loaded dataset metadata")
        policies = document.get("value_policies", {})
        records = document.get("provenance", {})
        for identifier, record in records.items():
            if (
                sha256((canonical_json(record) + "\n").encode()).hexdigest()
                != identifier
            ):
                raise ContractError("Provenance receipt identity mismatch")
        for key, value in sources.items():
            if not isinstance(value, dict) or value.get("status") != "loaded":
                raise ContractError(f"Dataset {key} was not successfully loaded")
            metadata = DatasetMetadata.from_dict(value.get("metadata"))
            policy = metadata.value_policy
            if policy and policy not in policies:
                raise ContractError(
                    f"Dataset {key} refers to missing value policy {policy}"
                )
            artifact = value.get("artifact", {})
            if (
                not isinstance(artifact.get("sha256"), str)
                or len(artifact["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in artifact["sha256"])
            ):
                raise ContractError(f"Dataset {key} has no input checksum")
            if not metadata.provenance_ids:
                raise ContractError(f"Dataset {key} has no acquisition observation")
            for identifier in metadata.provenance_ids:
                record = records.get(identifier, {})
                if (
                    record.get("dataset") != key
                    or record.get("artifact", {}).get("sha256") != artifact["sha256"]
                ):
                    raise ContractError(
                        f"Dataset {key} does not match its acquisition receipt"
                    )
        self._build_id = identity
        self._serialized = canonical_json(document)
        self._policies = {
            key: ValuePolicy.from_dict(value) for key, value in policies.items()
        }
        self._sources = sources

    @classmethod
    def create(
        cls,
        *,
        datasets: Mapping[str, Any],
        value_policies: Mapping[str, Any],
        processor: str,
        records: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> DatasetCatalog:
        content = {
            "schema_version": CATALOG_SCHEMA,
            "processor_sha256": processor,
            "datasets": dict(datasets),
            "value_policies": dict(value_policies),
            "provenance": dict(records or {}),
            "build_environment": dict(environment or {}),
        }
        return cls(
            {
                **content,
                "build_id": sha256(canonical_json(content).encode()).hexdigest(),
            }
        )

    @property
    def build_id(self) -> str:
        return self._build_id

    @property
    def datasets(self) -> dict[str, Any]:
        return self.to_dict()["datasets"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._serialized)

    def metadata(self, dataset: str) -> DatasetMetadata:
        try:
            return DatasetMetadata.from_dict(self._sources[dataset]["metadata"])
        except KeyError as exc:
            raise ContractError(
                f"Dataset was not loaded in this release: {dataset}"
            ) from exc

    def acquisitions(self, dataset: str) -> list[dict[str, Any]]:
        """Return immutable acquisition observations, including unknown download dates."""
        records = self.to_dict()["provenance"]
        return [
            {"record_id": identifier, **records[identifier]}
            for identifier in self.metadata(dataset).provenance_ids
        ]

    def period_label(self, dataset: str, *, short: bool = False) -> str:
        period = self.metadata(dataset).period
        if period.label is None:
            return "Period not specified"
        if short and period.start_year is not None:
            return f"{period.start_year}-{str(period.end_year)[-2:]}"
        return period.label

    def field_year(self, dataset: str) -> int:
        year = self.metadata(dataset).field_year
        if year is None:
            raise ContractError(f"Dataset {dataset} has no declared field year")
        return year

    def policy(self, dataset: str) -> ValuePolicy:
        policy_id = self.metadata(dataset).value_policy
        if policy_id is None:
            raise ContractError(f"Dataset {dataset} has no value policy")
        return ValuePolicy.from_dict(self._policies[policy_id].to_dict())

    def source_for_metric(self, metric: str, scope: str) -> tuple[str, str]:
        if scope not in {"campus", "district"}:
            raise ContractError(f"Unknown entity scope: {scope}")
        if metric.startswith("tapr."):
            dataset = f"{scope}_tapr_student_staff_profile"
            return dataset, self.field(dataset, metric[5:], scope=scope)
        if metric.startswith("baseline."):
            if scope != "campus":
                raise ContractError("Historical campus enrollment is a campus metric")
            dataset = "campus_tapr_historical_enrollment"
            return dataset, self.field(dataset, metric[9:], scope=scope)
        if metric.startswith("financials."):
            return "campus_peims_financials", metric[11:]
        if metric in {"rating", "overall_rating"}:
            return "accountability", "rating"
        if metric == "enrollment":
            return ("campuses" if scope == "campus" else "accountability"), "enrollment"
        if metric.startswith("transfer."):
            return "campus_transfer_reports", metric[9:]
        raise ContractError(f"Unknown canonical metric: {metric}")

    def field(self, dataset: str, metric: str, *, scope: str = "campus") -> str:
        metadata = self.metadata(dataset)
        definition = metadata.metrics.get(metric)
        return (
            definition.field
            if definition
            else f"{scope}_{self.field_year(dataset)}_{metric}"
        )

    def metric_period(self, metric: str, *, scope: str = "campus") -> DataPeriod:
        dataset, _ = self.source_for_metric(metric, scope)
        metadata = self.metadata(dataset)
        definition = metadata.metrics.get(metric.partition(".")[2] or metric)
        return definition.period if definition else metadata.period

    def measurement(
        self, entity: Any, metric: str, *, scope: str | None = None
    ) -> Measurement:
        scope = scope or ("campus" if hasattr(entity, "campus_number") else "district")
        dataset, key = self.source_for_metric(metric, scope)
        raw = getattr(entity, "meta", {}).get(key)
        if metric in {"rating", "overall_rating", "enrollment"}:
            raw = getattr(entity, key, None)
        unit = "percent" if key.endswith("_percent") else None
        if metric.startswith("financials.") and key.endswith("_per_student"):
            unit = "currency"
        definition = self.metadata(dataset).metrics.get(
            metric.partition(".")[2] or metric
        )
        if definition and definition.unit is not None:
            unit = definition.unit
        if metric in {"rating", "overall_rating"}:
            unit = "text"
        return measure(
            raw,
            self.policy(dataset),
            unit=unit,
            minimum=None
            if metric.startswith("financials.") or unit in {"currency", "text"}
            else 0,
            maximum=100 if unit == "percent" else None,
        )

    def metric_label(
        self, metric: str, label: str, *, scope: str = "campus", short: bool = True
    ) -> str:
        dataset, _ = self.source_for_metric(metric, scope)
        value = self.metric_period(metric, scope=scope)
        period = value.label or "Period not specified"
        if short and value.start_year is not None:
            period = f"{value.start_year}-{str(value.end_year)[-2:]}"
        return f"{period} {label}"

    def charter_transfer_summary(self, campus: Any) -> dict[str, Any]:
        """Compute one campus's observed and estimated flows at the declared grain."""
        policy = self.policy("campus_transfer_reports")
        model = self.metadata("campus_transfer_reports").estimation
        if model is None or model.grain != "origin_campus_destination_campus":
            raise ContractError(
                "Campus transfer estimates require a model at campus-pair grain"
            )
        destinations = {}
        for destination, count, masked in campus._repo.transfers_out(campus):
            if (
                destination is None
                or not destination.is_charter
                or destination.is_private
            ):
                continue
            value = measure(
                "-999" if masked else count, policy, unit="count", minimum=0
            )
            old = destinations.get(destination.id)
            if old is not None and old != value:
                raise ContractError("Conflicting duplicate campus transfer flow")
            destinations[destination.id] = value
        summary = aggregate(destinations.values())
        unresolved = [
            row
            for row in getattr(campus._repo, "_xfers_unresolved", [])
            if row["source_campus"] == campus.campus_number
        ]
        summary = replace(
            summary, missing_count=summary.missing_count + len(unresolved)
        )
        estimate = model.estimate(summary)
        return {
            "unresolved_transfers": unresolved,
            "unresolved_transfer_count": len(unresolved),
            "num_charter_transfer_destinations": len(destinations),
            "num_charter_transfer_destinations_masked": summary.suppressed_count,
            "total_unmasked_charter_transfers_out": summary.observed_sum,
            "total_estimated_masked_charter_transfers_out": estimate[
                "estimated_contribution"
            ],
            "total_estimated_charter_transfers_out": estimate["value"],
            "charter_transfer_aggregate": summary.to_dict(),
            "charter_transfer_estimate": estimate,
        }


def write_sqlite_catalog(
    connection: sqlite3.Connection, catalog: DatasetCatalog
) -> None:
    connection.execute(
        f"CREATE TABLE {CATALOG_TABLE} (singleton INTEGER PRIMARY KEY CHECK(singleton=1), document TEXT NOT NULL)"
    )
    connection.execute(
        f"INSERT INTO {CATALOG_TABLE} VALUES (1, ?)",
        (canonical_json(catalog.to_dict()),),
    )


def read_sqlite_catalog(path: str | Path) -> DatasetCatalog:
    path = Path(path).resolve()
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(
                f"SELECT document FROM {CATALOG_TABLE} WHERE singleton=1"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ContractError(
            f"Artifact lacks the required dataset catalog: {path.name}"
        ) from exc
    if len(rows) != 1:
        raise ContractError("Artifact must contain exactly one dataset catalog")
    return DatasetCatalog(json.loads(rows[0][0]))


ACTIVE_BUILD: ContextVar[BuildRecorder | None] = ContextVar(
    "teadata_build", default=None
)


class BuildRecorder:
    """Bind successful source reads to verified acquisition receipts."""

    def __init__(self, cfg: Any, store: ProvenanceStore):
        self.cfg = cfg
        self.store = store
        self.datasets: dict[str, Any] = {}
        self._token = None

    def __enter__(self) -> BuildRecorder:
        if ACTIVE_BUILD.get() is not None:
            raise ContractError("Nested data builds are not supported")
        self._token = ACTIVE_BUILD.set(self)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._token is not None:
            ACTIVE_BUILD.reset(self._token)

    def before_read(self, source: ResolvedDataset) -> dict[str, Any]:
        expected = self.cfg.resolve_dataset(
            source.dataset, source.requested_year, section=source.section
        )
        if expected.to_dict() != source.to_dict():
            raise ContractError("Source configuration changed during the build")
        policy = self.cfg.value_policies.get(source.metadata.value_policy)
        if policy is not None and policy.reference_id:
            references = {
                reference.reference_id: reference
                for reference in source.metadata.references
            }
            reference = references.get(policy.reference_id)
            if reference is None or reference.status != "available":
                raise ContractError(
                    f"Value policy requires reviewed methodology: {source.dataset}"
                )
        return verify_source_receipts(
            source, self.store, reference_base=self.cfg.base_dir
        )

    def after_read(
        self,
        source: ResolvedDataset,
        identity: Mapping[str, Any],
        rows: int,
        profile: str | None,
        effective_reader: Mapping[str, Any],
        columns: list[str],
    ) -> None:
        if file_identity(source.path) != identity:
            raise ContractError(f"Source changed while being read: {source.dataset}")
        if rows == 0:
            raise ContractError(f"Required dataset is empty: {source.dataset}")
        document: dict[str, Any] = {
            "dataset": source.dataset,
            "section": source.section,
            "requested_year": source.requested_year,
            "resolved_year": source.resolved_year,
            "metadata": source.metadata.to_dict(),
            "reader": dict(source.reader),
            "artifact": dict(identity),
            "status": "loaded",
            "reads": [],
        }
        old = self.datasets.get(source.dataset)
        if old is not None:
            if {k: v for k, v in old.items() if k != "reads"} != {
                k: v for k, v in document.items() if k != "reads"
            }:
                raise ContractError(
                    f"Mixed source versions in one build: {source.dataset}"
                )
            document = old
        read = {
            "profile": profile,
            "rows": rows,
            "reader": dict(effective_reader),
            "columns": columns,
        }
        if read not in document["reads"]:
            document["reads"].append(read)
        self.datasets[source.dataset] = document

    def catalog(self) -> DatasetCatalog:
        identifiers = set()
        for source in self.datasets.values():
            identifiers.update(source["metadata"]["provenance_ids"])
            for reference in source["metadata"]["references"]:
                identifiers.update(reference["provenance_ids"])
        return DatasetCatalog.create(
            datasets=self.datasets,
            value_policies={
                key: value.to_dict() for key, value in self.cfg.value_policies.items()
            },
            processor=processor_identity(),
            records={
                identifier: self.store.read_record(identifier)
                for identifier in sorted(identifiers)
            },
            environment={
                "python": platform.python_version(),
                "hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "architecture": platform.machine(),
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "teadata",
                        "pandas",
                        "numpy",
                        "geopandas",
                        "shapely",
                        "pyogrio",
                        "openpyxl",
                        "sqlalchemy",
                    )
                },
            },
        )


def verify_source_receipts(
    source: ResolvedDataset,
    store: ProvenanceStore,
    *,
    reference_base: Path | None = None,
) -> dict[str, Any]:
    identity = file_identity(source.path)
    if not source.metadata.provenance_ids:
        raise ContractError(
            f"Source {source.dataset} needs an acquisition receipt; run the source-observation stage"
        )
    matched = False
    for record_id in source.metadata.provenance_ids:
        record = store.read_record(record_id)
        if (
            record.get("kind") != "acquisition"
            or record.get("dataset") != source.dataset
        ):
            raise ContractError("Source receipt belongs to a different dataset")
        if record["artifact"]["sha256"] == identity["sha256"]:
            current_metadata = source.metadata.to_dict()
            recorded_metadata = DatasetMetadata.from_dict(
                record["source_metadata"]
            ).to_dict()
            current_metadata.pop("provenance_ids", None)
            recorded_metadata.pop("provenance_ids", None)
            if current_metadata != recorded_metadata:
                raise ContractError(
                    f"Source metadata differs from its receipt: {source.dataset}"
                )
            store.verify_artifact(identity["sha256"])
            matched = True
    if not matched:
        raise ContractError(
            f"Input bytes do not match declared provenance for {source.dataset}"
        )
    for reference in source.metadata.references:
        if reference.status != "available":
            continue
        if not reference.provenance_ids:
            raise ContractError(
                f"Reference document {reference.reference_id} needs an acquisition receipt"
            )
        for record_id in reference.provenance_ids:
            record = store.read_record(record_id)
            if reference.url not in {
                record.get("requested_url"),
                record.get("resolved_url"),
                record.get("source_metadata", {}).get("source_url"),
            }:
                raise ContractError("Reference receipt belongs to a different URL")
            store.verify_artifact(record["artifact"]["sha256"])
            if reference.path is not None:
                if reference_base is None:
                    raise ContractError(
                        "Local reference needs a source contract directory"
                    )
                path = (reference_base / reference.path).resolve()
                if (
                    not path.is_relative_to(reference_base.resolve())
                    or file_identity(path)["sha256"] != record["artifact"]["sha256"]
                ):
                    raise ContractError("Local methodology differs from its receipt")
    return identity
