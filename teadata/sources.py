"""Explicit source onboarding: archive existing bytes and reference documents.

Observation of migrated files leaves their original download date unknown.
Reference downloads are opt-in and never run implicitly during an ETL build.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import yaml

from .dataset_metadata import DatasetMetadata, ResolvedDataset
from .acquisition import AcquisitionClient
from .manifest import verify_reference_receipts, verify_source_receipts
from .provenance import ProvenanceStore
from .teadata_config import load_config


def observe_sources(
    config: str | Path, store: str | Path, *, fetch_references: bool = False,
    all_versions: bool = False, datasets: set[str] | None = None,
) -> dict[str, str]:
    """Record selected local sources and, explicitly, fetch their documentation.

    Writes the configuration only after all requested observations succeed.
    Immutable receipts from interrupted attempts remain valid historical events.
    """
    config = Path(config).resolve()
    cfg = load_config(config)
    raw = yaml.safe_load(config.read_text())
    archive = ProvenanceStore(store)
    client = AcquisitionClient(archive, max_bytes=64 * 1024 * 1024)
    references = {}
    observed = {}
    if datasets is not None and datasets - (set(cfg.data_sources) | set(cfg.spatial)):
        raise ValueError("Unknown dataset selection")
    for section in ("data_sources", "spatial"):
        catalog = getattr(cfg, section)
        selections = [(dataset, year) for dataset, versions in catalog.items()
                      if datasets is None or dataset in datasets
                      for year in (sorted(versions.entries) if all_versions else [None])]
        for dataset, year in selections:
            source = cfg.resolve_dataset(dataset, year, section=section)
            if year == 9999:
                source = replace(source, requested_year=None)
            entry = raw[section][dataset][
                source.resolved_year if source.resolved_year is not None else "current"
            ]
            if not isinstance(entry, dict):
                raise ValueError(
                    "Migrate source entries to structured contracts before observation"
                )
            updated_refs = []
            for reference in source.metadata.references:
                if reference.status != "available":
                    updated_refs.append(reference)
                    continue
                if reference.path:
                    path = (config.parent / reference.path).resolve()
                    if not path.is_relative_to(config.parent):
                        raise ValueError(
                            "Local methodology path escapes the contract directory"
                        )
                    identity = archive.retain_artifact(path)
                    matching = [
                        identifier
                        for identifier in reference.provenance_ids
                        if archive.read_record(identifier)["artifact"]["sha256"]
                        == identity["sha256"]
                    ]
                    if not matching:
                        document = ResolvedDataset(
                            f"reference:{reference.reference_id}",
                            "reference",
                            None,
                            source.resolved_year,
                            str(path),
                            DatasetMetadata(
                                title=reference.title, source_url=reference.url
                            ),
                        )
                        matching = [archive.record_acquisition(path, document)]
                    updated_refs.append(
                        replace(reference, provenance_ids=tuple(matching))
                    )
                    continue
                if reference.provenance_ids:
                    try:
                        for identifier in reference.provenance_ids:
                            archive.verify_artifact(
                                archive.read_record(identifier)["artifact"]["sha256"]
                            )
                    except FileNotFoundError:
                        pass
                    else:
                        updated_refs.append(reference)
                        continue
                if not fetch_references:
                    raise ValueError(
                        f"Reference {reference.reference_id} needs explicit --fetch-references"
                    )
                url = reference.url
                if url is None:
                    raise ValueError(f"Reference {reference.reference_id} has no remote URL")
                if url not in references:
                    document = ResolvedDataset(
                        dataset=f"reference:{reference.reference_id}",
                        section="reference",
                        requested_year=None,
                        resolved_year=source.resolved_year,
                        path="",
                        metadata=DatasetMetadata(title=reference.title, source_url=url),
                    )
                    extension = Path(urlparse(url).path).suffix
                    if extension not in {".pdf", ".html", ".csv", ".xlsx"}:
                        extension = ".html"
                    references[url] = client.fetch(
                        url, document, filename=reference.reference_id + extension
                    ).record_id
                updated_refs.append(
                    replace(reference, provenance_ids=(references[url],))
                )
            source = replace(
                source,
                metadata=replace(source.metadata, references=tuple(updated_refs)),
            )
            # Observing source files cannot approve changed methodology bytes.
            verify_reference_receipts(source, archive, reference_base=config.parent)
            identity = archive.retain_artifact(source.path)
            try:
                verify_source_receipts(source, archive, reference_base=config.parent)
            except (OSError, ValueError, RuntimeError):
                # Existing receipts remain in history; this contract selects the new observation.
                retained = tuple(
                    identifier for identifier in source.metadata.provenance_ids
                    if archive.read_record(identifier).get("kind") == "acquisition"
                    and archive.read_record(identifier).get("dataset") == source.dataset
                    and archive.read_record(identifier).get("section") == source.section
                    and archive.read_record(identifier)["artifact"]["sha256"] == identity["sha256"]
                )
                source = replace(
                    source, metadata=replace(source.metadata, provenance_ids=())
                )
                record_id = archive.record_acquisition(source.path, source)
                source = replace(
                    source,
                    metadata=replace(source.metadata, provenance_ids=(*retained, record_id)),
                )
            entry["metadata"] = source.metadata.to_dict()
            observed[f"{dataset}:{source.resolved_year or 'current'}" if all_versions else dataset] = source.metadata.provenance_ids[-1]
    with tempfile.NamedTemporaryFile(
        mode="w", dir=config.parent, delete=False
    ) as target:
        temporary_path = Path(target.name)
        yaml.safe_dump(raw, target, sort_keys=False, allow_unicode=True)
    try:
        os.replace(temporary_path, config)
    finally:
        temporary_path.unlink(missing_ok=True)
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--store", type=Path, default=Path("provenance"))
    parser.add_argument("--fetch-references", action="store_true")
    parser.add_argument("--all-versions", action="store_true", help="Observe every declared version, including historical inputs")
    parser.add_argument("--dataset", action="append", help="Restrict observation to named datasets")
    args = parser.parse_args()
    for name, receipt in observe_sources(
        args.config, args.store, fetch_references=args.fetch_references,
        all_versions=args.all_versions, datasets=set(args.dataset) if args.dataset else None,
    ).items():
        print(f"{name}: {receipt}")


if __name__ == "__main__":
    main()
