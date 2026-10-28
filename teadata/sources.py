"""Explicit source onboarding: archive existing bytes and reference documents.

Observation of migrated files leaves their original download date unknown.
Reference downloads are opt-in and never run implicitly during an ETL build.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import yaml

from .dataset_metadata import DatasetMetadata, ResolvedDataset
from .manifest import verify_source_receipts
from .provenance import ProvenanceStore
from .teadata_config import load_config


def observe_sources(
    config: str | Path, store: str | Path, *, fetch_references: bool = False
) -> dict[str, str]:
    """Record selected local sources and, explicitly, fetch their documentation.

    Writes the configuration only after all requested observations succeed.
    Immutable receipts from interrupted attempts remain valid historical events.
    """
    config = Path(config).resolve()
    cfg = load_config(config)
    raw = yaml.safe_load(config.read_text())
    archive = ProvenanceStore(store)
    references = {}
    observed = {}
    for section in ("data_sources", "spatial"):
        catalog = getattr(cfg, section)
        for dataset in catalog:
            source = cfg.resolve_dataset(dataset, section=section)
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
                if reference.url not in references:
                    with urlopen(reference.url, timeout=60) as response:
                        with tempfile.NamedTemporaryFile() as temporary:
                            while block := response.read(1024 * 1024):
                                temporary.write(block)
                            temporary.flush()
                            now = datetime.now(timezone.utc)
                            document = ResolvedDataset(
                                dataset=f"reference:{reference.reference_id}",
                                section="reference",
                                requested_year=None,
                                resolved_year=source.resolved_year,
                                path=temporary.name,
                                metadata=DatasetMetadata(
                                    title=reference.title, source_url=reference.url
                                ),
                            )
                            references[reference.url] = archive.record_acquisition(
                                temporary.name,
                                document,
                                downloaded_at=now,
                                requested_url=reference.url,
                                resolved_url=response.url,
                                response_headers=dict(response.headers),
                            )
                updated_refs.append(
                    replace(reference, provenance_ids=(references[reference.url],))
                )
            source = replace(
                source,
                metadata=replace(source.metadata, references=tuple(updated_refs)),
            )
            archive.retain_artifact(source.path)
            try:
                verify_source_receipts(source, archive, reference_base=config.parent)
            except (OSError, ValueError, RuntimeError):
                # Existing receipts remain in history; this contract selects the new observation.
                source = replace(
                    source, metadata=replace(source.metadata, provenance_ids=())
                )
                record_id = archive.record_acquisition(source.path, source)
                source = replace(
                    source,
                    metadata=replace(source.metadata, provenance_ids=(record_id,)),
                )
            entry["metadata"] = source.metadata.to_dict()
            observed[dataset] = source.metadata.provenance_ids[-1]
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
    args = parser.parse_args()
    for name, receipt in observe_sources(
        args.config, args.store, fetch_references=args.fetch_references
    ).items():
        print(f"{name}: {receipt}")


if __name__ == "__main__":
    main()
