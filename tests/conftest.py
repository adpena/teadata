import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest  # noqa: E402

from teadata.manifest import DatasetCatalog  # noqa: E402
from teadata.provenance import ProvenanceStore  # noqa: E402
from teadata.teadata_config import load_config  # noqa: E402


@pytest.fixture
def catalog():
    cfg = load_config(ROOT / "teadata" / "teadata_sources.yaml")
    datasets = {}
    records = {}
    store = ProvenanceStore(ROOT / "provenance")
    for section in ("data_sources", "spatial"):
        for name in getattr(cfg, section):
            source = cfg.resolve_dataset(name, section=section)
            for identifier in source.metadata.provenance_ids:
                records[identifier] = store.read_record(identifier)
            for reference in source.metadata.references:
                for identifier in reference.provenance_ids:
                    records[identifier] = store.read_record(identifier)
            datasets[name] = {
                "section": section,
                "metadata": source.metadata.to_dict(),
                "status": "loaded",
                "artifact": records[source.metadata.provenance_ids[-1]]["artifact"],
                "reads": [{"columns": cfg.column_schema(name, reader_profile=profile, section=section) or []}
                          for profile in (source.reader.get("profiles") or {None: {}})],
            }
    return DatasetCatalog.create(
        datasets=datasets,
        value_policies={
            key: value.to_dict() for key, value in cfg.value_policies.items()
        },
        processor="2" * 64,
        records=records,
        metric_bindings=cfg.resolved_metric_bindings(),
    )
