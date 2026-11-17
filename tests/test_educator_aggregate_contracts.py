"""The aggregate source scope stays distinct from its periods and detail."""

from unittest.mock import Mock

import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema
from teadata.educator_certification import ADMITTED, _dataset
from teadata.educator_retention import (
    SOURCE_WITNESS_RECEIPT, SOURCE_WITNESS_SHA, VIEWS, _source_witness,
)
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import load_config


def test_all_68_aggregate_editions_preserve_period_semantics_and_hold_boundary():
    config = load_config()
    editions = 0
    for dataset in (*VIEWS, *(_dataset(role) for role in ADMITTED)):
        source = config.resolve_dataset(dataset, 2026, reader_profile="source", exact=True)
        assert isinstance(source.metadata.version, str)
        assert isinstance(source.metadata.notes, str)
        resource = config.column_contract(dataset, 2026, reader_profile="source")
        assert resource is not None
        tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, source.metadata.version)["tables"]
        editions += len(tables)
        assert config.is_source_only(dataset, 2026)
        assert all(t["metric_period"]["kind"] == "school_year" for t in tables)
        if dataset in VIEWS:
            assert "first-teaching cohort selector" in source.metadata.notes
            assert "2024–25 Year 1 describes employment through 2025–26" in source.metadata.notes
            assert "blank Year 5 is unreported" in source.metadata.notes
            assert "56.44%" in source.metadata.notes and "56.38%" in source.metadata.notes
        else:
            assert {t["table"] for t in tables} == {f"state_{year}" for year in range(2016, 2027)}
            for phrase in ("September 1–August 31", "August 25, 2026", "not a data cutoff",
                           "not additive", "finality through August 31 are unknown", "detail remains unqualified"):
                assert phrase in source.metadata.notes
            if dataset.endswith("teacher"):
                for value in ("26,085", "26,086", "17,797", "17,801", "cause of the +1 and +4 differences is unknown"):
                    assert value in source.metadata.notes
    assert editions == 68


@pytest.mark.parametrize("change", [{"kind": "processing"}, {"method": "download"},
                                    {"artifact": {"sha256": "0" * 64}}])
def test_source_witness_refuses_unreviewed_observation_before_reading_payload(change):
    record = {"kind": "acquisition", "method": "local_observation", "artifact": {"sha256": SOURCE_WITNESS_SHA}}
    record.update(change)
    store = Mock(spec=ProvenanceStore)
    store.read_record.return_value = record
    store.verify_artifact.side_effect = AssertionError("Unadmitted witness opened")
    with pytest.raises(ContractError, match="reviewed observation"):
        _source_witness(store, SOURCE_WITNESS_RECEIPT)
