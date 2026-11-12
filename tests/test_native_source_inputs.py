"""Pinned native descriptors through the normal entity-build input path."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from teadata.dataset_metadata import NativeSourceRef, _native_manifest_location
from teadata.load_data import _native_sources_for_build
from teadata.manifest import ContractError
from teadata.release import DataRelease, publish_release

from test_catalog_projections import make_edition_fixture


@pytest.fixture(params=[3, 4])
def native(tmp_path, request):
    fixture = make_edition_fixture(tmp_path, normalized=True)
    path = fixture['root']
    if request.param == 4:
        path = publish_release(tmp_path, fixture['catalog'].externalize_editions(), schema_version=4)
    location = 'native.json'
    path.rename(tmp_path / location)
    selected = DataRelease(tmp_path / location, expected_sha256=sha256(path.with_name(location).read_bytes()).hexdigest(), offline=True)
    return selected, fixture


def test_native_descriptor_is_index_only_and_preserves_exact_source_keys(native, monkeypatch):
    selected, fixture = native
    expected = tuple(item[0] for item in fixture['leaves'].values())
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, 'verify', lambda *args, **kwargs: pytest.fail('Verified data during discovery'))
        guard.setattr(DataRelease, 'artifact', lambda *args, **kwargs: pytest.fail('Read native leaf'))
        guard.setattr(DataRelease, '_column_schema_bytes', lambda *args, **kwargs: pytest.fail('Read metadata payload'))
        guard.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('Descriptor opened a file'))
        actual = selected.native_source_ref(location='selected-native.json')
    assert set(actual.editions) == set(expected)
    assert actual.manifest_sha256 == selected.manifest_sha256 and actual.build_id == selected.build_id
    assert actual.location == 'selected-native.json'
    assert set(actual.to_dict()) == {'location', 'manifest_sha256', 'build_id', 'editions'}
    assert NativeSourceRef.from_dict(actual.to_dict()) == actual


def test_short_and_full_build_inputs_resolve_to_the_same_exact_descriptor(native):
    selected, _ = native
    full = selected.native_source_ref(location=selected.path.name)
    short = {'location': full.location, 'manifest_sha256': full.manifest_sha256}
    assert _native_sources_for_build([short], selected.path.parent) == (full,)
    # Preserve manual descriptors with an equivalent, differently ordered edition list.
    reordered = replace(full, editions=tuple(reversed(full.editions)))
    assert _native_sources_for_build([reordered.to_dict()], selected.path.parent) == (reordered,)
    assert _native_sources_for_build([], selected.path.parent) == ()


@pytest.mark.parametrize('location', [None, False, 3, '', 'release.json', '../native.json',
    'nested/native.json', r'folder\native.json', 'native?.json', 'native#.json', 'native .json',
    'https://example.org/native.json?download=1', 'https://user:secret@example.org/native.json'])
def test_shared_location_validation_rejects_unsafe_selections_before_opening(tmp_path, monkeypatch, location):
    opened = Mock(side_effect=AssertionError('Malformed source reached manifest I/O'))
    monkeypatch.setattr('teadata.load_data.DataRelease', opened)
    with pytest.raises(ValueError):
        _native_manifest_location(location)
    with pytest.raises(ContractError, match='options.native_sources'):
        _native_sources_for_build([{'location': location, 'manifest_sha256': '0' * 64}], tmp_path)
    opened.assert_not_called()


@pytest.mark.parametrize('raw', [None, (), {}, False, ['native.json'], [{'location': 'native.json'}],
    [{'location': 'native.json', 'manifest_sha256': '0' * 64, 'extra': True}]])
def test_build_selection_requires_explicit_known_fields(tmp_path, raw):
    with pytest.raises(ContractError, match='options.native_sources'):
        _native_sources_for_build(raw, tmp_path)


@pytest.mark.parametrize('pin', [False, '', '0' * 63, 'A' * 64, '０' * 64])
def test_bad_pins_are_actionable_and_never_open_sources(tmp_path, monkeypatch, pin):
    opened = Mock(side_effect=AssertionError('Bad pin reached manifest I/O'))
    monkeypatch.setattr('teadata.load_data.DataRelease', opened)
    with pytest.raises(ContractError, match='SHA-256'):
        _native_sources_for_build([{'location': 'native.json', 'manifest_sha256': pin}], tmp_path)
    opened.assert_not_called()


def test_short_input_rejects_changed_manifest_before_artifact_reads(native, monkeypatch):
    selected, _ = native
    selected.path.write_bytes(selected.path.read_bytes() + b'\n')
    monkeypatch.setattr(DataRelease, 'artifact', lambda *args, **kwargs: pytest.fail('Read data after changed pin'))
    with pytest.raises(ContractError, match='trusted pin'):
        _native_sources_for_build([{'location': selected.path.name, 'manifest_sha256': selected.manifest_sha256}], selected.path.parent)


@pytest.mark.parametrize('damage', ['extra', 'missing', 'build'])
def test_full_descriptor_still_rejects_wrong_manual_metadata(native, damage):
    selected, _ = native
    ref = selected.native_source_ref(location=selected.path.name)
    if damage == 'build':
        ref = replace(ref, build_id='f' * 64)
    elif damage == 'missing':
        ref = replace(ref, editions=ref.editions[:1])
    else:
        ref = replace(ref, editions=ref.editions + (replace(ref.editions[0], edition='2024'),))
    with pytest.raises(ContractError, match='build ID|editions differ'):
        _native_sources_for_build([ref.to_dict()], selected.path.parent)


def test_short_and_full_duplicates_fail_before_structural_replay(native, monkeypatch):
    selected, _ = native
    ref = selected.native_source_ref(location=selected.path.name)
    short = {'location': ref.location, 'manifest_sha256': ref.manifest_sha256}
    monkeypatch.setattr(DataRelease, 'verify', lambda *args, **kwargs: pytest.fail('Replayed duplicate source'))
    with pytest.raises(ContractError, match='cannot repeat'):
        _native_sources_for_build([short, ref.to_dict()], selected.path.parent)


def test_two_field_http_selection_remains_offline(tmp_path, monkeypatch):
    from teadata import release

    monkeypatch.setattr(release, '_download', lambda *args, **kwargs: pytest.fail('Implicit download'))
    monkeypatch.setattr(release, 'user_cache_path', lambda *args: tmp_path / 'empty-cache')
    with pytest.raises(ContractError, match='offline|Offline|cache'):
        _native_sources_for_build([{'location': 'https://example.invalid/native.json', 'manifest_sha256': 'a' * 64}], tmp_path)


def test_entity_release_does_not_become_a_native_attachment():
    selected = DataRelease.__new__(DataRelease)
    selected._v3 = False
    with pytest.raises(ContractError, match='native schema3/schema4'):
        selected.native_source_ref(location='native.json')


@pytest.mark.parametrize('fail', [False, True])
def test_build_verification_discards_child_metadata_before_next_input(native, monkeypatch, fail):
    selected, _ = native
    first = selected.native_source_ref(location=selected.path.name)
    second = replace(first, location='other.json', manifest_sha256='f' * 64,
                     editions=tuple(replace(key, dataset='other_' + key.dataset) for key in first.editions))
    children = []

    def open_child(*args, **kwargs):
        for previous in children:
            assert previous.metadata == previous._column_schema_data == {}
        child = SimpleNamespace(metadata={'edition': object()}, _column_schema_data={'schema': object()})
        child.catalog = SimpleNamespace(_clear_edition_caches=child.metadata.clear)
        child.verify = Mock(side_effect=ContractError('damaged child') if fail else None)
        children.append(child)
        return child

    monkeypatch.setattr('teadata.load_data._open_native_source', open_child)
    entries = [first.to_dict(), second.to_dict()]
    if fail:
        with pytest.raises(ContractError, match='damaged child'):
            _native_sources_for_build(entries, selected.path.parent)
    else:
        assert _native_sources_for_build(entries, selected.path.parent) == (first, second)
    assert len(children) == (1 if fail else 2)
    assert all(child.metadata == child._column_schema_data == {} for child in children)


def test_invalid_input_stops_before_entity_build_or_provenance_writes(tmp_path, monkeypatch):
    from teadata import load_data

    cfg = SimpleNamespace(options={'native_sources': [{'location': 'missing.json', 'manifest_sha256': 'a' * 64}]},
                          resolve_dataset=lambda name, **kwargs: SimpleNamespace(path=str(tmp_path / name)),
                          spatial_selection=lambda *args: {})
    monkeypatch.setattr(load_data, 'sys', SimpleNamespace(flags=SimpleNamespace(hash_randomization=0)))
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setenv('TEADATA_OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(load_data, 'load_config', lambda *args: cfg)
    build = Mock(side_effect=AssertionError('Constructed entities after invalid input'))
    store = Mock(side_effect=AssertionError('Opened output provenance after invalid input'))
    monkeypatch.setattr(load_data, '_build_repo', build)
    monkeypatch.setattr(load_data, 'ProvenanceStore', store)
    with pytest.raises(ContractError, match=r'options.native_sources\[0\]'):
        load_data.load_repo(str(tmp_path / 'districts'), str(tmp_path / 'campuses'))
    build.assert_not_called()
    store.assert_not_called()
