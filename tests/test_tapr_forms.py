from argparse import Namespace
from hashlib import sha256
import json
import os

import pytest

from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.scripts import refresh_tapr
from teadata.tapr_forms import extract_tapr_forms


def test_retained_menu_shape_preserves_order_and_malformed_publisher_labels():
    # Minimal source-shaped golden: TAPR uses a second opening label where a
    # closing tag was intended. Hidden values and radio choices remain separate.
    html = b'''Metadata preceding the actual form
<form method="post" action="/cgi/sas/broker">
<input type="hidden" name="ccyy" value="2024">
<input type="hidden" name="id" value="">
<table><tr><td><input type='radio' name='dsname' value='RHSP' id='dd23'></td>
<td><label for='dd23'>Annual RHSP/DAP/FHSP-E/FHSP-DLA/Texas First-DLA<label></td></tr>
<tr><td><input type='radio' name='dsname' value='FHSP' id='dd24' disabled></td>
<td><label for='dd24'>Longitudinal RHSP/DAP/FHSP-E/FHSP-DLA/Texas First-DLA<label></td></tr></table>
</form>Footer and masking notes stay in the retained original.'''
    form, = extract_tapr_forms(html, encoding="utf-8", base_url="https://example.org/source")
    report = form.to_dict()
    assert report["method"] == "post" and report["closed"] is True
    assert report["resolved_action"] == "https://example.org/cgi/sas/broker"
    assert [dict(control["attributes"])["value"] for control in report["controls"]] == ["2024", "", "RHSP", "FHSP"]
    assert dict(report["controls"][-1]["attributes"])["disabled"] is None
    labels = {dict(label["attributes"]).get("for"): label["text"] for label in report["labels"]}
    assert labels["dd23"].startswith("Annual RHSP")
    assert labels["dd24"].startswith("Longitudinal RHSP")
    assert report["position"] == [2, 0]


def test_controls_are_observations_without_submission_default_inference():
    form, = extract_tapr_forms(b'<form><input name="same" value="one"><input name="same" value="two"></form>', encoding="ascii")
    report = form.to_dict()
    assert report["method"] is None and report["action"] is None and report["resolved_action"] is None
    assert [dict(control["attributes"])["value"] for control in report["controls"]] == ["one", "two"]
    assert all("type" not in dict(control["attributes"]) for control in report["controls"])


def test_select_textarea_and_context_are_preserved_without_disabled_semantics():
    html = b'''<form><fieldset disabled><legend>Options</legend>
<select name="key" multiple><optgroup label="group" disabled>
<option value="one" selected>First &amp; second<option value="two">Other
</optgroup></select><textarea name="notes">\n Source notes  \n</textarea>
<button name="submit" value="yes">Continue</button></fieldset></form>'''
    form, = extract_tapr_forms(html, encoding="ascii")
    controls = form.to_dict()["controls"]
    assert [row["tag"] for row in controls] == ["select", "textarea", "button"]
    first, second = controls[0]["options"]
    assert first["text"] == "First & second" and second["text"] == "Other"
    contexts = form.to_dict()["contexts"]
    group = contexts[first["context_id"]]
    assert group["tag"] == "optgroup" and contexts[group["parent"]]["tag"] == "fieldset"
    assert dict(first["attributes"])["selected"] is None
    assert controls[1]["text"] == "\n Source notes  \n" and controls[2]["text"] == "Continue"


def test_only_literal_links_are_resolved_and_scripts_are_never_executed():
    html = b'''<script>document.write('<form action="evil"><input name="spoof"></form>')</script>
<form method="POST" action="/step"><input type="button" onclick="window.open('/dict?a=1&amp;b=2', 'dictionary')">
<input type="button" onclick="window.open(computed_url)">
<a href="/reference">Reference</a><a href="javascript:alert(1)">Script</a></form>'''
    form, = extract_tapr_forms(html, encoding="ascii", base_url="https://example.org/menu")
    report = form.to_dict()
    assert len(report["controls"]) == 2
    assert [link["resolved_url"] for link in report["links"]] == ["https://example.org/dict?a=1&b=2", "https://example.org/reference", None]
    assert dict(report["controls"][1]["attributes"])["onclick"] == "window.open(computed_url)"


def test_incomplete_forms_and_explicit_encoding_are_visible():
    form, = extract_tapr_forms('<form><label for="x">Élèves</label><input id="x">'.encode("iso-8859-1"), encoding="iso-8859-1")
    assert form.closed is False and form.to_dict()["labels"][0]["text"] == "Élèves"
    with pytest.raises(UnicodeDecodeError):
        extract_tapr_forms(b'<form>\xe9</form>', encoding="utf-8")


def test_textarea_uses_stdlib_rcdata_for_literal_markup_and_entities():
    form, = extract_tapr_forms(b'<form><textarea>x<b>y</b><!--note--></form> &amp;\n</textarea></form>', encoding="ascii")
    assert form.to_dict()["controls"][0]["text"] == 'x<b>y</b><!--note--></form> &\n'
    assert form.closed is True


def test_optional_optgroup_ends_do_not_leak_or_nest():
    html = b'<form><fieldset><select><optgroup label="A" disabled><option value="a">A<optgroup label="B"><option value="b">B</select><input name="x"></fieldset></form>'
    form, = extract_tapr_forms(html, encoding="ascii")
    report = form.to_dict()
    groups = [row for row in report["contexts"] if row["tag"] == "optgroup"]
    assert len(groups) == 2 and groups[0]["parent"] == groups[1]["parent"] == 0
    select, following = report["controls"]
    assert [option["context_id"] for option in select["options"]] == [1, 2]
    assert following["context_id"] == 0  # The surrounding fieldset, not either group.


def test_context_projection_is_linear_and_has_a_total_node_budget():
    depth, count = 100, 100
    html = ('<form>' + '<fieldset>' * depth + '<input name="x">' * count
            + '</fieldset>' * depth + '</form>').encode()
    form, = extract_tapr_forms(html, encoding="ascii", max_nodes=201)
    report = form.to_dict()
    assert len(report["contexts"]) == depth and len(report["controls"]) == count
    assert all(row["context_id"] == depth - 1 and "context" not in row for row in report["controls"])
    assert [row["parent"] for row in report["contexts"]] == [None, *range(depth - 1)]
    assert len(json.dumps(report)) < 40_000
    with pytest.raises(ValueError, match="exceeds max-nodes"):
        extract_tapr_forms(html, encoding="ascii", max_nodes=200)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_node_budget_must_be_explicit_positive_integer(limit):
    with pytest.raises(ValueError, match="positive integer"):
        extract_tapr_forms(b'<form></form>', encoding="ascii", max_nodes=limit)


@pytest.mark.parametrize("html", [b'<form action="one" action="two"></form>', b'<form><form></form></form>', b'<form><select><button>bad</button></select></form>', b'<form/>'])
def test_ambiguous_structure_is_rejected(html):
    with pytest.raises(ValueError):
        extract_tapr_forms(html, encoding="ascii")


def test_self_closing_void_inputs_are_supported():
    form, = extract_tapr_forms(b'<form><input name="id" value="1" /></form>', encoding="ascii")
    assert dict(form.to_dict()["controls"][0]["attributes"])["value"] == "1"


def test_export_is_detached():
    form, = extract_tapr_forms(b'<form><input name="id" value="1"></form>', encoding="ascii")
    report = form.to_dict()
    report["controls"][0]["attributes"].append(("changed", "yes"))
    assert "changed" not in dict(form.to_dict()["controls"][0]["attributes"])


def inspection_fixture(tmp_path):
    original = tmp_path / "form.html"
    original.write_bytes(b'<form method="post" action="/next"><input name="key" value="all"></form>')
    source = ProvenanceStore(tmp_path / "source")
    dataset = ResolvedDataset("synthetic_tapr_form", "reference", None, None, "",
                              DatasetMetadata(title="Synthetic form fixture"))
    record_id = source.record_acquisition(original, dataset)
    args = Namespace(source_store=source.root, store=tmp_path / "inspection", record=record_id,
                     output=tmp_path / "observed.json", encoding="ascii", max_bytes=1024)
    return source, args


def test_cli_inspection_is_offline_retained_and_never_overwrites(tmp_path, monkeypatch):
    source, args = inspection_fixture(tmp_path)
    original = source.read_record(args.record)
    monkeypatch.setattr(refresh_tapr, "AcquisitionClient", lambda *a, **k: pytest.fail("Unexpected network client"))
    argv = ["form", "--source-store", str(args.source_store), "--store", str(args.store),
            "--record", args.record, "--output", str(args.output), "--encoding", args.encoding]
    assert refresh_tapr.main(argv) == 0
    report = json.loads(args.output.read_bytes())
    assert report["source_record_id"] == args.record and report["source_artifact"] == original["artifact"]
    assert report["forms"][0]["resolved_action"] is None
    assert report["javascript_executed"] is False and report["submission_controls_inferred"] is False
    assert source.read_record(args.record) == original and original["downloaded_at"] is None
    archive = ProvenanceStore(args.store)
    processing, = archive.history("tapr_explicit_form_inspection")
    assert set(archive.verify_record_closure([processing["record_id"]])) == {processing["record_id"], args.record}
    assert processing["outputs"]["form_occurrences"]["sha256"] == sha256(args.output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        refresh_tapr.main(argv)


def test_retained_input_caps_and_checksum_refuse_changed_bytes(tmp_path):
    source, args = inspection_fixture(tmp_path)
    args.max_bytes = 1
    with pytest.raises(ValueError, match="exceeds max-bytes"):
        refresh_tapr.inspect_form(args)
    assert not args.output.exists()
    args.max_bytes = 1024
    artifact = source.read_record(args.record)["artifact"]
    (source.root / "objects" / artifact["sha256"]).write_bytes(b'<form method="get"></form>')
    with pytest.raises(ValueError, match="differ from the selected acquisition"):
        refresh_tapr.inspect_form(args)
    assert not args.output.exists()


def test_output_cap_refuses_before_processing_publication_and_removes_staging(tmp_path):
    _source, args = inspection_fixture(tmp_path)
    args.max_output_bytes = 1
    with pytest.raises(ValueError, match="exceeds max-output-bytes"):
        refresh_tapr.inspect_form(args)
    assert not args.output.exists()
    assert not list(tmp_path.glob('.tapr-form-*'))
    assert ProvenanceStore(args.store).history("tapr_explicit_form_inspection") == []


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_nonregular_retained_inputs_refuse_without_following_or_blocking(tmp_path, kind):
    source, args = inspection_fixture(tmp_path)
    artifact = source.read_record(args.record)["artifact"]
    selected = source.root / "objects" / artifact["sha256"]
    selected.unlink()
    if kind == "symlink":
        selected.symlink_to(tmp_path / "form.html")
    else:
        os.mkfifo(selected)
    with pytest.raises((OSError, ValueError)):
        refresh_tapr.inspect_form(args)
    assert not args.output.exists()
