import pytest

from teadata import EntitySelectionError
from teadata.selectors import _EntityNames, _entity_suggestions, _field_suggestions


@pytest.mark.parametrize(
    ("query", "target", "alternatives"),
    [
        ("asutin isd", "austin isd", ("palestine isd", "seguin isd")),
        ("travis lake isd", "lake travis isd", ("rains isd", "paradise isd", "travis isd")),
        ("trvais lake isd", "lake travis isd", ("travis isd", "travis acd", "lake isd", "rains isd",
                                             "paradise isd", "prairie lea isd", "travis lake academy")),
        ("cypress fairbanks isd", "cypress-fairbanks isd", ("cross plains isd",)),
        ("oconnor high school", "o'connor h s", ("hill country high school", "carroll high school")),
        ("high school oconnr", "o'connor h s", ("hill country high school", "carroll high school")),
        ("oconnr high school", "o'connor h s", ("hill country high school", "carroll high school")),
        ("alief learing ctr ６–１２", "alief learning ctr (6-12)", ("alief learning ctr (k-6)",)),
        ("academy 3d", "3d academy", ("academy el", "academy int", "academy h s")),
        ("alief learning ctr 6–12", "alief learning ctr (6-12)", ("alief learning ctr (k-6)",)),
        ("academy 12 elemntary", "elementary academy 12",
         ("academy 13 elementary", "academy 12 preparatory", "academy 12 elms", "academy 12 center",
          "academy 12 scholars", "academy 12 learning", "academy 12 montessori")),
    ],
)
def test_suggestions_rank_hard_names_without_resolving_them(query, target, alternatives):
    choices = (*alternatives, target)
    assert _entity_suggestions(query, choices)[0] == target
    assert _entity_suggestions(query, choices) == _entity_suggestions(query, reversed(choices))
    directory = _EntityNames(((name, f"{index:09d}") for index, name in enumerate(choices)), str)
    with pytest.raises(EntitySelectionError, match="No exact match") as error:
        directory.resolve(query, scope="campus")
    assert error.value.candidates[0][0] == target


def test_same_name_candidates_retain_ids_and_dont_crowd_out_other_labels():
    entries = [("SUMMER SCHOOL", f"{i:09d}") for i in reversed(range(1, 12))]
    directory = _EntityNames([*entries, ("SMYER SCHOOL", "000000100")], str)
    with pytest.raises(EntitySelectionError, match=r"SUMMER SCHOOL \(000000001\).*6 more") as exact:
        directory.resolve("summer school", scope="campus")
    assert exact.value.to_dict()["candidates"][-1] == {"name": "SUMMER SCHOOL", "id": "000000011"}
    with pytest.raises(EntitySelectionError, match="Did you mean") as miss:
        directory.resolve("summer shcool", scope="campus")
    assert len(miss.value.candidates) == 5
    assert miss.value.candidates[1] == ("SMYER SCHOOL", "000000100")


def test_duplicate_district_and_literal_numbered_campus():
    districts = _EntityNames([("NORTHSIDE ISD", "244905"), ("NORTHSIDE ISD", "015915")], str)
    with pytest.raises(EntitySelectionError, match=r"NORTHSIDE ISD \(015915\), NORTHSIDE ISD \(244905\)"):
        districts.resolve("Northside ISD", scope="district")

    campus = _EntityNames([("ALIEF LEARNING CTR (6-12)", "101903005"), ("3D ACADEMY", "101912385")], str)
    assert campus.resolve(" alief learning ctr (6-12) ", scope="campus") == "101903005"
    assert campus.resolve("3d academy", scope="campus") == "101912385"
    assert _field_suggestions("amount_sttus", ["amount_status"]) == ("amount_status",)
    long_name = "." * 300
    assert _entity_suggestions(long_name, ["alief"]) == ()
    with pytest.raises(EntitySelectionError) as error:
        campus.resolve(long_name, scope="campus")
    assert len(str(error.value)) < 300
    assert error.value.to_dict()["selector"] == long_name


def test_long_candidate_labels_are_previewed_without_truncating_structured_errors():
    name = "A" * 2000
    directory = _EntityNames([(name, "010901001"), (name, "010901002")], str)
    with pytest.raises(EntitySelectionError) as error:
        directory.resolve(name, scope="campus")
    assert len(str(error.value)) < 400
    assert error.value.to_dict()["candidates"][0]["name"] == name
