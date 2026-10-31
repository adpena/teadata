# Manually checked publisher cases

`teadata/reference_cases/publisher_2025.json` contains small, manually selected source cells and literal
expectations. No expected value was recorded from a TEA Data export, snapshot,
`to_dict`, or round trip. The source SHA-256, retained relative path, sheet/cell
or CSV record/column, raw token, reference period and units travel with each
case. CSV record numbers include the human-header and code-header records;
Excel cell addresses use ordinary one-based coordinates.

The TAPR publisher facts were independently read from the retained two-header
CSVs. The active release instead selects the separately retained normalized
CAMPPROF/DISTPROF workbooks. `release_input` binds those exact input hashes;
the original CSV hashes remain distinct, and no original transformation or
acquisition event is invented. Changing a selected workbook requires manual
review even when its file name or school year stays the same.

The original acquisition date and exact download URL are unknown for the
retained spreadsheets/CSVs. Portal URLs identify the publisher, not a verified
download event. The enhanced 2025 accountability workbook is the retained
reissued vintage, not a claim about its original publication. The legacy TEFA
configuration's 2025 label does not establish a school year; its period is
unknown. These limitations are intentional, not fields to fill by inference.

| Case | Independent cells and expected interpretation |
| --- | --- |
| Austin ISD `227901` | District Student CSV record 1093: DPNTALLC = 72272 enrollment; DPETALLC = 72175 membership. Summary G9333 = C. |
| Kiker EL `227901180` | Campus Student CSV record 8350: CPNTALLC = 750 enrollment, CPETALLC = 748 membership; CPETSAUC = -3 and CPETSNOC = -1 are suppressed, not counts. Summary G9440 = A. |
| BASIS San Antonio–Shavano `015834001` | Campus record 379: CPNTALLC = CPETALLC = 1057. CPETATTP = 21.2 percent describes **2023–24**, despite the source's 2025 label. Summary G448 = A. |
| Rocketship Explore EL `220820102` | Campus record 7474: enrollment 289, membership 288. Summary G8507 = `Not Rated: Minimum Size rules`, an observed official category; numeric score H8507 is `.`. |
| Ace Academy `rec1kXXvIucG7UkoA` | Legacy Vendors row 62 identifies a private school. Public TAPR enrollment and rating are not applicable, not missing or zero. This does not validate its released district affiliation. |

Enrollment/membership refer to school year 2024–25. The ratings refer to
calendar year 2025. The 2024–25 TAPR glossary distinguishes enrollment and
membership; it does not establish why exactly two pupils differ at Kiker.
The year-specific masking document establishes -1/-3 suppression; these cases
infer no disclosure interval. References and exact hashes are in `sources`.
The generated `data_elements.xlsx` crosswalk is not publisher truth.

The four Kiker transfer rows are 18581–18584 of the 2024–25 campus transfer
report, column 14. Three distinct BASIS destination **campuses** have -999;
Valor South Austin has observed 52. AskTED district rows 24 and 182 identify
their charter providers. The preserved Charter Cost Tracker scenario is
**52 + 3 × 5 = 67**, including modeled contribution 15. It is neither an exact
count nor a documented disclosure bound. Grouping the three BASIS campuses
into their common district before estimating would incorrectly yield 57.
The canonical estimation policy remains the implementation authority; the
literal arithmetic here is an independent oracle.

The mixed accountability excerpt retains original source row offsets after
district-summary rows are removed. It detects a Polars/pandas index alignment
bug: the old cleaner changes BASIS A to C, Kiker A to missing, and Rocketship's
minimum-size category to D on the complete retained workbook. The excerpt
regression must fail that old implementation and pass both optional backends.
Type controls separately preserve nullable integers/booleans, wide integers,
signed zero, duplicate indices and row order.

## Acceptance work still open

These items are **not** xfailed/skipped passing tests, and this small pack does
not certify release 0.0.125. That release is held for the accountability failure
and the unresolved affiliation below. Rebuilding has not been authorized here.

- **TAPR field units:** the publisher explicitly labels the curated CPET
  membership/autism/non-categorical fields as counts. The catalog's wildcard
  fields currently lack unit declarations. Their tests check values, raw tokens,
  statuses and periods, while the fixture retains independent `count` truth.
  Tests check API units for governed canonical enrollment, rating and attrition
  metrics. Fill the unit gap through a reviewed source-field dictionary covering
  supported TAPR fields, not suffix guesses or a few case-specific declarations.
- **Ace institutional affiliation:** the retained TEFA row contains geographic
  Austin ISD enrichment. A point inside Austin ISD is not evidence that this
  private school belongs to the public district. The desired governing public
  district is absent. The released `227901` value must not be blessed as truth.
  Accept and implement the domain repair, then add the actual retained-entity
  regression. The population-applicability test currently covers only TAPR N/A.
- **Reporting-only Rocketship:** it appears in TAPR and accountability but not
  among the 9,739 retained GIS campus identifiers. Preserve it with no invented
  point; absence in this file alone does not establish closure or status. The
  current tests cover its publisher metrics, not full spatial ingestion.
- **BASIS geography:** its retained point (-98.564832, 29.5714620000001) lies
  south of Austin ISD's minimum latitude 30.0971041803831. Its legitimate
  incoming Kiker transfer remains included. Its charter district has no retained
  district polygon. Affiliation and containment need separate retained-bundle
  assertions; the transfer test does not pretend to validate GIS ingestion.
- **Current TEFA missing location:** current official JSON index 264 identifies
  Cornerstone Christian Academy (`m3dnV0ENXzzNKH`), a Bastrop private school
  with null location. It belongs to the staged 2026–27 program, not release
  0.0.125. Retain it without inventing a point or nearest public district when
  that source is adopted. No legacy/current identifier crosswalk is accepted.
- **True unresolved public parent/transfer endpoint:** no real example was
  established in the retained release (its endpoint gate reported zero).
  Existing synthetic fail-closed controls remain necessary; do not manufacture
  a real entity or conflate a private school's lack of public affiliation with
  an unresolved public identifier.

## Retained-release gate

`scripts/check --release <manifest>` remains the single entry point. Its existing
retained-release validation reads the four public cases from the actual
EntityStore, checks enrollment/rating attributes and canonical measurements,
and refuses changed input SHA-256 values until the publisher examples receive
manual review. The packaged JSON is the single reference used by both tests
and release checks, including installed consumers. There is no auto-record
command or whole-store hydration. The known 0.0.125 artifact must fail these
rating assertions; matching producer/repeat bytes was not scientific evidence.

This is a small publisher-example gate. The uncovered affiliation, complete
field-unit dictionary, spatial and current TEFA cases above remain open and are
reported as outside this gate's coverage. A passing example gate does not
certify every entity or dataset.
