# Facility usage: 2025–26 source review

Source reviewed November 12–13, 2026 UTC; implementation checked November 16.
The two district source contracts are registered, independently qualified and
attached to the retained normal159 composition. Its 16 native source gates and
actual district namespace, DSL and export workflow passed. **Publication and
downstream adoption remain separate gates.** The [exact inventory](source-integration-status-0.0.159.json)
pins this unpublished composition; an older selected release does not gain
these observations automatically. This page documents the as-published source,
implemented access contract and remaining limitations.

## Population and observation grain

TEA's [collection page](https://tea.texas.gov/data-reports/financial-compliance/facility-usage-data-collection)
says charter schools are not currently required to report. The retained
[Facility Usage Report](https://tea4avcastro.tea.state.tx.us/fud/facility-usage-report.html)
contains a statewide all-districts workbook for **2025–26**. Its roster covers
ISDs, CISDs, CSDs and an MSD; it is not the statewide public campus or charter
roster. The collection page lists 2026–27 as coming soon, not as a published
data edition.

| Published submission status | Districts | Facility rows |
| --- | ---: | ---: |
| Submitted | 946 | 17,763 |
| Incomplete | 36 | No facility records displayed |
| Not submitted | 35 | No facility records displayed |
| Total collection roster | 1,017 | 17,763 |

The Facilities Data worksheet has 17,834 data rows: the 17,763 submitted facility
records and 71 status-only district rows. The latter have only district name,
district ID, map ESC region and submission status. Incomplete can mean a started
submission or data quality issues that prevented use; it is not interchangeable
with not submitted. Neither state means zero facilities. The collection does
not establish complete charter facility coverage. A charter's absence cannot be
converted into zero facilities or a missing required submission. The review did
not classify every campus in the bundled Lookup by charter status.

The observation is **a reported facility record belonging to a district**.
Properties, buildings and campuses are not interchangeable. FAQ questions 14
and 21 describe grouped address entries and separate building rows under the
same property ID. The data contain blank and repeated property IDs, repeated
organization codes and identical rows. Do not deduplicate by property, address,
campus or full row, and do not create additional district or campus entities.
A reproducible record locator needs the workbook revision, worksheet and
original worksheet row number; it is not a permanent physical-facility ID.

## What campus linkage can support

The workbook's Lookup worksheet contains 8,523 distinct nine-digit campus IDs
across the 1,017 collection districts. TEA's report metadata and legislative
Appendix A identify the AskTED reference date as October 2, 2026. TEA adds a
Campus Name (Campus ID) label to facility rows.
Matching the **entire supplied label** against this same workbook's lookup gives:

| Supplied label in a facility row | Rows | Interpretation |
| --- | ---: | --- |
| Exactly one matching campus label | 8,861 | 6,567 distinct campus IDs; potential explicit relationship |
| One `NA (campus ID)` label | 477 | ID was not found in the bundled campus lookup |
| Literal `NA` | 8,248 | No identified campus in the supplied label |
| Multiple or other labels | 177 | Retain the entire label; no single-campus assignment |

These counts check internal publisher consistency. They do not independently
verify campus existence, current status, location or the facility assignment.
The exact single matches have no district-ID conflict with the bundled lookup.
The other rows can reflect non-campus facilities, multiple campuses, historical
codes or unresolved input; absence from this lookup does not prove an invalid ID.
No fuzzy name matching, concatenation of guessed campus codes, or current
directory filtering is justified by these counts.

A later campus view can reference a reviewed facility relationship. It must
preserve the original facility observation and shared relationships, without
duplicating students/capacity as campus measures or allocating them among
campuses. The source does not supply a one-row-per-campus capacity series.

## Source fields and handling

All 32 source columns must remain attributable to their original header and
worksheet row. Columns A–AC are the 29 published submission fields, including
TEA's recoded categories. AD–AF are added by TEA. The Validation worksheet
provides allowed labels; it is not proof that every submitted value conforms.
The following handling is the implemented source contract. Public availability
depends on the selected normal release containing its qualified native child.

| Column | Publisher header | Meaning and handling |
| --- | --- | --- |
| A | Name of the district | District name from TEA's AskTED directory; retain alongside the ID, not as a join key. |
| B | County District Number | Six-digit district ID; normalize with the project's shared identifier rules and retain its raw token. |
| C | Contact Email Address | Excluded from all public tables, metadata and exports. Original publisher bytes remain private qualification evidence. |
| D | Type of Submission | Literal submission-method label, not publication or review approval. |
| E | Active used facility | Literal YES/NO/blank response; blank is not NO. |
| F | Property ID | Identifier text, possibly blank or repeated; never a campus ID or unique facility key. |
| G | Street Address | Reported facility address; no automatic geocoding or substitution for canonical site coordinates. |
| H | City, State | Literal reported address text; do not infer a separate standardized city or geography. |
| I | Zip Code | Identifier text; preserve leading zeros and source formatting. |
| J | TSDS/PEIMS Org Code | Literal local code or code list, including blank and 999; not automatically a nine-digit campus ID. |
| K | Type of Building (Recoded) | Publisher-recoded primary building classification. Keep separate from Facility Usage. |
| L | Number of Classrooms | Reported classroom quantity, not a class-size measure. |
| M | Original Year Completed | Reported construction-year attribute, not the observation year. |
| N | Year of Most Recent Renovation Completed | Reported renovation-year attribute, not the observation year; blank remains missing. |
| O | Does the facility house portables | Literal YES/NO/blank response; do not infer the portable count. |
| P | How many Portables | Reported portable quantity; blank is not zero. |
| Q | Square Footage | Reported total building area in square feet. |
| R | Acreage | Reported land area in acres. |
| S | Ownership Status | Literal OWN/RENT/SHARED/blank response. |
| T | Shared Usage | Current-cycle response about use by more than one campus; not a count of campuses. |
| U | Shared Building Type 1 (Recoded) | First publisher shared-building label; preserve free text as well as listed categories. |
| V | Shared Building Type 2 (Recoded) | Second label; do not collapse it into the primary building type. |
| W | Shared Building Type 3 (Recoded) | Third label; retain its position and literal value. |
| X | Shared Building Type 4 (Recoded) | Fourth label; retain its position and literal value. |
| Y | Facility Usage | Literal instructional/non-instructional/blank response; distinct from building type. |
| Z | Grades for this Facility | Literal grade list; no per-grade counts or allocation is provided. |
| AA | Maximum Instructional Capacity | Reported maximum students served at one time; distinct from adjusted capacity. |
| AB | Adjusted Instructional Capacity | Reported capacity adjusted for instructional/operational practices; retain negative, fractional and blank values as published, with quality limitations. |
| AC | Total Current Enrollment | Facility-reported students under mixed source rules; never an alias or replacement for canonical Enrollment or Membership. |
| AD | Campus Name (Campus ID) | Publisher-added AskTED match text; preserve NA and multiple labels without treating them as validated entity relationships. |
| AE | ESC Region (Map) | Publisher-added district map-region attribution, not the campus's physical ESC or another directory ESC field. |
| AF | Submission Status | Publisher district submission status; distinguish Submitted, Incomplete and Not Submitted. |

Blank, zero, literal NA and 999 must remain distinct. FAQ question 8 asks about a
missing organization code but answers in terms of property identification, so
it does not resolve a universal sentinel policy for either field. Do not apply
TAPR suppression rules or interpret every NA string as a masked value. The
published materials do not establish cell-specific confidentiality reasons for
blank cells.

Numeric cells require their original XML tokens and exact decimal interpretation.
The workbook contains fractional student counts and capacities, negative adjusted
capacities, and acreage tokens with up to **18 decimal places**. Web display
rounding is not the raw data. Do not truncate to integers, binary floats or an
assumed three-place precision. Round-trip fidelity does not establish that the
district reported the correct quantity. The reported construction years include
future years; do not silently substitute a year or discard those rows.

There are 152 distinct facility rows with a value in U–X outside the Validation
worksheet's literal shared-type list. Some are free-text facility descriptions.
Preserve them and expose the difference; an enum-only parser would lose or reject
published records. This is a source conformance observation, not a finding that
each such response is substantively wrong.

## Definitions, periods and version differences

Use the [2025–26 FAQ](https://tea.texas.gov/data-reports/financial-compliance/faq-facility-use-4-23-2026.pdf),
not the newer cycle's guide, for these data. Question 29 says enrollment is the
Fall Snapshot; question 27 specifies average daily students served for
instructional facilities that do not report enrollment in PEIMS. Therefore AC is
not one universally comparable campus census population, and no single exact
snapshot date can be assigned to every row. Questions 20–24 describe shared use,
building rows and capacity. Keep construction years, the collection school year,
AskTED date, prepared date and retrieval dates separate.

The [legislative report](https://tea.texas.gov/data-reports/financial-compliance/fud-legislative-report.pdf),
Appendix A (pages 8–9), says its figures use facility values TEA updated after
review. Appendix C (page 14) compares original and updated figures. The retained
bulk workbook still has these source cells:

| Workbook cell | District | Published value | Related report comparison |
| --- | --- | ---: | --- |
| AC16356 | Texarkana ISD, 019907, org 042 | 13,721,607 reported students | C1 original maximum 13,721,607; updated maximum 5,127 |
| AA11029; AB11029 blank | Merkel ISD, 221904, org 001 | 18,700 maximum capacity | C2 original maximum 18,700; updated maximum 6,000 |

The report statistics describe selected instructional facilities with enrollment
and capacity reported; they are not extrema over every workbook row. Its sample
changes from 7,219 to 7,218. An updated global maximum is **not a replacement
value for either identified cell**. No row-level correction map has been found
in the reviewed materials. Correction coverage for the bulk revision remains
unknown; do not claim every row is original, unchanged, corrected or verified.
Retain the values and this limitation; do not repair cells or back-solve updates.

The web report calculates enrollment as a percentage of maximum capacity for
instructional facilities. The legislative report uses adjusted capacity, falling
back to maximum when adjusted is zero or blank, and uses updated values. These
are different calculations and data revisions. Do not silently reproduce one
under the other's label or publish a replacement statewide utilization rate.

## Frozen evidence and integration gates

The HTML was retrieved November 12, 2026 at 23:57:31 UTC. Its embedded workbook
was decoded without rewriting the XLSX; there was no separate workbook HTTP
request. The workbook README says prepared October 16, 2026. The report metadata
and legislative Appendix A identify AskTED as of October 2, 2026.
Publication date, correction revision date and finality are
unknown. An HTTP Last-Modified value does not establish any of those dates.
The currently linked quick guide and FAQ describe 2026–27; they are retained as
separate references, not backported definitions or available observations.

| Retained input | SHA-256 |
| --- | --- |
| Report HTML | `4c1b808d8c3501a70a2205e3e554d5b1a57666afcd8a322fef85707123702789` |
| Embedded FUD_2025-2026_All_Districts.xlsx | `a71e0919f2451e2343be2600c8bccf2cea72fffa21641d4a241ce18c6528b944` |
| 2025–26 FAQ | `12abf52300f03c30aba96bdc3576561289ba8d2add01b3be71068f97ab5a4edc` |
| Legislative report | `7119730bfb3684b84cf309a752296a54ad498556e9b16ef4112e9d7f8b7b54e6` |

Source bytes, request times, response headers, hashes and extraction lineage are
retained through the existing ProvenanceStore. Source inspection covers every
worksheet row and header, status coverage, lookup consistency, numeric lexical
types and selected source cells. This is **not** whole-dataset semantic
qualification or independent confirmation of real-world facilities or students.

The implementation uses the existing source-only Config, schema4 native catalog,
lazy district namespaces, DSL, series and provenance/export paths. It creates
`facility_usage` with all 17,834 worksheet rows and `facility_usage_submission`
with 1,017 district rows. Both use publisher edition `2026`, district scope and
the 2025–26 collection school year. Submission rows retain the unique publisher
name, district number, map region and AF status only after proving agreement
across every source row in that district. Group row counts, bounds and a hash
of the original row numbers provide lineage; the immutable processing chain
binds the submission table to the facility artifact. Submission field selection
opens only the submission leaf.

Strings retain their exact raw tokens and normalize only empty/missing cells to
null, apart from the shared district identifier normalization. They use the
existing raw-only support contract. Numeric fields retain raw tokens and
`observed`/`missing` status columns. Source cell-type context distinguishes
absent XML cells, empty cells and observed values. Numeric zeros remain observed.
Area and capacity values use fixed Arrow decimal types selected from the complete
XML lexeme scan: Q `(19,12)`, R `(24,18)`, AA/AB `(18,13)` and AC `(9,1)`.
The original worksheet coordinate and row number remain attributable through
the recipe and row context. Validation conformance flags retain 169 out-of-list
cells in 152 rows, without changing their values or treating them as missing.

For a release-bound district whose selected release contains this child:

```python
district.available_datasets()
district.facility_usage_submission.submission_status
district.dataset("facility_usage").variables()
records = district.facility_usage.select("acreage", "maximum_instructional_capacity", school_year=2026)
records.to_df()
records.to_csv("facility-records.csv")
records.to_excel("facility-records.xlsx")
```

The ordinary `.dataset(...)` spelling also avoids entity metadata name
collisions. Entity `.meta` remains unchanged; facility arrays are not copied
into it. A scalar access fails helpfully when a district has multiple facility
rows. Use `.select(...).query(...)` with original row locators to select records.
The exact-edition `release.table(key)` path can also select identity fields.
The existing `Query.observations` and `release.series` paths preserve source
grain and raw support. Excel stores exact decimals as text. For SQLite series
exports, use Text/String destination columns for decimals; SQLite Numeric is
not an exact decimal transport. A status-only district has null facility facts;
a district outside the collection has no published rows. Neither means zero.

The offline producer is `uv run python -m teadata.scripts.refresh_facility_usage
--store <retained-store> --output <new-child-directory>` (one shell command).
It requires the retained four inputs and their original acquisition/extraction
records, plus the pinned public schema reference. It does not download data.
Independent qualification reads the original XML without using the producer's
parser or normalizer, compares every public cell and numeric status, checks
grouping lineage, and checks 12 frozen publisher examples. Its scope is exact
as-published district records, not corrected statistics or confirmation of
real-world facilities.

Qualification and qualification-receipt reuse require both the publisher
evidence and child-adjacent processing store; missing or altered evidence fails
closed. Public reading and export require only the descriptor-reachable
manifest, edition metadata, column schemas and Parquet files. Raw XLSX/HTML/PDF
objects and the whole private `provenance` directory are excluded from the
public asset allowlist. Normal release composition must use this same reviewed
qualification gate and the actual final producer identities; a development
child is never relabeled as a later release.

Campus relationship admission, derived utilization, statewide aggregates,
corrected-series claims and scalar enrollment/capacity aliases remain on hold.
The as-published observations can be preserved without endorsing their
accuracy; any intentional numeric correction needs identified evidence and
separate authorization. No application profile activation follows from library
registration or source qualification.
