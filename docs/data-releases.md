# Dataset contracts and data releases

TEA Data keeps one source contract in `teadata/teadata_sources.yaml`. Each version has a path, declared reference period, reader settings, an ordered column schema, a value policy, and documentation references. Reviewed, column-only JSON resources under `teadata/schemas/` define the expected headers; their hashes bind the reviewed documentation bytes. A catalog year selects a version; it does not establish its reference period. TAPR fields such as attrition and mobility can refer to an earlier period than the report. Their field definitions carry explicit overrides.

A build records the files it actually reads. The runtime catalog is embedded in the snapshot and SQLite sidecars. `release.json` selects the source and analytical artifacts by content hash and one build identity. Runtime consumers use that catalog, rather than interpreting an old snapshot with the current YAML. Independent asset URLs, newest-file discovery, legacy pickle formats, and cache rebuilds that leave partial changes on failure have been retired.

## Researcher API

```python
from teadata import current_release, DataEngine

release = current_release()
catalog = release.catalog
catalog.metadata("campus_tapr_student_staff_profile").period.label
catalog.metadata("campus_peims_financials").period.label
catalog.metric_period("tapr.student_membership_attrition_all_students_percent")
catalog.acquisitions("campus_transfer_reports")
release.processing["record"]["validation"]

engine = DataEngine.load_default(indexes_enabled=False)
campus = engine.campus_by_number("101902001")
campus.metric("tapr.student_enrollment_special_ed_percent").to_dict()
campus.metric("financials.students_w_disabilities_af_per_student").to_dict()
catalog.charter_transfer_summary(campus)
catalog.measurements_frame([campus], ["enrollment", "accountability.student_count"])
```

Values carry `observed`, `suppressed`, `missing`, `not_applicable`, or `invalid` status, the raw token, reason, policy identifier, and any documented bounds. Bound measurements also identify their release, dataset, physical field, period, and input receipts. Detached caller values are explicitly marked `caller_supplied`; attaching metadata does not prove their input bytes came from a released entity. Known entities and bound measurements from another release are rejected. Zero is observed. A negative financial adjustment can be observed. A period has a meaning only under its source policy. Missing and invalid values do not become suppressed values or zeros. Public percentages are not used to reconstruct protected counts.

Catalog schema 2 stores `metric_bindings` with the release: canonical names and aliases resolve to source fields, populations, units, and bounds without consulting current configuration. Old schema-1 bundles require a producer rebuild; they are never relabeled under new rules. `measure_field(raw, physical_field, scope="campus", entity=campus)` applies the same source rules and population check to a governed metadata field, returning `None` for opaque or ambiguous fields. `measurements_frame()` retains value, status, raw token, reason, unit, period, source, and build identity in separate columns.

Canonical public enrollment uses the TAPR student enrollment field for both campuses and districts. The accountability report's Number of Students is a separate `accountability.student_count` population and must not replace enrollment. Public-source metrics are not applicable to private schools. A metric without its own period inherits the dataset period; an explicitly unknown period stays unknown.

### Exact retained-source parity

An optional offline check compares every raw cell in explicitly selected,
complete TAPR raw-string native leaves against their exact retained original
CSV and prepared Parquet. Select the existing release, provenance store and
exact edition IDs; repeat `--source-edition` to check a finite set:

```sh
uv run --locked python -m teadata.verification --kind library \
  --source-parity /path/to/release.json --source-store /path/to/provenance \
  --source-edition EXACT_EDITION_ID
```

The catalog, reviewed crosswalk and historical processing recipes select the
source bytes, fields, layout, keys and reader budgets. Native joins compare
all rows and fields without assuming row order; original IDs, whitespace,
zeros and masking tokens remain exact. Reports name source receipts/URLs,
field/row/cell coverage, key differences and bounded, explicitly marked
examples. No current-code hash is substituted for a recorded historical
producer identity. Selected artifact integrity remains checked by the existing
release reader.

The checker reads the original once and each target stage once, then uses one
native key-to-row index and per-column comparisons. Its default 128 MiB
`working_bytes` budget is split between retained original, one target at a time,
and alignment/comparison buffers. Larger inputs remain uncovered/HOLD; increasing
this explicit core-API budget does not change the comparison rules. Source
decoding retains its recorded batch/key limits. These are Arrow-buffer admission
limits, not a process RSS guarantee.

Prepared TAPR inputs may retain a different historical global or script digest.
Those identities remain provenance; matching today's script alone would not
prove unchanged transitive dependencies. Before any native leaf staging/writer,
the canonical producer replays each exact original and bound dictionary through
the current bounded reader and compares every column, row group and completed
layout/EOF with the retained preparation. It writes no replacement preparation.
Receipt, input, crosswalk, row/key and reference gates remain strict. The new
leaf records current raw revalidation separately while retaining the actual
historical preparation identities. Any disagreement blocks; this raw readback
does not confer scientific qualification or a user-approved exception.

`PASS_RAW_PARITY` establishes only the named raw fidelity scope. Typed or
normalized leaves, unsupported readers and missing lineage are uncovered/HOLD;
they are never counted as checked. Scientific mappings, periods, units, policy
applicability, source authenticity and publisher correctness remain separate
qualifications. Existing independent publisher goldens and ordinary release
checks remain unchanged.

Every unexplained raw, key or contract mismatch blocks qualification. This
checker supports no exceptions, tolerances or automatic golden updates.
Repairing ETL to match the publisher needs no exception. A future intentional
difference would require specific user approval of the identified anomaly,
evidence and repair; general authorization to fix data is insufficient. E3's
printed-versus-TEA research comparisons are not ETL waivers.

### Native fields with retained technical support

An optional `column_support` in a native leaf contract separates real reviewed
publisher-header fields from physical raw/status/row-context columns. Each
`fields` entry names a normalized public field and its exact `raw` companion,
optional `status`, and optional field-specific `auxiliary` columns. Ordered
`context` columns carry only shared row/grain/locator information. These names
form one disjoint partition with the public fields and account for the complete
physical Arrow schema. Technical columns receive no invented publisher codes,
labels, positions, or additional research-variable entries. Extracting a district
ID from a combined name/number label is a recorded transformation; the original
label remains in its raw companion.

For these leaves, `variables()` and `field_metadata()` return detached `support`
pointers. `resolve_artifacts()` automatically selects a requested normalized
field's raw/status/auxiliary companions, required row context, and keys. It does
not select unrelated fields' support. Normalized count fields require both raw
and status columns: nullable values alone cannot distinguish suppression from
missing data. Given an exact selected `EditionKey` and count field:

```python
from teadata.values import measure

variable = next(row for row in catalog.variables(edition=key)
                if row["field"] == count_field)
plan, = catalog.resolve_artifacts([(key, count_field)])
table = release.analytical_dataset(plan["artifact"].identity).to_table(
    columns=list(plan["columns"]))
support = variable["support"]
raw = table[support["raw"]][0].as_py()
status = table[support["status"]][0].as_py()
value = measure(raw, catalog.policy(key.dataset, edition=key),
                unit="count", minimum=0)
assert value.status.value == status
```

The existing value policy remains the interpretation authority. Zero remains
observed; a suppressed token and a missing token may both have a null normalized
value and retain distinct statuses. Source-preparation/producer qualification
binds those transformations to the pinned processing recipe and inputs. The
native writer checks status vocabulary, value-presence consistency, and support
types; it does not invent or reapply a second masking policy. Publisher totals,
details and separators still require their explicit row-kind scope before any
aggregation. Schema3/4 scalar measurement APIs remain unavailable; selected
native projection and canonical `values.measure(raw, policy, ...)` are the
documented route. Existing leaves without `column_support` retain their original
contract bytes, field discovery, and projection behavior. This contract support
does not itself activate transfer datasets or qualify a new release.

### Publisher source qualifiers

`DatasetMetadata.source_qualifiers` is optional, frozen source metadata. It
distinguishes the `sof`, `peims_financial`, `accountability`, and
`ccmr_outcomes_bonus` families without
creating another provenance registry. A known publisher state names an existing
reviewed source reference and its retained acquisition evidence. The ordinary
catalog/reference gates verify that binding; storing a label does not establish
external field correctness or successful release qualification.

For an exact selected edition with these declarations:

```python
metadata = catalog.metadata(key.dataset, edition=key)
qualifiers = metadata.source_qualifiers
observations = catalog.acquisitions(key.dataset, edition=key)
field = catalog.field_metadata(selected_field, edition=key)
if qualifiers is not None:
    publisher_status = qualifiers.publisher_release_status
    publication_run = qualifiers.publication_run_id
    row_run_column = qualifiers.row_run_column
    financial_basis = qualifiers.financial_basis
    graduate_class_year = qualifiers.graduate_class_year
```

The same detached qualifier data appears in `metadata.to_dict()`, acquisition
`source_metadata`, and `field["metadata"]`. SOF publication run and the actual
per-row run are distinct: mixed row runs are retained, never rewritten to the
advertised publication run. A declared `row_run_column` must name a real
publisher-bound physical field or existing shared leaf context. Selecting a
field automatically projects that column, and `field["support"]["row_run_column"]`
identifies it. It does not add a key, copy cells into entity metadata, or load
unrelated fields. Source editions without a declared row-run column do not
acquire one by inference.

PEIMS `financial_basis` is explicitly `actual` or `budget`; SOF and accountability
publisher release wording is retained verbatim, including preliminary, near-final
or other source-specific labels. It is not a universal status enum or our
processing PASS/HOLD. Publication run IDs are explicit strings, not numerical
ordering rules. Edition identity, source periods, publisher release state,
acquisition time and processing qualification remain separate. Unknown values
stay unknown. Earlier immutable metadata, receipts and leaf semantic hashes
retain their original bytes when `source_qualifiers` is absent. These APIs do
not activate or qualify a new financial dataset.

For `ccmr_outcomes_bonus`, `graduate_class_year` is an explicitly declared
annual-graduate class year, validated as an integer from 1 to 9999. It requires
the existing reviewed source reference, as does known publisher release wording
such as `Early` or `Final`. It is distinct from a four-year graduation cohort,
the outcome-data windows, a calendar observation year and a funding fiscal year;
none of those are inferred from it. Keep Early and Final source contexts
separate, and retain an unknown measurement period when its window is not
established. The family does not accept SOF run or PEIMS financial-basis fields.
An absent class year remains omitted, including on existing qualifier families;
their serialized metadata and leaf semantic hashes remain unchanged.

## File organization and names

TEA Data uses the following project conventions. They apply the consistency,
descriptive context, documented organization and unique identifiers discussed by
[Harvard](https://guides.library.harvard.edu/hgsefilenaming),
[Princeton](https://researchdata.princeton.edu/research-lifecycle-guide/data-acquisition/file-organization)
and [Stanford](https://guides.library.stanford.edu/c.php?g=1186700); those guides
describe best practices, not one mandated universal naming schema.

| Namespace | Existing name or identity | Authority |
| --- | --- | --- |
| Retained original bytes | `provenance/objects/<byte-SHA-256>` | Exact bytes; descriptive discovery comes from the receipt and catalog. |
| Acquisition and processing receipts | `provenance/records/<record-SHA-256>.json` | Canonical receipt identity, including its actual observation and typed parents. |
| Logical source editions | `EditionKey(dataset, edition, revision_sha256, scope, table, grain)` | Qualified source identity; the revision binds ordered input bytes. |
| Native leaves and external metadata | `<leaf-contract-SHA-256>.parquet`; `<resource-byte-SHA-256>.json` | `ArtifactRef` separates contract identity from physical byte identity; metadata descriptors bind their own bytes. |
| Release bundles | `<role>_<build_id>.pkl.gz`, `.sqlite` or `.parquet`, beside `release.json` | The release manifest binds roles, exact names, bytes and one build identity. |

Opaque internal CAS names remain useful unique identifiers. Dataset titles,
periods, publisher state, source URLs and exact edition selections in the same
catalog supply their descriptive context. An artifact's retained basename may
be chosen during acquisition; it is not automatically the publisher's original
attachment name. Retained URLs and response headers provide separate evidence.

Use ISO dates such as `YYYY-MM-DD` when a known date is needed, with its meaning
documented. Unknown publication dates stay unknown; an acquisition timestamp
must not supply one. A mixed-period release snapshot has no implied global
source year. Future display or download labels should derive from the existing
`EditionKey` and `DatasetMetadata`, using the existing publication name/location
guards. These conventions introduce no name registry or aliases and do not
rename datasets, retained evidence or published assets.

## Private school-finder listings

The private producer selects the latest curated `tefa` edition independently of
the public build year. Catalog key `2027` identifies the official Comptroller
school-finder snapshot for the stated 2026–27 program school year. It includes
only records explicitly tagged `Schools`, `Private School`, and `EFA-Eligible`;
these are directory listings, including pre-K and online providers. They do not
establish one listing per physical campus, all Texas private schools, actual
student enrollment, payments, approval dates, or governing public districts.
The snapshot's actual download time is retained separately from unknown
publisher publication/revision dates, upstream as-of time, and reuse license.

One `reader.columns` contract preserves all 16 publisher fields under
`tefa_listing_*` names, including original arrays, nested objects, nulls, and
coordinate values. Exact source IDs and names also project into the existing
`Campus.source_id` and `Campus.name` attributes; the latter uses the model's
existing uppercase presentation normalization. The namespaced source name
retains its original case. IDs remain opaque and
case-sensitive in the `tefa` namespace; matching names or addresses do not merge
listings or prove continuity with older identifiers. Common address, website,
and display-grade presentation fields derive only from those source fields.

An online-only listing has no physical map point even when the source supplies
coordinates. Local-capable listings use valid supplied coordinates without
geocoding, axis swapping, or sign repair. A Texas address with nonnegative
longitude or nonpositive latitude is quarantined from mapping; its raw
coordinates remain available. Missing, online-only, invalid, and unverified
source-provided points have distinct `private_listing_location_status` values.
Unrecognized or reversed display-grade ranges remain unknown in `grade_range`,
with the original label and `private_listing_grade_status` retained. Private
listings remain outside public TAPR, accountability, and finance populations.

The legacy `2025` spreadsheet remains accessible only through its explicit
`source` reader profile for historical research. Its original acquisition URL,
download date, and reference period remain unknown. It is not a fallback for the
official roster and is not retroactively attributed to the Comptroller. Older
released catalogs retain their original source contract and identities.

## Arrow analytical datasets

New schema-2 releases materialize the canonical scalar catalog once, from the
completed immutable entities SQLite artifact. They add campus and district
Parquet cells, named entity attributes, and a source/definition table. Source
hydration, canonical evaluation, and Arrow writing are timed separately in the
processing receipt's `diagnostics.analytical`. Stable source/count checks remain
in `validation.analytical`; measured runtime durations are per-run evidence,
not deterministic scientific validation. No second masking interpreter runs
during a scan.

```python
import pyarrow.dataset as ds
from teadata import current_release
from teadata.analytical import measurement_from_row

release = current_release()
cells = release.analytical_dataset("campus")  # ordinary Arrow Dataset
selection = (ds.field("kind") == "metric") & (ds.field("name") == "enrollment")
for batch in cells.to_batches(
    columns=["entity_id", "value_integer", "status", "reason", "dataset", "period"],
    filter=selection, batch_size=4096,
):
    pass  # pass a standard Arrow batch to pandas, Polars or a database loader
sources = release.analytical_dataset("sources")
attributes = release.analytical_dataset("campus_attributes")
path = release.artifact("campus_analytics")  # standard Parquet path
catalog.field_metadata(catalog.source_for_metric("enrollment", "campus")[1])
```

Cells use named `field`/`metric`, `dataset`, report `dataset_period`, effective
field `period`, `field_year`, `status`, `reason`, policy and unit columns. Source
rows include acquisition dates and URLs, dictionaries, glossaries and masking
references as native Arrow structs/lists, plus complete archival receipt JSON.
Known opaque fields retain catalog provenance; an unbound field has an unknown
period rather than a year inferred from its name. Derived attributes bind the
entities artifact, without inventing per-attribute source periods.

Raw, interpreted value, and presentation columns have explicit typed lanes.
Integers fitting Int64 remain native integers; wider integers use a separately
tagged canonical decimal lane, never a float. Raw dict/list JSON preserves
insertion order and nested types. Signed floating zero and integer versus float
disclosure bounds survive transport. `raw_present` means actual source-key
membership for field rows and is null for metric rows, whose absent-key
attribute fallback is evaluated by the canonical scalar API. Private-school
not-applicable results can retain the actual source raw cell while their sparse
Measurement keeps its canonical null raw value. `measurement_from_row()`
materializes that scalar object only when requested; it does not reclassify.

Source-boundary values must be normalized JSON types accepted by EntityStore.
Arbitrary Python objects remain supported only through the scalar APIs. There
is no object-dtype column or request-time scalar fallback for a missing read
model. Older four-artifact schema-1 releases remain readable through source
APIs; explicit analytical requests require a rebuilt release.

Each Parquet footer binds schema version, role, build ID, catalog checksum,
entities checksum and actual producer code/lock/runtime inputs, separately from
the source catalog's processor ancestry. Catalog, enumeration and hydration use
one store-owned immutable SQLite connection with percent-safe file URIs. Before
publication the producer refuses WAL/journal sidecars or changed main-file
signatures/checksums. EntityStore uses one strict JSON decoder everywhere:
nullable SQL/JSON null becomes empty metadata, while malformed/nonobject JSON,
nonfinite numbers and unsupported integer conversion limits raise ContractError
rather than silently dropping source keys. The final manifest owns child checksums;
children do not contain their final manifest checksum, avoiding a hash cycle.
Verification rejects mixed sources, schemas, roles and row counts before
returning a standard Arrow Dataset. Release-local ordinals are optimization
columns; named public columns suffice for ordinary research queries.

The TAPR profile policy distinguishes primary suppression (`-1`), complementary suppression (`-3`), and invalid or unreasonable reported results (`-2`). There are no universal numeric bounds for those tokens. Year-specific masking references are retained with the contract. Unreviewed masking methodology blocks a build using that policy. Generic dataset reference periods describe the report; complete field-level dictionary review remains part of each dataset intake.

## Charter transfer estimates

The model remains `observed sum + 5 × suppressed flow count`, at origin-campus/destination-campus grain. Observed totals, protected counts, modeled contributions, and the model identifier stay separate. Five is the historical midpoint scenario, not a TEA disclosure bound or a confidence interval. Larger counts may also be masked. Missing, invalid, or unresolved flows are not silently assigned five. See the [versioned methodology](https://github.com/adpena/teadata/blob/c5cd4fe8b24d61bdcfe0f92faad3e7074f1d34d2/teadata/methodologies/charter-transfer-midpoint.md).

Unresolved entity joins are retained for investigation and block release publication. Affected complete totals and estimates are unavailable; the observed partial sum remains inspectable. The initial migration found 59 transfer rows for Rocketship Explore EL (`220820102`) absent from the spatial roster despite a matching TAPR record. The reporting roster now supplies historical entity identity and classification independently of geometry. Missing geometry does not discard analytical rows or invent coordinates. This repair does not establish an active/closed date or complete the dataset refresh.

## Rebuild from retained inputs

```bash
uv sync --locked --all-extras
# Set TEADATA_DATA_DIR if data is outside teadata/data/.
uv run python -m teadata.sources teadata/teadata_sources.yaml --all-versions --fetch-references
scripts/check
uv run python -m teadata.load_data
```

Source observation is explicit. It archives existing bytes without inventing original download dates. Metadata improvements preserve earlier actual download receipts for the same bytes alongside the new observation. Reference HTTP requests happen only with `--fetch-references`; builds make no implicit network requests. Local methodology files are versioned and archived. Changing reviewed methodology bytes requires explicit review and a changed `reviewed_sha256`; observation cannot approve that change. Receipts under `provenance/records/` are immutable and tracked; raw content-addressed objects under `provenance/objects/` are retained separately. Original acquisition URLs, download dates, licensing, and upstream transformation lineage remain unknown where migration evidence is insufficient. Local observation dates are not freshness evidence.

New retained-format builds embed the complete selected receipt graph, including
historical acquisition parents, nested reference receipts and processing inputs.
The explicit release qualification gate verifies that embedded graph with the
same canonical record checker before opening release artifacts. This checks
record identities, typed references and graph integrity; it does not establish
historical artifact availability or publisher authenticity. Existing consumer
constructors remain unchanged, and original receipt bytes and unknown timestamps
are preserved.

A changed ordered header layout fails its column schema. Review the source dictionary, parser, period, units, masking rules, and joins before updating that resource and its review hash. Do not replace a schema simply to silence a failure. The processing receipt separates structural coverage and join checks from pending source review; structural success is not researcher-grade certification. Current processing starts from the retained curated inputs; recovering and validating every upstream download/transformation is the next dataset-by-dataset audit.

Repeat-build comparisons apply to the recorded producer environment. Spreadsheet numeric inference can differ by one floating-point unit between operating systems, even with the same Python package versions. The current dependency migration preserves the retained macOS producer's analytical values exactly; it does not establish cross-platform floating-point identity. Dataset intake must review numeric parsing explicitly rather than silently accept comparison tolerances or change conversion rules. The catalog currently records Python, platform, and selected package versions; native-library versions and installed-wheel identities remain additional provenance work.

The AskTED adapter discovers archive controls from publisher year/season labels, verifies the response attachment year/season and enrollment header periods, and records its code identity. It acquires evidence without activating a configured dataset. The HTTP client bounds transfers/retries and honors `Retry-After`, deferring acquisition if the requested wait exceeds its limit. Publisher archive labels alone do not establish a bitemporal snapshot; row update dates and enrollment periods require separate review.

The builder sets `PYTHONHASHSEED=0`, records code and lockfile content identity and software versions, uses deterministic entity IDs, and writes deterministic gzip framing. It stages the entire bundle, verifies artifact bindings and a processing receipt, and atomically replaces the release pointer only after success. A failure leaves the previous pointer active. Exact-byte reproducibility is gated by publication refusing conflicting artifact bytes for an existing build identity; source and environment identities must match when comparing runs.

`TEADATA_CONFIG`, `TEADATA_PROVENANCE_DIR`, and `TEADATA_OUTPUT_DIR` select alternate build locations. Defaults are portable. The default output is `teadata/.cache/`, the sole runtime release authority. Older tracked assets are archival rollback inputs and are not discovered automatically.

## Runtime and rollout

Set `TEADATA_RELEASE` to a local manifest or `TEADATA_RELEASE_URL` to a version-pinned remote manifest URL. A source checkout defaults to `teadata/.cache/release.json` and must hydrate its Git LFS artifacts. Wheels contain code, source contracts, and methodology documents; configure their data release explicitly. For HTTP delivery, publish the unchanged `release.json` and all of its hash-bound artifacts as adjacent files under one base URL, such as a GitHub Release asset directory. Raw GitHub URLs for Git LFS files serve pointer text rather than artifact bytes, so they fail checksum validation; the reader does not combine a raw manifest URL with a separate media base. Remote files are cached by content hash and verified before use. A process pins a configured remote release until restart; requests and cache rebuilds pin one release for their duration.

The application requires the coordinated library version, schema migrations, and release manifest. Publish and pin those together before deployment. Never merge the app against an older library tag. Preserve the previous deployed code and data bundle for rollback. The migration's missing joins, unreviewed documentation, unknown original download dates, and incomplete upstream lineage must remain visible; they prevent claiming this baseline has completed researcher-grade dataset certification.
