# Dataset contracts and data releases

TEA Data keeps one source contract in `teadata/teadata_sources.yaml`. Each version has a path, declared reference period, reader settings, an ordered column schema, a value policy, and documentation references. Reviewed, column-only JSON resources under `teadata/schemas/` define the expected headers; their hashes bind the reviewed documentation bytes. A catalog year selects a version; it does not establish its reference period. TAPR fields such as attrition and mobility can refer to an earlier period than the report. Their field definitions carry explicit overrides.

A build records the files it actually reads. The runtime catalog is embedded in the snapshot and SQLite sidecars. `release.json` selects all four artifacts by content hash and one build identity. Runtime consumers use that catalog, rather than interpreting an old snapshot with the current YAML. Independent asset URLs, newest-file discovery, legacy pickle formats, and cache rebuilds that leave partial changes on failure have been retired.

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

A changed ordered header layout fails its column schema. Review the source dictionary, parser, period, units, masking rules, and joins before updating that resource and its review hash. Do not replace a schema simply to silence a failure. The processing receipt separates structural coverage and join checks from pending source review; structural success is not researcher-grade certification. Current processing starts from the retained curated inputs; recovering and validating every upstream download/transformation is the next dataset-by-dataset audit.

Repeat-build comparisons apply to the recorded producer environment. Spreadsheet numeric inference can differ by one floating-point unit between operating systems, even with the same Python package versions. The current dependency migration preserves the retained macOS producer's analytical values exactly; it does not establish cross-platform floating-point identity. Dataset intake must review numeric parsing explicitly rather than silently accept comparison tolerances or change conversion rules. The catalog currently records Python, platform, and selected package versions; native-library versions and installed-wheel identities remain additional provenance work.

The AskTED adapter discovers archive controls from publisher year/season labels, verifies the response attachment year/season and enrollment header periods, and records its code identity. It acquires evidence without activating a configured dataset. The HTTP client bounds transfers/retries and honors `Retry-After`, deferring acquisition if the requested wait exceeds its limit. Publisher archive labels alone do not establish a bitemporal snapshot; row update dates and enrollment periods require separate review.

The builder sets `PYTHONHASHSEED=0`, records code and lockfile content identity and software versions, uses deterministic entity IDs, and writes deterministic gzip framing. It stages the entire bundle, verifies artifact bindings and a processing receipt, and atomically replaces the release pointer only after success. A failure leaves the previous pointer active. Exact-byte reproducibility is gated by publication refusing conflicting artifact bytes for an existing build identity; source and environment identities must match when comparing runs.

`TEADATA_CONFIG`, `TEADATA_PROVENANCE_DIR`, and `TEADATA_OUTPUT_DIR` select alternate build locations. Defaults are portable. The default output is `teadata/.cache/`, the sole runtime release authority. Older tracked assets are archival rollback inputs and are not discovered automatically.

## Runtime and rollout

Set `TEADATA_RELEASE` to a local manifest or `TEADATA_RELEASE_URL` to an immutable remote manifest URL. A source checkout defaults to `teadata/.cache/release.json`. Wheels contain code, source contracts, and methodology documents; configure their release explicitly because data artifacts are distributed separately through Git LFS. Remote files are cached by content hash and verified before use. A process pins a configured remote release until restart; requests and cache rebuilds pin one release for their duration.

The application requires the coordinated library version, schema migrations, and release manifest. Publish and pin those together before deployment. Never merge the app against an older library tag. Preserve the previous deployed code and data bundle for rollback. The migration's missing joins, unreviewed documentation, unknown original download dates, and incomplete upstream lineage must remain visible; they prevent claiming this baseline has completed researcher-grade dataset certification.
