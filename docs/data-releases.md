# Dataset contracts and data releases

TEA Data keeps one source contract in `teadata/teadata_sources.yaml`. Each version has a path, declared reference period, reader settings and schema fingerprint, value policy, and documentation references. A catalog year selects a version; it does not establish its reference period. TAPR fields such as attrition and mobility can refer to an earlier period than the report. Their field definitions carry explicit overrides.

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
```

Values carry `observed`, `suppressed`, `missing`, `not_applicable`, or `invalid` status, the raw token, reason, policy identifier, and any documented bounds. Zero is observed. A negative financial adjustment can be observed. A period has a meaning only under its source policy. Missing and invalid values do not become suppressed values or zeros. Public percentages are not used to reconstruct protected counts.

The TAPR profile policy distinguishes primary suppression (`-1`), complementary suppression (`-3`), and invalid or unreasonable reported results (`-2`). There are no universal numeric bounds for those tokens. Year-specific masking references are retained with the contract. Unreviewed masking methodology blocks a build using that policy. Generic dataset reference periods describe the report; complete field-level dictionary review remains part of each dataset intake.

## Charter transfer estimates

The model remains `observed sum + 5 × suppressed flow count`, at origin-campus/destination-campus grain. Observed totals, protected counts, modeled contributions, and the model identifier stay separate. Five is the historical midpoint scenario, not a TEA disclosure bound or a confidence interval. Larger counts may also be masked. Missing, invalid, or unresolved flows are not silently assigned five. See the [canonical methodology](../teadata/methodologies/charter-transfer-midpoint.md).

Unresolved entity joins are retained. Affected complete totals and estimates are unavailable; the observed partial sum remains inspectable. The initial migration found 59 transfer rows for Rocketship Explore EL (`220820102`) missing from the spatial/entity roster, although the campus is present in TAPR. That is an explicit dataset-review item, not a completed data refresh.

## Rebuild from retained inputs

```bash
uv sync --locked --all-extras
# Set TEADATA_DATA_DIR if data is outside teadata/data/.
uv run python -m teadata.sources teadata/teadata_sources.yaml --fetch-references
uv run pytest
uv run python -m teadata.load_data
```

Source observation is explicit. It archives existing bytes without inventing original download dates. Reference HTTP requests happen only with `--fetch-references`; builds make no implicit network requests. Local methodology files are versioned and archived. Receipts under `provenance/records/` are immutable and tracked; raw content-addressed objects under `provenance/objects/` are retained separately. Original acquisition URLs, download dates, licensing, and upstream transformation lineage remain unknown where migration evidence is insufficient. Local observation dates are not freshness evidence.

A changed header layout fails its schema fingerprint. Review the source dictionary, parser, period, units, masking rules, and joins before updating that contract. Do not refresh a fingerprint simply to silence a failure. Current processing starts from the retained curated inputs; recovering and validating every upstream download/transformation is the next dataset-by-dataset audit.

The builder sets `PYTHONHASHSEED=0`, records code and lockfile content identity and software versions, uses deterministic entity IDs, and writes deterministic gzip framing. It stages the entire bundle, verifies artifact bindings and a processing receipt, and atomically replaces the release pointer only after success. A failure leaves the previous pointer active. Exact-byte reproducibility is gated by publication refusing conflicting artifact bytes for an existing build identity; source and environment identities must match when comparing runs.

`TEADATA_CONFIG`, `TEADATA_PROVENANCE_DIR`, and `TEADATA_OUTPUT_DIR` select alternate build locations. Defaults are portable. The default output is `teadata/.cache/`, the sole runtime release authority. Older tracked assets are archival rollback inputs and are not discovered automatically.

## Runtime and rollout

Set `TEADATA_RELEASE` to a local manifest or `TEADATA_RELEASE_URL` to an immutable remote manifest URL. A source checkout defaults to `teadata/.cache/release.json`. Wheels contain code, source contracts, and methodology documents; configure their release explicitly because data artifacts are distributed separately through Git LFS. Remote files are cached by content hash and verified before use. A process pins a configured remote release until restart; requests and cache rebuilds pin one release for their duration.

The application requires the coordinated library version, schema migrations, and release manifest. Publish and pin those together before deployment. Never merge the app against an older library tag. Preserve the previous deployed code and data bundle for rollback. The migration's missing joins, unreviewed documentation, unknown original download dates, and incomplete upstream lineage must remain visible; they prevent claiming this baseline has completed researcher-grade dataset certification.
