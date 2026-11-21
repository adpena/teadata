# TEA Data

TEA Data is an independent Python library for working with Texas public
education data. Use it to select district and campus cohorts, query published
observations, and export results with their source and measurement period.

## Export a published observation

In an existing uv project, add the exact software version used in this example.
Keep `uv.lock` with the analysis.

```bash
uv add "teadata==0.0.173"
```

Start in a shell without release overrides to use the packaged manifest pin.
Save the example as `texas-algebra.py` and run `uv run --locked texas-algebra.py`.
First use can fetch the required metadata and source files. Run it in a
directory where `texas-algebra-2025-26.zip` does not already exist.

```python
from teadata import citation, current_release

release = current_release()
measure = (
    "state_2026_staar_performance_eoc_algebra_i_"
    "all_students_percent_at_meets_gl_standard_or_above"
)
view = release.observations(
    "state_tapr_staar_end_of_course",
    measure,
    scope="state",
    school_year="2025-26",
)

print(view.to_df()[measure].tolist())  # ['51'], the publisher's raw percent token
print(release.manifest_sha256)
view.to_csv_bundle("texas-algebra-2025-26.zip")
print(citation(release).text)
```

TEA reports that 51% of its Texas Algebra I End-of-Course accountability
subset met or exceeded the grade-level standard in 2025–26. The source table
keeps the publisher's raw token as text. Use the [statewide comparison
API](querying.md#compare-a-published-measure-with-texas) for the reviewed
numeric comparison.

The ZIP contains `data.csv`, `provenance.json` with the exact selection and
publisher references, and `manifest.json` with both member hashes. Keep it with
your code and cite the publisher report, TEA Data software, and frozen data build.
[Data and software citations](data-releases.md#citing-tea-data)

## Choose a route

| Goal | Guide |
| --- | --- |
| Find a dataset, measurement year, or geographic scope | [Source integration inventory](source-integration-status.md) |
| Understand a population, masking rule, or series break | [Source coverage and methodology](source-coverage.md) |
| Select entities, filter a table, or compare with Texas | [Query guide](querying.md) |
| Export a series or materialize observations into SQL | [Observations and exports](enrichment.md#observations-across-school-years) |
| Reopen a historical release or work offline | [Release selection and caching](data-releases.md#runtime-and-rollout) |
| Prepare and validate a source | [Configuration](config.md) and [qualification](data-releases.md#qualifying-reviewed-native-bundles) |
