# Federal Report Card spending per pupil

The FY2025 source uses Part (x) of TEA's 2024–25 Federal Report Card. It reports
actual expenditures in dollars per pupil, with state and local and federal
funding shown separately. The published site and district allocation values
remain separate. TEA reports the enrollment used alongside these values.
The report's figures are specific to FRC and use a different population and
expenditure scope from PEIMS Actual PWR and TAPR. [TEA FRC definitions](https://tea.texas.gov/texas-schools/accountability/academic-accountability/performance-reporting/2024-25-frc-definitions.pdf)

| Scope | Rows | Published source |
| --- | ---: | --- |
| Texas | 1 | [State report, Part (x)](../provenance/records/4e46558c32fa4cf97baef5dc41a1f3c8168ad06c72a65cd734177c5aae572937.json) |
| District | 1,202 | [Statewide district summary](../provenance/records/29b8ca8721fdb8c512b96261b7ee8aa43f0d4cda3c3a2051fdaf6a036d98160e.json) |
| Campus | 8,866 | [Statewide campus summary](../provenance/records/3a120c76f02aea19904a9e5ff6bb0c6d835de8b598140afd732e24bd5a7fd9af.json) |

The state observation is the direct published total with enrollment of
5,530,672 and spending of $12,147 per pupil. The district table's enrollments
sum to 5,529,048 and the campus table's to 5,516,204. Use the direct state
observation for comparisons. The source reports dollar rates at whole-dollar
display precision, so the published components can differ from their total
after rounding. Original amounts and the unrounded numerators are outside the
published PPE totals contract.

The source-only crosswalk records the repeated publisher labels in their
physical positions. Each leaf provides `reported_enrollment` in students
and seven expenditure fields in `currency_per_pupil`. The campus leaf
retains both the literal district group and the nine-digit campus ID. Blank
currency cells have missing status and a null numeric value. The tokens `$0`,
`$-0` and negative amounts remain observed, with their original text available
through the native catalog's column support. `frc_source_table_row` locates
the parsed HTML row in Part (x) for Texas, or in the bulk table for districts
and campuses. It counts HTML table rows from one.

## Build and inspect the candidate

Use the exact acquisition objects and receipt closures referenced in the
packaged Config. The build runs offline and writes a new unpublished native
catalog. It compares eight cells each from the independent Hitchcock ISD and
Crosby Middle reports with the corresponding bulk rows. It also verifies
every written Arrow cell against the projected source before installing the
candidate. These publisher controls cover the named report selections.

```sh
FRC_CONFIG=teadata/teadata_sources.yaml
uv run --locked --offline python -m teadata.scripts.refresh_frc_ppe \
  --store "$TEADATA_PROVENANCE_DIR" \
  --config "$FRC_CONFIG" --config-sha256 "$(shasum -a 256 "$FRC_CONFIG" | cut -d ' ' -f1)" \
  --scope state --scope district --scope campus --output "$FRC_CANDIDATE_DIR"
```

Pin the resulting `release.json` and open it with `DataRelease` in offline
mode. Use `catalog.editions()` to select the requested exact edition and
scope, then `catalog.resolve_artifacts()` for the desired fields. The read
plan includes the identifier, raw value, status and source row needed to
audit the selected value. This candidate is available for explicit native
reads. Dataset activation and release publication are separate steps.
