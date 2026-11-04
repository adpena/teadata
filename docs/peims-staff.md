# PEIMS Staff FTE Counts and Salary

`peims_staff_fte_and_salary` registers the October PEIMS staff reports for
2022–23 through 2025–26. Each year has a separate Texas report and district
report. District rows retain the exact six-digit district identifier and the
publisher's literal staff category and role. Texas rows retain the published
state scope. Category and role are part of the key. The single space in the
category of certain publisher totals remains part of their identity. Absent
districts and roles have no row. [TEA report selector and criteria](https://rptsvr1.tea.texas.gov/adhocrpt/adpeb.html) ·
[TEA release listing](https://tea.texas.gov/data-reports/student-data/standard-reports/peims-standard-reports)

The catalog retains each year's criteria and role definitions. [2022–23 criteria](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=adhoc.std_download_selected_report.sas&rpt_subject=about&fname=abtstf23&ftype=html) ·
[2023–24 criteria](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=adhoc.std_download_selected_report.sas&rpt_subject=about&fname=abtstf24&ftype=html) ·
[2024–25 criteria](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=adhoc.std_download_selected_report.sas&rpt_subject=about&fname=abtstf25&ftype=html) ·
[2025–26 criteria](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=adhoc.std_download_selected_report.sas&rpt_subject=about&fname=abtstf26&ftype=html)

| Field | Meaning |
| --- | --- |
| `staff_fte` | Published full-time equivalents stored as exact scale-2 Arrow decimals |
| `staff_total_base_pay` | Published total base pay in integer nominal dollars |
| `staff_average_base_pay` | Published average base pay in integer nominal dollars |
| `staff_employee_count` | Published 2025–26 count with an attached `peims_employee_count_basis` of `positions` for detail or `distinct_employees` for totals |

TEA excludes supplemental pay, contracted staff, and education service center
staff from the report's base-pay or staff populations as described in the
annual criteria. Charter partner teachers enter FTE beginning in 2023–24 and
are excluded from Average Base Pay. The published average is a separate
observation. Dividing displayed base pay by displayed FTE can differ from it.
Totals and detail rows can cover the same employees and cannot be added.
Counts in these broad role classifications do not identify service assignment,
student caseloads, or time dedicated to a particular program.

Raw tokens and original CSV record numbers travel with the typed values. A
district `.` for Average Base Pay in the 2024–25 or 2025–26 edition has
`missing` status. A published numeric zero has `observed` status. The latest
edition is listed as published March 19, 2026. TEA's listing does not supply a
finality label. Employee Count is absent from the three earlier editions.

Each source receipt preserves the publisher request, content digest, and a
link to the earlier capture of the same CSV bytes. The definition links return
the same content as the retained annual criteria receipts. The provenance
closure also includes the selector and reviewed column-layout evidence.

## Offline candidate

Select the exact school-year end and use the reviewed source configuration
with the retained provenance store. The normal snapshot does not open these
source-only editions.

```sh
uv run --locked python -m teadata.scripts.refresh_peims \
  --store /path/to/retained/provenance --config teadata/teadata_sources.yaml \
  --config-sha256 CONFIG_SHA256 \
  --source peims_staff_fte_and_salary 2023 \
  --source peims_staff_fte_and_salary 2024 \
  --source peims_staff_fte_and_salary 2025 \
  --source peims_staff_fte_and_salary 2026 \
  --output /path/to/new/staff-candidate
```

The existing `--source-parity` command independently decodes the retained
CSV, checks the typed values and source support, and reports coverage for the
exact requested edition IDs. Point it at the candidate's bundled `provenance`
directory. Separately selected publisher HTML reports can verify the values and
categories. These checks describe the source candidate and leave activation and
publication separate.
