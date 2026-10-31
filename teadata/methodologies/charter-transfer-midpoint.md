# Charter transfer midpoint scenario, version 1

For each origin campus, report the sum of observed charter transfer counts separately from the modeled total. For every distinct suppressed origin-campus/destination-campus flow, the midpoint scenario adds five students:

`estimated total = observed sum + 5 * suppressed flow count`

This preserves the historical Charter Cost Tracker approach. Five is an explicit modeling assumption, not an observed count, a TEA disclosure bound, or a confidence interval. TEA notes that larger counts can also be masked to prevent inference. Missing, inapplicable, invalid, or unmatched flows are not automatically assigned five students. If a contributing count is missing or invalid, the modeled total is unavailable.

The model's grain is an origin-campus/destination-campus pair. To obtain a district summary of campus-flow estimates, sum the estimates at that grain. Do not apply five again after aggregation or interpret that sum as the independently masked district-report estimate. The legacy project reads district reports; its grouping can differ from campus reports.

The source contract stores the model identifier, coefficient, grain, and assumption. `EstimationPolicy.estimate()` is the calculation authority. Exports retain observed sums, suppressed counts, modeled contributions, model identity, and release provenance. Suppressed source values remain suppressed when an estimate is requested.

Historical implementation: [Charter Cost Tracker, commit b9123138](https://github.com/adpena/CharterCostTracker/blob/b9123138fcbd2e3f3e51fffd0ed64ff688cd206e/gen_statewide_cost_of_charter_reports.py#L237-L244). The historical repository may require access. This document and the public TEA Data contract fully specify the scenario without requiring that repository.

TEA source methodology: [About Transfer Reports](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/About/about_transfer_reports.html). Year-specific CSV report footnotes are archived with the source contract; they identify `-999` as masked and explain that larger counts can also be masked.
