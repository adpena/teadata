# TEA Data agent handoff, 2026-11-14

## Purpose and scope

Continue the 2025–26 enrollment and finance enrichment work without repeating completed releases or losing source evidence. This document transfers the owner's priorities to the next agent. It is not a new deployment, publication, cleanup or credentials authorization. Read repository AGENTS.md and relevant local skills first. Inspect current work ownership, Git changes and live release state before acting; coordinate rather than duplicate an active lane.

This handoff was prepared from the coordinating agent's dated reports, not a fresh rerun of all cited tests. The local checkout was freshly inspected on 2026-11-14: clean main at `fb6b324ca6769f5945df263faf4ac571825fb333`, matching the locally cached origin/main. Live remote inspection was blocked by network policy, so cached refs are not proof of current remote state. This documentation change is intentionally committed locally only; pushing was not requested. No existing files or published artifacts should be discarded to force a closeout pass.

## Reported evidence to recover and verify

The coordinating agent reported public library v0.0.154 with exact release manifest SHA-256 `c6d3d8fa152e67d7b992a2ff73c56d4725abe7f0171f57fc40158b27337a8520`. Do not republish or replace released assets. Recover the retained acquisition records and receipts named `SOF_QUALIFIED.json`, `CHRONIC_QUALIFIED.json`, `OUTPUT_CUSTODY.json`, `RELEASE_READY154.json` and the final export results. The earlier READY_NOT_PUBLISHED receipt was superseded by a reported public manifest readback; confirm that readback before relying on it. Some retained evidence uses hardlinks: never edit payloads in place.

Reported normal parent coverage: 13 base contracts, 12 native children, 140 editions, 1,229 districts and 12,593 campuses. Reported canonical library checks: 3,092 passed, 2 skipped and 1 warning, with lint, types, dependencies and data-release checks passed. These are historical receipts, not a fresh certification of this documentation commit or future changes.

Required semantic distinctions:

- Default TPRS enrollment was reported as 2025–26, 5,467,642. Membership is a different population, 5,454,059. Never substitute one for the other.
- SOF 2025–26 contains 1,243 rows with seven source run IDs: 45777, 45811, 47223, 47224, 47242, 47299 and 47422. Keep row run identity, LPE/DPE, source period, decimals, raw tokens, scales, zero and missing states explicit.
- Generic adjusted ADA is missing in all 1,243 reported rows. `ADA_TOT_REFINED` is refined ADA. `ADJ_AVG_ADA` is SSRA-specific adjusted ADA, not generic adjusted ADA. `ADA_ADJ_TOT_REFINED` was null. Do not invent the requested general adjusted measure or claim that requirement satisfied by a different field.
- Latest actual chronic absence was reported as 2024–25, 18.0%, with 2023–24 at 19.0%. A request for 2025–26 correctly failed. Coverage is state plus 20 ESCs; district and campus remain held. Preserve subgroup dimensions and measurement year, not merely publication year.
- A 2025–26 TEA report's 1,232 populated district values matched DPE refined ADA, but its source vintage was not established. Historical region/county 2005–06 and 2006–07 column ordering conflicts remain unresolved. Do not use these observations to waive lineage or trend qualification.

## Bounded next assignment

1. Locate the exact retained release and acquisition evidence, verify hashes and the normal release's actual data access. Use the existing catalog, enrichment contracts and export architecture, not parallel per-dataset properties or fabricated values.
2. Inventory each requested dataset independently: source availability; parsing and semantic qualification; normal enrichment; real-data regression checks; actual output; downstream app ingestion. Record supported years, scopes, populations, source runs and remaining gaps separately.
3. Confirm latest available year is selected per dataset. Test explicit unavailable-year rejection and missing/suppressed/zero preservation, using publisher-grounded real values through query and export. A global 2025–26 label is not acceptable.
4. Coordinate app adoption against the exact manifest above. Reported app dependency pins already select package 154, while deployment configuration still selects data 153. Library publication does not prove production ingestion.
5. Preserve original provenance records and objects. The cloud qualification could not complete the canonical source-provenance gate without the adjacent original store. A bulk transfer attempt failed; do not infer successful custody from a prior receipt or inaccessible machine path. Use an authorized supported transfer and verify bytes on the consumer.

The cloud reportedly verified all 330 public assets (933,566,795 bytes), offline normal DataRelease loading and real queries on native AMD64/Python 3.12.14. This does not replace original publisher evidence, the app's runtime/image/resource gates or actual production readback.

## Done criteria and return report

Use the canonical checks and independent exact-diff review required by AGENTS.md. Report exact commits, release hashes, checked values, coverage, receipt identities, failed/skipped/never-run checks and the smallest remaining blocker. Completion requires requested datasets and latest-year behavior verified through actual normal enriched output and app ingestion; missing generic adjusted ADA or held scopes must remain explicit. Never declare completion from configuration or synthetic tests alone. Keep credentials, machine-local evidence paths and user data outside tracked documentation.
