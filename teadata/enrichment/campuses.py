from typing import Dict, Any
import pandas as pd
from . import enricher
from .base import Enricher

from teadata.teadata_config import load_config

from teadata.teadata_config import (
    canonical_campus_number,
    normalize_campus_number_column,
)


def _canon_campus_number(x) -> str | None:
    return canonical_campus_number(x)


def _canon_series(series: pd.Series) -> pd.Series:
    """Vectorized canonicalization for a pandas Series of campus numbers."""
    return series.map(canonical_campus_number)


def _build_campus_multi_index(repo) -> dict[str, Any]:
    """Return a dict mapping multiple key shapes to Campus.id for robust matching.
    Keys created per campus_number:
      - canonical with apostrophe: '#########
      - without apostrophe: #########
      - int form (no leading zeros): e.g., 123456789 as string
    """
    idx: dict[str, Any] = {}
    for c in repo._campuses.values():
        cn = getattr(c, "campus_number", None)
        if not cn:
            continue
        key_can = _canon_campus_number(cn)
        if not key_can:
            continue
        idx[key_can] = c.id
        # Without apostrophe
        no_tick = key_can[1:]
        idx[no_tick] = c.id
        # Int-like
        if no_tick.isdigit():
            idx[str(int(no_tick))] = c.id
    return idx


# Shared helper for campus accountability enrichment
def _apply_campus_accountability(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    aliases=None,
    reader_kwargs=None,
):
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    # Rename spreadsheet headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Ensure we have a campus_number column; try standard helper, then fallbacks
    df, found = normalize_campus_number_column(df, new_col="campus_number")
    if found is None and "campus_number" not in df.columns:
        # Try common variations
        for guess in ("Campus Number", "CAMPUS", "campus", "Campus"):
            if guess in df.columns:
                df = df.rename(columns={guess: "campus_number"})
                break
    if "campus_number" not in df.columns:
        return resolved_year, 0

    # Canonicalize to leading-apostrophe format used across the repo
    df["campus_number"] = _canon_series(df["campus_number"])
    df = df[df["campus_number"].notna()]

    # Column selection/cleanup
    if select is None:
        use_cols = [c for c in df.columns if c != "campus_number"]
    else:
        use_cols = [c for c in select if c in df.columns]
    for c in use_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: (
                    None if (pd.isna(x) or (isinstance(x, str) and x.strip() == ""))
                    else (x.strip() if isinstance(x, str) else x)
                )
            )

    # Build mapping from canonical campus_number -> selected attrs
    mapping: dict[str, dict[str, Any]] = {}
    sub = df[["campus_number"] + use_cols].drop_duplicates("campus_number")
    for r in sub.itertuples(index=False):
        key = getattr(r, "campus_number")
        if not key:
            continue
        record = {k: getattr(r, k) for k in use_cols}
        mapping[key] = record
        digits = key[1:]
        mapping.setdefault(digits, record)
        if digits.isdigit():
            mapping.setdefault(str(int(digits)), record)

    # Multi-key index for robust matching
    updated = 0
    missing = 0
    for cobj in repo._campuses.values():
        cn = getattr(cobj, "campus_number", None)
        if not cn:
            continue
        can = _canon_campus_number(cn)
        if not can:
            missing += 1
            continue
        digits = can[1:]
        attrs = mapping.get(can) or mapping.get(digits)
        if attrs is None and digits.isdigit():
            attrs = mapping.get(str(int(digits)))
        if not attrs:
            missing += 1
            continue
        if getattr(cobj, "meta", None) is None:
            cobj.meta = {}
        for k, v in attrs.items():
            cobj.meta[k] = v
            if aliases and k in aliases:
                try:
                    setattr(cobj, aliases[k], v)
                except Exception:
                    pass
        updated += 1
    # Optional visibility for debugging
    # print(f"[enrich:accountability] campuses updated={updated} missing={missing}")

    return resolved_year, updated


# Shared helper for campus PEIMS financials enrichment
def _apply_campus_peims_financials(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    reader_kwargs=None,
):
    """
    Load campus-level PEIMS financials and attach a small set of attributes to Campus.meta.

    - We match on the canonical campus key: a normalized, 9-digit, zero-padded string **with a leading apostrophe** (e.g., "'123456789"), as used by repository Campus objects.
    - If `select` is None, we try to auto-detect three canonical fields via fuzzy header matching:
        * peims_total_expenditures
        * peims_instructional_expenditures
        * peims_per_pupil_expenditure
    - If `select` is provided, we keep exactly those columns.
    - If `rename` is provided, it is applied before selection.

    Returns (resolved_year, updated_count).
    """
    import re
    from difflib import SequenceMatcher
    from teadata.teadata_config import normalize_campus_number_column, load_config

    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    # Optional caller-provided renames first
    if rename:
        df = df.rename(columns=rename)

    # Ensure we have a campus_number column, then canonicalize vectorized
    df, found = normalize_campus_number_column(df, new_col="campus_number")
    if found is None and "campus_number" not in df.columns:
        for guess in ("Campus Number", "CAMPUS", "campus", "Campus"):
            if guess in df.columns:
                df = df.rename(columns={guess: "campus_number"})
                break
    if "campus_number" not in df.columns:
        return resolved_year, 0

    df["campus_number"] = _canon_series(df["campus_number"])
    df = df[df["campus_number"].notna()]

    # --- Column auto-detection -------------------------------------------------
    # Build a normalization for column headers to be robust to spacing, case, and punctuation
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    actual_cols = {norm(c): c for c in df.columns}

    def _resolve_aliases(name: str) -> list[str]:
        """Generate possible header aliases for a canonical field name.

        We expand common tokens so that callers can pass snake_case keys while the
        dataset may use human-readable headers (e.g., ``Instruction (AF) Percent``).
        """

        base = name.replace("_", " ").lower()
        tokens = base.split()
        expanded = []
        token_map = {
            "perc": ["perc", "percent", "pct"],
            "per": ["per"],
            "student": ["student", "pupil"],
            "af": ["af", "actual financials"],
            "ccmr": ["ccmr", "college career military readiness"],
            "dyslexia": ["dyslexia"],
            "serv": ["services", "serv"],
            "ed": ["education", "ed"],
            "security": ["security"],
            "monitoring": ["monitoring", "monitor"],
            "guidance": ["guidance"],
            "counseling": ["counseling", "counsel"],
            "school": ["school"],
            "leadership": ["leadership"],
            "students": ["students", "student"],
            "disabilities": ["disabilities", "disability"],
            "w": ["with", "w"],
        }

        def _expand(idx: int, current: list[str]):
            if idx == len(tokens):
                expanded.append(" ".join(current))
                return
            tok = tokens[idx]
            choices = token_map.get(tok, [tok])
            for choice in choices:
                _expand(idx + 1, current + [choice])

        _expand(0, [])
        # Include direct underscored variant (for already-normalized headers)
        expanded.append(name)
        # Deduplicate while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for alias in expanded:
            if alias not in seen:
                uniq.append(alias)
                seen.add(alias)
        return uniq

    def _match_via_sequence(target: str) -> str | None:
        key = norm(target)
        best_col = None
        best_score = 0.0
        for col, normed in actual_cols.items():
            score = SequenceMatcher(None, key, col).ratio()
            if score > best_score:
                best_col = normed
                best_score = score
        if best_score >= 0.55:
            return best_col
        return None

    # Candidate source headers for each canonical field
    candidates = {
        "peims_total_expenditures": [
            "totalexpenditures",
            "totalexpendedamount",
            "totalexpend",
            "totalexpamt",
        ],
        "peims_instructional_expenditures": [
            "instructionalexpenditures",
            "function11instruction",
            "instructionexpendedamount",
            "instrexp",
        ],
        "peims_per_pupil_expenditure": [
            "perpupilexpenditures",
            "perpupilexpenditure",
            "perstudentexpenditure",
            "perpupil",
        ],
    }

    # If caller supplied an explicit select, honor it; otherwise auto-pick from candidates
    if select is None:
        selected_map = {}
        for canonical, opts in candidates.items():
            sel = next((actual_cols[k] for k in opts if k in actual_cols), None)
            if not sel:
                # try fuzzy matching on canonical key
                sel = _match_via_sequence(canonical)
            if sel:
                selected_map[canonical] = sel
        if not selected_map:
            # Display a tiny preview to help diagnose header names
            preview = list(df.columns)[:12]
            print(
                f"[enrich:campus_peims_financials] Could not auto-detect columns; first 12 headers: {preview}"
            )
            return resolved_year, 0
    else:
        selected_map: dict[str, str] = {}
        seen_cols: set[str] = set()
        for canonical in select:
            target_norm = norm(canonical)
            chosen = None
            # direct match on raw header
            if canonical in df.columns:
                chosen = canonical
            elif target_norm in actual_cols:
                chosen = actual_cols[target_norm]
            else:
                # try alias expansion
                for alias in _resolve_aliases(canonical):
                    alias_norm = norm(alias)
                    if alias_norm in actual_cols:
                        chosen = actual_cols[alias_norm]
                        break
                if chosen is None:
                    chosen = _match_via_sequence(canonical)
            if chosen and chosen not in seen_cols:
                selected_map[canonical] = chosen
                seen_cols.add(chosen)
        if not selected_map:
            preview = list(df.columns)[:12]
            print(
                "[enrich:campus_peims_financials] Requested columns not found; "
                f"select={select} first headers={preview}"
            )
            return resolved_year, 0

    keep_src_cols = list(selected_map.values())
    sub = df[["campus_number"] + keep_src_cols].drop_duplicates("campus_number")

    # Build mapping from canonical key -> record
    mapping: dict[str, dict[str, Any]] = {}
    for r in sub.itertuples(index=False):
        key = getattr(r, "campus_number")
        if not key:
            continue
        record = {canon: getattr(r, src) for canon, src in selected_map.items()}
        mapping[key] = record
        digits = key[1:]
        mapping.setdefault(digits, record)
        if digits.isdigit():
            mapping.setdefault(str(int(digits)), record)

    updated = 0
    missing = 0
    for campus in repo._campuses.values():
        cn = getattr(campus, "campus_number", None)
        if not cn:
            continue
        can = _canon_campus_number(cn)
        if not can:
            missing += 1
            continue
        digits = can[1:]
        attrs = mapping.get(can) or mapping.get(digits)
        if attrs is None and digits.isdigit():
            attrs = mapping.get(str(int(digits)))
        if not attrs:
            missing += 1
            continue
        if getattr(campus, "meta", None) is None:
            campus.meta = {}
        wrote = False
        for k, v in attrs.items():
            campus.meta[k] = v
            wrote = True
        if wrote:
            updated += 1

    # Optional debug
    # print(f"[enrich:peims] campuses updated={updated} missing={missing}")
    return resolved_year, updated

def enrich_campuses_from_config(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    aliases=None,
    reader_kwargs=None,
):
    """Dispatch to the appropriate enrichment routine based on dataset name."""

    ds_lower = dataset.lower()
    if ds_lower == "campus_peims_financials":
        return _apply_campus_peims_financials(
            repo,
            cfg_path,
            dataset,
            year,
            select=select,
            rename=rename,
            reader_kwargs=reader_kwargs,
        )

    return _apply_campus_accountability(
        repo,
        cfg_path,
        dataset,
        year,
        select=select,
        rename=rename,
        aliases=aliases,
        reader_kwargs=reader_kwargs,
    )


@enricher("campus_accountability")
class CampusAccountability(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_campus_accountability(
            repo,
            cfg_path,
            "campus_accountability",
            year,
            select=["overall_rating_2025"],
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        return {"updated": updated, "year": year_resolved}


@enricher("campus_peims_financials")
class CampusPEIMSFinancials(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        # Auto-detect sensible PEIMS columns; callers can override via direct function use
        yr, updated = _apply_campus_peims_financials(
            repo,
            cfg_path,
            "campus_peims_financials",
            year,
            select=None,
            rename=None,
            reader_kwargs=None,
        )
        return {"updated": updated, "year": yr}
