import pandas as pd

def pick_sheet_with_columns(df, expected_cols: set[str]):
    if isinstance(df, dict):  # already a sheet_name=None read
        for name, dfi in df.items():
            if expected_cols.intersection(set(dfi.columns)):
                return dfi, name
        # fallback
        name, dfi = next(iter(df.items()))
        return dfi, name
    return df, None

def prepare_columns(df, *, rename=None, select=None, transforms=None):
    if rename:
        df = df.rename(columns=rename)
    if select is None:
        cols = list(df.columns)
    else:
        cols = [c for c in select if c in df.columns]
    for c in cols:
        s = df[c]
        if s.dtype == object:
            df[c] = s.map(lambda x: None if (x is None or (isinstance(x, str) and x.strip()=="")) else (x.strip() if isinstance(x, str) else x))
    if transforms:
        for k, fn in transforms.items():
            if k in df.columns:
                df[k] = df[k].map(fn)
    return df