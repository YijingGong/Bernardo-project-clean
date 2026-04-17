#!/usr/bin/env python3
"""
Farm8 HFD vs LFD Analysis
=========================
For each of 8 CSV output files, across 4 replicates of HFD and LFD:
  1. Verify the run spans 10 years (2013-2022).
  2. Discard first 6 years (2013-2018) as burn-in.
  3. Average the last 4 years (2019-2022) per column → one value per run.
  4. Compute mean and SD across the 4 replicates.
  5. Present results in tables and save to Excel.

Usage:
    conda activate Rufas2025
    python analyze_farm8_hfd_lfd.py
"""

import re
import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

CSV_DIR    = Path("/Users/yijinggong/Library/CloudStorage/Box-Box/postdoc/writing/Bernardo project/MASM_BERNARDO/output/CSVs")
OUTPUT_DIR = Path("/Users/yijinggong/Library/CloudStorage/Box-Box/postdoc/writing/Bernardo project/MASM_BERNARDO/output")

SIM_START   = date(2013, 1, 1)   # simulation_day 0 = Jan 1, 2013
BURN_IN_YRS = 6                  # discard 2013–2018
N_RUNS      = 4

# simulation_day at which analysis period starts (first day of 2019):
#   2013:365 + 2014:365 + 2015:365 + 2016:366 + 2017:365 + 2018:365 = 2191 days
ANALYSIS_START_DAY = 2191        # simulation_day >= 2191 → year 2019+

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_file_type(filename: str) -> str | None:
    """Extract the CSV type key from the filename.

    'Farm8_HFD run 2_saved_variables_csv_ration.txt_14-Apr-2026_Tue_21-54-13.csv'
    → 'ration.txt'
    """
    m = re.search(r'_saved_variables_csv_(.+?)_\d{2}-[A-Za-z]+-\d{4}_', filename)
    return m.group(1) if m else None


def find_simday_col(df: pd.DataFrame) -> str | None:
    """Return the simulation-day column name (flexible match)."""
    for col in df.columns:
        cl = col.lower()
        if 'simulation_day' in cl or ('simulation' in cl and 'day' in cl):
            return col
    return None


def read_farm8_csv(filepath: Path) -> pd.DataFrame:
    """Read one Farm8 CSV.

    Structure:
      row 0  = column headers (first col is 'DISCLAIMER')
      row 1  = first data row (disclaimer text in DISCLAIMER col)
      rows 2+= data (DISCLAIMER col empty/NaN)

    Returns a DataFrame with DISCLAIMER dropped and all columns coerced to
    numeric (non-numeric columns are dropped).
    """
    df = pd.read_csv(filepath, low_memory=False)

    if 'DISCLAIMER' in df.columns:
        df = df.drop(columns=['DISCLAIMER'])

    # Coerce all columns to numeric; non-numeric become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that are entirely NaN after coercion
    df = df.dropna(axis=1, how='all')

    return df


def process_one_run(filepath: Path) -> dict[str, float]:
    """Return per-column means over the analysis period for a single run file.

    Two column types are handled:
    - Daily columns (~3652 non-NaN rows): filtered by simulation_day >= ANALYSIS_START_DAY.
    - Ration-period columns (~122 non-NaN rows, one per 30-day ration cycle spanning the
      full 10-year simulation): the last 40% of non-NaN rows represent the last 4 years.
    """
    df = read_farm8_csv(filepath)
    sim_col = find_simday_col(df)
    total_rows = len(df)

    # Expected number of ration-period rows (one per ~30-day cycle over the full run)
    expected_periods = total_rows / 30   # ≈ 121.7 for a 3652-row / 10-year file

    # Build analysis-period slice for daily columns
    if sim_col is not None:
        df_clean = df.dropna(subset=[sim_col]).copy()
        df_clean[sim_col] = df_clean[sim_col].astype(int)
        df_analysis = df_clean[df_clean[sim_col] >= ANALYSIS_START_DAY].drop(columns=[sim_col])
    else:
        df_analysis = df.iloc[ANALYSIS_START_DAY:]

    means: dict[str, float] = {}

    for col in df.columns:
        if col == sim_col:
            continue

        non_nan = pd.to_numeric(df[col], errors='coerce').dropna()
        n = len(non_nan)
        if n == 0:
            continue

        if 0.7 * expected_periods <= n <= 1.3 * expected_periods:
            # ── Ration-period column ──────────────────────────────────────────
            # Each of the n rows represents one 30-day ration period, so the
            # full set spans the entire 10-year simulation.
            # Last 4 years ≈ last 40% (4/10) of rows.
            n_analysis = max(1, round(n * 4 / 10))
            means[col] = float(non_nan.tail(n_analysis).mean())
        elif col in df_analysis.columns:
            # ── Daily column ──────────────────────────────────────────────────
            val = pd.to_numeric(df_analysis[col], errors='coerce').mean()
            if pd.notna(val):
                means[col] = float(val)

    return means


def verify_duration(filepath: Path) -> tuple[int, int]:
    """Return (n_data_rows, max_simday) for a quick sanity check."""
    df = pd.read_csv(filepath, low_memory=False)
    if 'DISCLAIMER' in df.columns:
        df = df.drop(columns=['DISCLAIMER'])
    sim_col = find_simday_col(df)
    n = len(df)
    if sim_col:
        df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce')
        return n, int(df[sim_col].max())
    return n, n - 1   # approximate when no sim_day column


def discover_file_types(diet: str) -> list[str]:
    """List the CSV type keys present for run 1 of a given diet."""
    types = set()
    for f in CSV_DIR.glob(f"Farm8_{diet} run 1_*.csv"):
        ft = get_file_type(f.name)
        if ft:
            types.add(ft)
    return sorted(types)


def find_file(diet: str, run: int, file_type: str) -> Path | None:
    """Locate the CSV file for a given diet/run/type."""
    candidates = list(CSV_DIR.glob(f"Farm8_{diet} run {run}_*{file_type}*.csv"))
    return candidates[0] if candidates else None


# ─── Main analysis ────────────────────────────────────────────────────────────

def analyze() -> dict[str, pd.DataFrame]:
    """Run the full analysis.  Returns {file_type: results_DataFrame}."""

    analysis_yr_start = (SIM_START + timedelta(days=ANALYSIS_START_DAY)).year   # 2019
    analysis_yr_end   = SIM_START.year + 9                                       # 2022

    print("=" * 72)
    print("Farm8 HFD vs LFD — Post-burn-in Analysis")
    print(f"  Simulation : {SIM_START.year}–{analysis_yr_end}")
    print(f"  Burn-in    : {SIM_START.year}–{analysis_yr_start - 1}  "
          f"(sim days 0–{ANALYSIS_START_DAY - 1})")
    print(f"  Analysis   : {analysis_yr_start}–{analysis_yr_end}  "
          f"(sim days {ANALYSIS_START_DAY}–end)")
    print("=" * 72)

    hfd_types = discover_file_types('HFD')
    lfd_types = discover_file_types('LFD')
    all_types = sorted(set(hfd_types) | set(lfd_types))

    print(f"\nFile types found — HFD: {len(hfd_types)}, LFD: {len(lfd_types)}")
    print(f"  HFD only : {sorted(set(hfd_types) - set(lfd_types))}")
    print(f"  Common   : {sorted(set(hfd_types) & set(lfd_types))}")

    # ── Verify run durations ────────────────────────────────────────────────
    print("\n── Duration check ──────────────────────────────────────────────────")
    for diet in ('HFD', 'LFD'):
        for run in range(1, N_RUNS + 1):
            f = find_file(diet, run, 'ration.txt')
            if f is None:
                f = find_file(diet, run, all_types[0])
            if f:
                n, max_day = verify_duration(f)
                ok = "✓" if n == 3652 else "!"
                print(f"  {ok} Farm8_{diet} run {run}: {n} rows, "
                      f"max sim_day={max_day}  "
                      f"({'10 yrs OK' if n == 3652 else 'CHECK LENGTH'})")

    # ── Process each diet / file type ───────────────────────────────────────
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    # all_results[diet][file_type][col] = {run_label: mean}

    for diet in ('HFD', 'LFD'):
        types = hfd_types if diet == 'HFD' else lfd_types
        print(f"\n── Processing {diet} ──────────────────────────────────────────────")
        for ft in types:
            run_means: dict[str, dict[str, float]] = {}
            for run in range(1, N_RUNS + 1):
                f = find_file(diet, run, ft)
                if f is None:
                    print(f"  MISSING: Farm8_{diet} run {run} / {ft}")
                    continue
                means = process_one_run(f)
                run_means[f"Run {run}"] = means
                print(f"  Farm8_{diet} run {run} / {ft}: "
                      f"{len(means)} numeric columns in analysis period")
            all_results.setdefault(diet, {})[ft] = run_means

    # ── Build summary DataFrames ─────────────────────────────────────────────
    output_tables: dict[str, pd.DataFrame] = {}

    common_types = sorted(set(hfd_types) & set(lfd_types))
    hfd_only     = sorted(set(hfd_types) - set(lfd_types))

    for ft in common_types:
        hfd_runs = all_results.get('HFD', {}).get(ft, {})
        lfd_runs = all_results.get('LFD', {}).get(ft, {})

        hfd_df = pd.DataFrame(hfd_runs)           # index=col, cols=Run1..4
        lfd_df = pd.DataFrame(lfd_runs)

        hfd_df['HFD_Mean'] = hfd_df.mean(axis=1)
        hfd_df['HFD_SD']   = hfd_df.std(axis=1, ddof=1)
        lfd_df['LFD_Mean'] = lfd_df.mean(axis=1)
        lfd_df['LFD_SD']   = lfd_df.std(axis=1, ddof=1)

        combined = hfd_df.join(lfd_df, how='outer', lsuffix='_hfd', rsuffix='_lfd')
        # Clean up duplicate run columns from join
        run_cols_hfd = [f"Run {i}" for i in range(1, N_RUNS + 1)]
        run_cols_lfd = [f"Run {i}" for i in range(1, N_RUNS + 1)]

        # Rebuild cleanly
        result = pd.DataFrame(index=combined.index)
        for run in range(1, N_RUNS + 1):
            col = f"Run {run}"
            result[f"HFD_{col}"] = hfd_df.get(col)
            result[f"LFD_{col}"] = lfd_df.get(col)
        result['HFD_Mean'] = hfd_df['HFD_Mean']
        result['HFD_SD']   = hfd_df['HFD_SD']
        result['LFD_Mean'] = lfd_df['LFD_Mean']
        result['LFD_SD']   = lfd_df['LFD_SD']
        result.index.name  = 'Metric'

        output_tables[ft] = result

    for ft in hfd_only:
        hfd_runs = all_results.get('HFD', {}).get(ft, {})
        hfd_df = pd.DataFrame(hfd_runs)
        hfd_df['HFD_Mean'] = hfd_df.mean(axis=1)
        hfd_df['HFD_SD']   = hfd_df.std(axis=1, ddof=1)
        hfd_df.index.name  = 'Metric'
        output_tables[ft]  = hfd_df

    return output_tables


# ─── Print tables ─────────────────────────────────────────────────────────────

def print_tables(output_tables: dict[str, pd.DataFrame]) -> None:
    pd.set_option('display.max_rows',    None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',       220)
    pd.set_option('display.float_format', '{:.5g}'.format)

    print("\n" + "=" * 72)
    print("RESULTS  —  4-year average (2019–2022) across 4 replicates")
    print("=" * 72)

    for ft, df in output_tables.items():
        label = ft + ("  [HFD only]" if 'HFD_Mean' in df.columns and 'LFD_Mean' not in df.columns else "")
        print(f"\n{'─'*72}")
        print(f"  {label}")
        print(f"{'─'*72}")

        # Show mean/SD summary (concise)
        summary_cols = [c for c in df.columns if 'Mean' in c or 'SD' in c]
        print(df[summary_cols].to_string())


# ─── Save to Excel ────────────────────────────────────────────────────────────

def save_excel(output_tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    try:
        import openpyxl
    except ImportError:
        print("\nWARNING: openpyxl not installed; skipping Excel export.")
        return

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for ft, df in output_tables.items():
            # Sheet names: strip '.txt', truncate to 31 chars (Excel limit)
            sheet = ft.replace('.txt', '').replace('_', ' ')[:31]
            df.to_excel(writer, sheet_name=sheet)

            ws = writer.sheets[sheet]
            for col_cells in ws.columns:
                max_w = max((len(str(c.value or '')) for c in col_cells), default=8)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_w + 2, 60)

    print(f"\nExcel saved → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    output_tables = analyze()
    print_tables(output_tables)
    save_excel(output_tables, OUTPUT_DIR / "Farm8_HFD_vs_LFD_analysis.xlsx")
    print("\nDone.")
