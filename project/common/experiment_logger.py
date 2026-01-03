"""
experiment_logger.py
====================

Lightweight experiment result logger.

Responsibilities:
- Standardize experiment result structure
- Append experiment results to a CSV file

This module is storage-agnostic and does not depend on external services.
"""

import os
import pandas as pd
from typing import Dict


def log_experiment(
    results: Dict[str, float],
    output_path: str,
) -> None:
    """
    Append a single experiment result to a CSV file.

    If the file does not exist, it is created with headers.
    If it exists, the result is appended as a new row.

    Args:
        results (dict): Dictionary of metric names to values
        output_path (str): Path to CSV results file
    """

    results_df = pd.DataFrame([results])

    if os.path.exists(output_path):
        results_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_path, index=False)


# --------------------------------------------------
# Google Sheets logging
# --------------------------------------------------
import gspread
from google.auth import default


def log_experiment_to_sheets(row: dict, sheet_id: str) -> None:
    """
    Append a single experiment row to Google Sheets.
    Assumes header already exists and matches row.keys() order.
    """
    creds, _ = default()
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    ws = sh.sheet1  # gid=0

    ws.append_row(list(row.values()), value_input_option="USER_ENTERED")
