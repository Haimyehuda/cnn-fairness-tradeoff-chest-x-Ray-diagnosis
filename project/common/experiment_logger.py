# project/common/experiment_logger.py

import os
import datetime as dt
from typing import Dict, Optional

import gspread
from google.auth import default as google_auth_default


# ⚙️ מזהה ה-Spreadsheet של Google Sheets
# שים את ה-ID פה או כ-ENV בשם EXPERIMENT_SHEET_ID
SPREADSHEET_ID = os.environ.get("EXPERIMENT_SHEET_ID", "PUT_YOUR_SHEET_ID_HERE")


def _get_gsheets_client():
    """
    יוצר לקוח ל-Google Sheets.
    ב-Colab: צריך להריץ פעם אחת במחברת:
        from google.colab import auth
        auth.authenticate_user()
    ואז הפונקציה הזו תעבוד עם הקרדנציאלס של החשבון שלך.
    """
    creds, _ = google_auth_default()
    client = gspread.authorize(creds)
    return client


def _get_or_create_worksheet(client, experiment_name: str):
    """
    מחזיר worksheet (גיליון) בשם הניסוי.
    אם הוא לא קיים – יוצר חדש.
    """
    sh = client.open_by_key(SPREADSHEET_ID)

    try:
        ws = sh.worksheet(experiment_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=experiment_name, rows=1000, cols=50)
        # כותרת ראשונה בסיסית – נשכתב אותה בקריאה הראשונה ל-log_experiment_to_sheet
        ws.update("A1", "timestamp")

    return ws


def log_experiment_to_sheet(
    experiment_name: str,
    metrics: Dict[str, float],
    config: Optional[Dict[str, str]] = None,
    notes: Optional[str] = None,
):
    """
    רושם שורה חדשה ב-Google Sheets בגיליון ששמו experiment_name.

    parameters:
    -----------
    experiment_name : str
        שם הניסוי (למשל "baseline", "weighted_loss", "smote"...)
        זה יהיה שם ה-tab ב-Google Sheets.

    metrics : dict
        מדדים מספריים של הניסוי, למשל:
        {
            "overall_acc": 0.96,
            "cat_acc": 0.00,
            "dog_acc": 1.00
        }

    config : dict, optional
        הגדרות/היפרפרמטרים חשובים של הניסוי:
        {
            "epochs": 5,
            "batch_size": 32,
            "lr": 0.001,
            "loss": "CrossEntropy",
            "method": "weighted_loss"
        }

    notes : str, optional
        טקסט חופשי – למשל "weighted CE with inverse frequency".
    """
    if config is None:
        config = {}

    client = _get_gsheets_client()
    ws = _get_or_create_worksheet(client, experiment_name)

    # זמן ריצה
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # כותרת ראשונה בגיליון (אם עדיין לא מסודרת)
    header_row = ws.row_values(1)  # רשימת שמות עמודות

    # נבנה header סטנדרטי: timestamp | metric_* | cfg_* | notes
    base_header = ["timestamp"]
    metric_keys = [f"metric_{k}" for k in metrics.keys()]
    config_keys = [f"cfg_{k}" for k in config.keys()]
    extra_header = ["notes"]

    desired_header = base_header + metric_keys + config_keys + extra_header

    # אם הכותרת הנוכחית לא תואמת – נעדכן
    if header_row != desired_header:
        ws.resize(rows=1000, cols=len(desired_header))
        ws.update("1:1", [desired_header])
        header_row = desired_header

    # לבנות שורה לפי סדר הכותרות
    row_dict = {
        "timestamp": timestamp,
        **{f"metric_{k}": v for k, v in metrics.items()},
        **{f"cfg_{k}": v for k, v in config.items()},
        "notes": notes or "",
    }

    row_values = [row_dict.get(col_name, "") for col_name in header_row]

    ws.append_row(row_values, value_input_option="RAW")
    print(
        f"[experiment_logger] Logged experiment '{experiment_name}' to Google Sheets."
    )
