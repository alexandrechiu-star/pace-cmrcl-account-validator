import pandas as pd

# Names confirmed against the real Raw Account Data export (2026-08-20) — not the
# original spec. FY27_CORE_ACV does not exist in the actual export (removed from
# the pull entirely); Q1/Q2 ACV columns use spaces, not underscores.
RAW_ACCOUNT_COLUMNS = [
    'COMBO_COMPANY_ID_18',
    'COMBO_COMPANY_NAME',
    'ACCT_ID_18',
    'ACCT_NM',
    'TEAM_TERRITORY_NAME',
    'ACCT_OWN_FULL_NM',
    'COMBO_LOCKED_CITY',
    'COMBO_LOCKED_STATE',
    'COMBO_LOCKED_INDUSTRY',
    'ACCT_OWN_LVL_4_USR_NM',   # AVP
    'ACCT_OWN_LVL_5_USR_NM',   # RVP
    'FY27_C360_AOV',
    'COMBO_EMPLOYEE_COUNT',
    'ACCT_TIER',
    'ATR',
    'FY27 Q1 ACV',
    'FY27 Q2 ACV',
]


def extract_avp_region(territory):
    territory = str(territory)
    if 'RCG' in territory:
        return 'RCG'
    if 'MFG_W' in territory:
        return 'MFG W'
    if 'MFG_C' in territory:
        return 'MFG C'
    if 'MFG_E' in territory:
        return 'MFG E'
    if 'AUE' in territory:
        return 'AUE'
    return 'Other'


def load_accounts(path='data/raw_accounts.csv'):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in RAW_ACCOUNT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw account CSV is missing expected columns: {missing}. "
            f"Actual columns found: {list(df.columns)}. "
            "Column header text must match the Raw Account Data tab exactly — "
            "re-check the export if the sheet's header row was edited."
        )

    accounts = df[RAW_ACCOUNT_COLUMNS].copy()

    accounts['ACCT_TIER'] = accounts['ACCT_TIER'].fillna('Untiered')
    accounts['H1_FY27_ACV'] = accounts['FY27 Q1 ACV'].fillna(0) + accounts['FY27 Q2 ACV'].fillna(0)
    accounts['AVP_REGION'] = accounts['TEAM_TERRITORY_NAME'].apply(extract_avp_region)

    return accounts
