import json
import os
import time

import anthropic
import pandas as pd

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

MODEL = "claude-sonnet-5"

MISMATCH_FRAMEWORK = """
This tool validates existing account tiers against external market signals.

MISMATCH CATEGORIES:

🔴 Overtiered Risk - High tier (1/2) but weak external signals
    Indicators: Zero news coverage, no recent ACV, low employee count (<500),
                no growth signals, stale relationship

🟢 Undertiered Opportunity - Low tier (3/4) but strong external signals
    Indicators: High news volume, recent funding/M&A, employee growth,
                market momentum, peer velocity

⚠️ Bad Data - Account should be cleaned up
    Indicators: "DUPE", "OOB", "(Acquired by", "Part of", "-", "Parent/Child duplicate"
                in company name. Tier 4 accounts with these flags.

✅ Tier Confirmed - Current tier matches available signals
    Indicators: Tier aligns with company size, news coverage, spending pattern,
                and industry fit

EVALUATION CRITERIA:
- Company size (employee count as proxy for scale)
- External momentum (news volume, funding, acquisitions, leadership changes)
- Salesforce engagement (AOV, recent ACV, ATR)
- Industry fit (does the industry typically buy Salesforce at this scale?)
- Account health signals (spending trajectory, relationship depth)
"""

MISMATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "mismatch_category": {
            "type": "string",
            "enum": ["Overtiered Risk", "Undertiered Opportunity", "Bad Data", "Tier Confirmed"],
        },
        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "key_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "action_recommended": {"type": "string"},
    },
    "required": [
        "mismatch_category",
        "confidence",
        "key_signals",
        "reasoning",
        "action_recommended",
    ],
    "additionalProperties": False,
}


def score_account_with_claude(account: dict) -> dict:
    """Detect tier mismatches by validating current tier against external signals"""

    # Check for bad data flags first
    company_name = account.get('COMBO_COMPANY_NAME', '')
    bad_data_flags = ['DUPE', 'OOB', '(Acquired by', 'Part of', '(dupe)', '(duplicate)']
    has_bad_data_flag = any(flag.lower() in company_name.lower() for flag in bad_data_flags)

    internal_signals = f"""
    Current Org62 Tier: {account.get('ACCT_TIER', 'Unknown')}
    Industry: {account.get('COMBO_LOCKED_INDUSTRY', 'Unknown')}
    Location: {account.get('COMBO_LOCKED_CITY', '')}, {account.get('COMBO_LOCKED_STATE', '')}
    Employee Count: {account.get('COMBO_EMPLOYEE_COUNT', 'Unknown')}
    FY27 AOV (All-In): ${account.get('FY27_C360_AOV', 0):,.0f}
    H1 FY27 ACV (Q1+Q2): ${account.get('H1_FY27_ACV', 0):,.0f}
    ATR (Available to Renew): ${account.get('ATR', 0):,.0f}
    Territory: {account.get('TEAM_TERRITORY_NAME', 'Unknown')}
    """

    external_signals = f"""
    Recent News Coverage: {'Yes' if account.get('news_has_coverage') else 'No'}
    Number of Recent Articles: {account.get('news_article_count', 0)}
    Recent Headlines: {account.get('news_headlines', 'No recent news found')}
    Most Recent Coverage: {account.get('news_latest_date', 'Unknown')}
    """

    bad_data_note = f"\nNOTE: Company name contains flag: {company_name} - likely bad data" if has_bad_data_flag else ""

    prompt = f"""You are validating Salesforce account tiers against external market signals.

ACCOUNT: {account.get('COMBO_COMPANY_NAME', 'Unknown')}{bad_data_note}

INTERNAL SALESFORCE DATA:
{internal_signals}

EXTERNAL SIGNALS:
{external_signals}

{MISMATCH_FRAMEWORK}

INSTRUCTIONS:
- Assess whether the current tier matches the available signals
- High tier (1/2) with zero spend + zero news = Overtiered Risk
- Low tier (3/4) with strong news + large employee count = Undertiered Opportunity
- Tier 4 with "DUPE"/"OOB"/"Acquired by" in name = Bad Data
- Otherwise, if tier seems reasonable = Tier Confirmed
- Be specific in your reasoning - cite exact signals
- Action recommended should be concrete (e.g., "Review for downgrade", "Investigate for upgrade", "Remove from CRM")"""

    try:
        # Using tool-based structured output for JSON schema enforcement
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            tools=[{
                "name": "report_mismatch",
                "description": "Report the tier mismatch analysis results",
                "input_schema": MISMATCH_SCHEMA
            }],
            tool_choice={"type": "tool", "name": "report_mismatch"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract result from tool use
        tool_use = next(block for block in response.content if block.type == "tool_use")
        result = tool_use.input

        return {
            **account,
            'mismatch_category': result.get('mismatch_category'),
            'confidence': result.get('confidence'),
            'key_signals': ' | '.join(result.get('key_signals', [])),
            'reasoning': result.get('reasoning'),
            'action_recommended': result.get('action_recommended'),
            'ai_scored': True
        }

    except Exception as e:
        return {
            **account,
            'mismatch_category': 'Error',
            'confidence': None,
            'key_signals': str(e),
            'reasoning': None,
            'action_recommended': None,
            'ai_scored': False
        }


def score_all_enriched_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Run mismatch detection on all enriched accounts"""

    results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f"Analyzing: {i}/{total} — {row.get('COMBO_COMPANY_NAME', '')}")

        scored = score_account_with_claude(row.to_dict())
        results.append(scored)
        time.sleep(0.3)

    result_df = pd.DataFrame(results)
    result_df.to_csv('data/scored_accounts.csv', index=False)

    # Print summary stats
    if 'mismatch_category' in result_df.columns:
        print("\nMismatch Detection Summary:")
        print(result_df['mismatch_category'].value_counts())

    return result_df
