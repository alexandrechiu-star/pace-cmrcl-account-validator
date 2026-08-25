import os
import time

import pandas as pd
import requests

NEWS_API_KEY = os.getenv('NEWS_API_KEY')


def get_news_signals(company_name: str) -> dict:
    """Pull recent news for a company from News API"""

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{company_name}"',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'language': 'en',
            'apiKey': NEWS_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get('status') == 'ok' and data.get('articles'):
            articles = data['articles']
            headlines = [a['title'] for a in articles[:5]]
            return {
                'has_news': True,
                'article_count': data['totalResults'],
                'recent_headlines': headlines,
                'latest_date': articles[0]['publishedAt'][:10] if articles else None
            }
        return {'has_news': False, 'article_count': 0, 'recent_headlines': [], 'latest_date': None}

    except Exception as e:
        return {'has_news': False, 'article_count': 0, 'recent_headlines': [], 'latest_date': None, 'error': str(e)}


def enrich_account(row: dict) -> dict:
    """Enrich a single account with all external signals"""

    company_name = row.get('COMBO_COMPANY_NAME', '')

    news = get_news_signals(company_name)

    # News API free tier: 100 requests/day
    time.sleep(0.5)

    return {
        **row,
        'news_has_coverage': news['has_news'],
        'news_article_count': news['article_count'],
        'news_headlines': ' | '.join(news['recent_headlines'][:3]),
        'news_latest_date': news['latest_date']
    }


def enrich_sample(df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """
    Enrich a sample of accounts — don't run all 18K at once.
    Prioritize accounts where the tier vs. ACV mismatch is already visible.
    """

    has_acv = df[df['H1_FY27_ACV'] > 0].copy()
    low_tier_with_acv = has_acv[has_acv['ACCT_TIER'].isin(['Tier 3', 'Tier 4', 'Untiered'])]

    no_acv = df[df['H1_FY27_ACV'] == 0].copy()
    high_tier_no_acv = no_acv[no_acv['ACCT_TIER'].isin(['Tier 1', 'Tier 2'])]

    half = max(sample_size // 2, 1)
    priority_accounts = pd.concat([
        low_tier_with_acv.head(half),
        high_tier_no_acv.head(sample_size - half)
    ])

    print(f"Enriching {len(priority_accounts)} priority accounts...")

    enriched = []
    for i, (_, row) in enumerate(priority_accounts.iterrows()):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(priority_accounts)}")
        enriched.append(enrich_account(row.to_dict()))

    result = pd.DataFrame(enriched)
    result.to_csv('data/enriched_accounts.csv', index=False)
    return result
