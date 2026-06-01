import pandas as pd
import numpy as np
import re

def compute_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes deterministic binary indicators for procurement irregularities.
    Adds 'single_bid', 'short_tender_period', and 'winner_concentration' to the dataframe.
    """
    # 1. Single Bidding Indicator
    if 'bids_count' in df.columns:
        df['single_bid'] = (df['bids_count'] == 1).astype(int)
    else:
        df['single_bid'] = 0

    # 2. Short Tender Period Indicator (e.g., less than 30 statutory days)
    if 'tender_duration_days' in df.columns:
        df['short_tender_period'] = (df['tender_duration_days'] < 30).astype(int)
    else:
        df['short_tender_period'] = 0

    # 3. Winner Concentration Indicator (>70% historical win rate with specific buyer)
    if 'buyer_id' in df.columns and 'supplier_id' in df.columns:
        buyer_total = df.groupby('buyer_id')['contract_id'].transform('count')
        pair_total = df.groupby(['buyer_id', 'supplier_id'])['contract_id'].transform('count')
        df['supplier_win_ratio'] = pair_total / buyer_total
        df['winner_concentration'] = (df['supplier_win_ratio'] > 0.70).astype(int)
    else:
        df['winner_concentration'] = 0

    # Combined master flag
    target_cols = ['single_bid', 'short_tender_period', 'winner_concentration']
    df['is_suspicious'] = df[target_cols].any(axis=1).astype(int)
    
    return df

def clean_tender_text(text: str) -> str:
    """Standardizes text, removing basic HTML fragments and white space boilerplate."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]*>', ' ', text)  # Strip basic HTML tags
    text = re.sub(r'\s+', ' ', text)     # Normalize spacing
    return text.strip()
