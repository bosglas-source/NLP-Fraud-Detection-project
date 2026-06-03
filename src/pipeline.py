import pandas as pd
import numpy as np
import re
import warnings

def _indicator_to_label(value):
    """
    Convert a raw OpenTender indicator value to a binary label.
    - value >= 50  → 1 (flagged: majority of lots triggered the flag)
    - value == 0   → 0 (clean: no lots triggered the flag)
    - value is None or 0 < value < 50 → NaN (insufficient or ambiguous data)
    """
    if pd.isna(value):
        return np.nan
    if value >= 50:
        return 1
    if value == 0:
        return 0
    return np.nan

def compute_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes deterministic binary indicators for procurement irregularities.
    Adds 'single_bid', 'short_tender_period', and 'winner_concentration' to the dataframe.
    """
    # 1. Single Bidding Indicator
    if 'ind_single_bid' in df.columns:
        df['single_bid'] = df['ind_single_bid'].apply(_indicator_to_label)
    else:
        warnings.warn("Column 'ind_single_bid' not found. 'single_bid' defaulted to NaN.")
        df['single_bid'] = np.nan

    # 2. Short Tender Period Indicator
    if 'ind_advertisement_period' in df.columns:
        df['short_tender_period'] = df['ind_advertisement_period'].apply(_indicator_to_label)
    else:
        warnings.warn("Column 'ind_advertisement_period' not found. 'short_tender_period' defaulted to NaN.")
        df['short_tender_period'] = np.nan

    # 3. Winner Concentration Indicator (Top 5% buyers by contract volume)
    if 'buyer_contracts_count' in df.columns:
        threshold = df['buyer_contracts_count'].quantile(0.95)
        df['winner_concentration'] = (df['buyer_contracts_count'] >= threshold).astype(int)
    else:
        warnings.warn("Column 'buyer_contracts_count' not found. 'winner_concentration' defaulted to NaN.")
        df['winner_concentration'] = np.nan

    # Combined master flag
    # .eq(1).any() ensures that NaNs are safely treated as non-suspicious
    target_cols = ['single_bid', 'short_tender_period', 'winner_concentration']
    df['is_suspicious'] = df[target_cols].eq(1).any(axis=1).astype(int)
    
    return df

def clean_tender_text(text: str) -> str:
    """Standardizes text, removing URLs, basic HTML fragments, and white space boilerplate."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"https?://\S+", "", text)  # Strip URLs
    text = re.sub(r'<[^>]*>', ' ', text)      # Strip basic HTML tags
    text = re.sub(r'\s+', ' ', text)          # Normalize spacing
    
    return text.strip()
