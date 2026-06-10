from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

DISCLAIMER = (
    "Academic prototype only. These outputs are statistical review signals, not accusations, "
    "not evidence of fraud, and not legal or regulatory conclusions. Any flagged contract "
    "would require human review against the original tender documents and market context."
)

FLAG_LABELS = {
    "single_bid": "Single bid",
    "winner_concentration": "Winner concentration",
    "copy_paste_description": "Copy-paste description",
    "shared_address_flag": "Shared address / location",
    "short_tender_period": "Short tender period",
    "isolation_forest_flag": "Embedding anomaly",
}

KNOWLEDGE_FALLBACK = [
    {
        "source": "EU procurement principle - competition",
        "topic": "single bidding",
        "text": (
            "Single-bid procedures are a procurement red flag because they may indicate weak "
            "competition, narrow specifications, limited market outreach, or tender design that "
            "discouraged alternative bidders."
        ),
    },
    {
        "source": "Procurement integrity typology - winner concentration",
        "topic": "winner concentration",
        "text": (
            "Winner concentration can reflect legitimate specialization, but it can also indicate "
            "supplier lock-in, weak competition, or favoritism. It should be interpreted with buyer "
            "and market context."
        ),
    },
    {
        "source": "NLP similarity red flag - copy-paste descriptions",
        "topic": "copy paste",
        "text": (
            "Near-identical descriptions across different buyers can indicate reused templates, "
            "centralized procurement, framework agreements, or potentially tailored specifications."
        ),
    },
    {
        "source": "NER entity-linking red flag - shared address or location",
        "topic": "entity linking",
        "text": (
            "Named Entity Recognition can extract organizations and locations from tender text. "
            "Shared addresses or repeated location mentions across supposedly independent records are "
            "review signals that may indicate legitimate shared administration, parent-company links, "
            "or potential shell-company patterns."
        ),
    },
    {
        "source": "Unsupervised anomaly detection guidance",
        "topic": "anomaly detection",
        "text": (
            "Embedding-space anomalies are unusual records compared with the broader corpus. "
            "Isolation Forest results should be used to prioritize review, not classify misconduct."
        ),
    },
]


def find_file(name: str) -> Path | None:
    for base in [
        ROOT,
        ROOT / "outputs_02",
        ROOT / "outputs_03",
        ROOT / "outputs_04",
        ROOT / "Notebook 2 Outputs",
        ROOT / "Notebook 3 Outputs",
        ROOT / "Notebook 4 Outputs",
    ]:
        path = base / name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_contracts(path_str: str) -> pd.DataFrame:
    usecols = [
        "contract_id",
        "country",
        "publication_date",
        "bid_deadline",
        "title",
        "description",
        "buyer_id",
        "buyer_name",
        "buyer_contracts_count",
        "score_integrity",
        "score_transparency",
        "score_tender",
        "single_bid",
        "short_tender_period",
        "winner_concentration",
        "copy_paste_description",
        "shared_address_flag",
        "extracted_companies",
        "extracted_locations",
        "num_extracted_companies",
        "num_extracted_locations",
        "has_extracted_company",
        "has_extracted_location",
    ]
    path = Path(path_str)
    header = pd.read_csv(path, nrows=0).columns
    available = [col for col in usecols if col in header]
    df = pd.read_csv(path, usecols=available, low_memory=False)

    for col in [
        "single_bid",
        "short_tender_period",
        "winner_concentration",
        "copy_paste_description",
        "shared_address_flag",
        "num_extracted_companies",
        "num_extracted_locations",
        "has_extracted_company",
        "has_extracted_location",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    for col in ["extracted_companies", "extracted_locations"]:
        if col not in df.columns:
            df[col] = "[]"

    df["target_suspicious"] = (
        df[["single_bid", "winner_concentration", "copy_paste_description", "shared_address_flag"]]
        .eq(1)
        .any(axis=1)
        .astype(int)
    )
    df["search_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["buyer_name"].fillna("").astype(str)
        + " "
        + df["contract_id"].fillna("").astype(str)
    ).str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_optional_csv(path_str: str | None) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame()
    return pd.read_csv(path_str, low_memory=False)


@st.cache_data(show_spinner=False)
def load_reports(path_str: str | None) -> dict[str, dict[str, Any]]:
    if not path_str:
        return {}
    with open(path_str, "r", encoding="utf-8") as f:
        reports = json.load(f)
    return {str(item.get("contract_id")): item for item in reports}


def as_flag(value: Any) -> bool:
    try:
        return int(float(value)) == 1
    except Exception:
        return False


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def clean_notice_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?i)\bNOTE\s*:", "\n\nNOTE:", text)
    text = re.sub(r"(?i)(documentation)(NOTE:)", r"\1\n\n\2", text)
    text = re.sub(r"(?i)(documents?)(NOTE:)", r"\1\n\n\2", text)
    text = re.sub(r"(?i)(attached documentation)(NOTE:)", r"\1\n\n\2", text)
    text = re.sub(r"(?i)(Please refer to attached documentation)\s*", r"\1.\n\n", text)
    text = re.sub(r"(?i)(https?://\S+)", r"\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" \.", ".", text)
    return text.strip()


def get_risk_factors(row: pd.Series) -> list[str]:
    factors = []
    for col in [
        "single_bid",
        "winner_concentration",
        "copy_paste_description",
        "shared_address_flag",
        "isolation_forest_flag",
    ]:
        if col in row and as_flag(row[col]):
            factors.append(col)
    if "short_tender_period" in row and as_flag(row["short_tender_period"]):
        factors.append("short_tender_period")
    return factors


def fallback_context(factors: list[str]) -> list[dict[str, Any]]:
    topic_map = {
        "single_bid": "single bidding",
        "winner_concentration": "winner concentration",
        "copy_paste_description": "copy paste",
        "shared_address_flag": "entity linking",
        "isolation_forest_flag": "anomaly detection",
    }
    topics = {topic_map.get(factor) for factor in factors}
    contexts = [item for item in KNOWLEDGE_FALLBACK if item["topic"] in topics]
    if not contexts:
        contexts = [KNOWLEDGE_FALLBACK[-1]]
    return [{**item, "score": 1.0} for item in contexts[:3]]


def fallback_report(row: pd.Series) -> dict[str, Any]:
    factors = get_risk_factors(row)
    score = 1
    score += 3 if "single_bid" in factors else 0
    score += 2 if "winner_concentration" in factors else 0
    score += 2 if "copy_paste_description" in factors else 0
    score += 2 if "shared_address_flag" in factors else 0
    score += 2 if "isolation_forest_flag" in factors else 0
    score += 1 if "short_tender_period" in factors else 0
    score = min(score, 10)
    level = "high" if score >= 7 else "medium" if score >= 4 else "low"

    readable = [FLAG_LABELS.get(factor, factor).lower() for factor in factors]
    if readable:
        explanation = (
            f"This contract is rated {level} risk because it triggered: {', '.join(readable)}. "
            "These are proxy indicators for human review and should not be interpreted as proof of wrongdoing."
        )
    else:
        explanation = (
            "This contract is rated low risk because it does not trigger the main proxy indicators used in this demo."
        )

    questions = [
        "Do the original tender documents support the model signal?",
        "Is there a legitimate procurement explanation for the flagged pattern?",
    ]
    if "copy_paste_description" in factors:
        questions.append("Are similar descriptions explained by a shared template, framework, or centralized buyer process?")
    if "shared_address_flag" in factors:
        questions.append("Do extracted company/location mentions suggest a legitimate shared site, parent entity, or administrative address?")
    if "winner_concentration" in factors:
        questions.append("Does the buyer repeatedly award similar contracts to the same supplier group?")
    if "single_bid" in factors:
        questions.append("Was market outreach broad enough to produce meaningful competition?")

    return {
        "contract_id": str(row.get("contract_id", "")),
        "title": str(row.get("title", "")),
        "buyer_name": None if pd.isna(row.get("buyer_name")) else str(row.get("buyer_name")),
        "risk_score": score,
        "risk_level": level,
        "risk_factors": factors,
        "retrieved_context": fallback_context(factors),
        "explanation": explanation,
        "human_review_questions": questions,
        "disclaimer": DISCLAIMER,
    }


def risk_badge(level: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}.get(str(level).lower(), "Unknown")


st.set_page_config(page_title="Procurement Risk Audit Assistant", layout="wide")

st.title("Procurement Risk Audit Assistant")
st.caption("NLP demo app for contract risk triage and explainability")
st.warning(DISCLAIMER)

features_path = find_file("contracts_ie_features.csv")
if features_path is None:
    st.error("Could not find contracts_ie_features.csv. Run Notebook 02 first.")
    st.stop()

contracts = load_contracts(str(features_path))

anomaly_path = find_file("anomaly_scores.csv")
anomalies = load_optional_csv(str(anomaly_path) if anomaly_path else None)
if not anomalies.empty:
    keep_cols = [col for col in ["contract_id", "anomaly_score", "isolation_forest_flag"] if col in anomalies.columns]
    contracts = contracts.merge(anomalies[keep_cols], on="contract_id", how="left")

reports = load_reports(str(find_file("risk_reports_sample.json")) if find_file("risk_reports_sample.json") else None)
similarity = load_optional_csv(str(find_file("similarity_scores.csv")) if find_file("similarity_scores.csv") else None)
model_comparison = load_optional_csv(str(find_file("model_comparison.csv")) if find_file("model_comparison.csv") else None)

with st.sidebar:
    st.header("Find A Contract")
    query = st.text_input("Search title, buyer, or ID", "")
    only_suspicious = st.checkbox("Only suspicious proxy labels", value=True)
    only_anomalies = st.checkbox("Only anomaly flags", value=False)
    max_rows = st.slider("Search results", 10, 200, 50, step=10)

    filtered = contracts
    if query.strip():
        filtered = filtered[filtered["search_text"].str.contains(query.strip().lower(), na=False, regex=False)]
    if only_suspicious:
        filtered = filtered[filtered["target_suspicious"] == 1]
    if only_anomalies and "isolation_forest_flag" in filtered.columns:
        filtered = filtered[filtered["isolation_forest_flag"].fillna(0).astype(float) == 1]

    filtered = filtered.head(max_rows).copy()
    if filtered.empty:
        st.info("No matching contracts. Try broadening the filters.")
        st.stop()

    if "selected_contract_id" not in st.session_state:
        st.session_state.selected_contract_id = str(filtered.iloc[0]["contract_id"])

    valid_ids = set(filtered["contract_id"].astype(str))
    if st.session_state.selected_contract_id not in valid_ids:
        st.session_state.selected_contract_id = str(filtered.iloc[0]["contract_id"])

    current_position = filtered.index[
        filtered["contract_id"].astype(str) == st.session_state.selected_contract_id
    ][0]
    current_list_position = list(filtered.index).index(current_position)

    nav_left, nav_right = st.columns(2)
    if nav_left.button("Previous", use_container_width=True, disabled=current_list_position == 0):
        st.session_state.selected_contract_id = str(filtered.iloc[current_list_position - 1]["contract_id"])
        st.rerun()
    if nav_right.button(
        "Next",
        use_container_width=True,
        disabled=current_list_position >= len(filtered) - 1,
    ):
        st.session_state.selected_contract_id = str(filtered.iloc[current_list_position + 1]["contract_id"])
        st.rerun()

    labels = (
        filtered["title"].fillna("(no title)").astype(str).str.slice(0, 75)
        + " | "
        + filtered["buyer_name"].fillna("unknown buyer").astype(str).str.slice(0, 35)
    )
    choice = st.selectbox(
        "Select contract",
        options=list(filtered.index),
        index=current_list_position,
        format_func=lambda idx: labels.loc[idx],
    )
    st.session_state.selected_contract_id = str(filtered.loc[choice, "contract_id"])
    selected_list_position = list(filtered.index).index(choice)
    st.caption(f"Showing {selected_list_position + 1} of {len(filtered):,} filtered contracts")

    preview_cols = ["title", "buyer_name", "target_suspicious"]
    if "anomaly_score" in filtered.columns:
        preview_cols.append("anomaly_score")
    st.dataframe(
        filtered[preview_cols].rename(
            columns={
                "title": "Title",
                "buyer_name": "Buyer",
                "target_suspicious": "Proxy",
                "anomaly_score": "Anomaly",
            }
        ),
        use_container_width=True,
        height=220,
    )

row = contracts.loc[choice]
contract_id = str(row["contract_id"])
report = fallback_report(row)

title_text = clean_notice_text(row.get("title", "")) or "(no title)"
description_text = clean_notice_text(row.get("description", ""))

left, right = st.columns([2.4, 1])
with left:
    st.subheader(title_text[:300])
    meta = " | ".join(
        part
        for part in [
            str(row.get("buyer_name", "") or "").strip(),
            str(row.get("country", "") or "").strip(),
            str(row.get("publication_date", "") or "")[:10],
        ]
        if part and part.lower() != "nan"
    )
    if meta:
        st.caption(meta)

with right:
    score = int(report.get("risk_score", 0))
    level = str(report.get("risk_level", "unknown")).lower()
    st.metric("RAG risk score", f"{score}/10", risk_badge(level))
    st.metric("Proxy suspicious", "Yes" if as_flag(row.get("target_suspicious")) else "No")
    if "anomaly_score" in row and not pd.isna(row.get("anomaly_score")):
        st.metric("Anomaly score", f"{float(row['anomaly_score']):.3f}")

st.divider()

with st.expander("Contract Description", expanded=True):
    if description_text:
        st.markdown(description_text[:2500].replace("\n", "  \n"))
        if len(description_text) > 2500:
            st.caption("Description truncated for display.")
    else:
        st.caption("No description available.")

st.subheader("Risk Signals")
flag_cols = st.columns(len(FLAG_LABELS))
for col, container in zip(FLAG_LABELS, flag_cols):
    value = as_flag(row.get(col))
    container.metric(FLAG_LABELS[col], "Flagged" if value else "Clear")

tab_report, tab_ner, tab_similarity, tab_model, tab_json = st.tabs(
    ["Audit Report", "NER Entities", "Similarity Evidence", "Model Summary", "JSON"]
)

with tab_report:
    st.markdown(f"**Risk level:** {risk_badge(level)}")
    st.markdown("**Risk factors:**")
    factors = report.get("risk_factors", [])
    if factors:
        for factor in factors:
            st.write(f"- {FLAG_LABELS.get(factor, factor)}")
    else:
        st.write("- No main risk factors triggered")

    st.markdown("**Explanation:**")
    st.write(report.get("explanation", "No explanation available."))

    st.markdown("**Retrieved guidance:**")
    for ctx in report.get("retrieved_context", []):
        with st.expander(f"{ctx.get('source', 'Context')}"):
            st.write(ctx.get("text", ""))

    st.markdown("**Human review questions:**")
    for question in report.get("human_review_questions", []):
        st.write(f"- {question}")

with tab_ner:
    st.markdown("**Extracted organizations / companies:**")
    companies_text = clean_notice_text(row.get("extracted_companies", "[]"))
    st.code(companies_text if companies_text else "[]", language="text")
    st.markdown("**Extracted locations:**")
    locations_text = clean_notice_text(row.get("extracted_locations", "[]"))
    st.code(locations_text if locations_text else "[]", language="text")
    entity_cols = st.columns(3)
    entity_cols[0].metric("Company entities", safe_int(row.get("num_extracted_companies", 0)))
    entity_cols[1].metric("Location entities", safe_int(row.get("num_extracted_locations", 0)))
    entity_cols[2].metric("Shared address flag", "Flagged" if as_flag(row.get("shared_address_flag")) else "Clear")

with tab_similarity:
    if similarity.empty:
        st.info("similarity_scores.csv was not found.")
    else:
        matches = similarity[
            (similarity["contract_id_a"].astype(str) == contract_id)
            | (similarity["contract_id_b"].astype(str) == contract_id)
        ].copy()
        if matches.empty:
            st.info("No near-duplicate matches found for this contract.")
        else:
            matches = matches.sort_values("cosine_sim", ascending=False).head(10)
            st.dataframe(
                matches[
                    [
                        col
                        for col in [
                            "cosine_sim",
                            "contract_id_a",
                            "buyer_name_a",
                            "title_a",
                            "contract_id_b",
                            "buyer_name_b",
                            "title_b",
                        ]
                        if col in matches.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with tab_model:
    if model_comparison.empty:
        st.info("model_comparison.csv was not found.")
    else:
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    pr_curve_path = find_file("precision_recall_curves.png")
    if pr_curve_path:
        st.image(str(pr_curve_path), caption="Notebook 03 precision-recall curves")

with tab_json:
    st.json(report)

st.divider()
st.caption(DISCLAIMER)
