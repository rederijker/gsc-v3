"""
GSC InsightHub — SEO analytics tool with Google Search Console data.
OAuth fixed: offline access, refresh token handling, no PKCE (confidential web client).
"""

import concurrent.futures
import io
import itertools
import json
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timedelta

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu
from streamlit_raw_echarts import JsCode, st_echarts

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GSC InsightHub: SEO analytic tool with GSC data — Cristiano Caggiula",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
OAUTH_SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
REDIRECT_URI = "https://gsc-seo.streamlit.app/"


def get_oauth_client_config():
    return {
        "web": {
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "client_secret": st.secrets["gcp_service_account"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }


def credentials_from_session():
    raw = st.session_state.get("credentials")
    if raw is None:
        return None
    if isinstance(raw, Credentials):
        return raw
    return Credentials(
        token=raw.get("token"),
        refresh_token=raw.get("refresh_token"),
        token_uri=raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=raw.get("client_id"),
        client_secret=raw.get("client_secret"),
        scopes=raw.get("scopes", OAUTH_SCOPES),
    )


def save_credentials_to_session(creds: Credentials) -> None:
    st.session_state.credentials = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else OAUTH_SCOPES,
    }


def refresh_credentials_if_needed(creds: Credentials) -> Credentials:
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_credentials_to_session(creds)
    return creds


def authorize_app() -> Credentials:
    creds = credentials_from_session()
    if creds:
        try:
            return refresh_credentials_if_needed(creds)
        except Exception as exc:
            st.warning(f"Token scaduto o non valido: {exc}. Effettua di nuovo il login.")
            st.session_state.credentials = None

    flow = Flow.from_client_config(
        get_oauth_client_config(),
        scopes=OAUTH_SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    auth_code = st.query_params.get("code")

    if auth_code:
        try:
            flow.fetch_token(code=auth_code)
        except Exception as exc:
            st.error(f"Errore durante fetch_token: {exc}")
            st.stop()

        creds = flow.credentials

        if not creds.refresh_token:
            st.error(
                "Google non ha inviato un refresh token. "
                "Vai su https://myaccount.google.com/permissions, revoca l'accesso a questa app, "
                "poi rifai login."
            )
            st.stop()

        save_credentials_to_session(creds)
        st.query_params.clear()
        st.rerun()

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="false",
    )

    gif_url = "https://github.com/rederijker/gsc-v3/blob/main/assets/back.gif?raw=true"
    st.markdown(
        f"""
        <div class="header" style="padding:5%;background-image:url({gif_url});">
            <h1 style="text-align:center;">GSC InsightHub</h1>
            <p style="text-align:center;">made with 🎈 by
                <a href="https://www.linkedin.com/in/cristiano-caggiula/">Cristiano Caggiula</a>
            </p>
            <h2 style="text-align:center;">Master Google Search Console Data like a Pro</h2>
            <p style="text-align:center;font-size:17px;">
                Explore <strong>Google Search Console data</strong>, generate detailed reports,
                customize searches, and access unlimited information without programming skills.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("🔑 Login con Google", auth_url)
    st.stop()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "credentials": None,
    "selected_site": None,
    "df": None,
    "available_sites": [],
    "dimension_filters": {},
    "selected_page": None,
    "page_data": None,
    "keyword_analysis": None,
    "data_loaded": False,
    "keyword_groups": None,
    "click_totals": None,
    "download_ready": False,
    "scan_started": False,
    "search_query": "",
    "show_heading": True,
    "show_keyword_metrics": True,
    "show_meta": True,
    "show_body_alt": True,
    "show_not_covered": False,
    "selected_group": None,
    "selected_page_on_page": None,
    "action": "Update URL",
    "urls": "",
    "json_file": None,
    "api_app": "***GET INSIGHT FROM MY GSC DATA***",
    "urls_to_inspect": "",
    "inspection_results": None,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

REQUIRED_COLUMNS = ["Page", "Query", "Clicks", "Impressions", "CTR", "Position"]


def clear_data():
    st.session_state.df = None


# ---------------------------------------------------------------------------
# Helpers — page scraping & keyword analysis
# ---------------------------------------------------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def fetch_page_data(page_url):
    warnings = []
    try:
        response = requests.get(page_url, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            meta_title = clean_text(soup.find("title").text) if soup.find("title") else ""
            if not meta_title:
                warnings.append("Meta Title is missing.")
            meta_description = soup.find("meta", attrs={"name": "description"})
            meta_description = clean_text(meta_description["content"]) if meta_description else ""
            if not meta_description:
                warnings.append("Meta Description is missing.")
            h1_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h1"))
            )
            if not h1_headings:
                warnings.append("H1 Headings are missing.")
            h2_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h2"))
            )
            if not h2_headings:
                warnings.append("H2 Headings are missing.")
            h3_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h3"))
            )
            h4_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h4"))
            )
            h5_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h5"))
            )
            h6_headings = " ".join(
                set(clean_text(tag.get_text(separator=" ")) for tag in soup.find_all("h6"))
            )
            body_content = " ".join(
                set(
                    clean_text(p.get_text(separator=" "))
                    for p in soup.find_all(["p", "div", "span", "li"])
                )
            )
            alt_tags = " ".join(set(clean_text(img.get("alt", "")) for img in soup.find_all("img")))
            return {
                "meta_title": meta_title,
                "meta_description": meta_description,
                "h1_headings": h1_headings,
                "h2_headings": h2_headings,
                "h3_headings": h3_headings,
                "h4_headings": h4_headings,
                "h5_headings": h5_headings,
                "h6_headings": h6_headings,
                "body_content": body_content,
                "alt_tags": alt_tags,
                "warnings": warnings,
            }
        st.error(f"Failed to fetch the page content. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the page content: {e}")
    return None


def aggregate_queries(df):
    return (
        df.groupby("Query")
        .agg({"Clicks": "sum", "Impressions": "sum", "CTR": "mean", "Position": "mean"})
        .reset_index()
    )


def determine_optimal_clusters(X, max_clusters=50):
    silhouette_scores = []
    K = range(2, min(max_clusters + 1, X.shape[0]))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    if len(K) == 0:
        return 1
    return silhouette_scores.index(max(silhouette_scores)) + 2


def cluster_keywords(keywords_df, max_clusters=20):
    vectorizer = TfidfVectorizer(stop_words="english")
    keywords_df = aggregate_queries(keywords_df)
    X = vectorizer.fit_transform(keywords_df["Query"])
    if X.shape[0] < 2:
        st.warning("Not enough samples to perform clustering. At least 2 unique queries are required.")
        return keywords_df, None
    optimal_k = determine_optimal_clusters(X, max_clusters)
    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    model.fit(X)
    keywords_df["Cluster"] = model.labels_
    return keywords_df, model


def analyze_topic_coverage(page_data, clustered_keywords):
    parts = [
        clean_text(page_data[k])
        for k in [
            "meta_title", "meta_description", "h1_headings", "h2_headings",
            "h3_headings", "h4_headings", "h5_headings", "h6_headings",
            "body_content", "alt_tags",
        ]
    ]
    full_content = " ".join(parts)
    clustered_keywords = clustered_keywords.copy()
    clustered_keywords["Covered"] = clustered_keywords["Query"].apply(
        lambda x: clean_text(x) in full_content
    )
    return clustered_keywords


def analyze_keywords(page_data, keywords_df):
    fields = {
        "Title": clean_text(page_data["meta_title"]),
        "Meta Description": clean_text(page_data["meta_description"]),
        "H1": clean_text(page_data["h1_headings"]),
        "H2": clean_text(page_data["h2_headings"]),
        "H3": clean_text(page_data["h3_headings"]),
        "H4": clean_text(page_data["h4_headings"]),
        "H5": clean_text(page_data["h5_headings"]),
        "H6": clean_text(page_data["h6_headings"]),
        "Body Content": clean_text(page_data["body_content"]),
        "Alt Tags": clean_text(page_data["alt_tags"]),
    }
    keyword_analysis = []
    for _, row in keywords_df.iterrows():
        keyword_clean = clean_text(row["Query"])
        entry = {
            "Keyword": row["Query"],
            "Clicks": row["Clicks"],
            "Impressions": row["Impressions"],
            "CTR": row["CTR"],
            "Position": row["Position"],
        }
        for col, content in fields.items():
            entry[col] = keyword_clean in content
        keyword_analysis.append(entry)
    return keyword_analysis


def get_opportunity_keywords(keywords_df, keyword_analysis):
    avg_impressions = keywords_df["Impressions"].mean()
    opportunity_keywords = keywords_df[
        (keywords_df["Impressions"] > avg_impressions)
        & (keywords_df["Position"] >= 1)
        & (keywords_df["Position"] <= 20)
    ]
    analysis_by_kw = {item["Keyword"]: item for item in keyword_analysis}
    not_covered = []
    for keyword in opportunity_keywords["Query"]:
        analysis_row = analysis_by_kw.get(keyword)
        if not analysis_row:
            continue
        if not any(
            analysis_row[k]
            for k in [
                "Title", "Meta Description", "H1", "H2", "H3",
                "H4", "H5", "H6", "Body Content", "Alt Tags",
            ]
        ):
            not_covered.append(keyword)
    return opportunity_keywords[opportunity_keywords["Query"].isin(not_covered)][
        ["Query", "Position", "Clicks", "Impressions", "CTR"]
    ]


def get_cluster_names(clustered_keywords):
    cluster_names = {}
    for cluster in clustered_keywords["Cluster"].unique():
        cluster_data = clustered_keywords[clustered_keywords["Cluster"] == cluster]
        top_keyword = cluster_data.loc[cluster_data["Impressions"].idxmax()]["Query"]
        cluster_names[cluster] = top_keyword
    return cluster_names


def remove_duplicates_and_sum_clicks(df, keyword_column, clicks_column):
    return df.groupby(keyword_column, as_index=False).agg({clicks_column: "sum"})


def group_keywords(df, stop_words, min_group_size, ngram_size, keyword_column):
    all_words = list(itertools.chain(*df[keyword_column].str.lower().str.split()))
    word_freq = Counter(all_words)
    common_terms = {
        word for word, freq in word_freq.items()
        if freq > 1 and word not in stop_words and len(word) > 2
    }
    grouped_dfs = []
    for keyword in df[keyword_column]:
        words = re.findall(r"\b\w+\b", keyword.lower())
        if len(words) >= ngram_size:
            ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
            groups = set()
            for ngram in ngrams:
                if all(term in common_terms or term.isdigit() for term in ngram):
                    groups.add(" ".join(ngram))
            if groups:
                grouped_dfs.extend(
                    [pd.DataFrame({"Group": [group], "Keywords": [keyword]}) for group in groups]
                )
    if not grouped_dfs:
        return pd.DataFrame(columns=["Group", "Keywords"])
    grouped_keywords_df = pd.concat(grouped_dfs, ignore_index=True)
    return grouped_keywords_df.groupby("Group").filter(lambda x: len(x) >= min_group_size)


def calculate_click_totals(df, grouped_df, keyword_column, clicks_column):
    click_totals = {}
    for group in grouped_df["Group"].unique():
        keywords_in_group = grouped_df[grouped_df["Group"] == group]["Keywords"].tolist()
        click_totals[group] = df[df[keyword_column].isin(keywords_in_group)][clicks_column].sum()
    return click_totals


def fetch_data_chunk(
    webmasters_service, site_url, start_date, end_date,
    dimensions, filters, selected_type, start_row, row_limit=None,
):
    request_body = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "dimensions": dimensions,
        "startRow": start_row,
        "type": selected_type,
        "rowLimit": min(row_limit, 25000) if row_limit else 25000,
    }
    for dimension, filter_info in filters.items():
        filter_value = filter_info["filter_value"]
        if filter_value:
            if "dimensionFilterGroups" not in request_body:
                request_body["dimensionFilterGroups"] = []
            request_body["dimensionFilterGroups"].append({
                "filters": [{
                    "dimension": dimension.lower(),
                    "expression": filter_value,
                    "operator": filter_info["operator"],
                }]
            })
    response_data = webmasters_service.searchanalytics().query(
        siteUrl=site_url, body=request_body
    ).execute()
    return response_data.get("rows", [])


def inspect_url(url_to_inspect, webmasters_service, selected_site, retries=3):
    request_body = {"inspectionUrl": url_to_inspect, "siteUrl": selected_site}
    error_fields = {
        "index_status_verdict": "ERROR",
        "index_status_coverage_state": "ERROR",
        "index_status_robots_txt_state": "ERROR",
        "index_status_indexing_state": "ERROR",
        "index_status_last_crawl_time": "ERROR",
        "index_status_page_fetch_state": "ERROR",
        "index_status_google_canonical": "ERROR",
        "index_status_user_canonical": "ERROR",
        "index_status_sitemap": "ERROR",
        "index_status_referring_urls": "ERROR",
        "index_status_crawled_as": "ERROR",
    }
    for attempt in range(retries):
        try:
            response = webmasters_service.urlInspection().index().inspect(body=request_body).execute()
            inspection_result = response.get("inspectionResult", {})
            index_status_result = inspection_result.get("indexStatusResult", {})
            return {
                "url": url_to_inspect,
                "index_status_verdict": index_status_result.get("verdict", "N/A"),
                "index_status_coverage_state": index_status_result.get("coverageState", "N/A"),
                "index_status_robots_txt_state": index_status_result.get("robotsTxtState", "N/A"),
                "index_status_indexing_state": index_status_result.get("indexingState", "N/A"),
                "index_status_last_crawl_time": index_status_result.get("lastCrawlTime", "N/A"),
                "index_status_page_fetch_state": index_status_result.get("pageFetchState", "N/A"),
                "index_status_google_canonical": index_status_result.get("googleCanonical", "N/A"),
                "index_status_user_canonical": index_status_result.get("userCanonical", "N/A"),
                "index_status_sitemap": ", ".join(index_status_result.get("sitemap", [])),
                "index_status_referring_urls": ", ".join(index_status_result.get("referringUrls", [])),
                "index_status_crawled_as": index_status_result.get("crawledAs", "N/A"),
                "response": response,
            }
        except HttpError as err:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"url": url_to_inspect, **error_fields, "response": str(err)}


def convert_df_to_zip(df, file_name="data.csv"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(file_name, df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)
    return buffer


def rows_to_dataframe(rows, dimensions):
    data_list = []
    for row in rows:
        data_entry = {dim: row["keys"][dimensions.index(dim)] for dim in dimensions}
        data_entry.update({
            "Clicks": row["clicks"],
            "Impressions": row["impressions"],
            "CTR": row["ctr"],
            "Position": row["position"],
        })
        data_list.append(data_entry)
    return pd.DataFrame(data_list)


# ---------------------------------------------------------------------------
# Performance reports (pages & queries)
# ---------------------------------------------------------------------------
def _half_period_metrics(df, group_col):
    if "Date" not in df.columns or group_col not in df.columns:
        st.warning(f"Add 'Date' and '{group_col}' to dimensions for this report.")
        return None

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    start_date = df["Date"].min().date()
    end_date = df["Date"].max().date()
    midpoint = start_date + (end_date - start_date) / 2

    first_half = df[df["Date"].dt.date <= midpoint]
    second_half = df[df["Date"].dt.date > midpoint]

    agg = {"Impressions": "sum", "Clicks": "sum", "CTR": "mean", "Position": "mean"}

    first = first_half.groupby(group_col).agg(agg).reset_index().rename(columns={
        "Impressions": "Impressions_First_Half",
        "Clicks": "Clicks_First_Half",
        "CTR": "CTR_First_Half",
        "Position": "Position_First_Half",
    })
    second = second_half.groupby(group_col).agg(agg).reset_index().rename(columns={
        "Impressions": "Impressions_Second_Half",
        "Clicks": "Clicks_Second_Half",
        "CTR": "CTR_Second_Half",
        "Position": "Position_Second_Half",
    })

    perf = pd.merge(first, second, on=group_col, how="outer").fillna(0)
    for metric in ["Impressions", "Clicks", "CTR", "Position"]:
        perf[f"{metric}_Change"] = perf[f"{metric}_Second_Half"] - perf[f"{metric}_First_Half"]

    return {
        "df": perf,
        "start_date": start_date,
        "end_date": end_date,
        "midpoint": midpoint,
        "first_half": first,
        "group_col": group_col,
    }


def _render_traffic_change_report(ctx, title, entity_label, section_number):
    if ctx is None:
        return

    perf = ctx["df"]
    group_col = ctx["group_col"]
    first_half = ctx["first_half"]
    start_date = ctx["start_date"]
    end_date = ctx["end_date"]
    midpoint = ctx["midpoint"]

    gained = perf[perf["Clicks_Change"] > 0]
    lost = perf[perf["Clicks_Change"] < 0]
    stable = perf[perf["Clicks_Change"] == 0]

    total_clicks_first = first_half["Clicks_First_Half"].sum()
    total_impressions_first = first_half["Impressions_First_Half"].sum()
    avg_ctr_first = first_half["CTR_First_Half"].mean()
    avg_position_first = first_half["Position_First_Half"].mean()

    overall_impressions = perf["Impressions_Change"].sum()
    overall_clicks = perf["Clicks_Change"].sum()
    overall_ctr = perf["CTR_Change"].mean()
    overall_position = perf["Position_Change"].mean()

    clicks_pct = (overall_clicks / total_clicks_first * 100) if total_clicks_first else 0
    impressions_pct = (overall_impressions / total_impressions_first * 100) if total_impressions_first else 0
    position_pct = (overall_position / avg_position_first * 100) if avg_position_first else 0

    display_cols = [
        group_col,
        "Clicks_First_Half", "Clicks_Second_Half", "Clicks_Change",
        "Impressions_First_Half", "Impressions_Second_Half", "Impressions_Change",
        "CTR_First_Half", "CTR_Second_Half", "CTR_Change",
        "Position_First_Half", "Position_Second_Half", "Position_Change",
    ]
    fmt = {
        "Clicks_First_Half": "{:.0f}", "Clicks_Second_Half": "{:.0f}", "Clicks_Change": "{:.0f}",
        "Impressions_First_Half": "{:.0f}", "Impressions_Second_Half": "{:.0f}", "Impressions_Change": "{:.0f}",
        "CTR_First_Half": "{:.2%}", "CTR_Second_Half": "{:.2%}", "CTR_Change": "{:.2%}",
        "Position_First_Half": "{:.2f}", "Position_Second_Half": "{:.2f}", "Position_Change": "{:.2f}",
    }

    with st.container(border=True):
        st.subheader(f"{section_number}. {title}")
        st.divider()
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        with col1:
            st.markdown(
                f"""
                This report divides the time period into two halves and compares them.
                - **First half**: {start_date} to {midpoint}
                - **Second half**: {midpoint + pd.Timedelta(days=1)} to {end_date}
                """
            )
        with col2:
            st.metric("Total Clicks Change", f"{overall_clicks:.0f}", f"{clicks_pct:.2f}%")
        with col3:
            st.metric("Total Impressions Change", f"{overall_impressions:.0f}", f"{impressions_pct:.2f}%")
        with col4:
            st.metric("Average CTR Change", f"{overall_ctr * 100:.2f}%", f"{overall_ctr * 100:.2f}%")
        with col5:
            delta_color = "inverse" if overall_position < 0 else "normal"
            st.metric("Average Position Change", f"{overall_position:.2f}", f"{position_pct:.2f}%", delta_color=delta_color)

        fig = px.bar(
            {"Traffic Change": ["Gained Traffic", "Lost Traffic", "No Changes"],
             "Count": [len(gained), len(lost), len(stable)]},
            x="Traffic Change", y="Count",
            color="Traffic Change",
            color_discrete_map={"Gained Traffic": "#32CD32", "Lost Traffic": "coral", "No Changes": "grey"},
            title="Traffic Change Overview",
        )
        fig.update_layout(showlegend=False, paper_bgcolor="rgb(10,14,18)", plot_bgcolor="rgb(10,14,18)")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            for label, value, positive_good in [
                ("impressions", overall_impressions, True),
                ("clicks", overall_clicks, True),
                ("CTR", overall_ctr, True),
            ]:
                fn = st.success if (value > 0) == positive_good else st.warning
                trend = "increasing" if value > 0 else "decreasing"
                fn(f"The overall trend of {label} is {trend}.")
            if overall_position < 0:
                st.success("The overall trend of average position is improving (lowering).")
            else:
                st.warning("The overall trend of average position is worsening (rising).")

        with st.expander(f"{entity_label.upper()} THAT GAINED TRAFFIC ⬆️"):
            st.dataframe(gained[display_cols].reset_index(drop=True).style.format(fmt))
        with st.expander(f"{entity_label.upper()} THAT LOST TRAFFIC ⬇️"):
            st.dataframe(lost[display_cols].reset_index(drop=True).style.format(fmt))
            threshold = st.slider(
                "Set the percentage of total loss",
                min_value=1, max_value=100, value=10, step=5,
                format="%d%%",
                key=f"threshold_{group_col}",
            ) / 100.0
            if overall_clicks != 0:
                significant = lost[abs(lost["Clicks_Change"]) > abs(overall_clicks) * threshold]
                if not significant.empty:
                    st.dataframe(significant[display_cols].reset_index(drop=True).style.format(fmt))
        with st.expander(f"{entity_label.upper()} WITH NO CHANGES TRAFFIC ➡️"):
            st.dataframe(stable[display_cols].reset_index(drop=True).style.format(fmt))


def analyze_page_performance(df):
    ctx = _half_period_metrics(df, "Page")
    _render_traffic_change_report(ctx, "Pages Traffic Changes Report", "Pages", "2")


def analyze_query_performance(df):
    ctx = _half_period_metrics(df, "Query")
    _render_traffic_change_report(ctx, "Queries Traffic Changes Report", "Queries", "4")


def analyze_query_position_changes(df):
    if "Date" not in df.columns or "Query" not in df.columns:
        st.warning("Add 'Date' and 'Query' to dimensions to show Query Position Changes Report")
        return

    df_pos = df.copy()
    df_pos["Date"] = pd.to_datetime(df_pos["Date"])
    start_date = df_pos["Date"].min().date()
    end_date = df_pos["Date"].max().date()
    midpoint = start_date + (end_date - start_date) / 2

    first_half = df_pos[df_pos["Date"].dt.date <= midpoint]
    second_half = df_pos[df_pos["Date"].dt.date > midpoint]

    first = first_half.groupby("Query").agg({
        "Position": "mean", "Clicks": "sum", "Impressions": "sum"
    }).reset_index().rename(columns={
        "Position": "Position_First_Half",
        "Clicks": "Clicks_First_Half",
        "Impressions": "Impressions_First_Half",
    })
    second = second_half.groupby("Query").agg({
        "Position": "mean", "Clicks": "sum", "Impressions": "sum"
    }).reset_index().rename(columns={
        "Position": "Position_Second_Half",
        "Clicks": "Clicks_Second_Half",
        "Impressions": "Impressions_Second_Half",
    })

    perf = pd.merge(first, second, on="Query", how="outer")
    perf["Position_Change"] = perf["Position_Second_Half"] - perf["Position_First_Half"]

    improved = perf[perf["Position_Change"] < 0]
    worsened = perf[perf["Position_Change"] > 0]
    stable = perf[perf["Position_Change"] == 0]
    out_of_serp = perf[(perf["Impressions_First_Half"] > 0) & (perf["Impressions_Second_Half"].isna())]

    overall_position = perf["Position_Change"].mean()
    avg_position_first = first["Position_First_Half"].mean()
    position_pct = (overall_position / avg_position_first * 100) if avg_position_first else 0

    include_missing = st.checkbox("Include queries with missing data in one of the periods", value=True)
    if not include_missing:
        perf = perf.dropna(subset=["Position_First_Half", "Position_Second_Half"])

    pos_cols = [
        "Query", "Position_First_Half", "Position_Second_Half", "Position_Change",
        "Clicks_First_Half", "Clicks_Second_Half", "Impressions_First_Half", "Impressions_Second_Half",
    ]
    pos_fmt = {
        "Position_First_Half": "{:.2f}", "Position_Second_Half": "{:.2f}", "Position_Change": "{:.2f}",
        "Clicks_First_Half": "{:.0f}", "Clicks_Second_Half": "{:.0f}",
        "Impressions_First_Half": "{:.0f}", "Impressions_Second_Half": "{:.0f}",
    }

    with st.container(border=True):
        st.subheader("5. Query Position Changes Report")
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""
                - **First half**: {start_date} to {midpoint}
                - **Second half**: {midpoint + pd.Timedelta(days=1)} to {end_date}
                """
            )
        with col2:
            delta_color = "inverse" if overall_position < 0 else "normal"
            st.metric("Average Position Change", f"{overall_position:.2f}", f"{position_pct:.2f}%", delta_color=delta_color)

        fig = px.bar(
            {"Position Change": ["Improved Position", "Worsened Position", "No Changes"],
             "Count": [len(improved), len(worsened), len(stable)]},
            x="Position Change", y="Count", color="Position Change",
            color_discrete_map={"Improved Position": "#32CD32", "Worsened Position": "coral", "No Changes": "grey"},
        )
        fig.update_layout(showlegend=False, paper_bgcolor="rgb(10,14,18)", plot_bgcolor="rgb(10,14,18)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("QUERIES THAT IMPROVED POSITION ⬆️"):
            st.dataframe(improved[pos_cols].reset_index(drop=True).style.format(pos_fmt))
        with st.expander("QUERIES THAT WORSENED POSITION ⬇️"):
            st.dataframe(worsened[pos_cols].reset_index(drop=True).style.format(pos_fmt))
        with st.expander("QUERIES WITH NO POSITION CHANGES ➡️"):
            st.dataframe(stable[pos_cols].reset_index(drop=True).style.format(pos_fmt))
        with st.expander("QUERIES THAT DROPPED OUT OF SERP ❌"):
            st.dataframe(out_of_serp[["Query", "Position_First_Half", "Clicks_First_Half", "Impressions_First_Half"]].reset_index(drop=True).style.format({
                "Position_First_Half": "{:.2f}", "Clicks_First_Half": "{:.0f}", "Impressions_First_Half": "{:.0f}",
            }))


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
hr { margin: 0em 0px; }
.st-emotion-cache-qcpnpn {
    border: 2px solid rgb(3 169 244 / 50%);
    border-radius: 0.5rem;
    padding: calc(-1px + 0.9rem);
    background: rgb(0 0 0 / 24%);
    box-shadow: 0 15px 25px rgba(0, 0, 0, .6);
}
h3 { color: #00BCD4; text-align: center; }
</style>
""", unsafe_allow_html=True)

pd.set_option("styler.render.max_elements", 20000000)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
credentials = authorize_app()

if credentials:
    webmasters_service = build("searchconsole", "v1", credentials=credentials)

    if not st.session_state.available_sites:
        site_list = webmasters_service.sites().list().execute()
        st.session_state.available_sites = [
            site["siteUrl"] for site in site_list.get("siteEntry", [])
        ]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.selected_site = st.selectbox(
            "Select a website:", st.session_state.available_sites
        )
    with col2:
        api_options = [
            "***GET INSIGHT FROM MY GSC DATA***",
            "***BULK INSPECT URLS***",
            "***INDEXING API***",
        ]
        api_app = st.radio(
            "What do you feel like doing?",
            api_options,
            captions=["Laugh out loud.", "Get the popcorn.", "I'm feel lucky."],
            horizontal=True,
            index=api_options.index(st.session_state.api_app),
        )
        if api_app != st.session_state.api_app:
            st.session_state.api_app = api_app
            st.rerun()

    st.divider()

    # ===================================================================
    # GSC DATA
    # ===================================================================
    if st.session_state.api_app == "***GET INSIGHT FROM MY GSC DATA***":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            options_type = {"Web": "web", "News": "news", "Discover": "discover", "Image": "image", "Video": "video"}
            today = datetime.now()

            def get_start_date(option):
                mapping = {
                    "Last 7 days": 7, "Last 28 days": 28, "Last 3 months": 90,
                    "Last 6 months": 180, "Last 12 months": 365, "Last 16 months": 480,
                }
                days = mapping.get(option)
                return today - timedelta(days=days) if days else None

            selected_type_key = st.selectbox("CHANNEL", list(options_type.keys()))
            selected_type = options_type[selected_type_key]
            time_options = [
                "Last 7 days", "Last 28 days", "Last 3 months", "Last 6 months",
                "Last 12 months", "Last 16 months", "Custom range",
            ]
            selected_time_option = st.selectbox("Select range", time_options)
            if selected_time_option == "Custom range":
                start_date = st.date_input("Start date", pd.to_datetime(today - timedelta(days=90)))
                end_date = st.date_input("End date", pd.to_datetime(today))
            else:
                start_date = get_start_date(selected_time_option)
                end_date = today

        with col2:
            selected_dimensions = st.multiselect(
                "DIMENSIONS", ["Date", "Page", "Query", "Device", "Country"],
                default=["Date", "Query", "Page"],
            )
            with st.expander("Filters for Dimensions"):
                unique_key = 0
                for dimension in selected_dimensions:
                    c1, c2 = st.columns(2)
                    with c1:
                        operator = st.selectbox(
                            f"{dimension} operator",
                            ["equals", "contains", "notEquals", "notContains", "includingRegex", "excludingRegex"],
                            key=f"op_{dimension}",
                        )
                    with c2:
                        filter_value = st.text_input("Value", placeholder="value", key=f"val_{unique_key}")
                    unique_key += 1
                    st.session_state.dimension_filters[dimension] = {
                        "operator": operator, "filter_value": filter_value,
                    }

        with col3:
            check_box_row = st.radio("SET ROW LIMIT?", ["No", "Yes"])
            row_limit = (
                st.number_input("Row limit", min_value=1, max_value=25000, value=25000)
                if check_box_row == "Yes" else None
            )

        if st.button("GET DATA ⬇️"):
            clear_data()
            if st.session_state.selected_site:
                dimensions = list(selected_dimensions)
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.session_state.df = pd.DataFrame()

                try:
                    if row_limit is not None:
                        with st.spinner("Downloading data with row limit..."):
                            rows = fetch_data_chunk(
                                webmasters_service, st.session_state.selected_site,
                                start_date, end_date, dimensions,
                                st.session_state.dimension_filters, selected_type, 0, row_limit,
                            )
                            if rows:
                                st.session_state.df = rows_to_dataframe(rows, dimensions)
                                st.session_state.data_loaded = True
                                st.session_state.download_ready = True
                            else:
                                st.warning("No data retrieved for the selected period.")
                            progress_bar.progress(1.0)
                    else:
                        total_days = (end_date - start_date).days + 1
                        completed_days = 0
                        with st.spinner("Downloading data without row limit..."):
                            current_date = start_date
                            while current_date <= end_date:
                                next_date = current_date + timedelta(days=1)
                                start_row = 0
                                while True:
                                    rows = fetch_data_chunk(
                                        webmasters_service, st.session_state.selected_site,
                                        current_date, next_date, dimensions,
                                        st.session_state.dimension_filters, selected_type,
                                        start_row, 25000,
                                    )
                                    if not rows:
                                        break
                                    chunk_df = rows_to_dataframe(rows, dimensions)
                                    st.session_state.df = pd.concat(
                                        [st.session_state.df, chunk_df], ignore_index=True
                                    )
                                    start_row += len(rows)
                                    status_text.text(f"Rows downloaded this batch: {len(rows)}")
                                    if len(rows) < 25000:
                                        break
                                completed_days += 1
                                current_date = next_date
                                progress_bar.progress(completed_days / total_days)
                            st.session_state.data_loaded = True
                            st.session_state.download_ready = True
                            progress_bar.progress(1.0)
                except HttpError as e:
                    st.warning(f"HTTP Error: {e}")

        if st.session_state.data_loaded and st.session_state.download_ready:
            zip_buffer = convert_df_to_zip(st.session_state.df)
            st.download_button(
                label="Download data ZIP", data=zip_buffer,
                file_name="data.zip", mime="application/zip",
            )

            selected2 = option_menu(
                None,
                ["GSC DATA OVERVIEW", "QUERIES REPORT", "PAGES REPORT", "PAGE OPTIMIZATION", "QUERIES GROUPER"],
                icons=["graph-up-arrow", "key", "file-earmark-text", "rocket", "intersect"],
                menu_icon="cast", default_index=0, orientation="horizontal",
            )

            df = st.session_state.df

            # ----------------------------------------------------------
            # OVERVIEW
            # ----------------------------------------------------------
            if selected2 == "GSC DATA OVERVIEW":
                average_ctr_perc = df["CTR"].mean() * 100
                formatted_ctr_m = f"{average_ctr_perc:.2f}%"
                st.session_state["average_ctr_perc"] = average_ctr_perc
                st.session_state["formatted_ctr_m"] = formatted_ctr_m

                with st.container(border=True):
                    copy_website_data = df.copy()
                    st.subheader("Website Data")
                    st.divider()
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    with col1:
                        st.write(
                            f"Performance overview **from** {start_date.strftime('%Y-%m-%d')} "
                            f"**to** {end_date.strftime('%Y-%m-%d')}"
                        )
                        with st.expander("Filters"):
                            dim_cols = [
                                c for c in copy_website_data.columns
                                if c not in ["Date", "Clicks", "Impressions", "CTR", "Position"]
                            ]
                            selected_dimension = st.selectbox("Select Dimension", dim_cols)
                            if selected_dimension == "Query":
                                search_query = st.text_input("Enter Query (supports regex)")
                                if search_query:
                                    copy_website_data = copy_website_data[
                                        copy_website_data["Query"].str.contains(
                                            search_query, case=False, regex=True, na=False
                                        )
                                    ]
                            else:
                                selected_values = st.multiselect(
                                    f"Select {selected_dimension}",
                                    options=copy_website_data[selected_dimension].unique(),
                                )
                                if selected_values:
                                    copy_website_data = copy_website_data[
                                        copy_website_data[selected_dimension].isin(selected_values)
                                    ]
                            if not copy_website_data.empty:
                                min_pos = int(copy_website_data["Position"].min())
                                max_pos = int(copy_website_data["Position"].max())
                                min_ctr = float(copy_website_data["CTR"].min() * 100)
                                max_ctr = float(copy_website_data["CTR"].max() * 100)
                            else:
                                min_pos, max_pos, min_ctr, max_ctr = 0, 1, 0.0, 1.0
                            if min_pos == max_pos:
                                max_pos += 1
                            pos_range = st.slider("Select Position Range", min_pos, max_pos, (min_pos, max_pos))
                            if min_ctr == max_ctr:
                                max_ctr += 1.0
                            ctr_range = st.slider("Select CTR Range (%)", min_ctr, max_ctr, (min_ctr, max_ctr))
                            if not copy_website_data.empty:
                                copy_website_data = copy_website_data[
                                    (copy_website_data["Position"] >= pos_range[0])
                                    & (copy_website_data["Position"] <= pos_range[1])
                                    & (copy_website_data["CTR"] * 100 >= ctr_range[0])
                                    & (copy_website_data["CTR"] * 100 <= ctr_range[1])
                                ]
                        total_clicks = copy_website_data["Clicks"].sum() if not copy_website_data.empty else 0
                        total_impressions = copy_website_data["Impressions"].sum() if not copy_website_data.empty else 0
                        average_position = copy_website_data["Position"].mean() if not copy_website_data.empty else 0
                        average_ctr = copy_website_data["CTR"].mean() * 100 if not copy_website_data.empty else 0
                    with col2:
                        st.metric("Total Clicks", total_clicks)
                    with col3:
                        st.metric("Total Impressions", total_impressions)
                    with col4:
                        st.metric("Average Position", f"{average_position:.2f}")
                    with col5:
                        st.metric("Average CTR", f"{average_ctr:.2f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(copy_website_data, width=2000, height=520)
                    with col2:
                        if not copy_website_data.empty and "Date" in copy_website_data.columns:
                            df_graf = copy_website_data.groupby("Date").agg({
                                "Clicks": "sum", "Impressions": "sum", "CTR": "mean", "Position": "mean",
                            }).reset_index()
                            df_graf["CTR"] = df_graf["CTR"].apply(lambda ctr: f"{ctr * 100:.2f}")
                            df_graf["Position"] = df_graf["Position"].apply(lambda pos: round(pos, 2))
                            options = {
                                "xAxis": {"type": "category", "data": df_graf["Date"].tolist()},
                                "yAxis": [
                                    {"type": "value"},
                                    {"type": "value", "inverse": True, "show": False},
                                ],
                                "grid": {"right": 20, "left": 65, "top": 45, "bottom": 50},
                                "legend": {"show": True, "top": "top"},
                                "tooltip": {"trigger": "axis"},
                                "series": [
                                    {"type": "line", "name": "Clicks", "data": df_graf["Clicks"].tolist(), "smooth": True},
                                    {"type": "line", "name": "Impressions", "data": df_graf["Impressions"].tolist(), "smooth": True},
                                    {"type": "line", "name": "CTR", "data": df_graf["CTR"].tolist(), "smooth": True},
                                    {"type": "line", "name": "Position", "data": df_graf["Position"].tolist(), "smooth": True, "yAxisIndex": 1},
                                ],
                                "backgroundColor": "#0a0e12",
                            }
                            st_echarts(option=options, theme="chalk", height=500, width="100%")
                        else:
                            st.warning("No graph available without date data.")

            # ----------------------------------------------------------
            # QUERIES REPORT
            # ----------------------------------------------------------
            elif selected2 == "QUERIES REPORT":
                formatted_ctr_m = st.session_state.get("formatted_ctr_m", "0.00%")

                with st.container(border=True):
                    st.subheader("1. Queries Performance Report")
                    st.divider()
                    if "Query" in selected_dimensions:
                        df_query_performance = df.groupby("Query").agg({
                            "Impressions": "sum", "Clicks": "sum", "CTR": "mean", "Position": "mean",
                        }).reset_index()
                        df_query_performance["CTR"] = df_query_performance["CTR"] * 100
                        avg_ctr_query = df_query_performance["CTR"].mean()
                        avg_position_query = df_query_performance["Position"].mean()

                        fig = px.scatter(
                            df_query_performance, x="CTR", y="Position", size="Clicks",
                            hover_data={"Query": True, "CTR": ":.2f", "Position": ":.2f", "Clicks": True},
                            custom_data=["Query"],
                        )
                        fig.update_yaxes(autorange="reversed")
                        fig.update_traces(
                            marker=dict(sizemin=4),
                            hovertemplate="<b>Query:</b> %{customdata[0]}<br><b>CTR:</b> %{x:.2f}%<br><b>Position:</b> %{y:.2f}<br><b>Clicks:</b> %{marker.size}",
                        )
                        fig.update_layout(
                            title="Query Performance Bubble Chart",
                            paper_bgcolor="rgb(10,14,18)", plot_bgcolor="rgb(10,14,18)",
                        )
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col2:
                            st.metric("Queries", df_query_performance["Query"].nunique())
                        with col3:
                            st.metric("AVG. CTR", f"{avg_ctr_query:.2f}%")
                        with col4:
                            st.metric("AVG. Position", f"{avg_position_query:.2f}")
                        st.plotly_chart(fig, use_container_width=True)

                if "Page" in df.columns and "Query" in df.columns:
                    with st.container():
                        st.subheader("3. Queries Cannibalization Report")
                        st.divider()
                        df_can = df.copy()
                        df_can["Cleaned_Page"] = df_can["Page"].apply(lambda x: x.split("#")[0])
                        query_page_metrics = df_can.groupby(["Query", "Cleaned_Page"]).agg({
                            "Position": "mean", "CTR": "mean", "Clicks": "sum", "Impressions": "sum",
                        }).reset_index()
                        query_page_group = query_page_metrics.groupby("Query")["Cleaned_Page"].apply(
                            lambda pages: list(set(pages))
                        ).reset_index()
                        query_page_group.columns = ["Query", "Pages"]
                        cannibalized = query_page_group[query_page_group["Pages"].apply(len) > 1]
                        cannibalization_report = cannibalized.explode("Pages").merge(
                            query_page_metrics, left_on=["Query", "Pages"], right_on=["Query", "Cleaned_Page"]
                        ).drop(columns=["Pages"])
                        cannibalized["Cannibals Pages"] = cannibalized["Pages"].apply(len)
                        st.metric("Unique cannibalized queries", cannibalized["Query"].nunique())
                        st.dataframe(
                            cannibalized[["Query", "Cannibals Pages"]].sort_values("Cannibals Pages", ascending=False)
                        )
                        query_selected = st.selectbox("Select a cannibalized Query", cannibalized["Query"])
                        st.dataframe(cannibalization_report[cannibalization_report["Query"] == query_selected])
                else:
                    st.warning("Add 'Page' to dimensions to show Queries Cannibalization Report")

                analyze_query_performance(df)
                analyze_query_position_changes(df)

            # ----------------------------------------------------------
            # PAGES REPORT
            # ----------------------------------------------------------
            elif selected2 == "PAGES REPORT":
                formatted_ctr_m = st.session_state.get("formatted_ctr_m", "0.00%")
                if "Page" not in df.columns:
                    st.warning("Add 'Page' to dimensions to show Pages Report")
                else:
                    agg = df.groupby("Page").agg({
                        "Impressions": "sum", "Clicks": "sum", "CTR": "mean", "Position": "mean",
                    }).reset_index()
                    agg["CTR_pct"] = agg["Clicks"] / agg["Impressions"]
                    avg_clicks = agg["Clicks"].mean()
                    avg_impressions = agg["Impressions"].mean()
                    avg_position = agg["Position"].mean()
                    avg_ctr_val = df["CTR"].mean()

                    with st.container(border=True):
                        st.subheader("1. Pages Health Check Report")
                        st.divider()
                        st.metric("Pages", len(agg))
                        popular = agg[
                            (agg["CTR_pct"] > avg_ctr_val) & (agg["Clicks"] > avg_clicks)
                            & (agg["Impressions"] > avg_impressions) & (agg["Position"] < 10)
                        ].sort_values("Clicks", ascending=False)
                        with st.expander(":green[BEST PAGES]"):
                            st.dataframe(popular)
                    analyze_page_performance(df)

            # ----------------------------------------------------------
            # PAGE OPTIMIZATION
            # ----------------------------------------------------------
            elif selected2 == "PAGE OPTIMIZATION":
                with st.container(border=True):
                    st.subheader("1. Queries Coverage Analysis")
                    st.divider()
                    if "Page" in df.columns and "Query" in df.columns:
                        selected_page_on_page = st.selectbox("Select a page", df["Page"].unique())
                        scan_button = st.button("Analyze Page🤖", key="scan_button")
                    else:
                        st.warning("Dimensions must include 'Page' and 'Query'.")
                        scan_button = False
                        selected_page_on_page = None

                    if scan_button or st.session_state.scan_started:
                        if scan_button:
                            st.session_state.scan_started = True
                        if selected_page_on_page and selected_page_on_page != st.session_state.selected_page_on_page:
                            st.session_state.selected_page_on_page = selected_page_on_page
                            with st.spinner("Fetching page data..."):
                                st.session_state.page_data = fetch_page_data(selected_page_on_page)

                        if st.session_state.page_data and st.session_state.selected_page_on_page:
                            page_data = df[df["Page"] == st.session_state.selected_page_on_page][
                                ["Query", "Clicks", "Impressions", "CTR", "Position"]
                            ]
                            grouped_page_data = aggregate_queries(page_data)
                            for warning in st.session_state.page_data.get("warnings", []):
                                st.warning(warning)
                            keyword_presence = analyze_keywords(st.session_state.page_data, grouped_page_data)
                            keyword_df = pd.DataFrame(keyword_presence)
                            st.dataframe(keyword_df)
                            st.subheader("Prioritize the Optimization of These Queries")
                            st.dataframe(get_opportunity_keywords(grouped_page_data, keyword_presence))

                with st.container(border=True):
                    st.subheader("2. Page Topics")
                    st.divider()
                    if st.session_state.page_data and st.session_state.selected_page_on_page:
                        page_data = df[df["Page"] == st.session_state.selected_page_on_page][
                            ["Query", "Clicks", "Impressions", "CTR", "Position"]
                        ]
                        grouped_page_data = aggregate_queries(page_data)
                        with st.spinner("Clustering topics..."):
                            clustered_keywords, _ = cluster_keywords(grouped_page_data)
                            if clustered_keywords is not None and "Cluster" in clustered_keywords.columns:
                                cluster_names = get_cluster_names(clustered_keywords)
                                clustered_keywords = analyze_topic_coverage(
                                    st.session_state.page_data, clustered_keywords
                                )
                                for cluster in clustered_keywords["Cluster"].unique():
                                    cluster_df = clustered_keywords[clustered_keywords["Cluster"] == cluster]
                                    covered = cluster_df["Covered"].sum()
                                    total = len(cluster_df)
                                    pct = (covered / total * 100) if total else 0
                                    with st.expander(
                                        f"**{cluster_names[cluster].upper()}** | "
                                        f"Clicks {cluster_df['Clicks'].sum()} | "
                                        f"Coverage {pct:.2f}%"
                                    ):
                                        st.dataframe(cluster_df)

            # ----------------------------------------------------------
            # QUERIES GROUPER
            # ----------------------------------------------------------
            elif selected2 == "QUERIES GROUPER":
                st.subheader("Queries Grouper")
                col1, col2 = st.columns([1, 2])
                with col1:
                    language = st.selectbox("Language", ["English", "Italian"])
                    default_stop_words = (
                        ["and", "but", "is", "the", "to", "in", "for", "on", "with", "as", "by", "at", "from"]
                        if language == "English"
                        else ["a", "adesso", "ai", "al", "alla", "che", "con", "da", "di", "e", "il", "la", "le", "per", "un", "una"]
                    )
                    with st.expander("Customize Stop Words"):
                        custom = st.text_area("One per line", "\n".join(default_stop_words))
                        stop_words = [w.strip() for w in custom.split("\n") if w.strip()]
                with col2:
                    min_group_size = st.slider("Minimum Group Size", 1, 50, 2)
                    ngram_size = st.slider("Length of the keyword", 1, 5, 2)

                if st.button("Start Grouping ▶"):
                    with st.spinner("Grouping..."):
                        if "Query" in df.columns and "Clicks" in df.columns:
                            df_cleaned = remove_duplicates_and_sum_clicks(df, "Query", "Clicks")
                            st.session_state.keyword_groups = group_keywords(
                                df_cleaned, stop_words, min_group_size, ngram_size, "Query"
                            )
                            st.session_state.click_totals = calculate_click_totals(
                                df_cleaned, st.session_state.keyword_groups, "Query", "Clicks"
                            )
                        else:
                            st.warning("DataFrame must contain 'Query' and 'Clicks'.")

                if st.session_state.keyword_groups is not None and st.session_state.click_totals:
                    sorted_groups = sorted(st.session_state.click_totals.items(), key=lambda x: x[1], reverse=True)
                    overview = []
                    for group, total_clicks in sorted_groups:
                        kws = st.session_state.keyword_groups[
                            st.session_state.keyword_groups["Group"] == group
                        ]["Keywords"].tolist()
                        overview.append({"Group": group, "Total Clicks": total_clicks, "Keywords": ", ".join(kws)})
                    overview_df = pd.DataFrame(overview)
                    st.dataframe(overview_df.drop(columns=["Keywords"]))
                    selected_group = st.selectbox("Select a Group", overview_df["Group"])
                    if selected_group:
                        kws = st.session_state.keyword_groups[
                            st.session_state.keyword_groups["Group"] == selected_group
                        ]["Keywords"].tolist()
                        st.dataframe(df[df["Query"].isin(kws)][["Query", "Clicks"]])

    # ===================================================================
    # BULK URL INSPECTION
    # ===================================================================
    elif st.session_state.api_app == "***BULK INSPECT URLS***":
        st.subheader("BULK INSPECT URLs")
        st.session_state.urls_to_inspect = st.text_area(
            "Insert URLs to inspect (one per line):",
            value=st.session_state.urls_to_inspect,
            height=200,
        ).rstrip()

        if st.button("URL INSPECTION 🕵️‍♂️"):
            if st.session_state.selected_site:
                urls = [u.strip() for u in st.session_state.urls_to_inspect.splitlines() if u.strip()]
                results = []
                progress_placeholder = st.empty()
                start_time = time.time()

                with st.spinner("Inspecting URLs..."):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        futures = {
                            executor.submit(
                                inspect_url, url, webmasters_service, st.session_state.selected_site
                            ): url for url in urls
                        }
                        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                            try:
                                results.append(future.result())
                            except Exception as e:
                                results.append({"url": futures[future], "response": str(e)})
                            elapsed = time.time() - start_time
                            eta = (elapsed / (idx + 1)) * (len(urls) - idx - 1)
                            progress_placeholder.write(
                                f"Processing {idx + 1}/{len(urls)} — ETA {int(eta // 60)}m {int(eta % 60)}s"
                            )

                progress_placeholder.empty()
                index_results = pd.DataFrame(results)
                st.dataframe(index_results.drop(columns=["response"], errors="ignore"))
                st.download_button(
                    "Download results as CSV",
                    index_results.to_csv(index=False),
                    file_name="inspection_results.csv",
                    mime="text/csv",
                )

    # ===================================================================
    # INDEXING API
    # ===================================================================
    elif st.session_state.api_app == "***INDEXING API***":
        with st.expander("How to use the Google Indexing API"):
            st.markdown("""
            1. Create a project on [Google Cloud Console](https://console.cloud.google.com/)
            2. Enable **Indexing API**
            3. Create a **Service Account** and download the JSON key
            4. Add the service account to **Google Search Console** as owner/user
            """)

        st.session_state.json_file = st.file_uploader(
            "Upload Google Cloud JSON credentials file", type=["json"]
        )
        st.session_state.action = st.selectbox(
            "Action", ["Update URL", "Remove URL", "Check URL Status"],
            index=["Update URL", "Remove URL", "Check URL Status"].index(st.session_state.action),
        )
        st.session_state.urls = st.text_area("Enter URLs:", st.session_state.urls)

        if st.button("Execute"):
            if not st.session_state.urls or not st.session_state.json_file:
                st.error("Enter URLs and upload the JSON file.")
            else:
                sa_creds = service_account.Credentials.from_service_account_info(
                    json.load(st.session_state.json_file)
                )
                indexing_service = build("indexing", "v3", credentials=sa_creds)
                for url in st.session_state.urls.splitlines():
                    url = url.strip()
                    if not url:
                        continue
                    try:
                        if st.session_state.action == "Update URL":
                            body = {"url": url, "type": "URL_UPDATED"}
                            response = indexing_service.urlNotifications().publish(body=body).execute()
                            st.success(f"Updated: {url}")
                        elif st.session_state.action == "Remove URL":
                            body = {"url": url, "type": "URL_DELETED"}
                            response = indexing_service.urlNotifications().publish(body=body).execute()
                            st.success(f"Removed: {url}")
                        else:
                            response = indexing_service.urlNotifications().getMetadata(url=url).execute()
                        st.json(response)
                    except HttpError as e:
                        error_content = json.loads(e.content.decode("utf-8"))
                        st.error(f"HTTP {e.status_code}: {error_content['error']['message']}")
                    except Exception as e:
                        st.error(str(e))
