from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import streamlit as st
import httplib2
import pandas as pd
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow
from oauth2client.file import Storage
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go  # Importa il modulo go da Plotly
import time
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_raw_echarts import st_echarts, JsCode
from datetime import datetime, timedelta
import altair as alt
# clustering
from collections import Counter
import itertools
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Google Search Console API",
    page_icon="🔍",
    layout="wide"
)


# Initialize session state
if 'credentials' not in st.session_state:
    st.session_state.credentials = None
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'available_sites' not in st.session_state:
    st.session_state.available_sites = []
if 'dimension_filters' not in st.session_state:
    st.session_state.dimension_filters = {}
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = None
if 'page_data' not in st.session_state:
    st.session_state.page_data = None
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'keyword_groups' not in st.session_state:
    st.session_state.keyword_groups = None
if 'click_totals' not in st.session_state:
    st.session_state.click_totals = None


#AUTH APP
OAUTH_SCOPE = ['https://www.googleapis.com/auth/webmasters.readonly']
REDIRECT_URI = 'https://seo-tool.streamlit.app/'  # Updated redirect URI


def authorize_app():
    client_config = {
        "web": {
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_secret": st.secrets["gcp_service_account"]["client_secret"],
            "redirect_uris": st.secrets["gcp_service_account"]["redirect_uris"]
        }
    }

    flow = Flow.from_client_config(client_config, scopes=OAUTH_SCOPE)
    flow.redirect_uri = REDIRECT_URI

    query_params = st.experimental_get_query_params()
    auth_code = query_params.get('code', None)

    if auth_code and not st.session_state.credentials:
        try:
            flow.fetch_token(code=auth_code[0])
            credentials = flow.credentials
            st.session_state.credentials = credentials
        except Exception as e:
            st.write(f"Error during authorization: {e}")

    if st.session_state.credentials is None:
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.write(f"➡️ Go to [this link]({auth_url}) and authorize app")
    
    return st.session_state.credentials

@st.cache_data(show_spinner=False)
def fetch_data_chunk(_webmasters_service, site_url, start_date, end_date, dimensions, filters, selected_type, start_row, row_limit=None):
    request_body = {
        "startDate": start_date.strftime('%Y-%m-%d'),
        "endDate": end_date.strftime('%Y-%m-%d'),
        "dimensions": dimensions,
        "startRow": start_row,
        "type": selected_type,
        "rowLimit": min(row_limit, 25000) if row_limit else 25000
    }

    for dimension, filter_info in filters.items():
        filter_operator = filter_info['operator']
        filter_value = filter_info['filter_value']
        if filter_value:
            if 'dimensionFilterGroups' not in request_body:
                request_body['dimensionFilterGroups'] = []
            request_body['dimensionFilterGroups'].append({
                'filters': [{
                    'dimension': dimension,
                    'expression': filter_value,
                    'operator': filter_operator
                }]
            })

    response_data = _webmasters_service.searchanalytics().query(siteUrl=site_url, body=request_body).execute()
    return response_data.get('rows', [])

#TAB3 PAGE OPTIMIZATION
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

# Funzione per estrarre i dati della pagina
def fetch_page_data(page_url):
    warnings = []
    try:
        response = requests.get(page_url)
        if response.status_code == 200:
            html_content = response.content
            soup = BeautifulSoup(html_content, 'html.parser')

            meta_title = clean_text(soup.find('title').text) if soup.find('title') else ''
            if not meta_title:
                warnings.append("Meta Title is missing.")
            meta_description = soup.find('meta', attrs={'name': 'description'})
            meta_description = clean_text(meta_description['content']) if meta_description else ''
            if not meta_description:
                warnings.append("Meta Description is missing.")
            
            h1_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h1')]))
            if not h1_headings:
                warnings.append("H1 Headings are missing.")
            h2_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h2')]))
            if not h2_headings:
                warnings.append("H2 Headings are missing.")
            h3_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h3')]))
            h4_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h4')]))
            h5_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h5')]))
            h6_headings = ' '.join(set([clean_text(tag.get_text(separator=" ")) for tag in soup.find_all('h6')]))
            body_content = ' '.join(set([clean_text(p.get_text(separator=" ")) for p in soup.find_all(['p', 'div', 'span', 'li'])]))
            alt_tags = ' '.join(set([clean_text(img.get('alt', '')) for img in soup.find_all('img')]))

            return {
                'meta_title': meta_title,
                'meta_description': meta_description,
                'h1_headings': h1_headings,
                'h2_headings': h2_headings,
                'h3_headings': h3_headings,
                'h4_headings': h4_headings,
                'h5_headings': h5_headings,
                'h6_headings': h6_headings,
                'body_content': body_content,
                'alt_tags': alt_tags,
                'warnings': warnings
            }
        else:
            st.error(f"Failed to fetch the page content. Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the page content: {e}")
    return None

# Funzione per aggregare le query duplicate
def aggregate_queries(df):
    return df.groupby('Query').agg({
        'Clicks': 'sum',
        'Impressions': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index()

#TAB3 TOPIC CLUSTER K-MEAN

def determine_optimal_clusters(X, max_clusters=20):
    distortions = []
    silhouette_scores = []
    K = range(2, min(max_clusters + 1, X.shape[0]))  # Assicurati che max_clusters non sia maggiore del numero di campioni
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    if len(K) == 0:
        return 1  # Se ci sono meno di 2 campioni, ritorna 1 come numero di cluster   

    
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # since range starts from 2
    return optimal_k

def cluster_keywords(keywords_df, max_clusters=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Rimuovere parole chiave duplicate
    keywords_df = aggregate_queries(keywords_df)
    
    X = vectorizer.fit_transform(keywords_df['Query'])
    
    if X.shape[0] < 2:
        st.warning("Not enough samples to perform clustering. At least 2 unique queries are required.")
        return keywords_df, None
    
    optimal_k = determine_optimal_clusters(X, max_clusters)
    
    model = KMeans(n_clusters=optimal_k, random_state=42)
    model.fit(X)
    
    keywords_df['Cluster'] = model.labels_
    return keywords_df, model
  
# Funzione per analizzare la copertura degli argomenti
def analyze_topic_coverage(page_data, clustered_keywords):
    meta_title_clean = clean_text(page_data['meta_title'])
    meta_description_clean = clean_text(page_data['meta_description'])
    h1_headings_clean = clean_text(page_data['h1_headings'])
    h2_headings_clean = clean_text(page_data['h2_headings'])
    h3_headings_clean = clean_text(page_data['h3_headings'])
    h4_headings_clean = clean_text(page_data['h4_headings'])
    h5_headings_clean = clean_text(page_data['h5_headings'])
    h6_headings_clean = clean_text(page_data['h6_headings'])
    body_content_clean = clean_text(page_data['body_content'])
    alt_tags_clean = clean_text(page_data['alt_tags'])
    full_content = f"{meta_title_clean} {meta_description_clean} {h1_headings_clean} {h2_headings_clean} {h3_headings_clean} {h4_headings_clean} {h5_headings_clean} {h6_headings_clean} {body_content_clean} {alt_tags_clean}"

    clustered_keywords['Covered'] = clustered_keywords['Query'].apply(lambda x: True if clean_text(x) in full_content else False)
    return clustered_keywords
  

# Funzione per ottenere parole chiave con opportunità
def get_opportunity_keywords(keywords_df, keyword_analysis):
    avg_impressions = keywords_df['Impressions'].mean()
    opportunity_keywords = keywords_df[(keywords_df['Impressions'] > avg_impressions) & (keywords_df['Position'] >= 1) & (keywords_df['Position'] <= 20)]
    
    # Filtra le keyword che non sono presenti in nessun elemento della pagina
    not_covered_keywords = []
    for keyword in opportunity_keywords['Query']:
        analysis_row = next(item for item in keyword_analysis if item['Keyword'] == keyword)
        if not any([analysis_row['Title'], analysis_row['Meta Description'], analysis_row['H1'], analysis_row['H2'], analysis_row['H3'], analysis_row['H4'], analysis_row['H5'], analysis_row['H6'], analysis_row['Body Content'], analysis_row['Alt Tags']]):
            not_covered_keywords.append(keyword)
    
    opportunity_keywords = opportunity_keywords[opportunity_keywords['Query'].isin(not_covered_keywords)]
    return opportunity_keywords[['Query', 'Position', 'Clicks', 'Impressions', 'CTR']]

# Funzione per analizzare la presenza delle parole chiave nei vari elementi della pagina
def analyze_keywords(page_data, keywords_df):
    meta_title_clean = clean_text(page_data['meta_title'])
    meta_description_clean = clean_text(page_data['meta_description'])
    h1_headings_clean = clean_text(page_data['h1_headings'])
    h2_headings_clean = clean_text(page_data['h2_headings'])
    h3_headings_clean = clean_text(page_data['h3_headings'])
    h4_headings_clean = clean_text(page_data['h4_headings'])
    h5_headings_clean = clean_text(page_data['h5_headings'])
    h6_headings_clean = clean_text(page_data['h6_headings'])
    body_content_clean = clean_text(page_data['body_content'])
    alt_tags_clean = clean_text(page_data['alt_tags'])

    keyword_analysis = []
    for _, row in keywords_df.iterrows():
        keyword = row['Query']
        keyword_clean = clean_text(keyword)

        in_meta_title = keyword_clean in meta_title_clean
        in_meta_description = keyword_clean in meta_description_clean
        in_h1_headings = keyword_clean in h1_headings_clean
        in_h2_headings = keyword_clean in h2_headings_clean
        in_h3_headings = keyword_clean in h3_headings_clean
        in_h4_headings = keyword_clean in h4_headings_clean
        in_h5_headings = keyword_clean in h5_headings_clean
        in_h6_headings = keyword_clean in h6_headings_clean
        in_body_content = keyword_clean in body_content_clean
        in_alt_tags = keyword_clean in alt_tags_clean

        keyword_analysis.append({
            'Keyword': keyword,
            'Clicks':row['Clicks'],
            'Impressions': row['Impressions'],
            'CTR': row['CTR'],
            'Position': row['Position'],
            'Title': in_meta_title,
            'Meta Description': in_meta_description,
            'H1': in_h1_headings,
            'H2': in_h2_headings,
            'H3': in_h3_headings,
            'H4': in_h4_headings,
            'H5': in_h5_headings,
            'H6': in_h6_headings,
            'Body Content': in_body_content,
            'Alt Tags': in_alt_tags
        })

    return keyword_analysis

# Funzione per ottenere il nome del cluster basato sulla parola chiave con le impressioni più alte
def get_cluster_names(clustered_keywords):
    cluster_names = {}
    for cluster in clustered_keywords['Cluster'].unique():
        cluster_data = clustered_keywords[clustered_keywords['Cluster'] == cluster]
        top_keyword = cluster_data.loc[cluster_data['Impressions'].idxmax()]['Query']
        cluster_names[cluster] = top_keyword
    return cluster_names

#TAB4 KEYWORD GROUPING
#TAB4 KEYWORD GROUPING
#TAB4 KEYWORD GROUPING
#TAB4 KEYWORD GROUPING

# Function to remove duplicates and sum clicks
def remove_duplicates_and_sum_clicks(df, keyword_column, clicks_column):
    # Raggruppa per parola chiave e somma i clic
    df_grouped = df.groupby(keyword_column, as_index=False).agg({clicks_column: 'sum'})
    return df_grouped

# Function to determine keyword groups based on term frequency
def group_keywords(df, stop_words, min_group_size, ngram_size, keyword_column):
    # Count the frequency of all words in keywords
    all_words = list(itertools.chain(*df[keyword_column].str.lower().str.split()))
    word_freq = Counter(all_words)
    # Select only words that are common but not too common (exclude stop words and words that are too short)
    common_terms = {word for word, freq in word_freq.items() if freq > 1 and word not in stop_words and len(word) > 2}

    # Create a list of DataFrames for the groups
    grouped_dfs = []

    # Associate each keyword with the most common term it contains
    for keyword in df[keyword_column]:
        words = re.findall(r'\b\w+\b', keyword.lower())  # Extract complete words from the keyword
        if len(words) >= ngram_size:
            ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
            groups = set()
            for ngram in ngrams:
                if all(term in common_terms or term.isdigit() for term in ngram):  # Also consider numbers
                    groups.add(" ".join(ngram))
            if groups:
                grouped_dfs.extend([pd.DataFrame({'Group': [group], 'Keywords': [keyword]}) for group in groups])

    # Concatenate the DataFrames in the list into a single DataFrame
    grouped_keywords_df = pd.concat(grouped_dfs, ignore_index=True)

    # Filter groups with a minimum size
    filtered_groups = grouped_keywords_df.groupby('Group').filter(lambda x: len(x) >= min_group_size)

    return filtered_groups

# Function to calculate the total clicks for each group
def calculate_click_totals(df, grouped_df, keyword_column, clicks_column):
    click_totals = {}
    for group in grouped_df['Group'].unique():
        keywords_in_group = grouped_df[grouped_df['Group'] == group]['Keywords'].tolist()
        click_total = df[df[keyword_column].isin(keywords_in_group)][clicks_column].sum()
        click_totals[group] = click_total
    return click_totals

# Funzione per analizzare il trend di impressions e clicks
def analyze_page_performance(df):
    # Creare una copia del DataFrame di partenza per evitare conflitti con altre analisi
    df_page_performance_analysis = df.copy()
    
    # Assicurati che il DataFrame contenga una colonna "Date"
    if 'Date' not in df_page_performance_analysis.columns:
        st.error("The DataFrame must contain a 'Date' column.")
        return
    
    # Conversione della colonna 'Date' in datetime
    df_page_performance_analysis['Date'] = pd.to_datetime(df_page_performance_analysis['Date'])
    
    # Utilizza l'intervallo di date nel DataFrame
    start_date_page_performance_analysis = df_page_performance_analysis['Date'].min().date() # Converte in oggetto date
    end_date_page_performance_analysis = df_page_performance_analysis['Date'].max().date()  # Converte in oggetto date
    
    # Filtra il DataFrame in base alle date disponibili
    filtered_df_page_performance_analysis = df_page_performance_analysis[(df_page_performance_analysis['Date'].dt.date >= start_date_page_performance_analysis) & (df_page_performance_analysis['Date'].dt.date <= end_date_page_performance_analysis)]
    
    # Definisci i periodi di confronto
    midpoint_page_performance_analysis = start_date_page_performance_analysis + (end_date_page_performance_analysis - start_date_page_performance_analysis) / 2
    
    first_half_df_page_performance_analysis = filtered_df_page_performance_analysis[filtered_df_page_performance_analysis['Date'].dt.date <= midpoint_page_performance_analysis]
    second_half_df_page_performance_analysis = filtered_df_page_performance_analysis[filtered_df_page_performance_analysis['Date'].dt.date > midpoint_page_performance_analysis]
    
    # Calcola le somme di impressions e clicks, e le medie di CTR e posizione media per ogni periodo e ogni pagina
    first_half_performance_page_performance_analysis = first_half_df_page_performance_analysis.groupby('Page').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index().rename(columns={
        'Impressions': 'Impressions_First_Half',
        'Clicks': 'Clicks_First_Half',
        'CTR': 'CTR_First_Half',
        'Position': 'Position_First_Half'
    })
    
    second_half_performance_page_performance_analysis = second_half_df_page_performance_analysis.groupby('Page').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index().rename(columns={
        'Impressions': 'Impressions_Second_Half',
        'Clicks': 'Clicks_Second_Half',
        'CTR': 'CTR_Second_Half',
        'Position': 'Position_Second_Half'
    })
    
    # Unisci i dati dei due periodi
    performance_df_page_performance_analysis = pd.merge(first_half_performance_page_performance_analysis, second_half_performance_page_performance_analysis, on='Page', how='outer').fillna(0)
    
    # Calcola la variazione di impressions, clicks, CTR e posizione media tra i periodi
    performance_df_page_performance_analysis['Impressions_Change'] = performance_df_page_performance_analysis['Impressions_Second_Half'] - performance_df_page_performance_analysis['Impressions_First_Half']
    performance_df_page_performance_analysis['Clicks_Change'] = performance_df_page_performance_analysis['Clicks_Second_Half'] - performance_df_page_performance_analysis['Clicks_First_Half']
    performance_df_page_performance_analysis['CTR_Change'] = performance_df_page_performance_analysis['CTR_Second_Half'] - performance_df_page_performance_analysis['CTR_First_Half']
    performance_df_page_performance_analysis['Position_Change'] = performance_df_page_performance_analysis['Position_Second_Half'] - performance_df_page_performance_analysis['Position_First_Half']
    
    # Identifica le pagine che hanno guadagnato, perso o sono rimaste stabili in termini di traffico basandosi sui clic
    gained_traffic_page_performance_analysis = performance_df_page_performance_analysis[performance_df_page_performance_analysis['Clicks_Change'] > 0]
    lost_traffic_page_performance_analysis = performance_df_page_performance_analysis[performance_df_page_performance_analysis['Clicks_Change'] < 0]
    stable_traffic_page_performance_analysis = performance_df_page_performance_analysis[performance_df_page_performance_analysis['Clicks_Change'] == 0]
    
    # Analizza il trend generale di impressions, clicks, CTR e posizione media per ottenere metriche
    total_clicks_first_half = first_half_performance_page_performance_analysis['Clicks_First_Half'].sum()
    total_impressions_first_half = first_half_performance_page_performance_analysis['Impressions_First_Half'].sum()
    avg_ctr_first_half = first_half_performance_page_performance_analysis['CTR_First_Half'].mean()
    avg_position_first_half = first_half_performance_page_performance_analysis['Position_First_Half'].mean()
    
    overall_impressions_trend_page_performance_analysis = performance_df_page_performance_analysis['Impressions_Change'].sum()
    overall_clicks_trend_page_performance_analysis = performance_df_page_performance_analysis['Clicks_Change'].sum()
    overall_ctr_trend_page_performance_analysis = performance_df_page_performance_analysis['CTR_Change'].mean()
    overall_position_trend_page_performance_analysis = performance_df_page_performance_analysis['Position_Change'].mean()
    
    # Calcola la variazione percentuale
    clicks_percentage_change = (overall_clicks_trend_page_performance_analysis / total_clicks_first_half) * 100
    impressions_percentage_change = (overall_impressions_trend_page_performance_analysis / total_impressions_first_half) * 100
    ctr_percentage_change = (overall_ctr_trend_page_performance_analysis / avg_ctr_first_half) * 100
    position_percentage_change = (overall_position_trend_page_performance_analysis / avg_position_first_half) * 100

    # Variazione posizione inversa per la scheda punteggio
    position_score_change = -overall_position_trend_page_performance_analysis
    position_score_percentage_change = -position_percentage_change

    
    st.subheader("2. Pages Traffic Changes Report")
    st.divider()
    col1, col2, col3, col4, col5 =st.columns([3, 1, 1, 1, 1])
    with col1:        
        st.markdown("""
        This report divides the time period into two halves and compares them.
        The comparison is made between the following date ranges:
        """)
        st.markdown(f"""
        - **First half period**: {start_date_page_performance_analysis} to {midpoint_page_performance_analysis}
        - **Second half period**: {midpoint_page_performance_analysis + pd.Timedelta(days=1)} to {end_date_page_performance_analysis}
        """)
    with col2:
        st.metric("Total Clicks Change", f"{overall_clicks_trend_page_performance_analysis:.0f}", f"{clicks_percentage_change:.2f}%")
    with col3:
        st.metric("Total Impressions Change", f"{overall_impressions_trend_page_performance_analysis:.0f}", f"{impressions_percentage_change:.2f}%")
    with col4:
        st.metric("Average CTR Change", f"{overall_ctr_trend_page_performance_analysis * 100:.2f}%", f"{overall_ctr_trend_page_performance_analysis * 100:.2f}%")
    with col5:
        if overall_position_trend_page_performance_analysis < 0:
            st.metric("Average Position Change", f"{overall_position_trend_page_performance_analysis:.2f}", f"{-position_percentage_change:.2f}%", delta_color="normal")
        else:
            st.metric("Average Position Change", f"{overall_position_trend_page_performance_analysis:.2f}", f"{position_percentage_change:.2f}%", delta_color="inverse")

    # Creazione del grafico a barre con plotly
    bar_data_page_performance_analysis = {
        "Traffic Change": ["Gained Traffic", "Lost Traffic", "No Changes"],
        "Count": [gained_traffic_page_performance_analysis.shape[0], lost_traffic_page_performance_analysis.shape[0], stable_traffic_page_performance_analysis.shape[0]]
    }
    color_discrete_map = {
    "Gained Traffic": "#32CD32",
    "Lost Traffic": "coral",
    "No Changes": "grey"
    }
    fig_page_performance_analysis = px.bar(
        bar_data_page_performance_analysis, 
        x="Traffic Change", 
        y="Count", 
        title="Traffic Change Overview",
        labels={"Traffic Change": "Traffic Change Type", "Count": "Number of Pages"},
        color="Traffic Change",
        color_discrete_map=color_discrete_map
    )

    fig_page_performance_analysis.update_layout(
        showlegend=False,
        title=dict(
            text="Traffic Change Overview",
            y=0.8,  # Alza il titolo più vicino al grafico
            yanchor='bottom'
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_page_performance_analysis, use_container_width=True)
    with col2:
        st.markdown("<br></br>", unsafe_allow_html=True)
        if overall_impressions_trend_page_performance_analysis > 0:
            st.success("The overall trend of impressions is increasing.")
        else:
            st.warning("The overall trend of impressions is decreasing.")
        
        if overall_clicks_trend_page_performance_analysis > 0:
            st.success("The overall trend of clicks is increasing.")
        else:
            st.warning("The overall trend of clicks is decreasing.")
        
        if overall_ctr_trend_page_performance_analysis > 0:
            st.success("The overall trend of CTR is increasing.")
        else:
            st.warning("The overall trend of CTR is decreasing.")
        
        if overall_position_trend_page_performance_analysis < 0:
            st.success(f"The overall trend of average position is improving (lowering). Variation: {position_score_change:.2f} positions, improvement of {position_score_percentage_change:.2f}%.")
        else:
            st.warning(f"The overall trend of average position is worsening (rising). Variation: {position_score_change:.2f} positions, worsening of {position_score_percentage_change:.2f}%.")

    # Visualizza i risultati
    with st.expander("PAGES THAT GAINED TRAFFIC ⬆️"):
        st.dataframe(gained_traffic_page_performance_analysis[[
            'Page', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
            'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
            'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
            'Position_First_Half', 'Position_Second_Half', 'Position_Change'
        ]].reset_index(drop=True).style.format({
            'Clicks_First_Half': '{:.0f}',
            'Clicks_Second_Half': '{:.0f}',
            'Clicks_Change': '{:.0f}',
            'Impressions_First_Half': '{:.0f}',
            'Impressions_Second_Half': '{:.0f}',
            'Impressions_Change': '{:.0f}',
            'CTR_First_Half': '{:.2%}',
            'CTR_Second_Half': '{:.2%}',
            'CTR_Change': '{:.2%}',
            'Position_First_Half': '{:.2f}',
            'Position_Second_Half': '{:.2f}',
            'Position_Change': '{:.2f}'
        }))
    
    with st.expander("PAGES THAT LOST TRAFFIC ⬇️"):
        
        st.dataframe(lost_traffic_page_performance_analysis[[
            'Page', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
            'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
            'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
            'Position_First_Half', 'Position_Second_Half', 'Position_Change'
        ]].reset_index(drop=True).style.format({
            'Clicks_First_Half': '{:.0f}',
            'Clicks_Second_Half': '{:.0f}',
            'Clicks_Change': '{:.0f}',
            'Impressions_First_Half': '{:.0f}',
            'Impressions_Second_Half': '{:.0f}',
            'Impressions_Change': '{:.0f}',
            'CTR_First_Half': '{:.2%}',
            'CTR_Second_Half': '{:.2%}',
            'CTR_Change': '{:.2%}',
            'Position_First_Half': '{:.2f}',
            'Position_Second_Half': '{:.2f}',
            'Position_Change': '{:.2f}'
        }))
            # Aggiungi un selettore per stabilire la soglia
        st.write("")
        st.markdown(f"<h4>Which pages have contributed the most to the traffic loss?</h4>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Identify the pages that have contributed the most to your website's traffic loss. You can select a specific percentage of the total loss to determine the influence threshold. This will show you which pages had the biggest impact on the overall traffic decrease.")
        with col2:
            threshold_page_performance_analysis = st.slider(
                "Set the percentage of total loss",
                min_value=1, max_value=100, value=10, step=5,
                format="%d%%",
                help="Adjust the threshold to determine which pages are considered to have significant loss in traffic."
            ) / 100.0       
        
        # Identifica le pagine che causano la perdita di traffico
        if overall_clicks_trend_page_performance_analysis != 0:
            significant_lost_traffic_page_performance_analysis = lost_traffic_page_performance_analysis[abs(lost_traffic_page_performance_analysis['Clicks_Change']) > abs(overall_clicks_trend_page_performance_analysis) * threshold_page_performance_analysis]  # Perdita significativa > soglia della perdita totale
            if not significant_lost_traffic_page_performance_analysis.empty:
                st.write("RESULTS")
                st.dataframe(significant_lost_traffic_page_performance_analysis[[
                    'Page', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
                    'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
                    'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
                    'Position_First_Half', 'Position_Second_Half', 'Position_Change'
                ]].reset_index(drop=True).style.format({
                    'Clicks_First_Half': '{:.0f}',
                    'Clicks_Second_Half': '{:.0f}',
                    'Clicks_Change': '{:.0f}',
                    'Impressions_First_Half': '{:.0f}',
                    'Impressions_Second_Half': '{:.0f}',
                    'Impressions_Change': '{:.0f}',
                    'CTR_First_Half': '{:.2%}',
                    'CTR_Second_Half': '{:.2%}',
                    'CTR_Change': '{:.2%}',
                    'Position_First_Half': '{:.2f}',
                    'Position_Second_Half': '{:.2f}',
                    'Position_Change': '{:.2f}'
                }))
        else:
            st.write("No significant loss in traffic detected.")
        
    
    with st.expander("PAGES WITH NO CHANGES TRAFFIC ➡️"):
        st.dataframe(stable_traffic_page_performance_analysis[[
            'Page', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
            'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
            'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
            'Position_First_Half', 'Position_Second_Half', 'Position_Change'
        ]].reset_index(drop=True).style.format({
            'Clicks_First_Half': '{:.0f}',
            'Clicks_Second_Half': '{:.0f}',
            'Clicks_Change': '{:.0f}',
            'Impressions_First_Half': '{:.0f}',
            'Impressions_Second_Half': '{:.0f}',
            'Impressions_Change': '{:.0f}',
            'CTR_First_Half': '{:.2%}',
            'CTR_Second_Half': '{:.2%}',
            'CTR_Change': '{:.2%}',
            'Position_First_Half': '{:.2f}',
            'Position_Second_Half': '{:.2f}',
            'Position_Change': '{:.2f}'
        }))
    

# Page Layout
st.markdown("""
            <style>hr {margin:0em 0px;}</style>
            """, unsafe_allow_html=True
          )

#queries traffic channge
def analyze_query_performance(df):
    # Creare una copia del DataFrame di partenza per evitare conflitti con altre analisi
    df_query_performance_analysis = df.copy()
    
    # Assicurati che il DataFrame contenga una colonna "Date"
    if 'Date' not in df_query_performance_analysis.columns:
        st.error("The DataFrame must contain a 'Date' column.")
        return
    
    # Conversione della colonna 'Date' in datetime
    df_query_performance_analysis['Date'] = pd.to_datetime(df_query_performance_analysis['Date'])
    
    # Utilizza l'intervallo di date nel DataFrame
    start_date_query_performance_analysis = df_query_performance_analysis['Date'].min().date() # Converte in oggetto date
    end_date_query_performance_analysis = df_query_performance_analysis['Date'].max().date()  # Converte in oggetto date
    
    # Filtra il DataFrame in base alle date disponibili
    filtered_df_query_performance_analysis = df_query_performance_analysis[(df_query_performance_analysis['Date'].dt.date >= start_date_query_performance_analysis) & (df_query_performance_analysis['Date'].dt.date <= end_date_query_performance_analysis)]
    
    # Definisci i periodi di confronto
    midpoint_query_performance_analysis = start_date_query_performance_analysis + (end_date_query_performance_analysis - start_date_query_performance_analysis) / 2
    
    first_half_df_query_performance_analysis = filtered_df_query_performance_analysis[filtered_df_query_performance_analysis['Date'].dt.date <= midpoint_query_performance_analysis]
    second_half_df_query_performance_analysis = filtered_df_query_performance_analysis[filtered_df_query_performance_analysis['Date'].dt.date > midpoint_query_performance_analysis]
    
    # Calcola le somme di impressions e clicks, e le medie di CTR e posizione media per ogni periodo e ogni query
    first_half_performance_query_performance_analysis = first_half_df_query_performance_analysis.groupby('Query').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index().rename(columns={
        'Impressions': 'Impressions_First_Half',
        'Clicks': 'Clicks_First_Half',
        'CTR': 'CTR_First_Half',
        'Position': 'Position_First_Half'
    })
    
    second_half_performance_query_performance_analysis = second_half_df_query_performance_analysis.groupby('Query').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index().rename(columns={
        'Impressions': 'Impressions_Second_Half',
        'Clicks': 'Clicks_Second_Half',
        'CTR': 'CTR_Second_Half',
        'Position': 'Position_Second_Half'
    })
    
    # Unisci i dati dei due periodi
    performance_df_query_performance_analysis = pd.merge(first_half_performance_query_performance_analysis, second_half_performance_query_performance_analysis, on='Query', how='outer').fillna(0)
    
    # Calcola la variazione di impressions, clicks, CTR e posizione media tra i periodi
    performance_df_query_performance_analysis['Impressions_Change'] = performance_df_query_performance_analysis['Impressions_Second_Half'] - performance_df_query_performance_analysis['Impressions_First_Half']
    performance_df_query_performance_analysis['Clicks_Change'] = performance_df_query_performance_analysis['Clicks_Second_Half'] - performance_df_query_performance_analysis['Clicks_First_Half']
    performance_df_query_performance_analysis['CTR_Change'] = performance_df_query_performance_analysis['CTR_Second_Half'] - performance_df_query_performance_analysis['CTR_First_Half']
    performance_df_query_performance_analysis['Position_Change'] = performance_df_query_performance_analysis['Position_Second_Half'] - performance_df_query_performance_analysis['Position_First_Half']
    
    # Identifica le query che hanno guadagnato, perso o sono rimaste stabili in termini di traffico basandosi sui clic
    gained_traffic_query_performance_analysis = performance_df_query_performance_analysis[performance_df_query_performance_analysis['Clicks_Change'] > 0]
    lost_traffic_query_performance_analysis = performance_df_query_performance_analysis[performance_df_query_performance_analysis['Clicks_Change'] < 0]
    stable_traffic_query_performance_analysis = performance_df_query_performance_analysis[performance_df_query_performance_analysis['Clicks_Change'] == 0]
    
    # Analizza il trend generale di impressions, clicks, CTR e posizione media per ottenere metriche
    total_clicks_first_half = first_half_performance_query_performance_analysis['Clicks_First_Half'].sum()
    total_impressions_first_half = first_half_performance_query_performance_analysis['Impressions_First_Half'].sum()
    avg_ctr_first_half = first_half_performance_query_performance_analysis['CTR_First_Half'].mean()
    avg_position_first_half = first_half_performance_query_performance_analysis['Position_First_Half'].mean()
    
    overall_impressions_trend_query_performance_analysis = performance_df_query_performance_analysis['Impressions_Change'].sum()
    overall_clicks_trend_query_performance_analysis = performance_df_query_performance_analysis['Clicks_Change'].sum()
    overall_ctr_trend_query_performance_analysis = performance_df_query_performance_analysis['CTR_Change'].mean()
    overall_position_trend_query_performance_analysis = performance_df_query_performance_analysis['Position_Change'].mean()
    
    # Calcola la variazione percentuale
    clicks_percentage_change = (overall_clicks_trend_query_performance_analysis / total_clicks_first_half) * 100
    impressions_percentage_change = (overall_impressions_trend_query_performance_analysis / total_impressions_first_half) * 100
    ctr_percentage_change = (overall_ctr_trend_query_performance_analysis / avg_ctr_first_half) * 100
    position_percentage_change = (overall_position_trend_query_performance_analysis / avg_position_first_half) * 100

    # Variazione posizione inversa per la scheda punteggio
    position_score_change = -overall_position_trend_query_performance_analysis
    position_score_percentage_change = -position_percentage_change

    
   
    with st.container(border=True):
        st.subheader("4. Queries Traffic Changes Report")
        st.divider()
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        with col1:        
            st.markdown("""
            This report divides the time period into two halves and compares them.
            The comparison is made between the following date ranges:
            """)
            st.markdown(f"""
            - **First half period**: {start_date_query_performance_analysis} to {midpoint_query_performance_analysis}
            - **Second half period**: {midpoint_query_performance_analysis + pd.Timedelta(days=1)} to {end_date_query_performance_analysis}
            """)
        with col2:
            st.metric("Total Clicks Change", f"{overall_clicks_trend_query_performance_analysis:.0f}", f"{clicks_percentage_change:.2f}%")
        with col3:
            st.metric("Total Impressions Change", f"{overall_impressions_trend_query_performance_analysis:.0f}", f"{impressions_percentage_change:.2f}%")
        with col4:
            st.metric("Average CTR Change", f"{overall_ctr_trend_query_performance_analysis * 100:.2f}%", f"{overall_ctr_trend_query_performance_analysis * 100:.2f}%")
        with col5:
            if overall_position_trend_query_performance_analysis < 0:
                st.metric("Average Position Change", f"{overall_position_trend_query_performance_analysis:.2f}", f"{-position_percentage_change:.2f}%", delta_color="normal")
            else:
                st.metric("Average Position Change", f"{overall_position_trend_query_performance_analysis:.2f}", f"{position_percentage_change:.2f}%", delta_color="inverse")
    
        # Creazione del grafico a barre con plotly
        bar_data_query_performance_analysis = {
            "Traffic Change": ["Gained Traffic", "Lost Traffic", "No Changes"],
            "Count": [gained_traffic_query_performance_analysis.shape[0], lost_traffic_query_performance_analysis.shape[0], stable_traffic_query_performance_analysis.shape[0]]
        }
        
        fig_query_performance_analysis = px.bar(
            bar_data_query_performance_analysis, 
            x="Traffic Change", 
            y="Count", 
            title="Traffic Change Overview",
            labels={"Traffic Change": "Traffic Change Type", "Count": "Number of Queries"},
            color="Traffic Change",
            color_discrete_map={
                "Gained Traffic": "#32CD32",
                "Lost Traffic": "coral",
                "No Changes": "grey"
            }
        )
        fig_query_performance_analysis.update_layout(
            showlegend=False,
            title=dict(
                text="Traffic Change Overview",
                y=0.8,  # Alza il titolo più vicino al grafico
                yanchor='bottom'
            )
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_query_performance_analysis, use_container_width=True)
        with col2:
            st.markdown("<br></br>", unsafe_allow_html=True)
            if overall_impressions_trend_query_performance_analysis > 0:
                st.success("The overall trend of impressions is increasing.")
            else:
                st.warning("The overall trend of impressions is decreasing.")
            
            if overall_clicks_trend_query_performance_analysis > 0:
                st.success("The overall trend of clicks is increasing.")
            else:
                st.warning("The overall trend of clicks is decreasing.")
            
            if overall_ctr_trend_query_performance_analysis > 0:
                st.success("The overall trend of CTR is increasing.")
            else:
                st.warning("The overall trend of CTR is decreasing.")
            
            if overall_position_trend_query_performance_analysis < 0:
                st.success(f"The overall trend of average position is improving (lowering). Variation: {position_score_change:.2f} positions, improvement of {position_score_percentage_change:.2f}%.")
            else:
                st.warning(f"The overall trend of average position is worsening (rising). Variation: {position_score_change:.2f} positions, worsening of {position_score_percentage_change:.2f}%.")
    
        # Visualizza i risultati
        with st.expander("QUERIES THAT GAINED TRAFFIC ⬆️"):
            st.dataframe(gained_traffic_query_performance_analysis[[
                'Query', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
                'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
                'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
                'Position_First_Half', 'Position_Second_Half', 'Position_Change'
            ]].reset_index(drop=True).style.format({
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Clicks_Change': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}',
                'Impressions_Change': '{:.0f}',
                'CTR_First_Half': '{:.2%}',
                'CTR_Second_Half': '{:.2%}',
                'CTR_Change': '{:.2%}',
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}'
            }))
        
        with st.expander("QUERIES THAT LOST TRAFFIC ⬇️"):
            st.dataframe(lost_traffic_query_performance_analysis[[
                'Query', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
                'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
                'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
                'Position_First_Half', 'Position_Second_Half', 'Position_Change'
            ]].reset_index(drop=True).style.format({
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Clicks_Change': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}',
                'Impressions_Change': '{:.0f}',
                'CTR_First_Half': '{:.2%}',
                'CTR_Second_Half': '{:.2%}',
                'CTR_Change': '{:.2%}',
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}'
            }))
            # Aggiungi un selettore per stabilire la soglia
            st.write("")
            st.markdown(f"<h4>Which queries have contributed the most to the traffic loss?</h4>",
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("Identify the queries that have contributed the most to your website's traffic loss. You can select a specific percentage of the total loss to determine the influence threshold. This will show you which queries had the biggest impact on the overall traffic decrease.")
            with col2:
                threshold_query_performance_analysis = st.slider(
                    "Set the threshold for significant loss in traffic (percentage of total loss)",
                    min_value=1, max_value=100, value=10, step=5,
                    format="%d%%",
                    help="Adjust the threshold to determine which queries are considered to have significant loss in traffic."
                ) / 100.0
            
            # Identifica le query che causano la perdita di traffico
            if overall_clicks_trend_query_performance_analysis != 0:
                significant_lost_traffic_query_performance_analysis = lost_traffic_query_performance_analysis[abs(lost_traffic_query_performance_analysis['Clicks_Change']) > abs(overall_clicks_trend_query_performance_analysis) * threshold_query_performance_analysis]  # Perdita significativa > soglia della perdita totale
                if not significant_lost_traffic_query_performance_analysis.empty:
                    st.write("Queries causing the loss in traffic:")
                    st.dataframe(significant_lost_traffic_query_performance_analysis[[
                        'Query', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
                        'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
                        'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
                        'Position_First_Half', 'Position_Second_Half', 'Position_Change'
                    ]].reset_index(drop=True).style.format({
                        'Clicks_First_Half': '{:.0f}',
                        'Clicks_Second_Half': '{:.0f}',
                        'Clicks_Change': '{:.0f}',
                        'Impressions_First_Half': '{:.0f}',
                        'Impressions_Second_Half': '{:.0f}',
                        'Impressions_Change': '{:.0f}',
                        'CTR_First_Half': '{:.2%}',
                        'CTR_Second_Half': '{:.2%}',
                        'CTR_Change': '{:.2%}',
                        'Position_First_Half': '{:.2f}',
                        'Position_Second_Half': '{:.2f}',
                        'Position_Change': '{:.2f}'
                    }))
            else:
                st.write("No significant loss in traffic detected.")
        with st.expander("QUERIES WITH NO CHANGES TRAFFIC ➡️"):
            st.dataframe(stable_traffic_query_performance_analysis[[
                'Query', 'Clicks_First_Half', 'Clicks_Second_Half', 'Clicks_Change',
                'Impressions_First_Half', 'Impressions_Second_Half', 'Impressions_Change',
                'CTR_First_Half', 'CTR_Second_Half', 'CTR_Change',
                'Position_First_Half', 'Position_Second_Half', 'Position_Change'
            ]].reset_index(drop=True).style.format({
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Clicks_Change': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}',
                'Impressions_Change': '{:.0f}',
                'CTR_First_Half': '{:.2%}',
                'CTR_Second_Half': '{:.2%}',
                'CTR_Change': '{:.2%}',
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}'
            }))

st.subheader('Authenticate with Google Account')
st.write("➡️ [Google Cloud Console](https://console.cloud.google.com/apis/credentials)")
with st.expander("How to Get credential?"):
    st.text("Add your credentials to the secrets.toml file.")

credentials = authorize_app()

if credentials:
    webmasters_service = build('searchconsole', 'v1', credentials=credentials)

    if not st.session_state.available_sites:
        site_list = webmasters_service.sites().list().execute()
        st.session_state.available_sites = [site['siteUrl'] for site in site_list.get('siteEntry', [])]

    st.session_state.selected_site = st.selectbox('Select a website:', st.session_state.available_sites)

    tab1, tab2 = st.tabs(["SEARCH ANALYTICS", "URL INSPECTION"])

    with tab2:
        url_to_inspect = st.text_input("Insert URL to inspect:")
        if st.button('URL INSPECTION 🕵️‍♂️'):
            with st.spinner("Inspecting URL..."):
                if st.session_state.selected_site:
                    request_body = {'inspectionUrl': url_to_inspect, 'siteUrl': st.session_state.selected_site}
                    response = webmasters_service.urlInspection().index().inspect(body=request_body).execute()

                inspection_result = response.get('inspectionResult', {})
                index_status_result = inspection_result.get('indexStatusResult', {})
                mobile_usability_result = inspection_result.get('mobileUsabilityResult', {})
                rich_results_result = inspection_result.get('richResultsResult', {})

                st.write("### Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("🤖INDEX STATE")
                    st.write(f"Verdict: {index_status_result.get('verdict', 'N/A')}")
                    st.write(f"Coverage State: {index_status_result.get('coverageState', 'N/A')}")
                    st.write(f"Robots.txt State: {index_status_result.get('robotsTxtState', 'N/A')}")
                with col2:
                    st.write("📱MOBILE USABILTY")
                    st.write(f"Verdict: {mobile_usability_result.get('verdict', 'N/A')}")
                with col3:
                    st.write("⭐RICH RESULTS")
                    st.write(f"Verdict: {rich_results_result.get('verdict', 'N/A')}")

                st.write(inspection_result.get('inspectionResultLink', 'N/A'))
                with st.expander("Complete response"):
                    st.write(f'Response: {response}')

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            options_type = {'Web': 'web', 'News': 'news', 'Discovery': 'discovery', 'Image': 'image', 'Video': 'video'}
            today = datetime.now()
            three_months_ago = today - timedelta(days=90)

            selected_type = st.selectbox('CHANNEL', list(options_type.keys()))
            start_date = st.date_input('Start date', pd.to_datetime(three_months_ago))
            end_date = st.date_input('End date', pd.to_datetime(today))

        with col2:
            selected_dimensions = st.multiselect('DIMENSIONS', ['Date', 'Page', 'Query', 'Device', 'Country'], default=['Date', 'Query', 'Page'])
            with st.expander("Filters for Dimensions"):
                unique_key = 0
                for dimension in selected_dimensions:
                    col1, col2 = st.columns(2)
                    with col1:
                        operator = st.selectbox(f'{dimension}', ['equals', 'contains', 'notEquals', 'notContains', 'includingRegex', 'excludingRegex'])
                    with col2:
                        filter_value = st.text_input(label="", placeholder=" value", key=unique_key)
                    unique_key += 1
                    st.session_state.dimension_filters[dimension] = {'operator': operator, 'filter_value': filter_value}

        with col3:
            row_limit_options = ['No', 'Yes']
            check_box_row = st.radio('SET ROW LIMIT?', row_limit_options)
            row_limit = st.number_input('Row limit', min_value=1, max_value=25000, value=25000) if check_box_row == 'Yes' else None

        if st.button('GET DATA ⬇️'):
            if st.session_state.selected_site:
                dimensions = [dim.upper() for dim in selected_dimensions]

                progress_bar = st.progress(0)
                status_text = st.empty()
                total_downloaded_rows = 0
                start_row = 0

                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame()

                with st.spinner("Downloading data..."):
                    while True:
                        rows = fetch_data_chunk(webmasters_service, st.session_state.selected_site, start_date, end_date, dimensions, st.session_state.dimension_filters, selected_type, start_row, row_limit)
                        
                        chunk_df = pd.DataFrame(rows, columns=[*dimensions, 'Clicks', 'Impressions', 'CTR', 'Position'])
                        st.session_state.df = pd.concat([st.session_state.df, chunk_df])
                        
                        total_downloaded_rows += len(rows)
                        start_row += len(rows)
                        status_text.text(f"Total rows downloaded: {total_downloaded_rows}")
                        
                        if len(rows) < 25000 or (row_limit and total_downloaded_rows >= row_limit):
                            break
                        
                        progress_bar.progress(min(total_downloaded_rows / (row_limit if row_limit else total_downloaded_rows + len(rows)), 1.0))
                    
                    st.session_state.data_loaded = True
                    progress_bar.progress(100)

                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(st.session_state.df)
                st.download_button(label="Download data CSV", data=csv, file_name='data.csv', mime='text/csv')

if st.session_state.data_loaded:
    df = st.session_state.df


