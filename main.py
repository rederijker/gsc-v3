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
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

#PAGE CONFIGURATION
st.set_page_config(
    page_title="SEO Gnosis: Master Search Console Data Like a Pro-by Cristiano Caggiula",
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
if 'download_ready' not in st.session_state:
    st.session_state.download_ready = False
if 'scan_started' not in st.session_state:
    st.session_state.scan_started = False
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''
if 'show_heading' not in st.session_state:
    st.session_state.show_heading = True
if 'show_keyword_metrics' not in st.session_state:
    st.session_state.show_keyword_metrics = True
if 'show_meta' not in st.session_state:
    st.session_state.show_meta = True
if 'show_body_alt' not in st.session_state:
    st.session_state.show_body_alt = True
if 'show_not_covered' not in st.session_state:
    st.session_state.show_not_covered = False
required_columns = ['Page', 'Query', 'Clicks', 'Impressions', 'CTR', 'Position']

def clear_data():
    st.session_state.df = None
    
#TAB3 PAGE OPTIMIZATION
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

#EXTRACT data from 
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
# Funzione per analizzare il trend di impressions, clicks, CTR e posizione media
def analyze_page_performance(df):
    # Creare una copia del DataFrame di partenza per evitare conflitti con altre analisi
    df_page_performance_analysis = df.copy()
    
    # Assicurati che il DataFrame contenga una colonna "Date"
    if 'Date' not in df_page_performance_analysis.columns:
        st.warning("The DataFrame must contain a 'Date' column.")
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
        ),
        paper_bgcolor='rgb(10,14,18)',  # Colore di sfondo del layout
        plot_bgcolor='rgb(10,14,18)'    # Colore di sfondo dell'area del grafico
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
        st.warning("Add 'Date' and 'Query' to dimensions to show 4. Queries Traffic Changes Report")
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
            ),
            paper_bgcolor='rgb(10,14,18)',  # Colore di sfondo del layout
            plot_bgcolor='rgb(10,14,18)'    # Colore di sfondo dell'area del grafico
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
#REPORT CAMBI DI POSIZIONAMENTO
pd.set_option("styler.render.max_elements", 20000000)

def analyze_query_position_changes(df):
    # Creare una copia del DataFrame di partenza per evitare conflitti con altre analisi
    df_position_analysis = df.copy()
    
    # Assicurati che il DataFrame contenga una colonna "Date"
    if 'Date' not in df_position_analysis.columns:
        st.warning("Add 'Date' and 'Query' to dimensions to show 5. Query Position Changes Report")
        return
    
    # Conversione della colonna 'Date' in datetime
    df_position_analysis['Date'] = pd.to_datetime(df_position_analysis['Date'])
    
    # Utilizza l'intervallo di date nel DataFrame
    start_date_position_analysis = df_position_analysis['Date'].min().date()
    end_date_position_analysis = df_position_analysis['Date'].max().date()
    
    # Filtra il DataFrame in base alle date disponibili
    filtered_df_position_analysis = df_position_analysis[(df_position_analysis['Date'].dt.date >= start_date_position_analysis) & (df_position_analysis['Date'].dt.date <= end_date_position_analysis)]
    
    # Definisci i periodi di confronto
    midpoint_position_analysis = start_date_position_analysis + (end_date_position_analysis - start_date_position_analysis) / 2
    
    first_half_df_position_analysis = filtered_df_position_analysis[filtered_df_position_analysis['Date'].dt.date <= midpoint_position_analysis]
    second_half_df_position_analysis = filtered_df_position_analysis[filtered_df_position_analysis['Date'].dt.date > midpoint_position_analysis]
    
    # Calcola le medie di posizione media, somma di click e impression per ogni periodo e ogni query
    first_half_performance_position_analysis = first_half_df_position_analysis.groupby('Query').agg({
        'Position': 'mean',
        'Clicks': 'sum',
        'Impressions': 'sum'
    }).reset_index().rename(columns={
        'Position': 'Position_First_Half',
        'Clicks': 'Clicks_First_Half',
        'Impressions': 'Impressions_First_Half'
    })
    
    second_half_performance_position_analysis = second_half_df_position_analysis.groupby('Query').agg({
        'Position': 'mean',
        'Clicks': 'sum',
        'Impressions': 'sum'
    }).reset_index().rename(columns={
        'Position': 'Position_Second_Half',
        'Clicks': 'Clicks_Second_Half',
        'Impressions': 'Impressions_Second_Half'
    })
    
    # Unisci i dati dei due periodi
    performance_df_position_analysis = pd.merge(first_half_performance_position_analysis, second_half_performance_position_analysis, on='Query', how='outer')
    
    # Calcola la variazione di posizione media tra i periodi
    performance_df_position_analysis['Position_Change'] = performance_df_position_analysis['Position_Second_Half'] - performance_df_position_analysis['Position_First_Half']
    
    # Identifica le query che hanno migliorato, peggiorato o sono rimaste stabili in termini di posizione
    improved_position_queries = performance_df_position_analysis[performance_df_position_analysis['Position_Change'] < 0]
    worsened_position_queries = performance_df_position_analysis[performance_df_position_analysis['Position_Change'] > 0]
    stable_position_queries = performance_df_position_analysis[performance_df_position_analysis['Position_Change'] == 0]
    
    # Identifica le query che sono uscite dalla SERP
    queries_out_of_serp = performance_df_position_analysis[(performance_df_position_analysis['Impressions_First_Half'] > 0) & (performance_df_position_analysis['Impressions_Second_Half'].isna())]
    
    # Analizza il trend generale di posizione media
    avg_position_first_half = first_half_performance_position_analysis['Position_First_Half'].mean()
    overall_position_trend = performance_df_position_analysis['Position_Change'].mean()
    position_percentage_change = (overall_position_trend / avg_position_first_half) * 100
    position_score_change = -overall_position_trend
    position_score_percentage_change = -position_percentage_change
    
    # Filtro a toggle per includere/escludere query con dati mancanti
    st.markdown("### Filters")
    include_missing_data = st.checkbox("Include queries with missing data in one of the periods", value=True)
    
    if not include_missing_data:
        performance_df_position_analysis = performance_df_position_analysis.dropna(subset=['Position_First_Half', 'Position_Second_Half'])

    with st.container(border=True):
        st.subheader("5. Query Position Changes Report")
        st.divider()
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:        
            st.markdown("""
            This report divides the time period into two halves and compares them.
            The comparison is made between the following date ranges:
            """)
            st.markdown(f"""
            - **First half period**: {start_date_position_analysis} to {midpoint_position_analysis}
            - **Second half period**: {midpoint_position_analysis + pd.Timedelta(days=1)} to {end_date_position_analysis}
            """)
        with col2:
            if overall_position_trend < 0:
                st.metric("Average Position Change", f"{overall_position_trend:.2f}", f"{position_percentage_change:.2f}%", delta_color="normal")
            else:
                st.metric("Average Position Change", f"{overall_position_trend:.2f}", f"{position_percentage_change:.2f}%", delta_color="inverse")
    
        # Creazione del grafico a barre con plotly
        bar_data_position_analysis = {
            "Position Change": ["Improved Position", "Worsened Position", "No Changes"],
            "Count": [improved_position_queries.shape[0], worsened_position_queries.shape[0], stable_position_queries.shape[0]]
        }
        
        fig_position_analysis = px.bar(
            bar_data_position_analysis, 
            x="Position Change", 
            y="Count", 
            title="Position Change Overview",
            labels={"Position Change": "Position Change Type", "Count": "Number of Queries"},
            color="Position Change",
            color_discrete_map={
                "Improved Position": "#32CD32",
                "Worsened Position": "coral",
                "No Changes": "grey"
            }
        )
        fig_position_analysis.update_layout(
            showlegend=False,
            title=dict(
                text="Position Change Overview",
                y=0.8,
                yanchor='bottom'
            ),
            paper_bgcolor='rgb(10,14,18)',  # Colore di sfondo del layout
            plot_bgcolor='rgb(10,14,18)'    # Colore di sfondo dell'area del grafico
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_position_analysis, use_container_width=True)
        with col2:
            st.markdown("<br></br>", unsafe_allow_html=True)
            if overall_position_trend < 0:
                st.success(f"The overall trend of average position is improving (lowering). Variation: {position_score_change:.2f} positions, improvement of {position_score_percentage_change:.2f}%.")
            else:
                st.warning(f"The overall trend of average position is worsening (rising). Variation: {position_score_change:.2f} positions, worsening of {position_score_percentage_change:.2f}%.")
    
        # Visualizza i risultati
        with st.expander("QUERIES THAT IMPROVED POSITION ⬆️"):
            st.dataframe(improved_position_queries[[
                'Query', 'Position_First_Half', 'Position_Second_Half', 'Position_Change', 'Clicks_First_Half', 'Clicks_Second_Half', 'Impressions_First_Half', 'Impressions_Second_Half'
            ]].reset_index(drop=True).style.format({
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}',
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}'
            }))
        
        with st.expander("QUERIES THAT WORSENED POSITION ⬇️"):
            st.dataframe(worsened_position_queries[[
                'Query', 'Position_First_Half', 'Position_Second_Half', 'Position_Change', 'Clicks_First_Half', 'Clicks_Second_Half', 'Impressions_First_Half', 'Impressions_Second_Half'
            ]].reset_index(drop=True).style.format({
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}',
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}'
            }))
        
        with st.expander("QUERIES WITH NO POSITION CHANGES ➡️"):
            st.dataframe(stable_position_queries[[
                'Query', 'Position_First_Half', 'Position_Second_Half', 'Position_Change', 'Clicks_First_Half', 'Clicks_Second_Half', 'Impressions_First_Half', 'Impressions_Second_Half'
            ]].reset_index(drop=True).style.format({
                'Position_First_Half': '{:.2f}',
                'Position_Second_Half': '{:.2f}',
                'Position_Change': '{:.2f}',
                'Clicks_First_Half': '{:.0f}',
                'Clicks_Second_Half': '{:.0f}',
                'Impressions_First_Half': '{:.0f}',
                'Impressions_Second_Half': '{:.0f}'
            }))

        with st.expander("QUERIES THAT DROPPED OUT OF SERP ❌"):
            st.markdown("""
            Per determinare se una query è uscita dalla SERP, possiamo basarci sull'assenza di impression e clic nel secondo periodo, dopo essere stata presente nel primo periodo. Questo scenario suggerisce che la query non sta più ricevendo traffico, il che potrebbe essere dovuto a:

            - **Riduzione delle Ricerche**: La query potrebbe non essere più rilevante o cercata dagli utenti.
            - **Perdita di Posizione**: La query potrebbe aver perso visibilità nelle SERP, finendo su pagine successive dove riceve meno traffico.
            - **Modifica dell'Algoritmo**: Un cambiamento nell'algoritmo di ricerca di Google potrebbe aver influenzato la visibilità della query.
            - **Rimozione del Contenuto**: Il contenuto che rispondeva a quella query potrebbe essere stato rimosso o deindicizzato.
            """)
            st.dataframe(queries_out_of_serp[[
                'Query', 'Position_First_Half', 'Clicks_First_Half', 'Impressions_First_Half'
            ]].reset_index(drop=True).style.format({
                'Position_First_Half': '{:.2f}',
                'Clicks_First_Half': '{:.0f}',
                'Impressions_First_Half': '{:.0f}'
            }))


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

    query_params = st.query_params
    auth_code = query_params.get('code', None)

    if auth_code:
        st.markdown(f"""
        <h1 style="text-align:center;">SEO Gnosis</h1>""",
        unsafe_allow_html=True
        )        
        if not st.session_state.credentials:
            
            try:
                flow.fetch_token(code=auth_code)
                credentials = flow.credentials
                st.session_state.credentials = credentials
                st.write("✅ Auth code received:. Now you are connected with Google Search Console API")
            except Exception as e:
                st.write(f"Error during authorization: {e}")
                st.write(f"Please reauthenticate")

    if st.session_state.credentials is None:
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        gif_url = "https://github.com/rederijker/gsc-v3/blob/main/assets/back.gif?raw=true"
        st.markdown(f"""
        <div class="header" style="padding:5%;background-image:url({gif_url});">
        <h1 style="text-align:center;">SEO Gnosis</h1>
        <h2 style="text-align:center;">Master Google Search Console Data like a Pro with a Free SEO tool</h2>
        <p style="text-align:center;font-size:17px;">Explore Google Search Console data, generate detailed reports, customize searches, and access unlimited information without needing programming skills. Ideal for webmasters, SEO experts, and digital marketers.</p>
        
        <div style="text-align:center;">
            <a href="{auth_url}" style="text-decoration:none;">
                <button style="background-color: white; color: #4285F4; border: 1px solid #4285F4; padding: 10px 20px; font-size: 17px; border-radius: 5px; cursor: pointer; display: inline-flex; align-items: center;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1200px-Google_%22G%22_logo.svg.png" alt="Google logo" style="width: 47px; height: 47px; margin-right: 8px;">
                    Login with Google
                </button>
            </a>
        </div></div>
        """, unsafe_allow_html=True)
        
            
    return st.session_state.credentials

def fetch_data_chunk(webmasters_service, site_url, start_date, end_date, dimensions, filters, selected_type, start_row, row_limit=None):
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
                    'dimension': dimension.lower(),
                    'expression': filter_value,
                    'operator': filter_operator
                }]
            })

    response_data = webmasters_service.searchanalytics().query(siteUrl=site_url, body=request_body).execute()
    rows = response_data.get('rows', [])
    return rows

st.markdown("""
    <style>
.st-emotion-cache-qcpnpn {
    border: 2px solid rgb(3 169 244 / 50%);
    border-radius: 0.5rem;
    padding: calc(-1px + 0.9rem);
    background: rgb(0 0 0 / 24%);
    box-shadow: 0 15px 25px rgba(0, 0, 0, .6);}
    h3 {
    color: #00BCD4;

    text-align: center;}
        
    </style>
    """, unsafe_allow_html=True)

# Intestazione dell'app


# Carica e visualizza l'immagine del logo

credentials = authorize_app()

if credentials:
    webmasters_service = build('searchconsole', 'v1', credentials=credentials)

    if not st.session_state.available_sites:
        site_list = webmasters_service.sites().list().execute()
        st.session_state.available_sites = [site['siteUrl'] for site in site_list.get('siteEntry', [])]
    col1, col2 =st.columns([1,2])
    with col1:
        st.session_state.selected_site = st.selectbox('Select a website:', st.session_state.available_sites)
    with col2:
        st.write("")
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
        col1, col2, col3 = st.columns([1,2,1])
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
                        operator = st.selectbox(f'{dimension} operator', ['equals', 'contains', 'notEquals', 'notContains', 'includingRegex', 'excludingRegex'])
                    with col2:
                        filter_value = st.text_input(label="Value", placeholder="value", key=unique_key)
                    unique_key += 1
                    st.session_state.dimension_filters[dimension] = {'operator': operator, 'filter_value': filter_value}

        with col3:
            row_limit_options = ['No', 'Yes']
            check_box_row = st.radio('SET ROW LIMIT?', row_limit_options)
            row_limit = st.number_input('Row limit', min_value=1, max_value=25000, value=25000) if check_box_row == 'Yes' else None

        

        if st.button('GET DATA ⬇️'):
            clear_data()
            if st.session_state.selected_site:
                dimensions = [dim for dim in selected_dimensions]

                progress_bar = st.progress(0)
                status_text = st.empty()
                total_downloaded_rows = 0
                start_row = 0

                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame()

                with st.spinner("Downloading data..."):
                    while True:
                        rows = fetch_data_chunk(webmasters_service, st.session_state.selected_site, start_date, end_date, dimensions, st.session_state.dimension_filters, selected_type, start_row, row_limit)
                        
                        if not rows:
                            st.warning("No data retrieved from API.")
                            break

                        data_list = []
                        for row in rows:
                            data_entry = {dimension: row['keys'][dimensions.index(dimension)] for dimension in dimensions}
                            data_entry.update({
                                'Clicks': row['clicks'],
                                'Impressions': row['impressions'],
                                'CTR': row['ctr'],
                                'Position': row['position']
                            })
                            data_list.append(data_entry)

                        chunk_df = pd.DataFrame(data_list)
                        st.session_state.df = pd.concat([st.session_state.df, chunk_df], ignore_index=True)
                        
                        total_downloaded_rows += len(rows)
                        start_row += len(rows)
                        status_text.text(f"Total rows downloaded: {total_downloaded_rows}")
                        
                        if len(rows) < 25000 or (row_limit and total_downloaded_rows >= row_limit):
                            break
                        
                        progress_bar.progress(min(total_downloaded_rows / (row_limit if row_limit else total_downloaded_rows + len(rows)), 1.0))
                    
                    st.session_state.data_loaded = True
                    st.session_state.download_ready = True
                    progress_bar.progress(100)

                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

if st.session_state.data_loaded and st.session_state.download_ready:
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download data CSV", data=csv, file_name='data.csv', mime='text/csv')

if st.session_state.data_loaded:
    df = st.session_state.df

    average_position = df['Position'].mean()
    total_clicks = df['Clicks'].sum()
    average_ctr = df['CTR'].mean() * 100
    total_impressions = df['Impressions'].sum()                    
    formatted_average_m= "{:.2f}".format(average_position)
    total_clicks_m = df['Clicks'].sum()
    average_ctr_m = df['CTR'].mean()
    average_ctr_perc= average_ctr_m * 100
    formatted_ctr_m = "{:.2f}%".format(average_ctr_perc)
    total_impressions_m = df['Impressions'].sum()
    with st.container(border=True):
        copy_website_data = df.copy()
        
        # Rimuovi la colonna "Cleaned_Page" se esiste nella copia
        if 'Cleaned_Page' in copy_website_data.columns:
            copy_website_data = copy_website_data.drop(columns=['Cleaned_Page'])
        
        st.subheader("Website Data")
        st.divider()      
   
     
        
        # Aggiornamento delle metriche
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        with col1:
            st.write(f" Performance overview of your website **from** {start_date.strftime('%Y-%m-%d')} **to** {end_date.strftime('%Y-%m-%d')}")
            dimensions = [col for col in copy_website_data.columns if col not in ['Date', 'Clicks', 'Impressions', 'CTR', 'Position']]
            selected_dimension = st.selectbox("Select Dimension", dimensions)
        
            if selected_dimension == 'Query':
                search_query = st.text_input("Enter Query")
                if search_query:
                    copy_website_data = copy_website_data[copy_website_data['Query'].str.contains(search_query, case=False)]
            else:
                selected_values = st.multiselect(f"Select {selected_dimension}", options=copy_website_data[selected_dimension].unique())
                if selected_values:
                    copy_website_data = copy_website_data[copy_website_data[selected_dimension].isin(selected_values)]
        
            
            
            # Calcola le metriche in base ai dati filtrati
            total_clicks = copy_website_data['Clicks'].sum()
            total_impressions = copy_website_data['Impressions'].sum()
            average_position = copy_website_data['Position'].mean()
            average_ctr = copy_website_data['CTR'].mean() * 100
        
        with col2:
            st.metric(label="Total Clicks", value=total_clicks)
        with col3:
            st.metric(label="Total Impressions", value=total_impressions)
        with col4:
            st.metric(label="Average Position", value=f"{average_position:.2f}")
        with col5:
            st.metric(label="Average CTR", value=f"{average_ctr:.2f}%")
        col1, col2=st.columns(2)
        with col1:
            st.dataframe(copy_website_data, width=2000, height=520)
        with col2:
            # Creazione del grafico
            if 'Date' in copy_website_data.columns:
                df_graf = copy_website_data.groupby('Date').agg({
                    'Clicks': 'sum',
                    'Impressions': 'sum',
                    'CTR': 'mean',
                    'Position': 'mean'
                }).reset_index()
            
                def traffic_report(df_graf):
                    df_graf['CTR'] = df_graf['CTR'].apply(lambda ctr: f"{ctr * 100:.2f}")
                    df_graf['Position'] = df_graf['Position'].apply(lambda pos: round(pos, 2))
            
                    options = {
                        "xAxis": {
                            "type": "category",
                            "data": df_graf['Date'].tolist(),
                            "axisLabel": {"formatter": "{value}"}
                        },
                        "yAxis": {"type": "value", "name": ""},
                        "grid": {"right": 20, "left": 65, "top": 45, "bottom": 50},
                        "legend": {
                            "show": True,
                            "top": "top",
                            "align": "auto",
                            "selected": {"Clicks": True, "Impressions": True, "CTR": False, "Position": False}
                        },
                        "tooltip": {"trigger": "axis"},
                        "series": [
                            {"type": "line", "name": "Clicks", "data": df_graf['Clicks'].tolist(), "smooth": True, "lineStyle": {"width": 1, "color": "#D5A021"}, "showSymbol": True},
                            {"type": "line", "name": "Impressions", "data": df_graf['Impressions'].tolist(), "smooth": True, "lineStyle": {"width": 1, "color": "#F06449"}, "showSymbol": False},
                            {"type": "line", "name": "CTR", "data": df_graf['CTR'].tolist(), "smooth": True, "lineStyle": {"width": 1, "color": "#91C499"}, "showSymbol": False},
                            {"type": "line", "name": "Position", "data": df_graf['Position'].tolist(), "smooth": True, "lineStyle": {"width": 1, "color": "#5BC3EB"}, "showSymbol": False, "yAxisIndex": 1, "axisLabel": {"show": "Position"}}
                        ],
                        "yAxis": [{"type": "value", "name": ""}, {"type": "value", "inverse": True, "show": False}],
                        "backgroundColor": "#0a0e12",
                        "color": ["#D5A021", "#F06449", "#91C499", "#5BC3EB"],
                    }
            
                    st_echarts(option=options, theme='chalk', height=500, width='100%')
            
                traffic_report(df_graf)
            else:
                st.write("### Traffic Trend")
                st.warning("No graph available without date data.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["QUERIES REPORT", "PAGES REPORT", "PAGE OPTIMIZATION","QUERIES GROUPER"])
   
    
    with tab1:
            with st.container(border=True):
                st.subheader("1. Queries Performance Report")
                st.divider()
               
                
                if all(dim in selected_dimensions for dim in ['Query']):
                    query_funcs = {
                        'Impressions': 'sum',
                        'Clicks': 'sum',
                        'CTR': 'mean',
                        'Position': 'mean'
                    }
                    # Raggruppiamo df per query per il bubble chart
                    df_query_performance = df.groupby('Query').agg(query_funcs).reset_index()
                    
                    # Converti il CTR in percentuale
                    df_query_performance['CTR'] = df_query_performance['CTR'] * 100
                    
                    # Calcola i valori minimi e massimi per il grafico
                    min_ctr = df_query_performance['CTR'].min()
                    max_ctr = df_query_performance['CTR'].max()
                    min_position = df_query_performance['Position'].min()
                    max_position = df_query_performance['Position'].max()
                    
                    # Calcola i valori medi di CTR e Posizione solo per le query selezionate
                    average_ctr = df_query_performance['CTR'].mean()
                    average_position = df_query_performance['Position'].mean()
                    
                    # Arrotonda la posizione media a due cifre decimali
                    df_query_performance['Position'] = df_query_performance['Position'].round(2)
                    
                    # Crea il grafico a bolle con Plotly utilizzando il DataFrame filtrato
                    fig = px.scatter(
                        df_query_performance, 
                        x='CTR', 
                        y='Position', 
                        size='Clicks', 
                        hover_data={'Query': True, 'CTR': ':.2f%', 'Position': ':.2f', 'Clicks': True},
                        custom_data=['Query']
                    )
                    
                    # Configura gli assi
                    fig.update_yaxes(autorange="reversed")
                    fig.update_yaxes(range=[min_position, max_position])
                    fig.update_xaxes(range=[min_ctr, max_ctr], tickformat=".2f%%") # Formatta l'asse X come percentuale
                    
                    # Aggiungi rettangoli colorati per i quadranti
                    fig.add_shape(type='rect', x0=min_ctr, x1=average_ctr, y0=min_position, y1=average_position,
                                  fillcolor='rgba(0, 0, 255, 0.2)', line=dict(width=0), layer='below')  # Bottom left quadrant - blue
                    fig.add_shape(type='rect', x0=average_ctr, x1=max_ctr, y0=min_position, y1=average_position,
                                  fillcolor='rgba(0, 255, 0, 0.2)', line=dict(width=0), layer='below')  # top right quadrant - green
                    fig.add_shape(type='rect', x0=min_ctr, x1=average_ctr, y0=average_position, y1=max_position,
                                  fillcolor='rgba(255, 0, 0, 0.2)', line=dict(width=0), layer='below')  # Top left quadrant - red
                    fig.add_shape(type='rect', x0=average_ctr, x1=max_ctr, y0=average_position, y1=max_position,
                                  fillcolor='rgba(255, 255, 0, 0.2)', line=dict(width=0), layer='below')  # bottom right quadrant - yellow
                    
                    # Aggiungi linee di riferimento per la media di CTR e posizione
                    fig.add_shape(type='line', x0=average_ctr, x1=average_ctr, y0=min_position, y1=max_position, line=dict(color='red', dash='dash'))
                    fig.add_annotation(x=average_ctr, y=max_position, text="Average", showarrow=False, yshift=10, font=dict(color='white'))
                    
                    fig.add_shape(type='line', x0=min_ctr, x1=max_ctr, y0=average_position, y1=average_position, line=dict(color='red', dash='dash'))
                    fig.add_annotation(x=max_ctr, y=average_position, text="Average", showarrow=False, xshift=10, font=dict(color='white'))
                
                    # Aggiorna le tracce delle bolle
                    fig.update_traces(marker=dict(sizemin=4), hovertemplate='<b>Query:</b> %{customdata[0]}<br><b>CTR:</b> %{x:.2f}%<br><b>Position:</b> %{y:.2f}<br><b>Clicks:</b> %{marker.size}')
                         
                    
                    # Aggiungi titolo
                    fig.update_layout(
                        title="Query Performance Bubble Chart",
                        paper_bgcolor='rgb(10,14,18)',  
                        plot_bgcolor='rgb(10,14,18)' 
                    )
                    
                    # Aggiungi la mappa di colori per la dimensione delle bolle
                 
                    # Mostra il grafico interattivo
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write("This section provides a visual representation of how different search queries are performing. It analyzes and displays metrics such as the position of queries in search results and their click-through rates (CTR). By comparing these metrics to overall averages, it helps identify which queries are performing well and which need improvement.")

                    with col2:
                        unique_query_count_metric = df_query_performance['Query'].nunique()
                        st.metric("Queries", f"{unique_query_count_metric}")
                    with col3:                      
                        st.metric("AVG. CTR", f"{average_ctr_perc:.2f}%")
                    with col4:                                             
                        st.metric("AVG. Position", f"{average_position:.2f}")
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("")
                        with st.popover("How to read the graph?🤔"):
                            st.markdown("""                            
                            
                            #### Chart Elements
                            - **Bubbles:** Each bubble represents a single search query.
                            - **Axes:** 
                              - **X-Axis (CTR):** Positions the bubbles based on the click-through rate (CTR) of the queries.
                              - **Y-Axis (Average Position):** Positions the bubbles based on the average position in search results.
                            - **Bubble Size:** Indicates the number of clicks received by the query. The larger the bubble, the more clicks it has received.
                            - **Dashed Lines:** Represent the average values of CTR and position. The vertical dashed line indicates the average CTR, while the horizontal dashed line indicates the average position.
                            
                            #### Quadrant Interpretation
                            - **Green Quadrant (Top Right):** Queries with CTR and position equal to or above the average. These queries have excellent visibility and attract many clicks.
                            - **Yellow Quadrant (Bottom Right):** Queries with above-average CTR but low position. These indicate user interest despite less favorable ranking.
                            - **Blue Quadrant (Top Left):** Queries with equal to or above-average position but below-average CTR. These queries are visible but receive fewer clicks.
                            - **Red Quadrant (Bottom Left):** Queries with below-average CTR and position. These queries have poor visibility and attract few clicks.
                            
                            This structure helps quickly identify where queries stand relative to the overall average, facilitating the identification of areas for improvement and optimization opportunities.
                            """)                            
                      
                        st.plotly_chart(fig, use_container_width=False)
                        fig.show()
                    with col2:
                        # Suddividere i dati in quattro DataFrame in base ai quadranti specificati e fornire all'utente la lista delle query in ciascun quadrante
                        upper_high_ctr = df[(df['Position'] <= average_position) & (df['CTR'] > average_ctr)]
                        lower_high_ctr = df[(df['Position'] > average_position) & (df['CTR'] > average_ctr)]
                        lower_low_ctr = df[(df['Position'] > average_position) & (df['CTR'] <= average_ctr)]
                        upper_low_ctr = df[(df['Position'] <= average_position) & (df['CTR'] <= average_ctr)]
                    
                        def unique_pages(series):
                            return ', '.join(series.unique())
                    
                        try:
                            agg_funcs2 = {
                                'Impressions': 'sum',
                                'Clicks': 'sum',
                                'CTR': 'mean',
                                'Position': 'mean',
                                'Page': unique_pages
                            }
                    
                            # Raggruppiamo e aggreghiamo i DataFrame dei quadranti
                            df_upper_high_ctr = upper_high_ctr.groupby('Query').agg(agg_funcs2).reset_index()
                            df_lower_high_ctr = lower_high_ctr.groupby('Query').agg(agg_funcs2).reset_index()
                            df_lower_low_ctr = lower_low_ctr.groupby('Query').agg(agg_funcs2).reset_index()
                            df_upper_low_ctr = upper_low_ctr.groupby('Query').agg(agg_funcs2).reset_index()
                    
                            # Mostrare df
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            with st.expander(":green[GREEN QUADRANT: Queries with Above-Average Position and CTR]"):
                                st.write("Queries with CTR and position equal or grather then the average")
                                st.write(df_upper_high_ctr)    
                                
                            with st.expander(":orange[YELLO QUADRANT: Queries with Below-Average Position and Above-Average CTR]"):
                                st.write("""
                                    Those queries appear to be highly relevant to users. They achieve a high click-through rate (CTR) even when they rank lower than the average query on your website. If the average position of these queries improves, it could significantly impact your website's performance. It's advisable to focus on enhancing the SEO for these queries. For instance, consider a prominent query in quadrant 2 for a gardening website, such as "how to build a wooden shed." Check if you already have a dedicated page for this topic and proceed in two ways:
                    
                                    - If you don't have a dedicated page, think about creating one to consolidate all the information on your website related to this subject.
                    
                                    - If you already have a page, contemplate adding more content to better address the needs of users searching for this query.
                                """)
                                st.write(df_lower_high_ctr)
        
                            with st.expander(":blue[BLUE QUADRANT: Top position and low CTR Queries]"):
                                st.write("""
                                    These queries might have a low click-through rate (CTR) for various reasons. Check the largest bubbles to find signs of the following:
                    
                                    Your competitors may be using structured data markup and appearing with rich results, attracting users to click on their results instead of yours. Consider optimizing for the most common visual elements in Google Search.
                    
                                    You may have optimized, or be "accidentally" ranking for a query that users are not interested in relation to your site. This might not be an issue for you, in which case you can ignore those queries. If you prefer people not to find you through those queries (for example, they contain offensive words), try to fine-tune your content to remove mentions that could be seen as synonyms or related queries to the one bringing traffic.
                    
                                    People may have already found the information they needed, for example, your company's opening hours, address, or phone number. Check the queries that were used and the URLs that contained the information. If one of your website goals is to drive people to your stores, this is working as intended; if you believe that people should visit your website for extra information, you could try to optimize your titles and descriptions to make that clear. See the next section for more details.
                                """)
                                st.write(df_upper_low_ctr)
                            with st.expander(":red[RED QUADRANT: Low position and low CTR Queries]"):
                                st.write("""
                                    When looking at queries with low CTR (both with low and top position), it's especially interesting to look at the bubble sizes to understand which queries have a low CTR but are still driving significant traffic. While the queries in this quadrant might seem unworthy of your effort, they can be divided into two main groups:
                                    
                                    **Related queries**: If the query in question is important to you, it's a good start to have it appearing in Search already. Prioritize these queries over queries that are not appearing in Search results at all, as they'll be easier to optimize.
                                    
                                    **Unrelated queries**: If your site doesn't cover content related to this query, maybe it's a good opportunity to fine tune your content or focus on queries that will bring relevant traffic.
                                """)
                                st.write(df_lower_low_ctr)
                        except KeyError as e:
                            st.warning("To obtain insights on both queries and pages, consider adding 'Page' to the dimensions in your analysis.")
     
                


            # Controllo se il DataFrame contiene le colonne 'Query' e 'Page'
            if 'Query' in df.columns and 'Page' in df.columns:
                with st.container(border=True):
                    st.subheader("2. Queries distribution on SERP Pages Report")
                    st.divider()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("")
            
                        def assign_page(position):
                            if position <= 10:
                                return 'Page 1'
                            elif position <= 20:
                                return 'Page 2'
                            elif position <= 30:
                                return 'Page 3'
                            elif position <= 40:
                                return 'Page 4'
                            elif position <= 50:
                                return 'Page 5'
                            elif position <= 60:
                                return 'Page 6'
                            elif position <= 70:
                                return 'Page 7'
                            elif position <= 80:
                                return 'Page 8'
                            elif position <= 90:
                                return 'Page 9'
                            elif position <= 100:
                                return 'Page 10'
                            else:
                                return 'Beyond Page 10'
            
                        # Assicurati che il DataFrame contenga una colonna 'Position'
                        df_query_page_serp = df.copy()
                        df_query_page_serp['SERP_Page'] = df_query_page_serp['Position'].apply(assign_page)
            
                        # Rimuovere i duplicati dalle query basandosi sulla combinazione di 'Query' e 'SERP_Page'
                        df_query_performance_unique = df_query_page_serp.drop_duplicates(subset=['Query', 'SERP_Page'])
            
                        # Conta il numero totale di query uniche
                        total_unique_queries = df_query_performance_unique['Query'].nunique()
            
                        # Raggruppa per pagina e conta il numero di query uniche
                        page_distribution = df_query_performance_unique.groupby('SERP_Page').size().reset_index(name='Num_Queries')
            
                        # Calcola la percentuale del totale
                        page_distribution['Percentage_of_Total'] = (page_distribution['Num_Queries'] / total_unique_queries) * 100
            
                        # Ordina il DataFrame per pagina
                        page_order = ['Page 1', 'Page 2', 'Page 3', 'Page 4', 'Page 5', 'Page 6', 'Page 7', 'Page 8', 'Page 9', 'Page 10', 'Beyond Page 10']
                        page_distribution['SERP_Page'] = pd.Categorical(page_distribution['SERP_Page'], categories=page_order, ordered=True)
                        page_distribution = page_distribution.sort_values('SERP_Page')
            
                        # Visualizza il numero totale di query uniche
                        st.write(f"Total Unique Queries: {total_unique_queries}")
            
                        # Creare il grafico a barre
                        fig_bar = px.bar(page_distribution, x='SERP_Page', y='Num_Queries', title='Number of Queries per Google SERP Page', text='Percentage_of_Total', color='SERP_Page', category_orders={'SERP_Page': page_order})
                        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            
                        # Rimuovere la legenda
                        fig_bar.update_layout(showlegend=False,paper_bgcolor='rgb(10,14,18)',plot_bgcolor='rgb(10,14,18)' )          
                        # Visualizzare il grafico utilizzando Streamlit
                        st.plotly_chart(fig_bar)
            
                    with col2:
                        # Visualizza il DataFrame
                        st.dataframe(page_distribution)
            else:
                st.warning("Add 'Query' or 'Page' dimension to generate the 2. Queries distribution on SERP Pages Report")
        
            try:
                # Controllo se la colonna 'Page' è presente
                if 'Page' in df.columns:
                    df['Cleaned_Page'] = df['Page'].apply(lambda x: x.split('#')[0])
            
                    with st.container(border=True):
                        st.subheader("3. Queries Cannibalization Report")            
                        st.divider()
                        
                        # Group by the cleaned page and query, and calculate the metrics
                        query_page_metrics = df.groupby(['Query', 'Cleaned_Page']).agg({
                            'Position': 'mean',
                            'CTR': 'mean',
                            'Clicks': 'sum',
                            'Impressions': 'sum'
                        }).reset_index()
                        
                        # Group queries by page
                        query_page_group = query_page_metrics.groupby('Query')['Cleaned_Page'].apply(lambda pages: list(set(pages))).reset_index()
                        query_page_group.columns = ['Query', 'Pages']
                        
                        # Identify cannibalized queries
                        cannibalized_queries = query_page_group[query_page_group['Pages'].apply(lambda x: len(x) > 1)]
                        
                        # Create a cannibalization report
                        cannibalization_report = cannibalized_queries.explode('Pages').merge(query_page_metrics, left_on=['Query', 'Pages'], right_on=['Query', 'Cleaned_Page']).drop(columns=['Pages'])
                        
                        # Count unique cannibalized queries
                        num_unique_queries = cannibalized_queries['Query'].nunique()
                        
                        # Count the number of pages that are cannibalizing queries
                        num_cannibalizing_pages = cannibalization_report['Cleaned_Page'].nunique()
                        
                        # Calculate the average number of pages cannibalizing each query
                        num_pages_per_query = cannibalization_report.groupby('Query')['Cleaned_Page'].nunique()
                        average_pages_per_query = num_pages_per_query.mean()
                    
                        # Create a DataFrame with the columns 'Query' and 'Number of Pages'
                        cannibalized_queries['Cannibals Pages'] = cannibalized_queries['Pages'].apply(len)
                        cannibalized_query_page_counts = cannibalized_queries[['Query', 'Cannibals Pages']]      
                        cannibalized_query_page_counts_ordered = cannibalized_query_page_counts.sort_values('Cannibals Pages', ascending=False)
                    
                        # Display the report
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.write("It shows queries appearing on multiple pages and the number of pages per query. Select a query to see detailed metrics for each page (CTR, clicks, position, impressions) aiding in resolving cannibalization issues.")
                            
                            def convert_cannibalization_report_to_csv(cannibalization_report):
                                return cannibalization_report.to_csv(index=False).encode('utf-8')
                            
                            csvcann = convert_cannibalization_report_to_csv(cannibalization_report)
                            st.download_button(label="Download cannibalization report CSV", data=csvcann, file_name='cannibalization_report.csv', mime='text/csv')
                        
                        with col2:
                            st.metric("Unique cannibalized queries", f"{num_unique_queries}")
                        with col3:
                            st.metric("Pages cannibalizing queries", f"{num_cannibalizing_pages}")
                        with col4:
                            st.metric("AVG n° of pages cannibalizing each query", f"{average_pages_per_query:.2f}")
                    
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(cannibalized_query_page_counts_ordered)
                        with col2:
                            query_selected = st.selectbox("Select a cannibalized Query", cannibalized_queries['Query'])
                            # Filter the report based on the selected query
                            filtered_report = cannibalization_report[cannibalization_report['Query'] == query_selected]
                            
                            # Display the filtered DataFrame
                            st.write(f"Metrics for the selected Query: {query_selected}")
                            st.dataframe(filtered_report)
            

                    
            
                else:
                    # Mostra un messaggio di avviso se la colonna 'Page' non è presente
                    st.warning("Add 'Page' to dimensions to show 3. Queries Cannibalization Report")
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")
            
            analyze_query_performance(df)

            analyze_query_position_changes(df)

                

        
        
    
    with tab2:
       # Supponiamo che `df` sia già definito e contenga i dati necessari
        try:
            # Raggruppamento dei dati
            agg_funcs = {
                'Impressions': 'sum',
                'Clicks': 'sum',
                'CTR': 'mean',
                'Position': 'mean'
            }
            df_aggregated_popular_page = df.groupby('Page').agg(agg_funcs).reset_index()
            
            df_aggregated_popular_page['CTR'] = (df_aggregated_popular_page['Clicks'] / df_aggregated_popular_page['Impressions'])
            df_aggregated_popular_page['CTR'] = df_aggregated_popular_page['CTR'].map('{:.2%}'.format)
            df_aggregated_popular_page = df_aggregated_popular_page.rename(columns={'CTR': 'Average CTR'})
            df_aggregated_popular_page['Position'] = df_aggregated_popular_page['Position'].round(2)
            average_position_popular = df_aggregated_popular_page['Position'].mean()
            df_aggregated_popular_page = df_aggregated_popular_page.rename(columns={'Position': 'Average Position'})
            average_clic_df_popular = df_aggregated_popular_page['Clicks'].mean()
            average_impression_df_pupular = df_aggregated_popular_page['Impressions'].mean()
        
            popular_pages = df_aggregated_popular_page[
                (df_aggregated_popular_page['Average CTR'] > formatted_ctr_m) &
                (df_aggregated_popular_page['Clicks'] > average_clic_df_popular) &
                (df_aggregated_popular_page['Impressions'] > average_impression_df_pupular) &
                (df_aggregated_popular_page['Average Position'] < 10)
            ].sort_values(by='Clicks', ascending=False)
            
            less_pages = df_aggregated_popular_page[
                (df_aggregated_popular_page['Average CTR'] < formatted_ctr_m) &
                (df_aggregated_popular_page['Clicks'] > average_clic_df_popular) &
                (df_aggregated_popular_page['Impressions'] > average_impression_df_pupular) &
                (df_aggregated_popular_page['Average Position'] < 10)
            ]
            
            opp_pages = df_aggregated_popular_page[
                (df_aggregated_popular_page['Clicks'] > average_clic_df_popular) &
                (df_aggregated_popular_page['Impressions'] > average_impression_df_pupular) &
                (df_aggregated_popular_page['Average Position'] > 10 ) &
                (df_aggregated_popular_page['Average Position'] <= 20)
            ]
            
            worst_pages = df_aggregated_popular_page[
                (df_aggregated_popular_page['Clicks'] < average_clic_df_popular) &
                (df_aggregated_popular_page['Impressions'] < average_impression_df_pupular) &
                (df_aggregated_popular_page['Average CTR'] < formatted_ctr_m) &
                (df_aggregated_popular_page['Average Position'] > average_position_popular)
            ]
            with st.container(border=True):
                st.subheader("1. Pages Health Check Report")            
                st.divider()
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                format_average_clicks_popular = "{:.2f}".format(average_clic_df_popular)
                format_average_impression_popular = "{:.2f}".format(average_impression_df_pupular)
                format_average_position_popular = "{:.2f}".format(average_position_popular)
    
                with col1: 
                    st.write("This report provides an overview of your website's page performance and help identify areas for improvement.")                
                with col2:
                    st.metric("Pages Average Clicks", value=format_average_clicks_popular)
                with col3:
                    st.metric("Pages Average Impressions", value=format_average_impression_popular)
                with col4:
                    st.metric("Pages Average CTR", value=formatted_ctr_m)
                with col5:
                    st.metric("Pages Average Position", value=format_average_position_popular)
                    st.text("")      
               
                
                worst_pages_count = worst_pages.shape[0]
                opp_pages_count = opp_pages.shape[0]
                less_pages_count = less_pages.shape[0]
                popular_pages_count = popular_pages.shape[0]
                
                chart_data = {
                    "Set": ["Best Pages", "Less Effective Pages", "Ranking opportunities", "Require attention"],
                    "N°Pages": [popular_pages_count, less_pages_count, opp_pages_count, worst_pages_count]
                }
            
                # Creazione del grafico a barre colorato con plotly
                fig = px.bar(
                    x=chart_data["Set"],
                    y=chart_data["N°Pages"],
                    labels={"x": "Set", "y": "N°Pages"},
                    title="🏥 Pages health check graph",
                    color=chart_data["Set"],
                    color_discrete_map={
                        "Best Pages": "green",
                        "Less Effective Pages": "yellow",
                        "Ranking opportunities": "blue",
                        "Require attention": "red"
                    }
                )
                           
                fig.update_layout(showlegend=False, paper_bgcolor='rgb(10,14,18)', plot_bgcolor='rgb(10,14,18)' )
                
                
                col1, col2=st.columns([2,2])
                with col1:
                    # Visualizzare il grafico in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    with st.expander(":green[BEST PAGES]"):
                        st.write("Pages with an elevated Click-Through Rate (CTR), a significant volume of Clicks, and a substantial number of Impressions (exceeding the average), with Average position within the top 10 search engine result positions.")
                        st.write(popular_pages)
                    with st.expander(":orange[LESS EFFECTIVE PAGES]"):
                        st.write("Page with High Clicks, High Impressions and Average position within the top 10 search engine result positions, but low CTR")
                        st.write(less_pages)
                    with st.expander(":blue[PAGES WITH RANKING OPPORTUNITES]"):
                        st.write("Page with High Clicks, High Impressions but average position between 10-20 in SERP")
                        st.write(opp_pages)
                    with st.expander(":red[PAGES THAT REQUIRE ATTENTION]"):
                        st.write("Page low Clicks, Low Impression, Low CTR and Low Position in comparison to the average")
                        st.write(worst_pages)

            with st.container(border=True):
               
                analyze_page_performance(df)
            
        except KeyError as e:
            st.warning("Add 'Page' to dimensions to show 1. Pages Health Check Report and 2. Pages Traffic Changes Report")
                    
            st.markdown("<br>", unsafe_allow_html=True)
        
 


        with tab3:
            with st.container():
                st.subheader("1. Queries Coverage Analysis")
                st.divider()
            
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(
                        "This report checks if Google's considered queries are present in various webpage elements such as the title, meta description, headings, body content, and ALT tags. This helps you identify gaps by finding missing important keywords.")
            
                with col2:
                    if st.session_state.df is not None and 'Page' in st.session_state.df.columns and 'Query' in st.session_state.df.columns:
                        selected_page = st.selectbox("Select a page", st.session_state.df['Page'].unique(), key='select_page')
                        scan_button = st.button("Analyze Page🤖", key='scan_button')
                    else:
                        st.warning("To use this feature, ensure that the dimensions contains the 'Page' and 'Query'.")
                        scan_button = False
    
                if scan_button or st.session_state.scan_started:
                    if scan_button:
                        st.session_state.scan_started = True
                    if selected_page and (selected_page != st.session_state.get('selected_page', None)):
                        st.session_state.selected_page = selected_page
                        with st.spinner("Fetching page data..."):
                            st.session_state.page_data = fetch_page_data(selected_page)
                        st.session_state.keyword_analysis = None
            
                    if 'page_data' in st.session_state and st.session_state.page_data is not None:
                        page_data = st.session_state.df[st.session_state.df['Page'] == st.session_state.selected_page][['Query', 'Clicks', 'Impressions', 'CTR', 'Position']]
                        grouped_page_data = aggregate_queries(page_data)
                  
                        # Analisi della copertura delle parole chiave
                        with st.container():
                            st.markdown(
                                f"<h4>📄 {st.session_state.page_data['meta_title']} | <a href='{st.session_state.selected_page}'>Go to the page</a></h4>",
                                unsafe_allow_html=True)
            
                            # Visualizzare i warning se presenti
                            if 'warnings' in st.session_state.page_data:
                                for warning in st.session_state.page_data['warnings']:
                                    st.warning(warning)
            
                            keyword_presence = analyze_keywords(st.session_state.page_data, grouped_page_data)
                            keyword_df = pd.DataFrame(keyword_presence)
            
                            # Controllo se 'Query' è presente nelle colonne
                            if 'Keyword' not in keyword_df.columns:
                                st.error("The DataFrame does not contain the required column 'Keyword'. Please check the data processing.")
                            else:
                                # Gestione dello stato del filtro di ricerca per query
                                search_query = st.text_input(
                                    label="",
                                    placeholder="Filter queries containing:",
                                    value=st.session_state.get('search_query', '')
                                )
                                st.session_state.search_query = search_query
                                if search_query:
                                    keyword_df = keyword_df[keyword_df['Keyword'].str.contains(search_query, case=False, na=False)]
    
                                # Checkbox per mostrare/nascondere colonne
                                show_heading = st.checkbox("Show/hide heading", st.session_state.show_heading)
                                show_keyword_metrics = st.checkbox("Show/hide keyword metrics", st.session_state.show_keyword_metrics)
                                show_meta = st.checkbox("Show/hide meta", st.session_state.show_meta)
                                show_body_alt = st.checkbox("Show/hide body alt", st.session_state.show_body_alt)
            
                                st.session_state.show_heading = show_heading
                                st.session_state.show_keyword_metrics = show_keyword_metrics
                                st.session_state.show_meta = show_meta
                                st.session_state.show_body_alt = show_body_alt
            
                                # Creare la lista delle colonne da mostrare
                                columns_to_show = ['Keyword']  # La colonna Keyword deve essere sempre visibile
                                if show_keyword_metrics:
                                    columns_to_show.extend(['Clicks', 'Impressions', 'CTR', 'Position'])
                                if show_meta:
                                    columns_to_show.extend(['Title', 'Meta Description'])
                                if show_heading:
                                    columns_to_show.extend(['H1', 'H2', 'H3', 'H4', 'H5', 'H6'])
                                if show_body_alt:
                                    columns_to_show.extend(['Body Content', 'Alt Tags'])
            
                                keyword_df = keyword_df[columns_to_show]
            
                                # Gestione dello stato del filtro per query non presenti in nessun elemento
                                show_not_covered = st.checkbox("Show only queries not covered in any element", st.session_state.show_not_covered)
                                st.session_state.show_not_covered = show_not_covered
            
                                if show_not_covered:
                                    keyword_df = keyword_df[(keyword_df['Title'] == False) &
                                                            (keyword_df['Meta Description'] == False) &
                                                            (keyword_df['H1'] == False) &
                                                            (keyword_df['H2'] == False) &
                                                            (keyword_df['H3'] == False) &
                                                            (keyword_df['Body Content'] == False) &
                                                            (keyword_df['Alt Tags'] == False)]
            
                                # Ordina il dataframe per la colonna 'Body Content'
                                if 'Body Content' in keyword_df.columns:
                                    keyword_df_sorted = keyword_df.sort_values(by='Body Content', ascending=False)
                                else:
                                    keyword_df_sorted = keyword_df.sort_values(by='Keyword', ascending=False)
            
                                # Visualizza il dataframe ordinato
                                st.dataframe(keyword_df_sorted)
            
                                # Parole chiave con opportunità
            
                                st.markdown(
                                    f"<h4>Prioritize the Optimization of These Queries</h4>",
                                    unsafe_allow_html=True)
                                st.write("These queries are currently ranked between positions 1 and 20 and do not appear in any key elements of your page. Optimize your page for these queries to improve their ranking")
                                opportunity_keywords = get_opportunity_keywords(grouped_page_data, keyword_presence)
                                st.dataframe(opportunity_keywords)
        
            with st.container():
                st.subheader("2. Page Topics")
                st.divider()
                st.write(
                    "We have grouped the keywords for which Google is considering your page to identify the main themes. For each theme, you can check the total clicks and impressions, as well as the coverage percentage of the theme by your page content.")
                if 'page_data' in st.session_state and st.session_state.page_data is not None:
                    with st.spinner("Clustering topics..."):
                        clustered_keywords, model = cluster_keywords(grouped_page_data)
                        cluster_names = get_cluster_names(clustered_keywords)
                        clustered_keywords = analyze_topic_coverage(st.session_state.page_data, clustered_keywords)
                    
                        for cluster in clustered_keywords['Cluster'].unique():
                            cluster_name = cluster_names[cluster]
                            cluster_df = clustered_keywords[clustered_keywords['Cluster'] == cluster]
                            total_keywords = len(cluster_df)
                            covered_keywords = cluster_df['Covered'].sum()
                            coverage_percentage = (covered_keywords / total_keywords) * 100
                            cluster_name = cluster_name.upper()
                    
                            with st.expander(
                                    f"**:blue[{cluster_name}]**  | __Clicks {cluster_df['Clicks'].sum()}__  | Impressions {cluster_df['Impressions'].sum()} |  Coverage by page content {coverage_percentage:.2f}%"):
                                st.dataframe(cluster_df[['Query', 'Clicks', 'Impressions', 'CTR', 'Position', 'Covered']])
                
                if 'scan_started' in st.session_state and st.session_state.scan_started:
                    st.write("")
           
    
        with tab4:
            st.subheader("Queries Grouper")        
            col1, col2 = st.columns(2)
            with col1:
                st.write("💬 Select language")
                language = st.selectbox("", ["English", "Italian"])
            
                if language == "English":
                    default_stop_words = [
                        'and', 'but', 'is', 'the', 'to', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
                        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                        'own', 'same', 'so', 'than', 'too', ' very', 's', 't', 'can', 'will', 'just', 'don', 
                        'should', 'now'
                    ]
                else:
                    default_stop_words = [
                        'a', 'adesso', 'ai', 'al', 'alla', 'allo', 'allora', 'altre',
                        'altri', 'altro', 'anche', 'ancora', 'avere', 'aveva', 'avevano',
                        'ben', 'buono', 'che', 'chi', 'cinque', 'comprare', 'con',
                        'consecutivi', 'consecutivo', 'cosa', 'cui', 'da', 'del', 'della',
                        'dello', 'dentro', 'deve', 'devo', 'di', 'doppio', 'due', 'e',
                        'ecco', 'fare', 'fine', 'fino', 'fra', 'gente', 'giu', 'ha', 'hai',
                        'hanno', 'ho', 'il', 'indietro', 'invece', 'io', 'la', 'lavoro',
                        'le', 'lei', 'lo', 'loro', 'lui', 'lungo', 'ma', 'me', 'meglio',
                        'molta', 'molti', 'molto', 'nei', 'nella', 'no', 'noi', 'nome',
                        'nostro', 'più', 'se', 'o', 'per', 'un', 'una'
                    ]
            
            # Text area for custom stop words
            with st.expander("Customize Stop Words"):
                custom_stop_words = st.text_area("One per line", "\n".join(default_stop_words))
                stop_words = [word.strip() for word in custom_stop_words.split('\n') if word.strip()]
            
            with col2:
                st.text("")
            st.divider()
            
            # Control for minimum group size and tuple length
            min_group_size, ngram_size = st.columns(2)
            with min_group_size:
                min_group_size = st.slider("Minimum Group Size",
                                           min_value=1,
                                           max_value=50,
                                           value=2,
                                           help="The minimum group size is the minimum number of keywords required in a group for it to be displayed in the results. Increase this value to show only larger keyword groups."
                                          )
            
            with ngram_size:
                ngram_size = st.slider(
                "Length of the keyword", 
                min_value=1, 
                max_value=5, 
                value=2, 
                help="Drag the slider to choose the n-gram size for the keyword. An n-gram size of 1 means a single word, whereas 5 means a phrase of up to 5 words."
            )
            
            keyword_column = 'Query'
            clicks_column = 'Clicks'
            
            if st.button("Group Keywords with Clicks ✨"):
                with st.spinner("Grouping..."):
                    try:
                        if keyword_column in df.columns and clicks_column in df.columns:
                            # Remove duplicates and sum clicks
                            df_cleaned = remove_duplicates_and_sum_clicks(df, keyword_column, clicks_column)
                            # Group keywords
                            st.session_state.keyword_groups = group_keywords(df_cleaned, stop_words, min_group_size, ngram_size, keyword_column=keyword_column)
                            # Calculate click totals
                            st.session_state.click_totals = calculate_click_totals(df_cleaned, st.session_state.keyword_groups, keyword_column=keyword_column, clicks_column=clicks_column)
                        else:
                            st.warning("The DataFrame must contain 'Query' and 'Clicks' columns to proceed.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            
                if st.session_state.keyword_groups is not None and st.session_state.click_totals is not None:
                    sorted_groups = sorted(st.session_state.click_totals.items(), key=lambda x: x[1], reverse=True)
                    top_groups = sorted_groups[:5]
                    tab1, tab2 = st.columns([2, 2])
                    with tab1:
                        try:
                            st.subheader("🔑 Groups")
                            for group, total_clicks in sorted_groups:
                                with st.expander(f"{group} - Total Clicks: {total_clicks}"):
                                    keywords_list = st.session_state.keyword_groups[st.session_state.keyword_groups['Group'] == group]['Keywords'].tolist()
                                    keyword_clicks_df = df_cleaned[df_cleaned[keyword_column].isin(keywords_list)][[keyword_column, clicks_column]]
                                    st.write(keyword_clicks_df)
                        except KeyError as e:
                            st.warning(str(e))
       
                    with tab2:
                        top_groups_clicks = [click for group, click in top_groups]
                        top_group_names = [group for group, click in top_groups]
                
                        if len(top_groups) > 0:
                            fig, ax = plt.subplots()
                            ax.barh(top_group_names, top_groups_clicks)
                            ax.set_xlabel('Total Clicks')
                            ax.set_ylabel('Group Name')
                            ax.set_title('Top 5 Groups by Clicks')
                            st.pyplot(fig)
            
                st.subheader("🔑 Groups Overview")
                top_groups_data = []
                for group, total_clicks in sorted_groups:
                    keywords_list = st.session_state.keyword_groups[st.session_state.keyword_groups['Group'] == group]['Keywords'].tolist()
                    top_groups_data.append({
                        "Group": group,
                        "Total Clicks": total_clicks,
                        "Keywords": ", ".join(keywords_list)
                    })
            
                top_groups_df = pd.DataFrame(top_groups_data)
                st.dataframe(top_groups_df)
            
                # Sezione per visualizzare i dettagli di ciascun gruppo
                st.subheader("🔍 Group Details")
                selected_group = st.selectbox("Select a Group to View Details", top_groups_df["Group"])
            
                if selected_group:
                    try:
                        group_details_df = st.session_state.keyword_groups[st.session_state.keyword_groups['Group'] == selected_group]
                        keyword_clicks_df = df_cleaned[df_cleaned[keyword_column].isin(group_details_df['Keywords'])][[keyword_column, clicks_column]]
                        st.write(f"Details for group: {selected_group}")
                        st.dataframe(keyword_clicks_df)
                    except KeyError as e:
                        st.warning(str(e))
            
                # Grafico dei Top 5 gruppi per clic
                st.subheader("📊 Top 5 Groups by Clicks")
                top_groups_clicks = [click for group, click in top_groups]
                top_group_names = [group for group, click in top_groups]
            
                if len(top_groups) > 0:
                    fig, ax = plt.subplots()
                    ax.barh(top_group_names, top_groups_clicks)
                    ax.set_xlabel('Total Clicks')
                    ax.set_ylabel('Group Name')
                    ax.set_title('Top 5 Groups by Clicks')
                    st.pyplot(fig)
