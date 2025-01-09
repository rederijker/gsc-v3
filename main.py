import streamlit as st
import httplib2
import pandas as pd
import zipfile
import io
from apiclient.discovery import build
from datetime import timedelta
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
import streamlit.components.v1 as components
import concurrent.futures 
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from googleapiclient.http import BatchHttpRequest
import json
import urllib.parse
from googleapiclient.errors import HttpError
from streamlit_option_menu import option_menu



#PAGE CONFIGURATION
st.set_page_config(
    page_title="GSC InsightHub:SEO analytic tool with GSC data-by Cristiano Caggiula",
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
if 'selected_group' not in st.session_state:
    st.session_state.selected_group = None

if 'selected_page_on_page' not in st.session_state:
    st.session_state.selected_page = None
# Inizializzazione dello stato della sessione indexing api
if 'action' not in st.session_state:
    st.session_state['action'] = "Update URL"
if 'urls' not in st.session_state:
    st.session_state['urls'] = ""
if 'json_file' not in st.session_state:
            st.session_state['json_file'] = None
# Inizializzazione dello stato della sessione URL INSPECTION
if "api_app" not in st.session_state:
    st.session_state.api_app = "***GET INSIGHT FROM MY GSC DATA***"  # Imposta il valore predefinito
if "urls_to_inspect" not in st.session_state:
    st.session_state.urls_to_inspect = ""
#Initialize download csv urls bulk inspection tool
# All'inizio dello script, inizializza il flag se non esiste
if 'inspection_results' not in st.session_state:
    st.session_state['inspection_results'] = None

def handle_tab_selection(tab_index):
    st.session_state.selected_tab = tab_index

    
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

def determine_optimal_clusters(X, max_clusters=50):
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



# Page Layout
st.markdown("""
            <style>hr {margin:0em 0px;}</style>
            """, unsafe_allow_html=True
          )



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
        <h1 style="text-align:center;">GSC InsightHub</h1><br>""",
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
            <h1 style="text-align:center;">GSC InsightHub</h1>
            <p style="text-align:center;">made with 🎈 by <a href="https://www.linkedin.com/in/cristiano-caggiula/">Cristiano Caggiula</a></p>
            <h2 style="text-align:center;">Master Google Search Console Data like a Pro with a Free SEO tool</h2>
            <p style="text-align:center;font-size:17px;">
                Explore <strong>Google Search Console data</strong>, generate detailed reports, customize searches, and access unlimited information without programming skills required for <strong>Google Search Console API</strong>. Perfect for webmasters, SEO experts, and digital marketers.
            </p>
            <div style="text-align:center;">
                <a href="{auth_url}" style="text-decoration:none;">
                    <button style="background-color: white; color: #4285F4; border: 1px solid #4285F4; padding: 10px 20px; font-size: 17px; border-radius: 5px; cursor: pointer; display: inline-flex; align-items: center;">
                        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1200px-Google_%22G%22_logo.svg.png" alt="Google logo" style="width: 47px; height: 47px; margin-right: 8px;">
                        Login with Google
                    </button>
                </a>
            </div>
            <br></br>
            <hr>
            <br>
            <p>GSC InsightHub leverages Google Search Console data to provide comprehensive SEO analytics. Here's a breakdown of its core functionalities:</p>    
            <h2>1. Easy Access Search Analytics API</h2>
            <p>With GSC InsightHub, you can easily access the Search Analytics API to:</p>
            <ul>
                <li>Generate performance reports on queries and pages.</li>
                <li>Analyze query distribution in search engine results pages (SERPs).</li>
            </ul>    
            <h2>2. On-Page SEO</h2>
            <p>Our tool helps you optimize on-page SEO by:</p>
            <ul>
                <li>Checking if the queries for which Google considers your webpage are present in the text.</li>
                <li>Identifying which topics you are covering and which ones you are missing.</li>
            </ul>    
            <h2>3. Keyword Grouper</h2>
            <p>Group and manage your keywords efficiently to enhance your SEO strategy.</p>    
            <h2>4. Bulk URL Inspection</h2>
            <p>Perform bulk URL inspections to ensure all your web pages meet SEO standards and are indexed properly.</p>
        </div>
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

        api_app = st.radio(
            "What do you feel like doing?",
            ["***GET INSIGHT FROM MY GSC DATA***", "***BULK INSPECT URLS***", "***INDEXING API***"],
            captions=["Laugh out loud.", "Get the popcorn.", "I'm feel lucky."],
            horizontal=True,
            index=["***GET INSIGHT FROM MY GSC DATA***", "***BULK INSPECT URLS***", "***INDEXING API***"].index(st.session_state.api_app)  # Mantiene il valore selezionato
        )
        # Aggiorna lo stato della sessione con la selezione corrente
        if api_app != st.session_state.api_app:
            st.session_state.api_app = api_app
            st.rerun() 
    st.divider()


    if st.session_state.api_app == "***GET INSIGHT FROM MY GSC DATA***":
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            # Opzioni per i tipi di dati
            options_type = {'Web': 'web', 'News': 'news', 'Discover': 'discover', 'Image': 'image', 'Video': 'video'}
            
            # Data di oggi
            today = datetime.now()
            
            # Funzione per calcolare la data di inizio in base all'opzione selezionata
            def get_start_date(option):
                if option == 'Last 7 days':
                    return today - timedelta(days=7)
                elif option == 'Last 28 days':
                    return today - timedelta(days=28)
                elif option == 'Last 3 months':
                    return today - timedelta(days=90)
                elif option == 'Last 6 months':
                    return today - timedelta(days=180)
                elif option == 'Last 12 months':
                    return today - timedelta(days=365)
                elif option == 'Last 16 months':
                    return today - timedelta(days=480)
                else:
                    return None
            
            # Selezione del tipo di canale
            selected_type = st.selectbox('CHANNEL', list(options_type.keys()))
            
            # Opzioni per il periodo di tempo
            time_options = ['Last 7 days', 'Last 28 days', 'Last 3 months', 'Last 6 months', 'Last 12 months', 'Last 16 months', 'Custom range',]
            selected_time_option = st.selectbox('Select range', time_options)
            
            if selected_time_option == 'Custom range':
                # Input per date personalizzate
                start_date = st.date_input('Start date', pd.to_datetime(today - timedelta(days=90)))
                end_date = st.date_input('End date', pd.to_datetime(today))
            else:
                # Calcolo delle date basato sull'opzione selezionata
                start_date = get_start_date(selected_time_option)
                end_date = today

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
                
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame()
                
                # Verifica se row_limit è impostato
                if row_limit is not None:
                    # Caso in cui row_limit è impostato: esegui una sola chiamata API
                    with st.spinner("Downloading data with row limit..."):
                        try:
                            rows = fetch_data_chunk(
                                webmasters_service,
                                st.session_state.selected_site,
                                start_date,
                                end_date,
                                dimensions,
                                st.session_state.dimension_filters,
                                selected_type,
                                0,           # Inizia da riga 0
                                row_limit    # Numero massimo di righe imposto dall'utente
                            )
                            
                            if not rows:
                                st.warning("No data retrieved for the selected period.")
                            else:
                                # Costruisci DataFrame con le righe ottenute
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
                                
                                status_text.text(f"Total rows downloaded: {len(rows)}")
                                st.session_state.data_loaded = True
                                st.session_state.download_ready = True
                                progress_bar.progress(1.0)
                
                        except HttpError as e:
                            st.warning(f"HTTP Error: {e}")
                
                else:
                    # Caso in cui row_limit non è impostato: utilizza la logica di download incrementale
                    total_days = (end_date - start_date).days + 1  # Include il giorno finale
                    completed_days = 0
                
                    with st.spinner("Downloading data without row limit..."):
                        try:
                            current_date = start_date
                            
                            while current_date <= end_date:
                                next_date = current_date + timedelta(days=1)
                                
                                daily_downloaded_rows = 0
                                start_row = 0
                                
                                while daily_downloaded_rows < 25000:
                                    rows = fetch_data_chunk(
                                        webmasters_service,
                                        st.session_state.selected_site,
                                        current_date,
                                        next_date,
                                        dimensions,
                                        st.session_state.dimension_filters,
                                        selected_type,
                                        start_row,
                                        25000
                                    )
                                    
                                    if not rows:
                                        st.warning(f"No data retrieved for {current_date}.")
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
                                    
                                    daily_downloaded_rows += len(rows)
                                    start_row += len(rows)
                                    status_text.text(f"Total rows downloaded: {len(rows)}")
                                    
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
                                



        def convert_df_to_zip(df, file_name="data.csv"):
            # Crea un buffer in memoria per il file ZIP
            buffer = io.BytesIO()
            
            # Crea un file ZIP
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Converti il DataFrame in CSV e scrivilo nel file ZIP
                csv_data = df.to_csv(index=False).encode('utf-8')
                zf.writestr(file_name, csv_data)
            
            # Spostati all'inizio del buffer
            buffer.seek(0)
            
            return buffer
        
        # Download CSV
        if st.session_state.data_loaded and st.session_state.download_ready:
            # Converti il DataFrame in un file ZIP compresso
            zip_buffer = convert_df_to_zip(st.session_state.df)
            
            # Fornisci il pulsante per scaricare il file ZIP
            st.download_button(label="Download data ZIP", 
                               data=zip_buffer, 
                               file_name="data.zip", 
                               mime="application/zip")

                

            with st.container(border=True):
                st.subheader("1. Queries Coverage Analysis")
                st.divider()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(
                        "This report checks if Google's considered queries are present in various webpage elements such as the title, meta description, headings, body content, and ALT tags. This helps you identify gaps by finding missing important keywords.")
                
                with col2:
                    if st.session_state.df is not None and 'Page' in st.session_state.df.columns and 'Query' in st.session_state.df.columns:
                        selected_page_on_page = st.selectbox("Select a page", st.session_state.df['Page'].unique())
                        scan_button = st.button("Analyze Page🤖", key='scan_button')
                    else:
                        st.warning("To use this feature, ensure that the dimensions contain the 'Page' and 'Query'.")
                        scan_button = False
                        
                
                if scan_button or st.session_state.get('scan_started', False):
                    if scan_button:
                        st.session_state.selected_tab = 3
                        st.session_state.scan_started = True
                    if selected_page_on_page != st.session_state.get('selected_page_on_page', None):
                        st.session_state.selected_page_on_page = selected_page_on_page
                        with st.spinner("Fetching page data..."):
                            st.session_state.page_data = fetch_page_data(selected_page_on_page)
                        st.session_state.keyword_analysis = None
                    
                    if 'page_data' in st.session_state and st.session_state.page_data is not None:
                        page_data = st.session_state.df[st.session_state.df['Page'] == st.session_state.selected_page_on_page]
                        page_data = page_data[['Query', 'Clicks', 'Impressions', 'CTR', 'Position']]
                        grouped_page_data = aggregate_queries(page_data)
                
                        # Analisi della copertura delle parole chiave
                        with st.container():
                            st.markdown(
                                f"<h4 style='text-align:center;'>📄 {st.session_state.page_data.get('meta_title', 'No Title')} | <a href='{selected_page_on_page}'>Go to the page</a></h4>",
                                unsafe_allow_html=True)
                            st.divider()
                            # Visualizzare i warning se presenti
                            if 'warnings' in st.session_state.page_data:
                                for warning in st.session_state.page_data['warnings']:
                                    st.warning(warning)
                
                            keyword_presence = analyze_keywords(st.session_state.page_data, grouped_page_data)
                            keyword_df = pd.DataFrame(keyword_presence)
                
                            # Controllo se 'Keyword' è presente nelle colonne
                            if 'Keyword' not in keyword_df.columns:
                                st.error("The DataFrame does not contain the required column 'Keyword'. Please check the data processing.")
                            else:
                                col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
                                
                                with col1:
                                    # Gestione dello stato del filtro di ricerca per query
                                    search_query = st.text_input(
                                        label="Filter queries containing",
                                        placeholder="Insert a query",
                                        value=st.session_state.get('search_query', '')
                                    )
                                    st.session_state.search_query = search_query
                                    if search_query:
                                        keyword_df = keyword_df[keyword_df['Keyword'].str.contains(search_query, case=False, na=False)]
                                    
                
                                with col2:
                                    # Gestione dello stato del filtro per query non presenti in nessun elemento
                                    show_not_covered = st.checkbox("Only not covered queries", st.session_state.get('show_not_covered', False))
                                    st.session_state.show_not_covered = show_not_covered
                
                                # Disabilitare le checkbox se "Only not covered queries" è attivo
                                is_disabled = st.session_state.show_not_covered
                
                                with col3:
                                    show_heading = st.checkbox("Show heading", st.session_state.get('show_heading', True), disabled=is_disabled)
                                with col4:
                                    show_keyword_metrics = st.checkbox("Show metrics", st.session_state.get('show_keyword_metrics', True), disabled=is_disabled)
                                with col5:
                                    show_meta = st.checkbox("Show meta", st.session_state.get('show_meta', True), disabled=is_disabled)
                                with col6:
                                    show_body_alt = st.checkbox("Show body alt", st.session_state.get('show_body_alt', True), disabled=is_disabled)
                
                                st.session_state.show_heading = show_heading
                                st.session_state.show_keyword_metrics = show_keyword_metrics
                                st.session_state.show_meta = show_meta
                                st.session_state.show_body_alt = show_body_alt
                
                                if st.session_state.show_not_covered:
                                    # Filtrare le query non coperte
                                    keyword_df = keyword_df[
                                        (keyword_df.get('Title', False) == False) &
                                        (keyword_df.get('Meta Description', False) == False) &
                                        (keyword_df.get('H1', False) == False) &
                                        (keyword_df.get('H2', False) == False) &
                                        (keyword_df.get('H3', False) == False) &
                                        (keyword_df.get('Body Content', False) == False) &
                                        (keyword_df.get('Alt Tags', False) == False)
                                    ]
                                    # Disabilitare le checkbox
                                    show_heading = False
                                    show_keyword_metrics = False
                                    show_meta = False
                                    show_body_alt = False
                                else:
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

            
            with st.container(border=True):
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
            
    
