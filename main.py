import streamlit as st
import httplib2
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import altair as alt
from collections import Counter
import itertools

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

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
    rows = response_data.get('rows', [])
    return rows

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
                        
                        if not rows:
                            st.warning("No data retrieved from API.")
                            break

                        # Log the rows for debugging
                        st.write("Fetched Rows: ", rows)

                        data_list = []
                        for row in rows:
                            data_entry = {dimension: row['keys'][dimensions.index(dimension.upper())] for dimension in dimensions}
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
                    progress_bar.progress(100)

                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(st.session_state.df)
                st.download_button(label="Download data CSV", data=csv, file_name='data.csv', mime='text/csv')
                
if st.session_state.data_loaded:
    df = st.session_state.df
