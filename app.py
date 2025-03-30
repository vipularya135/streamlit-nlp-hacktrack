import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from rank_bm25 import BM25Okapi

# Database path
db_path = "example.db"

# Connect to SQLite database
def get_data():
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data_table"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Convert date string to datetime for filtering
def filter_upcoming_hackathons(df):
    today = datetime.today().strftime('%Y-%m-%d')
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    return df[df['start_date'] >= today]

# Streamlit UI
st.set_page_config(page_title="HackTrack", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        h1.title {
            text-align: center;
            font-size: 48px;
            color: red;
            font-weight: bold;
        }
        .hackathon-card {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .styled-table {
            margin-top: 20px;
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
        .styled-table th {
            background: #1E90FF;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
            border-bottom: 2px solid #ddd;
        }
        .styled-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .styled-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .styled-table tr:hover {
            background-color: #e9f5ff;
        }
        .filter-button {
            background: linear-gradient(to right, #4facfe, #00f2fe);
            border: none;
            color: white;
            padding: 12px;
            font-size: 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: 0.3s;
            width: 100%;
            text-align: center;
            margin-top: 10px;
        }
        .filter-button:hover {
            background: linear-gradient(to right, #00c6fb, #005bea);
        }
                    .reset-button {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            border: none;
            color: white;
            padding: 12px;
            font-size: 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: 0.3s;
            width: 100%;
            text-align: center;
            margin-top: 10px;
        }
        .reset-button:hover {
            background: linear-gradient(to right, #ff5733, #ff2e63);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🚀 HackTrack: Discover Hackathons</p>', unsafe_allow_html=True)

# Load data
df = get_data()
df_upcoming = filter_upcoming_hackathons(df)

# Display first 5 upcoming hackathons
st.write("### 🎯 Upcoming Hackathons")
df_upcoming_sorted = df_upcoming.sort_values(by='start_date', ascending=True).head(5)

for _, row in df_upcoming_sorted.iterrows():
    st.markdown(f"""
        <div class="hackathon-card">
            <h2>🚀 {row['title']}</h2>
            <p><strong>📅 Date:</strong> {row['start_date'].strftime('%Y-%m-%d')}</p>
            <p><strong>📍 Location:</strong> {row['location'] if 'location' in df.columns else 'N/A'}</p>
            <p><strong>👥 Registrations:</strong> {row['register_count']}</p>
            <p><strong>📜 Description:</strong> {row['details']}</p>
            <a href="{row['url']}" target="_blank" style="color: yellow; font-weight: bold;">🔗 More Details</a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

today_date = pd.Timestamp.now(tz='UTC').normalize()
df['end_date'] = pd.to_datetime(df['end_date']).dt.tz_convert('UTC').dt.normalize()
df1 = df[df['end_date'] >= today_date].sort_values(by='start_date', ascending=True)

# Sidebar Filters
st.sidebar.markdown("### 📊 Search Settings")
search_method = st.sidebar.selectbox("Select Search Method", ["TF-IDF", "BM25", "Cosine Similarity"])
search_query = st.sidebar.text_input("🔍 Search Hackathons")

def search_hackathons(query, df, method):
    if query.strip() == "":
        return df  # Return full dataframe if search is empty
    
    descriptions = df['details'].fillna("").tolist()
    
    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    elif method == "BM25":
        tokenized_docs = [doc.split() for doc in descriptions]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = query.split()
        similarities = bm25.get_scores(query_tokens)
    
    elif method == "Cosine Similarity":
        vectorizer = CountVectorizer(stop_words='english')
        count_matrix = vectorizer.fit_transform(descriptions)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, count_matrix).flatten()
    
    df = df.copy()
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False).head(15)

filtered_df = search_hackathons(search_query, df, search_method)

# Most Popular Hackathons
if st.sidebar.button("🔥 Most Popular", key="popular", help="Show hackathons with the highest registrations"):
    filtered_df = filtered_df.sort_values(by="register_count", ascending=False)

# Sort by Start Date
if st.sidebar.button("📅 Sort by Earliest Date", key="date_asc", help="Show hackathons starting soonest"):
    filtered_df = filtered_df.sort_values(by="start_date", ascending=True)

# Sort by Latest Date
if st.sidebar.button("⏳ Sort by Latest Date", key="date_desc", help="Show hackathons starting latest"):
    filtered_df = filtered_df.sort_values(by="start_date", ascending=False)

if st.sidebar.button("🔄 Reset Filters", key="reset"):
    search_query = ""
    filtered_df = df  # Reset to original data

# Refresh View with Filters
st.write("### 🎟 All Hackathons")
st.dataframe(filtered_df, use_container_width=True)

# Add Presenters' Names to Sidebar with Styling
st.sidebar.markdown("""
    <style>
        .presenter-card {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
        }
    </style>
    <div class="presenter-card">
        👨‍💻 Presented by:<br>
        <b>Vipul Arya & Suhas HM</b>
    </div>
""", unsafe_allow_html=True)

