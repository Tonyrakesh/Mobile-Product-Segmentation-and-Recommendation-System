import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# PAGE SETUP

st.set_page_config(
    page_title="Mobile Segmentation",
    layout="wide"
)

# DATA CLEANING & PREPROCESSING

@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("Mobile Reviews Sentiment null.csv")
    df = df.drop_duplicates()

    # Clean local price and impute price_usd
    def clean_local(val):
        if pd.isnull(val): return np.nan
        cleaned = ''.join(c for c in str(val) if c.isdigit() or c == '.')
        try: return float(cleaned)
        except: return np.nan

    df['price_local_numeric'] = df['price_local'].apply(clean_local)
    missing_usd = df['price_usd'].isnull() & df['price_local_numeric'].notnull()
    df.loc[missing_usd, 'price_usd'] = df.loc[missing_usd, 'price_local_numeric'] / df.loc[missing_usd, 'exchange_rate_to_usd']
    df['price_usd'] = df.groupby('model')['price_usd'].transform(lambda x: x.fillna(x.median()))

    # Aspect ratings and sentiment imputation
    aspects = ['battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating']
    df['rating'] = df['rating'].fillna(df[aspects].mean(axis=1).round(1))
    sentiment_fill = df['rating'].apply(lambda r: 'Positive' if r >= 4.0 else ('Neutral' if r == 3.0 else 'Negative'))
    df['sentiment'] = df['sentiment'].fillna(sentiment_fill)
    df['source'] = df['source'].fillna(df['source'].mode()[0])
    df['engagement_score'] = np.log1p(df['helpful_votes'])

    # Aggregate Product Catalog
    catalog = df.groupby(['brand', 'model']).agg(
        median_price=('price_usd', 'median'),
        mean_rating=('rating', 'mean'),
        battery=('battery_life_rating', 'mean'),
        camera=('camera_rating', 'mean'),
        performance=('performance_rating', 'mean'),
        design=('design_rating', 'mean'),
        display=('display_rating', 'mean'),
        review_count=('rating', 'count')
    ).reset_index()

    # K-Means Clustering (k=4)
    cluster_features = ['price_usd', 'rating'] + aspects + ['helpful_votes']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    seg_names = {
        0: "Highly Satisfied Advocates",
        1: "Budget Value Seekers",
        2: "Critical Dissatisfied Users",
        3: "Premium Flagship Buyers"
    }
    df['segment_name'] = df['cluster'].map(seg_names)

    # 2D PCA for visual projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]

    # Cosine Similarity Matrix for Catalog
    sim_features = ['median_price', 'mean_rating', 'battery', 'camera', 'performance', 'design', 'display']
    minmax = MinMaxScaler()
    norm_specs = minmax.fit_transform(catalog[sim_features])
    cos_sim = cosine_similarity(norm_specs)
    sim_df = pd.DataFrame(cos_sim, index=catalog['model'], columns=catalog['model'])

    return df, catalog, sim_df

df, catalog, sim_df = load_and_preprocess_data()

# SIDEBAR NAVIGATION

st.sidebar.title("Navigate Pipeline")
#st.sidebar.markdown("Mobile Segmentation & Recommendation System")
page = st.sidebar.radio("Select a Page", [
    "Overview & Data Explorer",
    "Exploratory Data Analysis (EDA)",
    "Product & User Clustering",
    "Recommendation Engine"
])

# TAB 1: OVERVIEW & DATA

if page == "Overview & Data Explorer":
    st.title("Dataset Overview & Data Cleaning")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(df):,}")
    c2.metric("Phone Models", f"{len(catalog)}")
    c3.metric("Global Brands", f"{df['brand'].nunique()}")
    c4.metric("Avg Rating", f"{df['rating'].mean():.2f} / 5.0")
    
    #st.subheader("Aggregated Smartphone Catalog")
    st.dataframe(catalog.style.format({
        'median_price': '${:.2f}',
        'mean_rating': '{:.2f}',
        'battery': '{:.2f}',
        'camera': '{:.2f}',
        'performance': '{:.2f}'
    }))

# TAB 2: EXPLORATORY DATA ANALYSIS

elif page == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis")

    import plotly.express as px

    aspect_cols = [
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating",
    ]

    # Row 1: Two interactive charts side by side
    col1, col2 = st.columns(2)

    with col1:
        # 1. Interactive Box Plot (Price by Brand)
        fig1 = px.box(
            df,
            x="brand",
            y="price_usd",
            color="brand",
            title="Price Distribution by Brand (USD)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # 2. Interactive Bar Chart (Average Specs across Brands)
        brand_specs = (
            df.groupby("brand")[aspect_cols]
            .mean()
            .reset_index()
            .melt(
                id_vars="brand",
                var_name="Specification",
                value_name="Rating",
            )
        )
        fig2 = px.bar(
            brand_specs,
            x="brand",
            y="Rating",
            color="Specification",
            barmode="group",
            title="Specification Ratings across Brands",
        )
        fig2.update_yaxes(range=[2.5, 3.0])
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Two interactive charts side by side
    col3, col4 = st.columns(2)

    with col3:
        # 3. Interactive Correlation Heatmap
        corr_cols = ["price_usd", "rating"] + aspect_cols + ["helpful_votes"]
        corr = df[corr_cols].corr()
        fig3 = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # 4. Interactive Histogram/Bar (Ratings by Sentiment)
        fig4 = px.histogram(
            df,
            x="rating",
            color="sentiment",
            barmode="group",
            title="Rating Counts by Sentiment Class",
        )
        st.plotly_chart(fig4, use_container_width=True)

# TAB 3: CLUSTERING (SEGMENTATION)

elif page == "Product & User Clustering":
    st.title("Customer & Product Segmentation")
    fig_pca = px.scatter(
        df.sample(2500, random_state=42),
        x="pca_1", y="pca_2",
        color="segment_name",
        title="2D PCA Space Segmentation Projection",
        opacity=0.7
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# TAB 4: RECOMMENDATION ENGINE

elif page == "Recommendation Engine":
    st.title("Recommendation Engine")
    anchor_phone = st.selectbox("Select Target Phone:", catalog['model'].unique())
    top_n = st.slider("Number of Recommendations:", 2, 6, 4)
    
    # Recommendation logic
    sim_scores = sim_df[anchor_phone].drop(anchor_phone).sort_values(ascending=False).head(top_n)
    rec_results = sim_scores.reset_index()
    rec_results.columns = ['model', 'similarity_score']
    rec_results = rec_results.merge(catalog, on='model')
    
    st.subheader(f"Top Alternatives for {anchor_phone}")
    cols = st.columns(top_n)
    for i, row in rec_results.iterrows():
        with cols[i]:
            st.info(f"**{row['model']}** ({row['brand']})")
            st.metric("Similarity", f"{row['similarity_score']*100:.1f}%")
            st.write(f"▫️**Price:** ${row['median_price']:.2f}")
            st.write(f"▫️**Rating:** {row['mean_rating']:.2f}")
            st.write(f"▫️**Battery:** {row['battery']:.2f}")
            st.write(f"▫️**Camera:** {row['camera']:.2f}")
