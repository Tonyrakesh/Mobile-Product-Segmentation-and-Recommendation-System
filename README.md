Mobile Product Segmentation & Recommendation System
An end-to-end Machine Learning pipeline and Streamlit web application that analyzes 50,000 customer reviews across 22 smartphones to segment products and users and provide similarity-based recommendations.

Features
Data Cleaning & Imputation: Reconstructed missing prices via local exchange rates and imputed missing overall ratings using sub-aspect averages.

Exploratory Data Analysis: Identified brand pricing tiers and analyzed feature correlations across 7 brands.

Unsupervised Segmentation: Grouped products and consumers into 4 distinct segments using K-Means and PCA.

Cosine Recommendation Engine: Suggests the top 4 alternative smartphones based on normalized hardware specifications, ratings, and price.

Interactive Streamlit Dashboard: Complete web UI featuring dynamic Plotly charts and model recommendation cards.

Project Architecture
Preprocessing: Handled missing values, encoded categorical features (brand, model, country), and scaled numeric columns.

EDA: Discovered near-zero linear correlation (r ≈ 0.00) between device price and star rating.

Clustering (k=4):

Cluster 0: Highly Satisfied Advocates (High ratings, high specs)

Cluster 1: Budget Value Seekers (Lower price bracket, balanced specs)

Cluster 2: Critical Dissatisfied Users (Low satisfaction across tiers)

Cluster 3: Premium Flagship Buyers (High price bracket, premium build)

Recommender: Normalized multi-attribute feature vectors using MinMaxScaler and ranked items via cosine_similarity.

Tech Stack
Languages & Frameworks: Python, Streamlit

ML & Data Processing: Scikit-Learn, Pandas, NumPy

Visualizations: Plotly, Seaborn, Matplotlib
