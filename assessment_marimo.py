import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LDS7004M — Data Visualisation Assessment
    #### London Airbnb Market: A Spatial and Economic Analysis
    """)
    return


@app.cell
def _():
    # DEPENDENCIES
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import warnings
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import geopandas as gpd

    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore")
    return (
        RandomForestRegressor,
        cross_val_score,
        go,
        gpd,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        px,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(sns):
    # Global plot style
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    ACCENT  = "#E8474C" # primary highlight / superhost colour
    NEUTRAL = "#2C3E50" # dark neutral
    MAPBOX  = "carto-positron"

    # Unified 8-slot categorical palette shared across ALL figures
    PALETTE  = [
        "#E8474C", "#2C3E50",
        "#457B9D", "#2A9D8F", 
        "#E9C46A", "#F4A261", 
        "#9B59B6", "#264653"
        ]

    CAT_CMAP = "YlOrRd" # sequential — choropleths, ranked bars
    DIV_CMAP = "RdBu_r" # diverging — correlation matrices
    return ACCENT, CAT_CMAP, DIV_CMAP, MAPBOX, NEUTRAL, PALETTE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SECTION 1 — LOAD DATA
    """)
    return


@app.cell
def _(gpd, pd):
    # Load raw data
    listings_raw = pd.read_csv("data/listings.csv", low_memory=False)
    geo = gpd.read_file("data/neighbourhoods.geojson")

    print(f"Listings shape: {listings_raw.shape}")
    print(f"GeoJSON shape: {geo.shape}")
    return geo, listings_raw


@app.cell
def _(listings_raw):
    # Display the first few rows of listings
    listings_raw.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SECTION 2 — DATA CLEANING & FEATURE ENGINEERING
    """)
    return


@app.cell
def _(listings_raw):
    listings_raw.columns
    return


@app.cell
def _():
    COLS = [
        'id', 'name', 'host_id', 'host_name', 'host_since', 'host_location', 'host_response_time', 'host_is_superhost',
        'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count', 
        'latitude','longitude', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights', 
        'maximum_nights', 'has_availability', 'number_of_reviews', 'review_scores_rating', 'number_of_reviews_l30d',
        'estimated_occupancy_l365d', 'estimated_revenue_l365d', 'instant_bookable', 'review_scores_cleanliness', 
        'review_scores_location', 'review_scores_value', 'availability_365', 'calculated_host_listings_count', 'reviews_per_month'
        ]
    return (COLS,)


@app.cell
def _(COLS, listings_raw):
    df = listings_raw[COLS].copy()
    print(f"Working dataframe shape: {df.shape}")
    return (df,)


@app.cell
def _(df, listings_raw):
    # List of columns with missing values
    (df.isnull().sum()/len(listings_raw) * 100).sort_values(ascending=False)
    return


@app.cell
def _(df):
    # Drop the rows with missing price values
    df_1 = df.dropna(subset=['price'])
    return (df_1,)


@app.cell
def _(df_1, listings_raw):
    # List of columns with missing values
    (df_1.isnull().sum() / len(listings_raw) * 100).sort_values(ascending=False)
    return


@app.cell
def _(df_1, np):
    for _col in ['host_response_rate', 'host_acceptance_rate']:
        df_1[_col] = df_1[_col].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float) / 100.0
    return


@app.cell
def _(df_1, np):
    # Clean the price column
    df_1['price'] = df_1['price'].astype(str).str.replace('[\\$,]', '', regex=True).str.strip().replace('nan', np.nan).astype(float)
    return


@app.cell
def _(df_1):
    # View price column
    df_1['price'].describe()  # Dollars
    return


@app.cell
def _(df_1, pd):
    # View price column without scientific (exponential) notation
    pd.option_context('display.float_format', '{:.2f}'.format)
    df_1['price'].describe()  # Dollars
    return


@app.cell
def _(df_1):
    df_1['price'].value_counts()
    return


@app.cell
def _(df_1):
    df_1['price'].mode()
    return


@app.cell
def _(df_1, listings_raw):
    # List of columns with missing values
    (df_1.isnull().sum() / len(listings_raw) * 100).sort_values(ascending=False)
    return


@app.cell
def _(df_1):
    # View review scores rating column
    df_1.review_scores_rating.describe()
    return


@app.cell
def _(df_1):
    # View host response time column
    df_1.host_response_time.describe()
    return


@app.cell
def _(df_1):
    # View host response rate column
    df_1.host_response_rate.describe()
    return


@app.cell
def _(df_1):
    # View host acceptance rate column
    df_1.host_acceptance_rate.describe()
    return


@app.cell
def _(df_1):
    len(df_1.columns)
    return


@app.cell
def _(df_1):
    # Identify categorical and numerical columns
    categorical_cols = df_1.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df_1.select_dtypes(include=['number']).columns
    return categorical_cols, numerical_cols


@app.cell
def _(categorical_cols):
    categorical_cols
    return


@app.cell
def _(numerical_cols):
    numerical_cols
    return


@app.cell
def _(categorical_cols, df_1, numerical_cols):
    for _col in categorical_cols:
        if df_1[_col].isnull().any():
            mode = df_1[_col].mode(dropna=True)
            if not mode.empty:
                df_1[_col].fillna(mode[0], inplace=True)
    for _col in numerical_cols:
        if df_1[_col].isnull().any():
            mean = df_1[_col].mean()
            df_1[_col].fillna(mean, inplace=True)
    return


@app.cell
def _(df_1, listings_raw):
    # List of columns with missing values
    (df_1.isnull().sum() / len(listings_raw) * 100).sort_values(ascending=False)
    return


@app.function
def format_price(x, pos):
    if x == int(x):
        return f"{int(x):,}"
    else:
        return f"{x:,}"


@app.cell
def _(df_1, mo, plt):
    plt.figure(figsize=(10, 5))
    _n, _bins, _patches = plt.hist(df_1['price'], color='#457B9D')
    plt.title('Distribution of Price')
    plt.xlabel('Price ($)')
    plt.ylabel('Number of Listings')
    plt.grid(axis='y', alpha=0.3)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_1, mo, plt):
    plt.figure(figsize=(10, 5))
    _n, _bins, _patches = plt.hist(df_1['price'], color='#457B9D')
    plt.title('Distribution of Price')
    plt.xlabel('Price ($)')
    plt.ylabel('Number of Listings')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, df_1['price'].quantile(0.15))
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_1, mo, plt, sns):
    # Boxplot for price distribution
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_1['price'], color='#457B9D')
    plt.title('Boxplot of Price Distribution')
    plt.xlabel('Price ($)')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_1):
    Q1, Q3 = df_1['price'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = 5  # logical minimum — no credible listing is free
    upper = Q3 + 3 * IQR  # using ``3 * IQR`` to remove outliers
    before = len(df_1)
    df_2 = df_1[(df_1['price'] >= lower) & (df_1['price'] <= upper)]
    after = len(df_2)
    return after, before, df_2, lower, upper


@app.cell
def _(after, before, lower, upper):
    print(f"Outlier removal: {before - after} rows dropped "
          f"({(before-after)/before*100:.1f}%)")
    print(f"Price range retained: ${lower:.0f} – ${upper:.0f}")
    return


@app.cell
def _(df_2, mo, plt):
    plt.figure(figsize=(10, 5))
    _n, _bins, _patches = plt.hist(df_2['price'], bins=50, color='#457B9D')
    plt.title('Distribution of Price')
    plt.xlabel('Price ($)')
    plt.ylabel('Number of Listings')
    plt.grid(axis='y', alpha=0.3)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_2, mo, plt, sns):
    # Violin plot for price distribution
    plt.figure(figsize=(10, 5))
    sns.violinplot(x=df_2['price'], color='#457B9D')
    plt.title('Violin Plot of Price Distribution')
    plt.xlabel('Price ($)')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_2, mo, plt, sns):
    # Boxplot for price distribution
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_2['price'], color='#457B9D')
    plt.title('Boxplot of Price Distribution')
    plt.xlabel('Price ($)')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(format_price))
    plt.xticks(rotation=45)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(df_2):
    bool_cols = ['has_availability', 'instant_bookable']
    for _col in bool_cols:
        df_2[_col] = df_2[_col].map({'t': 1, 'f': 0, True: 1, False: 0})
    return


@app.cell
def _(df_2, np, pd):
    # Feature Engineering
    df_2['price_per_person'] = df_2['price'] / df_2['accommodates'].replace(0, np.nan)
    # Price per person (accommodation efficiency metric)
    df_2['host_country'] = df_2['host_location'].astype(str).str.split(',').str[-1].str.strip()
    reference_date = pd.Timestamp('2026-03-26')
    # Feature: Host country (extracted from host_location)
    df_2['host_since'] = pd.to_datetime(df_2['host_since'], errors='coerce')
    # Feature: Host age (years since host_since as of March 26, 2026)
    df_2['host_age'] = round((reference_date - df_2['host_since']).dt.days / 365.25)
    return


@app.cell
def _(df_2):
    df_2.columns
    return


@app.cell
def _(df_2, gpd):
    # Create a GeoDataFrame from the DataFrame for spatial analysis
    gdf = gpd.GeoDataFrame(df_2, geometry=gpd.points_from_xy(df_2.longitude, df_2.latitude), crs='EPSG:4326')
    return (gdf,)


@app.cell
def _(gdf):
    len(gdf.columns)
    return


@app.cell
def _(gdf, geo, gpd):
    # Join the listing GeoDataFrame with the neighbourhood GeoJSON data
    gdf_1 = gpd.sjoin(gdf, geo, how='left', predicate='within')
    return (gdf_1,)


@app.cell
def _(gdf_1):
    # Final cleaned dataset summary
    print(f'{gdf_1.shape[0]:,} rows × {gdf_1.shape[1]} columns')
    gdf_1.to_csv('data/listings_clean.csv', index=False)
    return


@app.cell
def _(gdf_1):
    gdf_1.neighbourhood.nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SECTION 3 — EXPLORATORY DATA ANALYSIS & VISUALISATIONS
    """)
    return


@app.cell
def _(ACCENT, NEUTRAL, PALETTE, gdf_1, mo, np, plt, sns):
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig.suptitle('Distribution of Airbnb Nightly Prices in London', fontsize=14, y=1.02)
    sns.histplot(gdf_1['price'], bins=100, kde=True, color=ACCENT, edgecolor='white', linewidth=0.4, ax=axes[0])
    axes[0].axvline(gdf_1['price'].median(), color=NEUTRAL, linestyle='--', lw=1.5, label=f"Median: ${gdf_1['price'].median():.0f}")
    axes[0].axvline(gdf_1['price'].mean(), color=PALETTE[2], linestyle='--', lw=1.5, label=f"Mean: ${gdf_1['price'].mean():.0f}")
    axes[0].set_title('Linear Scale')
    axes[0].set_xlabel('Nightly Price ($)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    sns.histplot(np.log1p(gdf_1['price']), bins=100, kde=True, color=PALETTE[2], edgecolor='white', linewidth=0.4, ax=axes[1])
    axes[1].set_title('Log-Transformed')
    axes[1].set_xlabel('log(1 + Nightly Price)')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(gdf_1):
    print('Skewness:', round(gdf_1['price'].skew(), 3))
    print('Kurtosis:', round(gdf_1['price'].kurt(), 3))
    return


@app.cell
def _(CAT_CMAP, MAPBOX, gdf_1, geo, mo, px):
    borough_stats = gdf_1.groupby('neighbourhood').agg(median_price=('price', 'median'), average_price=('price', 'mean'), listing_count=('id', 'count'), average_price_per_person=('price_per_person', 'mean'), instant_bookable_rate=('instant_bookable', 'mean'), average_rating=('review_scores_rating', 'mean'), average_revenue=('estimated_revenue_l365d', 'mean')).reset_index()
    _geo_merged = geo.merge(borough_stats, on='neighbourhood', how='left')
    fig2 = px.choropleth_mapbox(_geo_merged, geojson=_geo_merged.geometry.__geo_interface__, locations=_geo_merged.index, color='average_price', hover_name='neighbourhood', hover_data={'listing_count': True, 'average_price': ':.0f', 'average_rating': ':.2f', 'average_price_per_person': ':.0f', 'instant_bookable_rate': ':.2f'}, color_continuous_scale=CAT_CMAP, mapbox_style=MAPBOX, center={'lat': gdf_1.latitude.median(), 'lon': gdf_1.longitude.median()}, zoom=9, opacity=0.75, title='Average Airbnb Nightly Price by London Borough ($)')
    fig2.update_layout(margin={'r': 0, 't': 40, 'l': 0, 'b': 0}, coloraxis_colorbar_title='Average Price ($)/night')
    fig2.write_html('interactive_map_for_average_price_in_london_airbnb.html')
    mo.ui.plotly(fig2)
    return (fig2,)


@app.cell
def _(CAT_CMAP, MAPBOX, gdf_1, geo, mo, px):
    borough_stats_1 = gdf_1.groupby('neighbourhood').agg(median_price=('price', 'median'), average_price=('price', 'mean'), listing_count=('id', 'count'), average_price_per_person=('price_per_person', 'mean'), instant_bookable_rate=('instant_bookable', 'mean'), average_rating=('review_scores_rating', 'mean'), average_revenue=('estimated_revenue_l365d', 'mean')).reset_index()
    _geo_merged = geo.merge(borough_stats_1, on='neighbourhood', how='left')
    fig2_1 = px.choropleth_mapbox(_geo_merged, geojson=_geo_merged.geometry.__geo_interface__, locations=_geo_merged.index, color='average_rating', hover_name='neighbourhood', hover_data={'listing_count': True, 'average_price': ':.0f', 'average_rating': ':.2f', 'average_price_per_person': ':.0f', 'instant_bookable_rate': ':.2f'}, color_continuous_scale=CAT_CMAP, mapbox_style=MAPBOX, center={'lat': gdf_1.latitude.median(), 'lon': gdf_1.longitude.median()}, zoom=9, opacity=0.75, title='Average Airbnb Reviews by London Borough')
    fig2_1.update_layout(margin={'r': 0, 't': 40, 'l': 0, 'b': 0}, coloraxis_colorbar_title='Average Rating')
    fig2_1.write_html('interactive_map_for_average_reviews_in_london_airbnb.html')
    mo.ui.plotly(fig2_1)
    return (borough_stats_1,)


@app.cell
def _(PALETTE, borough_stats_1, mo, plt):
    top5 = borough_stats_1.nlargest(5, 'average_price')
    plt.figure(figsize=(7, 5))
    plt.bar(top5['neighbourhood'], top5['average_price'], color=PALETTE[2])
    plt.title('Top 5 Most Expensive London Boroughs Airbnbs', fontsize=12)
    plt.xlabel('Borough', fontsize=10)
    plt.ylabel('Average Price ($)', fontsize=10)
    for _i, _v in enumerate(top5['average_price']):
        plt.text(_i, _v + 1, f'{_v:.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.xticks(rotation=75)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(PALETTE, borough_stats_1, mo, plt):
    bottom5 = borough_stats_1.nsmallest(5, 'average_price')
    plt.figure(figsize=(7, 5))
    plt.bar(bottom5['neighbourhood'], bottom5['average_price'], color=PALETTE[2])
    plt.title('Top 5 Most Affordable London Boroughs Airbnbs', fontsize=12)
    plt.xlabel('Borough', fontsize=10)
    plt.ylabel('Average Price ($)', fontsize=10)
    for _i, _v in enumerate(bottom5['average_price']):
        plt.text(_i, _v + 1, f'{_v:.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.xticks(rotation=75)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(NEUTRAL, PALETTE, gdf_1, mo, plt, sns):
    _room_order = gdf_1.groupby('room_type')['price'].median().sort_values(ascending=False).index.tolist()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=gdf_1, x='room_type', y='price', order=_room_order, color=PALETTE[2], flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': NEUTRAL, 'alpha': 0.3}, ax=_ax)
    for _i, _rt in enumerate(_room_order):
        _subset = gdf_1[gdf_1['room_type'] == _rt]['price']
        _ax.text(_i, _subset.median() + 0.5, f'{len(_subset):,} listings\nM = {_subset.median():.0f}', ha='center', va='bottom', fontsize=9.5, color='white')
    _ax.set_title('Nightly Price Distribution by Room Type', fontsize=12)
    _ax.set_xlabel('Room Type', fontsize=10)
    _ax.set_ylabel('Nightly Price ($)', fontsize=10)
    _ax.set_ylim(0, gdf_1['price'].quantile(0.9))
    plt.tight_layout()
    plt.xticks(rotation=75)
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(NEUTRAL, PALETTE, gdf_1, mo, plt, sns):
    _room_order = gdf_1.groupby('room_type')['price'].median().sort_values(ascending=False).index.tolist()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=gdf_1, x='room_type', y='price', order=_room_order, color=PALETTE[2], flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': NEUTRAL, 'alpha': 0.3}, ax=_ax)
    for _i, _rt in enumerate(_room_order):
        _subset = gdf_1[gdf_1['room_type'] == _rt]['price']
        _ax.text(_i, _subset.median() + 0.5, f'{len(_subset):,} listings\nM = {_subset.median():.0f}', ha='center', va='bottom', fontsize=9.5, color='black')
    _ax.set_title('Nightly Price Distribution by Room Type', fontsize=12)
    _ax.set_xlabel('Room Type', fontsize=10)
    _ax.set_ylabel('Nightly Price ($)', fontsize=10)
    plt.tight_layout()
    plt.xticks(rotation=75)
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(DIV_CMAP, gdf_1, mo, np, plt, sns):
    heatmap_cols = ['price', 'accommodates', 'bedrooms', 'beds', 'bathrooms', 'index_right', 'maximum_nights', 'minimum_nights', 'number_of_reviews', 'review_scores_rating', 'instant_bookable']
    corr_matrix = gdf_1[heatmap_cols].corr()
    corr_matrix.rename(index={'index_right': 'neighborhood'}, columns={'index_right': 'neighborhood'}, inplace=True)
    _fig, _ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=DIV_CMAP, center=0, linewidths=0.5, annot_kws={'size': 10}, ax=_ax, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    _ax.set_title('Correlation Matrix of Listing Features', fontsize=12)
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return (corr_matrix,)


@app.cell
def _(PALETTE, corr_matrix, mo, plt):
    # Plot top correlations with price as a bar chart
    top_corr = corr_matrix['price'].drop('price').abs().sort_values(ascending=False).head(8).round(3)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    top_corr.plot(kind='bar', ax=_ax, color=PALETTE[2])
    _ax.set_ylabel('Absolute Pearson r', fontsize=10)
    _ax.set_xlabel('Feature', fontsize=11)
    _ax.set_title('Top 8 Features Most Correlated with Price', fontsize=12)
    _ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(CAT_CMAP, borough_stats_1, mo, plt):
    top20 = borough_stats_1.nlargest(20, 'listing_count').sort_values('listing_count')
    _fig, _ax = plt.subplots(figsize=(12, 8))
    _bars = _ax.barh(top20['neighbourhood'], top20['listing_count'], color=plt.cm.YlOrRd(top20['median_price'] / top20['median_price'].max()), edgecolor='white', linewidth=0.4)
    for _bar in _bars:
        _ax.text(_bar.get_width() + 11, _bar.get_y() + _bar.get_height() / 2, f'{_bar.get_width():,.0f}', va='center', fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=CAT_CMAP, norm=plt.Normalize(top20['median_price'].min(), top20['median_price'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=_ax, shrink=0.6)
    cbar.set_label('Median Nightly Price ($)')
    _ax.set_title('Top 20 London Boroughs by Airbnb Listing Count', fontsize=13)
    _ax.set_xlabel('Number of Active Listings')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### HOST ANALYSIS
    > Composition, origins, tenure, responsiveness and spatial distribution of London Airbnb hosts.
    """)
    return


@app.cell
def _(NEUTRAL, PALETTE, gdf_1, mo, plt):
    superhost_counts = gdf_1['host_is_superhost'].map({'t': 'Superhost', 'f': 'Regular Host'}).value_counts()
    _labels = superhost_counts.index.tolist()
    sizes = superhost_counts.values
    _fig, _ax = plt.subplots(figsize=(7, 7))
    _wedges, _texts, _autotexts = _ax.pie(sizes, labels=_labels, autopct='%1.1f%%', colors=[PALETTE[2], PALETTE[0]], startangle=90, pctdistance=0.78, wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2.5})
    for _at in _autotexts:
        _at.set(fontsize=12, color='white')
    for _t in _texts:
        _t.set(fontsize=13)
    _ax.text(0, 0, f'n = {sizes.sum():,}\nhosts', ha='center', va='center', fontsize=12, color=NEUTRAL)
    _ax.set_title('Superhost vs Regular Host Composition', fontsize=13, pad=20)
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(PALETTE, gdf_1, mo, plt):
    rt_order = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
    rt_counts = gdf_1['host_response_time'].value_counts().reindex(rt_order).dropna()
    _fig, _ax = plt.subplots(figsize=(8, 7))
    _wedges, _texts, _autotexts = _ax.pie(rt_counts.values, labels=[_t.title() for _t in rt_counts.index], autopct='%1.1f%%', colors=PALETTE[:len(rt_counts)], startangle=140, pctdistance=0.82, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for _at in _autotexts:
        _at.set(fontsize=10, color='white')
    for _t in _texts:
        _t.set(fontsize=10)
    _ax.set_title('Host Response Time Distribution', fontsize=13, pad=20)
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(NEUTRAL, PALETTE, gdf_1, mo, plt, sns):
    tenure = gdf_1['host_age'].dropna()
    _fig, _ax = plt.subplots(figsize=(12, 5))
    sns.histplot(tenure, kde=True, color=PALETTE[2], edgecolor='white', linewidth=0.4, alpha=0.75, ax=_ax)
    _ax.axvline(tenure.median(), color=NEUTRAL, linestyle='--', lw=1.8, label=f'Median: {tenure.median():.1f} yrs')
    _ax.axvline(tenure.mean(), color=PALETTE[2], linestyle='--', lw=1.8, label=f'Mean: {tenure.mean():.1f} yrs')
    _ax.set_title('Distribution of Host Tenure (Years on Platform)', fontsize=13)
    _ax.set_xlabel('Years Since Host Registration')
    _ax.set_ylabel('Number of Listings')
    _ax.legend(fontsize=10)
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(PALETTE, gdf_1, mo, plt):
    _uk_labels = {'United Kingdom', 'England', 'London', 'Uk', 'Scotland', 'Wales'}
    _top_locs = gdf_1['host_country'].value_counts().head(20).sort_values()
    _fig, _ax = plt.subplots(figsize=(12, 8))
    _bar_colors = [PALETTE[0] if loc in _uk_labels else PALETTE[2] for loc in _top_locs.index]
    _bars = _ax.barh(_top_locs.index, _top_locs.values, color=_bar_colors, edgecolor='white', linewidth=0.4)
    for _bar in _bars:
        _ax.text(_bar.get_width() + 20, _bar.get_y() + _bar.get_height() / 2, f'{_bar.get_width():,}', va='center', fontsize=10)
    from matplotlib.patches import Patch
    _legend_els = [Patch(facecolor=PALETTE[0], label='UK / London'), Patch(facecolor=PALETTE[2], label='International')]
    _ax.legend(handles=_legend_els, fontsize=10, loc='right')
    _ax.set_title('Top 20 Host Origin Locations', fontsize=13)
    _ax.set_xlabel('Number of Listings')
    _ax.set_ylabel('Reported Location')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return (Patch,)


@app.cell
def _(PALETTE, Patch, gdf_1, mo, plt):
    _uk_labels = {'United Kingdom', 'England', 'London', 'Uk', 'Scotland', 'Wales'}
    _top_locs = gdf_1['host_country'].value_counts().clip(upper=300).head(20).sort_values()
    _fig, _ax = plt.subplots(figsize=(12, 8))
    _bar_colors = [PALETTE[0] if loc in _uk_labels else PALETTE[2] for loc in _top_locs.index]
    _bars = _ax.barh(_top_locs.index, _top_locs.values, color=_bar_colors, edgecolor='white', linewidth=0.4)
    _ax.bar_label(_bars, fmt='{:,.0f}', padding=4, fontsize=10)
    _legend_els = [Patch(facecolor=PALETTE[0], label='UK / London'), Patch(facecolor=PALETTE[2], label='International')]
    _ax.legend(handles=_legend_els, fontsize=10, loc='right')
    _ax.set_title('Top 20 Host Origin Locations', fontsize=13)
    _ax.set_xlabel('Number of Listings')
    _ax.set_ylabel('Reported Location')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(PALETTE, gdf_1, mo, plt, sns):
    kde_df = gdf_1.dropna(subset=['host_acceptance_rate', 'host_response_rate', 'host_is_superhost']).copy()
    kde_df['Host Type'] = kde_df['host_is_superhost'].map({'t': 'Superhost', 'f': 'Regular Host'})
    _fig, _ax = plt.subplots(figsize=(10, 7))
    handles = []
    _labels = []
    for _htype, _color in zip(['Superhost', 'Regular Host'], [PALETTE[0], PALETTE[2]]):
        _subset = kde_df[kde_df['Host Type'] == _htype]
        h1 = sns.kdeplot(data=_subset, x='host_acceptance_rate', y='host_response_rate', ax=_ax, fill=True, alpha=0.3, color=_color, levels=6)
        h2 = sns.kdeplot(data=_subset, x='host_acceptance_rate', y='host_response_rate', ax=_ax, fill=False, alpha=0.8, color=_color, levels=6, linewidths=1.2)
        patch = plt.Line2D([0], [0], color=_color, lw=6, alpha=0.6)
        handles.append(patch)
        _labels.append(_htype)
    _ax.set_title('Bivariate KDE: Acceptance Rate vs Response Rate\n(Superhost vs Regular Host)', fontsize=13)
    _ax.set_xlabel('Host Acceptance Rate (0–1)')
    _ax.set_ylabel('Host Response Rate (0–1)')
    _ax.legend(handles=handles, labels=_labels, fontsize=11, loc='lower right', bbox_to_anchor=(0.67, 0.33))
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(PALETTE, gdf_1, go, mo):
    dims = {'Response Rate': 'host_response_rate', 'Acceptance Rate': 'host_acceptance_rate', 'Review Score': 'review_scores_rating', 'Instant Bookable': 'instant_bookable', 'Listings Count': 'host_listings_count', 'Occupancy Rate': 'estimated_occupancy_l365d'}
    radar_df = gdf_1.dropna(subset=list(dims.values())).copy()
    radar_df['Host Type'] = radar_df['host_is_superhost'].map({'t': 'Superhost', 'f': 'Regular Host'})
    group_means = radar_df.groupby('Host Type')[list(dims.values())].mean()
    group_norm = group_means.copy()
    for _col in group_means.columns:
        col_max = group_means[_col].max()
        if col_max > 0:
            group_norm[_col] = group_means[_col] / col_max
        else:
            group_norm[_col] = 0
    cats = list(dims.keys())
    fig12 = go.Figure()
    for _htype, _color in zip(['Superhost', 'Regular Host'], [PALETTE[0], PALETTE[2]]):
        if _htype in group_norm.index:
            vals = [group_norm.loc[_htype, _col] for _col in dims.values()]
            fig12.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself', name=_htype, line_color=_color, fillcolor=_color, opacity=0.6))
    fig12.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Host Performance Radar: Superhost vs Regular Host', showlegend=True, width=680, height=580)
    mo.ui.plotly(fig12)
    return


@app.cell
def _(CAT_CMAP, MAPBOX, fig2, gdf_1, geo, mo, px):
    # Geospatial — Superhost Rate Choropleth by Borough
    superhost_rate = gdf_1.groupby('neighbourhood').apply(lambda x: (x['host_is_superhost'] == 't').mean()).reset_index(name='superhost_rate')
    geo_sh = geo.merge(superhost_rate, on='neighbourhood', how='left')
    fig13 = px.choropleth_mapbox(geo_sh, geojson=geo_sh.geometry.__geo_interface__, locations=geo_sh.index, color='superhost_rate', hover_name='neighbourhood', hover_data={'superhost_rate': ':.1%'}, color_continuous_scale=CAT_CMAP, mapbox_style=MAPBOX, center={'lat': gdf_1.latitude.median(), 'lon': gdf_1.longitude.median()}, zoom=8.8, opacity=0.75, title='Superhost Rate by London Borough')
    fig13.update_layout(margin={'r': 0, 't': 40, 'l': 0, 'b': 0}, coloraxis_colorbar_title='Superhost Rate')
    mo.ui.plotly(fig13)
    fig2.write_html('interactive_map_superhost.html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ML training to identify the strongest predictors
    """)
    return


@app.cell
def _(gdf_1):
    room_type_labels = {_rt: _i for _i, _rt in enumerate(gdf_1.room_type.unique())}
    gdf_1['room_type_label'] = gdf_1.room_type.map(room_type_labels)
    return


@app.cell
def _(gdf_1):
    neighbourhood_labels = {_rt: _i for _i, _rt in enumerate(gdf_1.neighbourhood.unique())}
    gdf_1['neighbourhood_label'] = gdf_1.neighbourhood.map(neighbourhood_labels)
    return


@app.cell
def _(gdf_1):
    # Create a mapping from host_is_superhost to numerical labels
    host_is_superhost_labels = {'t': 1, 'f': 0}
    gdf_1['host_is_superhost_labels'] = gdf_1.host_is_superhost.map(host_is_superhost_labels)
    return (host_is_superhost_labels,)


@app.cell
def _(host_is_superhost_labels):
    host_is_superhost_labels
    return


@app.cell
def _():
    # Feature Importance (Random Forest Regressor)

    MODEL_FEATURES = [
        "accommodates", "bedrooms", "beds", "bathrooms", "room_type_label",
        "minimum_nights", "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location", "review_scores_value", "neighbourhood_label",
        "availability_365", "calculated_host_listings_count",
        "reviews_per_month", "host_is_superhost_labels", "instant_bookable"
    ]
    return (MODEL_FEATURES,)


@app.cell
def _(MODEL_FEATURES, gdf_1, train_test_split):
    model_df = gdf_1[MODEL_FEATURES + ['price']].dropna()
    X = model_df[MODEL_FEATURES]
    y = model_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(RandomForestRegressor, X_train, y_train):
    rf = RandomForestRegressor(max_depth=16, min_samples_leaf=3, n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    return (rf,)


@app.cell
def _(
    X,
    X_test,
    cross_val_score,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    rf,
    y,
    y_test,
):
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()
    return cv_r2, mae, r2, rmse, y_pred


@app.cell
def _(cv_r2, mae, r2, rmse):
    print(f"Random Forest Performance")
    print(f"MAE: £{mae:.2f}")
    print(f"RMSE: £{rmse:.2f}")
    print(f"R² (test): {r2:.3f}")
    print(f"R² (5-CV): {cv_r2:.3f}")
    return


@app.cell
def _(mo, plt, y_pred, y_test):
    # Plot regression line: True vs Predicted on test set

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, alpha=0.8, label="Perfect Prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Random Forest: Actual vs Predicted Prices")
    plt.legend()
    plt.tight_layout()
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(ACCENT, MODEL_FEATURES, mo, pd, plt, rf):
    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=MODEL_FEATURES).sort_values()
    _fig, _ax = plt.subplots(figsize=(10, 8))
    colors = [ACCENT if imp == importances.max() else 'steelblue' for imp in importances]
    importances.plot(kind='barh', color=colors, edgecolor='white', linewidth=0.4, ax=_ax)
    _ax.set_title('Random Forest Feature Importance for Price Prediction\n', fontsize=13)
    _ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)')
    _ax.set_ylabel('Feature')
    plt.tight_layout()
    mo.ui.matplotlib(_fig.gca())
    return


if __name__ == "__main__":
    app.run()
