"""
Pitcher Similarity Explorer - Streamlit App
CS 506 Final Project

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pybaseball import statcast
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Pitcher Similarity Explorer",
    page_icon="⚾",
    layout="wide"
)

# Title
st.title("⚾ MLB Pitcher Similarity Explorer")
st.markdown("### Compare pitchers based on their pitch characteristics")
st.markdown("Using  similarity calculations from MLP model preprocessing*")
st.markdown("---")

# ============================================================================
# PREPROCESSING FUNCTIONS (from MLP_feature_cleaning.ipynb)
# ============================================================================

def add_pitch_type_one_hot(dataframe, pitch_type_column='pitch_type'):
    """One-hot encode pitch types, grouping rare types (<1000) as 'other'"""
    df_with_features = dataframe.copy()
    pitch_type_counts = df_with_features[pitch_type_column].value_counts()
    common_pitch_types = pitch_type_counts[pitch_type_counts >= 1000].index.tolist()
    rare_pitch_types = pitch_type_counts[pitch_type_counts < 1000].index.tolist()

    for pitch_type in common_pitch_types:
        column_name = f"pitch_type_{pitch_type}"
        df_with_features[column_name] = (df_with_features[pitch_type_column] == pitch_type).astype(int)

    if rare_pitch_types:
        df_with_features['pitch_type_other'] = df_with_features[pitch_type_column].isin(rare_pitch_types).astype(int)

    return df_with_features

def add_count_one_hot(dataframe, balls_column='balls', strikes_column='strikes'):
    """One-hot encode ball-strike counts"""
    df_with_features = dataframe.copy()
    df_with_features['count'] = df_with_features[balls_column].astype(str) + '-' + df_with_features[strikes_column].astype(str)
    unique_counts = df_with_features['count'].dropna().unique()

    for count in unique_counts:
        column_name = f"count_{count}"
        if len(count) == 3 and int(count[0]) < 4 and int(count[2]) < 3:
            df_with_features[column_name] = (df_with_features['count'] == count).astype(int)

    df_with_features = df_with_features.drop('count', axis=1)
    return df_with_features

def add_handedness_one_hot(dataframe, batter_column='stand', pitcher_column='p_throws'):
    """Add matchup column and adjust release_pos_x for lefties"""
    df_with_features = dataframe.copy()
    df_with_features['matchup'] = (df_with_features[batter_column] != df_with_features[pitcher_column]).astype(int)

    if 'release_pos_x' in df_with_features.columns:
        lefty_mask = df_with_features[pitcher_column] == 'L'
        df_with_features.loc[lefty_mask, 'release_pos_x'] = df_with_features.loc[lefty_mask, 'release_pos_x'] * -1

    return df_with_features

def filter_columns(dataframe, columns_to_keep, min_pitch_count=None, pitcher_column='player_name'):
    """Filter columns and apply spin axis correction for left-handed pitchers"""
    available_columns = [col for col in columns_to_keep if col in dataframe.columns]
    df_filtered = dataframe[available_columns].copy()

    # Apply minimum pitch count filter
    if min_pitch_count is not None and pitcher_column in df_filtered.columns:
        pitch_counts = df_filtered[pitcher_column].value_counts()
        pitchers_to_keep = pitch_counts[pitch_counts >= min_pitch_count].index
        df_filtered = df_filtered[df_filtered[pitcher_column].isin(pitchers_to_keep)]

    # Spin axis correction for LHP
    if {'spin_axis', 'p_throws'}.issubset(df_filtered.columns):
        lhp_mask = df_filtered['p_throws'].eq('L')
        if lhp_mask.any():
            spin = pd.to_numeric(df_filtered.loc[lhp_mask, 'spin_axis'], errors='coerce')
            adjusted = (180 - spin) % 360
            df_filtered.loc[lhp_mask, 'spin_axis'] = adjusted
        df_filtered = df_filtered.drop('p_throws', axis=1)

    return df_filtered

# ============================================================================
# SIMILARITY CALCULATION (from MLP_feature_cleaning.ipynb)
# ============================================================================

@st.cache_data
def calculate_pitcher_vectors(df_processed):
    """
    Calculate normalized average vectors for each pitcher
    """
    players = df_processed['player_name'].unique().tolist()
    vectors = []

    for player in players:
        df_temp = df_processed[df_processed['player_name'] == player]
        df_temp = df_temp.drop('player_name', axis=1)

        # Average all pitches for this pitcher
        df_avg = (df_temp.sum(axis=0).to_numpy()) / len(df_temp)

        # NORMALIZE by L2 norm 
        df_avg = df_avg / np.linalg.norm(df_avg)

        vectors.append(df_avg)

    return players, vectors

@st.cache_data
def calculate_similarity_matrices(players, vectors):
    """
    Calculate cosine and Euclidean distance matrices
    """
    n_players = len(players)

    # Initialize matrices
    cosine_dist = np.zeros((n_players, n_players))
    euclidean_dist = np.zeros((n_players, n_players))

    # Calculate pairwise distances
    for u in range(n_players):
        for v in range(n_players):
            if u != v:
                # COSINE DISTANCE 
                cosine_dist[u][v] = 1 - (
                    np.dot(vectors[u], vectors[v]) /
                    (np.linalg.norm(vectors[u]) * np.linalg.norm(vectors[v]))
                )

                # EUCLIDEAN DISTANCE 
                euclidean_dist[u][v] = np.linalg.norm(vectors[u] - vectors[v], ord=2)

    return cosine_dist, euclidean_dist

# ============================================================================
# DATA LOADING WITH PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data"""
    with st.spinner("Loading MLB Statcast data and applying preprocessing..."):
        # Load 2025 season data
        data = statcast(start_dt='2025-01-01', end_dt='2025-09-30')

        # Apply preprocessing 
        data = add_pitch_type_one_hot(data)
        data = add_count_one_hot(data)
        data = add_handedness_one_hot(data)

        # Define columns to keep 
        columns_to_keep = [
            'player_name',
            'release_speed',
            'release_pos_x',
            'release_pos_z',
            'pfx_x',
            'pfx_z',
            'plate_x',
            'plate_z',
            'spin_axis',
            'p_throws'  # Will be dropped after spin axis correction
        ]

        # Add all pitch_type_* columns
        pitch_type_cols = [col for col in data.columns if col.startswith('pitch_type_')]
        columns_to_keep.extend(pitch_type_cols)

        # Add all count_* columns
        count_cols = [col for col in data.columns if col.startswith('count_')]
        columns_to_keep.extend(count_cols)

        # Add matchup column
        columns_to_keep.append('matchup')

        # Filter with min 2000 pitches
        df_filtered = filter_columns(data, columns_to_keep, min_pitch_count=2000)

        # Drop NaN
        df_filtered = df_filtered.dropna()

        return data, df_filtered

@st.cache_data
def get_pitcher_stats(raw_data, pitcher_name):
    """Calculate summary statistics for a pitcher"""
    pitcher_data = raw_data[raw_data['player_name'] == pitcher_name]

    if len(pitcher_data) == 0:
        return None, None

    stats = {
        'Total Pitches': len(pitcher_data),
        'Avg Velocity': f"{pitcher_data['release_speed'].mean():.1f} mph",
        'Avg Spin Rate': f"{pitcher_data['release_spin_rate'].mean():.0f} rpm" if 'release_spin_rate' in pitcher_data.columns else 'N/A',
        'Handedness': pitcher_data['p_throws'].iloc[0],
        'Unique Pitch Types': pitcher_data['pitch_type'].nunique()
    }

    return stats, pitcher_data

# ============================================================================
# MAIN APP
# ============================================================================

try:
    # Load data
    raw_data, df_processed = load_and_preprocess_data()

    # Calculate pitcher vectors and similarities 
    with st.spinner("Calculating pitcher similarities..."):
        players, vectors = calculate_pitcher_vectors(df_processed)
        cosine_distances, euclidean_distances = calculate_similarity_matrices(players, vectors)

    st.success(f"✅ Loaded {len(players)} qualified pitchers")

    # Sidebar - Pitcher Selection
    st.sidebar.header("Select Pitcher")

    selected_pitcher = st.sidebar.selectbox(
        "Choose a pitcher to analyze:",
        options=sorted(players),
        index=0
    )

    # Similarity metric selection
    st.sidebar.markdown("---")
    st.sidebar.header("Similarity Metric")
    similarity_metric = st.sidebar.radio(
        "Choose distance metric:",
        options=["Cosine Distance", "Euclidean Distance"],
        help="Cosine measures angular similarity, Euclidean measures straight-line distance."
    )

    # Get pitcher index
    pitcher_idx = players.index(selected_pitcher)

    # Get stats
    stats, pitcher_data = get_pitcher_stats(raw_data, selected_pitcher)

    if stats is None:
        st.error(f"No data found for {selected_pitcher}")
    else:
        # Main content - Two columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"📊 {selected_pitcher}")
            st.markdown("**Key Stats:**")

            for key, value in stats.items():
                st.metric(label=key, value=value)

        with col2:
            st.subheader("Pitch Type Distribution")

            # Pitch type breakdown
            pitch_counts = pitcher_data['pitch_type'].value_counts()

            fig_pie = px.pie(
                values=pitch_counts.values,
                names=pitch_counts.index,
                title=f"{selected_pitcher}'s Pitch Arsenal",
                hole=0.4
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # Pitch Characteristics
        st.subheader("⚡ Pitch Characteristics")

        tab1, tab2, tab3 = st.tabs(["Velocity", "Spin Rate", "Movement"])

        with tab1:
            # Velocity by pitch type
            fig_velo = px.box(
                pitcher_data,
                x='pitch_type',
                y='release_speed',
                title=f"{selected_pitcher} - Velocity by Pitch Type",
                labels={'release_speed': 'Velocity (mph)', 'pitch_type': 'Pitch Type'},
                color='pitch_type'
            )
            st.plotly_chart(fig_velo, use_container_width=True)

        with tab2:
            # Spin rate by pitch type
            if 'release_spin_rate' in pitcher_data.columns:
                fig_spin = px.box(
                    pitcher_data,
                    x='pitch_type',
                    y='release_spin_rate',
                    title=f"{selected_pitcher} - Spin Rate by Pitch Type",
                    labels={'release_spin_rate': 'Spin Rate (rpm)', 'pitch_type': 'Pitch Type'},
                    color='pitch_type'
                )
                st.plotly_chart(fig_spin, use_container_width=True)
            else:
                st.info("Spin rate data not available")

        with tab3:
            # Pitch movement
            fig_movement = px.scatter(
                pitcher_data,
                x='pfx_x',
                y='pfx_z',
                color='pitch_type',
                title=f"{selected_pitcher} - Pitch Movement",
                labels={'pfx_x': 'Horizontal Break (in)', 'pfx_z': 'Vertical Break (in)'},
                hover_data=['release_speed']
            )
            st.plotly_chart(fig_movement, use_container_width=True)

        st.markdown("---")

        # Similar Pitchers 
        st.subheader("🔍 Most Similar Pitchers")

        st.markdown(f"**Using {similarity_metric} ")

        # Get distances for selected pitcher
        if similarity_metric == "Cosine Distance":
            distances = cosine_distances[pitcher_idx]
        else:
            distances = euclidean_distances[pitcher_idx]

        # Create similarity dataframe
        similarities = []
        for i, player in enumerate(players):
            if i != pitcher_idx:
                similarities.append({
                    'Pitcher': player,
                    'Distance': round(distances[i], 4)
                })

        # Sort by distance (lower = more similar)
        similarities_df = pd.DataFrame(similarities).sort_values('Distance', ascending=True)

        # Convert to similarity score (inverse of distance)
        max_dist = similarities_df['Distance'].max()
        similarities_df['Similarity Score'] = 100 * (1 - similarities_df['Distance'] / max_dist)
        similarities_df['Similarity Score'] = similarities_df['Similarity Score'].round(1)

        # Show top 10 most similar
        st.markdown("**Top 10 Most Similar Pitchers:**")

        top_10 = similarities_df.head(10)

        # Create bar chart
        fig_similar = px.bar(
            top_10,
            x='Similarity Score',
            y='Pitcher',
            orientation='h',
            title=f"Pitchers Most Similar to {selected_pitcher} ({similarity_metric})",
            labels={'Similarity Score': 'Similarity Score (0-100)'},
            color='Similarity Score',
            color_continuous_scale='Blues',
            hover_data=['Distance']
        )

        fig_similar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_similar, use_container_width=True)

        # Show data table
        with st.expander("View Full Similarity Table"):
            display_df = similarities_df[['Pitcher', 'Distance', 'Similarity Score']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("---")

        # Pitch-by-Pitch Data
        st.subheader("📋 Pitch-by-Pitch Data")

        st.markdown(f"Showing recent pitches from **{selected_pitcher}**")

        # Select columns to display
        display_cols = [
            'game_date', 'pitch_type', 'release_speed', 'release_spin_rate',
            'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'description'
        ]

        # Filter to available columns
        available_cols = [col for col in display_cols if col in pitcher_data.columns]

        # Show latest 100 pitches
        recent_pitches = pitcher_data[available_cols].head(100)

        st.dataframe(recent_pitches, use_container_width=True, height=400)

        # Download button
        csv = recent_pitches.to_csv(index=False)
        st.download_button(
            label="Download Pitch Data as CSV",
            data=csv,
            file_name=f"{selected_pitcher.replace(', ', '_')}_pitch_data.csv",
            mime="text/csv"
        )

        st.markdown("---")

        st.subheader("📊 Team Analysis - Pitcher Similarity Network Graphs")

        st.markdown("*These network graphs were generated using the same distance calculations as above*")

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.image("images/cos_dist_pitcher_similarity.PNG",
                    caption="Cosine Distance Network Graph - Pitcher positions determined by MDS",
                    use_container_width=True)

        with col_img2:
            st.image("images/euc_dist_pitcher_similarity.PNG",
                    caption="Euclidean Distance Network Graph - Edge colors show distance quantiles",
                    use_container_width=True)

        st.info("💡 These graphs show all 110 pitchers positioned using Multidimensional Scaling (MDS) to preserve the calculated distances. Edge colors represent distance quantiles.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you have an internet connection to fetch Statcast data.")

    with st.expander("Show error details"):
        st.code(str(e), language='python')
        import traceback
        st.code(traceback.format_exc(), language='python')

# Footer
st.markdown("---")
st.markdown("**CS 506 Final Project** | Built with Streamlit | Data from MLB Statcast")
