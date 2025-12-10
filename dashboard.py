import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import MDS
import pickle
import os
from typing import Dict, List, Tuple

# Set page configuration
st.set_page_config(
    page_title="Pitcher Similarity Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the model architecture (must match training)
class PitcherClassifier(nn.Module):
    def __init__(self, num_features, num_pitchers, hidden1=128, hidden2=64, dropout_rate=0.3):
        super(PitcherClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_pitchers)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


@st.cache_resource
def load_model_and_artifacts():
    """Load trained model, scaler, label encoder, data, and confusion-based similarity matrices."""
    try:
        # Load artifacts
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load processed data
        df = pd.read_csv('processed_data.csv')
        
        # Load model
        num_features = df.drop(['player_name'], axis=1).shape[1]
        num_pitchers = len(label_encoder.classes_)
        
        model = PitcherClassifier(num_features, num_pitchers)
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
        model.eval()
        
        # Load confusion-based similarity matrices
        confusion_matrices = {}
        if os.path.exists('outputs/confusion_mass_matrix.npy'):
            confusion_matrices['C'] = np.load('outputs/confusion_mass_matrix.npy')
            confusion_matrices['m'] = np.load('outputs/off_diagonal_mass.npy')
            confusion_matrices['Sim'] = np.load('outputs/similarity_matrix_weighted_symmetric.npy')
            confusion_matrices['uniqueness'] = np.load('outputs/pitcher_uniqueness.npy')
            confusion_matrices['pitcher_names'] = np.load('outputs/pitcher_names.npy', allow_pickle=True)
            st.success("✓ Loaded confusion-based similarity matrices")
        else:
            st.warning("⚠️ Confusion matrices not found. Run the similarity computation cells in the notebook first.")
            confusion_matrices = None
        
        return model, scaler, label_encoder, df, confusion_matrices
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please ensure you have run the notebook cells to save model.pt, scaler.pkl, label_encoder.pkl, and processed_data.csv")
        return None, None, None, None, None


def compute_pitcher_similarity(model, scaler, label_encoder, df, pitcher_name, confusion_matrices=None, filter_pitch_type=None, filter_count=None):
    """
    Compute similarity scores using confusion-based weighted symmetric similarity.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        df: Processed DataFrame with all pitches
        pitcher_name: Name of the pitcher to analyze
        confusion_matrices: Dict with precomputed confusion matrices (C, Sim, m, uniqueness)
        filter_pitch_type: Optional pitch type filter
        filter_count: Optional count situation filter
    
    Returns:
        similarity_scores: Dict mapping pitcher names to weighted symmetric similarity scores
        uniqueness: Self-classification accuracy (higher = more unique)
        pitch_details: DataFrame with per-pitch predictions
        confusion_mass: Dict with directional confusion masses
    """
    # Filter pitches for the selected pitcher
    pitcher_pitches = df[df['player_name'] == pitcher_name].copy()
    
    if len(pitcher_pitches) == 0:
        return {}, 0.0, pd.DataFrame(), {}
    
    # Apply filters if provided
    if filter_pitch_type:
        pitch_type_cols = [col for col in pitcher_pitches.columns if col.startswith('pitch_type_')]
        if f'pitch_type_{filter_pitch_type}' in pitch_type_cols:
            pitcher_pitches = pitcher_pitches[pitcher_pitches[f'pitch_type_{filter_pitch_type}'] == 1]
    
    if filter_count:
        count_cols = [col for col in pitcher_pitches.columns if col.startswith('count_')]
        if f'count_{filter_count}' in count_cols:
            pitcher_pitches = pitcher_pitches[pitcher_pitches[f'count_{filter_count}'] == 1]
    
    if len(pitcher_pitches) == 0:
        return {}, 0.0, pd.DataFrame(), {}
    
    # Prepare features
    X = pitcher_pitches.drop(['player_name'], axis=1).values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()
    
    # Get pitcher index
    pitcher_idx = np.where(label_encoder.classes_ == pitcher_name)[0][0]
    
    # Calculate uniqueness (self-classification accuracy)
    predictions = np.argmax(probabilities, axis=1)
    uniqueness = np.mean(predictions == pitcher_idx)
    
    # Use precomputed confusion-based similarities if available
    if confusion_matrices is not None:
        Sim = confusion_matrices['Sim']
        C = confusion_matrices['C']
        m = confusion_matrices['m']
        
        # Get weighted symmetric similarities for this pitcher
        # Sim[i,j] = (C[i,j] + C[j,i]) / (m[i] + m[j])
        similarity_scores_array = Sim[pitcher_idx].copy()
        similarity_scores_array[pitcher_idx] = 0  # Zero out self
        
        # Get directional confusion masses
        confusion_mass_array = C[pitcher_idx].copy()
        confusion_mass_array[pitcher_idx] = 0
        
        # Create dictionaries
        similarity_scores = {
            label_encoder.classes_[i]: similarity_scores_array[i]
            for i in range(len(label_encoder.classes_))
            if i != pitcher_idx and similarity_scores_array[i] > 0
        }
        
        confusion_mass = {
            label_encoder.classes_[i]: confusion_mass_array[i]
            for i in range(len(label_encoder.classes_))
            if i != pitcher_idx and confusion_mass_array[i] > 0
        }
    else:
        # Fallback to old method if matrices not available
        mean_probs = np.mean(probabilities, axis=0)
        mean_probs[pitcher_idx] = 0
        
        if np.sum(mean_probs) > 0:
            similarity_scores_array = mean_probs / np.sum(mean_probs)
        else:
            similarity_scores_array = mean_probs
        
        similarity_scores = {
            label_encoder.classes_[i]: similarity_scores_array[i]
            for i in range(len(label_encoder.classes_))
            if i != pitcher_idx and similarity_scores_array[i] > 0
        }
        confusion_mass = {}
    
    # Create pitch details dataframe
    pitch_details = pitcher_pitches.copy()
    pitch_details['predicted_pitcher'] = label_encoder.inverse_transform(predictions)
    pitch_details['confidence'] = np.max(probabilities, axis=1)
    
    # Add top 3 predictions
    top3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    for i in range(3):
        pitch_details[f'top_{i+1}_pitcher'] = label_encoder.inverse_transform(top3_indices[:, i])
        pitch_details[f'top_{i+1}_prob'] = probabilities[np.arange(len(probabilities)), top3_indices[:, i]]
    
    return similarity_scores, uniqueness, pitch_details, confusion_mass


def plot_network_graph(df, pitcher_name, similarity_scores, distance_matrix, metric_name="Cosine", top_n=30, all_pitchers_in_matrix=None):
    """Create an interactive network graph showing pitcher similarities using force-directed layout."""
    # Get top N most similar pitchers
    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_pitchers = [pitcher_name] + [p for p, _ in sorted_scores]
    
    # Get indices - if all_pitchers_in_matrix provided, use that list, otherwise use full list
    if all_pitchers_in_matrix is not None:
        pitcher_indices = [all_pitchers_in_matrix.index(p) for p in top_pitchers if p in all_pitchers_in_matrix]
    else:
        all_pitchers = sorted(df['player_name'].unique())
        pitcher_indices = [all_pitchers.index(p) for p in top_pitchers if p in all_pitchers]
    
    if len(pitcher_indices) < 2:
        return None
    
    # Extract submatrix for these pitchers
    sub_matrix = distance_matrix[np.ix_(pitcher_indices, pitcher_indices)]
    
    # Use a circular layout to spread nodes out
    n_nodes = len(pitcher_indices)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 5.0
    
    # Put selected pitcher in center, others in circle
    positions = np.zeros((n_nodes, 2))
    positions[0] = [0, 0]  # Selected pitcher at origin
    for i in range(1, n_nodes):
        positions[i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
    
    # Create directed edges (all pairs, both directions, filter out near-zero distances)
    edges = []
    threshold = 0.01  # Only show edges with meaningful distance
    for i in range(len(pitcher_indices)):
        for j in range(len(pitcher_indices)):
            if i != j and sub_matrix[i][j] > threshold:
                edges.append((i, j, sub_matrix[i][j]))
    
    if len(edges) == 0:
        return None
    
    # Normalize distances for color mapping
    distances = [e[2] for e in edges]
    min_dist, max_dist = min(distances), max(distances)
    
    # Plot directed edges - use curved arrows to show directionality when A→B and B→A both exist
    edge_traces = []
    for i, j, dist in edges:
        # Color based on distance (blue=similar, red=dissimilar)
        norm_dist = (dist - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 0.5
        color = f'rgba({int(255*norm_dist)}, {int(100*(1-norm_dist))}, {int(255*(1-norm_dist))}, 0.5)'
        
        # Check if reverse edge exists to add curvature
        reverse_exists = any(e[0] == j and e[1] == i for e in edges)
        
        if reverse_exists:
            # Curve the edge slightly to show both directions
            mid_x = (positions[i, 0] + positions[j, 0]) / 2
            mid_y = (positions[i, 1] + positions[j, 1]) / 2
            # Perpendicular offset
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            offset = 0.3
            curve_x = mid_x - offset * dy / (np.sqrt(dx**2 + dy**2) + 1e-6)
            curve_y = mid_y + offset * dx / (np.sqrt(dx**2 + dy**2) + 1e-6)
            
            edge_traces.append(
                go.Scatter(
                    x=[positions[i, 0], curve_x, positions[j, 0], None],
                    y=[positions[i, 1], curve_y, positions[j, 1], None],
                    mode='lines',
                    line=dict(width=2, color=color),
                    hoverinfo='text',
                    text=f'{top_pitchers[i]} → {top_pitchers[j]}<br>Distance: {dist:.4f}',
                    showlegend=False
                )
            )
        else:
            # Straight edge
            edge_traces.append(
                go.Scatter(
                    x=[positions[i, 0], positions[j, 0], None],
                    y=[positions[i, 1], positions[j, 1], None],
                    mode='lines',
                    line=dict(width=2, color=color),
                    hoverinfo='text',
                    text=f'{top_pitchers[i]} → {top_pitchers[j]}<br>Distance: {dist:.4f}',
                    showlegend=False
                )
            )
    
    # Highlight selected pitcher and similar ones
    node_colors = ['red' if top_pitchers[i] == pitcher_name else 'lightblue' for i in range(len(top_pitchers))]
    node_sizes = [25 if top_pitchers[i] == pitcher_name else 15 for i in range(len(top_pitchers))]
    
    # Plot nodes
    node_trace = go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+text',
        text=top_pitchers,
        textposition='top center',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        hoverinfo='text',
        hovertext=[f'{p}<br>Pitches: {len(df[df["player_name"]==p])}' for p in top_pitchers],
        showlegend=False
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f'Directed Pitcher Similarity Network ({metric_name})<br><sub>Curved edges show bidirectional relationships (A→B and B→A). Edge color: blue=similar, red=dissimilar</sub>',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        hovermode='closest'
    )
    
    return fig


def calculate_uniqueness_percentile(uniqueness, all_pitcher_data, df, label_encoder):
    """Calculate the percentile rank of a pitcher's uniqueness score."""
    # Calculate uniqueness for all pitchers
    uniqueness_scores = []
    
    for pitcher_name in label_encoder.classes_:
        pitcher_data = df[df['player_name'] == pitcher_name]
        if len(pitcher_data) == 0:
            continue
        uniqueness_scores.append(uniqueness)  # Simplified - in production would calculate for each
    
    # Calculate percentile (what % of pitchers have lower uniqueness)
    if len(uniqueness_scores) > 0:
        percentile = (sum(1 for s in uniqueness_scores if s < uniqueness) / len(uniqueness_scores)) * 100
        return percentile
    return 50.0


def predict_custom_pitch(model, scaler, label_encoder, feature_values, feature_names):
    """
    Predict which pitcher a custom pitch most resembles.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_values: Dictionary of feature name -> value
        feature_names: List of all feature names in order
    
    Returns:
        top_predictions: List of tuples (pitcher_name, probability)
    """
    # Create feature vector
    X = np.zeros((1, len(feature_names)))
    for i, feature in enumerate(feature_names):
        if feature in feature_values:
            X[0, i] = feature_values[feature]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
    
    # Get top predictions
    top_indices = np.argsort(probabilities)[::-1][:10]
    top_predictions = [
        (label_encoder.classes_[i], probabilities[i])
        for i in top_indices
    ]
    
    return top_predictions


def plot_similarity_bar_chart(similarity_scores, top_n=10):
    """Create a bar chart of top N similar pitchers."""
    # Sort and get top N
    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_scores:
        return None
    
    pitchers, scores = zip(*sorted_scores)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(scores),
            y=list(pitchers),
            orientation='h',
            marker=dict(
                color=list(scores),
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Similarity Score")
            ),
            text=[f'{s:.3f}' for s in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Most Similar Pitchers',
        xaxis_title='Similarity Score',
        yaxis_title='Pitcher',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_pitch_characteristics_radar(df, pitcher_name, similar_pitchers, top_n=3):
    """Create a radar chart comparing pitch characteristics."""
    # Select numerical features for comparison
    numerical_features = ['release_speed', 'release_pos_x', 'release_pos_z', 
                         'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'spin_axis']
    
    available_features = [f for f in numerical_features if f in df.columns]
    
    if not available_features:
        return None
    
    # Get mean values for each pitcher
    pitchers_to_compare = [pitcher_name] + similar_pitchers[:top_n]
    
    fig = go.Figure()
    
    for pitcher in pitchers_to_compare:
        pitcher_data = df[df['player_name'] == pitcher]
        if len(pitcher_data) > 0:
            values = [pitcher_data[f].mean() for f in available_features]
            
            # Normalize values to 0-1 range for better visualization
            normalized_values = []
            for i, f in enumerate(available_features):
                min_val = df[f].min()
                max_val = df[f].max()
                if max_val != min_val:
                    normalized_values.append((values[i] - min_val) / (max_val - min_val))
                else:
                    normalized_values.append(0.5)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # Close the loop
                theta=available_features + [available_features[0]],
                fill='toself',
                name=pitcher,
                opacity=0.7
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Pitch Characteristics Comparison',
        height=500
    )
    
    return fig


def plot_pitch_type_similarity_heatmap(pitch_details, label_encoder):
    """Create a heatmap showing which pitchers are predicted for each pitch type."""
    # Get pitch type columns
    pitch_type_cols = [col for col in pitch_details.columns if col.startswith('pitch_type_') and pitch_details[col].sum() > 0]
    
    if len(pitch_type_cols) == 0:
        return None
    
    # Get top predicted pitchers
    top_pitchers = pitch_details['top_1_pitcher'].value_counts().head(10).index.tolist()
    
    # Create matrix: pitch types vs predicted pitchers
    matrix_data = []
    pitch_type_labels = []
    
    for pt_col in pitch_type_cols:
        pitch_type_name = pt_col.replace('pitch_type_', '')
        pitch_type_labels.append(pitch_type_name)
        
        filtered_pitches = pitch_details[pitch_details[pt_col] == 1]
        
        row = []
        for pitcher in top_pitchers:
            count = (filtered_pitches['top_1_pitcher'] == pitcher).sum()
            total = len(filtered_pitches)
            percentage = (count / total * 100) if total > 0 else 0
            row.append(percentage)
        
        matrix_data.append(row)
    
    if not matrix_data:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=top_pitchers,
        y=pitch_type_labels,
        colorscale='Blues',
        text=[[f'{val:.1f}%' for val in row] for row in matrix_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="% of Pitches")
    ))
    
    fig.update_layout(
        title='Pitch Type Similarity Breakdown',
        xaxis_title='Predicted Pitcher',
        yaxis_title='Actual Pitch Type',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig


def main():
    st.title("⚾ Pitcher Similarity Dashboard")
    st.markdown("Analyze pitcher similarities based on pitch characteristics and game context.")
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, scaler, label_encoder, df, confusion_matrices = load_model_and_artifacts()
    
    if model is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Get list of pitchers
    pitchers = sorted(df['player_name'].unique())
    
    # Mode selection
    mode = st.sidebar.radio("Mode", ["Pitcher Similarity", "Custom Pitch Prediction"])
    
    if mode == "Pitcher Similarity":
        # Pitcher selection
        selected_pitcher = st.sidebar.selectbox("Select Pitcher", pitchers)
        
        # Number of similar pitchers to display
        top_n = st.sidebar.slider("Number of Similar Pitchers", 5, 20, 10)
        
        # Filtering options
        st.sidebar.subheader("Filters")
        
        # Pitch type filter
        pitch_type_cols = [col.replace('pitch_type_', '') for col in df.columns if col.startswith('pitch_type_')]
        pitch_type_filter = st.sidebar.selectbox("Pitch Type", ["All"] + pitch_type_cols)
        
        # Count filter
        count_cols = [col.replace('count_', '') for col in df.columns if col.startswith('count_')]
        count_filter = st.sidebar.selectbox("Count", ["All"] + count_cols)
        
        # Apply filters
        filter_pitch_type = None if pitch_type_filter == "All" else pitch_type_filter
        filter_count = None if count_filter == "All" else count_filter
        
        # Compute similarity
        with st.spinner(f"Computing similarities for {selected_pitcher}..."):
            similarity_scores, uniqueness, pitch_details, confusion_mass = compute_pitcher_similarity(
                model, scaler, label_encoder, df, selected_pitcher,
                confusion_matrices=confusion_matrices,
                filter_pitch_type=filter_pitch_type,
                filter_count=filter_count
            )
        
        # Compute distance matrices - use model predictions for asymmetric similarities
        @st.cache_data
        def compute_model_similarity_matrix(_model, _scaler, _label_encoder, _df, top_pitchers_list):
            """Compute asymmetric similarity matrix using model predictions for subset of pitchers."""
            n_pitchers = len(top_pitchers_list)
            similarity_matrix = np.zeros((n_pitchers, n_pitchers))
            
            for i, pitcher_a in enumerate(top_pitchers_list):
                # Get similarity scores when analyzing pitcher A
                sim_scores, _, _, _ = compute_pitcher_similarity(
                    _model, _scaler, _label_encoder, _df, pitcher_a
                )
                
                # Fill in row i with similarities to other pitchers
                for j, pitcher_b in enumerate(top_pitchers_list):
                    if i != j:
                        similarity_matrix[i][j] = sim_scores.get(pitcher_b, 0)
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            return distance_matrix
        
        # Get top similar pitchers to focus computation
        top_similar_pitchers = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        top_pitcher_names = [selected_pitcher] + [p for p, _ in top_similar_pitchers]
        
        with st.spinner("Computing model-based distance matrix for network graph..."):
            model_dist_matrix = compute_model_similarity_matrix(
                model, scaler, label_encoder, df, top_pitcher_names
            )
            all_pitchers = top_pitcher_names
        
        # Compute Euclidean distance matrix for comparison
        with st.spinner("Computing Euclidean distance matrix..."):
            pitcher_vectors = []
            for p in all_pitchers:
                p_data = df[df['player_name'] == p].drop(['player_name'], axis=1)
                if len(p_data) > 0:
                    avg_vec = p_data.mean(axis=0).values
                    pitcher_vectors.append(avg_vec / np.linalg.norm(avg_vec))
                else:
                    pitcher_vectors.append(np.zeros(len(p_data.columns)))
            
            n_pitchers = len(all_pitchers)
            euclidean_dist_matrix = np.zeros((n_pitchers, n_pitchers))
            
            for i in range(n_pitchers):
                for j in range(n_pitchers):
                    if i != j:
                        euclidean_dist_matrix[i][j] = np.linalg.norm(pitcher_vectors[i] - pitcher_vectors[j])
            
            # Get Euclidean similarities for selected pitcher
            pitcher_idx = all_pitchers.index(selected_pitcher)
            euclidean_similarities = {}
            for i, p in enumerate(all_pitchers):
                if i != pitcher_idx:
                    euclidean_similarities[p] = 1 - (euclidean_dist_matrix[pitcher_idx][i] / euclidean_dist_matrix.max())
        
        # Display results
        st.header(f"Analysis for {selected_pitcher}")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pitches Analyzed", len(pitch_details))
        with col2:
            # Calculate percentile
            all_uniqueness = []
            for p in label_encoder.classes_:
                p_data = df[df['player_name'] == p]
                if len(p_data) > 0:
                    X_p = p_data.drop(['player_name'], axis=1).values
                    X_p_scaled = scaler.transform(X_p)
                    X_p_tensor = torch.tensor(X_p_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(X_p_tensor)
                        preds = torch.argmax(logits, dim=1).numpy()
                    p_idx = np.where(label_encoder.classes_ == p)[0][0]
                    p_uniqueness = np.mean(preds == p_idx)
                    all_uniqueness.append(p_uniqueness)
            
            percentile = (sum(1 for u in all_uniqueness if u < uniqueness) / len(all_uniqueness)) * 100 if all_uniqueness else 50
            st.metric("Uniqueness", f"{percentile:.0f}th percentile")
            st.caption(f"Model accuracy: {uniqueness:.1%}")
        with col3:
            avg_similarity = np.mean(list(similarity_scores.values())[:top_n]) if similarity_scores else 0
            st.metric("Avg Similarity (Top N)", f"{avg_similarity:.3f}")
        
        # Similarity comparison: Model vs Euclidean
        if similarity_scores:
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                st.subheader("Model-Based Similarity")
                fig_bar = plot_similarity_bar_chart(similarity_scores, top_n)
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_sim2:
                st.subheader("Euclidean Distance Similarity")
                fig_bar_euc = plot_similarity_bar_chart(euclidean_similarities, top_n)
                if fig_bar_euc:
                    fig_bar_euc.update_layout(title=f'Top {top_n} Most Similar (Euclidean)')
                    st.plotly_chart(fig_bar_euc, use_container_width=True)
            
            # Radar chart comparison
            similar_pitcher_names = [p for p, _ in sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)]
            fig_radar = plot_pitch_characteristics_radar(df, selected_pitcher, similar_pitcher_names, top_n=3)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Pitch type heatmap
            if len(pitch_details) > 0:
                fig_heatmap = plot_pitch_type_similarity_heatmap(pitch_details, label_encoder)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Network graphs
            st.subheader("Similarity Network Visualizations")
            tab_net1, tab_net2 = st.tabs(["Model-Based Network", "Euclidean Network"])
            
            with tab_net1:
                fig_network = plot_network_graph(
                    df, selected_pitcher, similarity_scores, 
                    model_dist_matrix, "Model-Based Similarity", top_n=min(top_n, 20),
                    all_pitchers_in_matrix=all_pitchers
                )
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)
                    st.caption("Network graph showing asymmetric model-based similarities. Curved edges show A→B may differ from B→A.")
                else:
                    st.info("No network graph available - insufficient similarity data for this pitcher.")
            
            with tab_net2:
                fig_network_euc = plot_network_graph(
                    df, selected_pitcher, euclidean_similarities,
                    euclidean_dist_matrix, "Euclidean Distance", top_n=min(top_n, 20),
                    all_pitchers_in_matrix=all_pitchers
                )
                if fig_network_euc:
                    st.plotly_chart(fig_network_euc, use_container_width=True)
                    st.caption("Network graph using Euclidean distance on averaged features.")
                else:
                    st.info("No network graph available - insufficient similarity data for this pitcher.")
                    st.plotly_chart(fig_network_euc, use_container_width=True)
                    st.caption("Network graph using Euclidean distance on averaged features.")
            
            # Detailed similarity table
            st.subheader("Similarity Scores")
            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            similarity_df = pd.DataFrame(sorted_scores, columns=['Pitcher', 'Similarity Score'])
            similarity_df['Rank'] = range(1, len(similarity_df) + 1)
            similarity_df = similarity_df[['Rank', 'Pitcher', 'Similarity Score']]
            st.dataframe(similarity_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No similarity data available with current filters.")
    
    else:  # Custom Pitch Prediction
        st.header("Custom Pitch Prediction")
        st.markdown("Enter pitch characteristics to see which pitcher it most resembles.")
        
        # Get feature names (excluding player_name)
        feature_names = [col for col in df.columns if col != 'player_name']
        
        # Separate numerical and categorical features
        numerical_features = ['release_speed', 'release_pos_x', 'release_pos_z', 
                             'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'spin_axis', 'release_spin']
        pitch_type_features = [col for col in feature_names if col.startswith('pitch_type_')]
        count_features = [col for col in feature_names if col.startswith('count_')]
        
        # Input form
        with st.form("custom_pitch_form"):
            st.subheader("Pitch Characteristics")
            
            col1, col2 = st.columns(2)
            
            feature_values = {}
            
            with col1:
                st.markdown("**Kinematic Features**")
                for feature in numerical_features:
                    if feature in df.columns:
                        default_val = float(df[feature].mean())
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        feature_values[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=(max_val - min_val) / 100
                        )
            
            with col2:
                st.markdown("**Contextual Features**")
                
                # Pitch type selection
                pitch_type_names = [col.replace('pitch_type_', '') for col in pitch_type_features]
                selected_pitch_type = st.selectbox("Pitch Type", pitch_type_names)
                for col in pitch_type_features:
                    if col == f'pitch_type_{selected_pitch_type}':
                        feature_values[col] = 1.0
                    else:
                        feature_values[col] = 0.0
                
                # Count selection
                count_names = [col.replace('count_', '') for col in count_features]
                selected_count = st.selectbox("Count", count_names)
                for col in count_features:
                    if col == f'count_{selected_count}':
                        feature_values[col] = 1.0
                    else:
                        feature_values[col] = 0.0
                
                # Matchup (if exists)
                if 'matchup' in feature_names:
                    feature_values['matchup'] = float(st.selectbox("Matchup", ["Same Hand (0)", "Opposite Hand (1)"]).split('(')[1][0])
            
            submit_button = st.form_submit_button("Predict Pitcher")
        
        if submit_button:
            with st.spinner("Predicting..."):
                top_predictions = predict_custom_pitch(model, scaler, label_encoder, feature_values, feature_names)
            
            st.subheader("Top 10 Most Similar Pitchers")
            
            # Display as bar chart
            pitchers_pred, probs_pred = zip(*top_predictions)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(probs_pred),
                    y=list(pitchers_pred),
                    orientation='h',
                    marker=dict(
                        color=list(probs_pred),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Probability")
                    ),
                    text=[f'{p:.2%}' for p in probs_pred],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Most Likely Pitchers',
                xaxis_title='Probability',
                yaxis_title='Pitcher',
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display as table
            pred_df = pd.DataFrame(top_predictions, columns=['Pitcher', 'Probability'])
            pred_df['Rank'] = range(1, len(pred_df) + 1)
            pred_df['Probability'] = pred_df['Probability'].apply(lambda x: f'{x:.2%}')
            pred_df = pred_df[['Rank', 'Pitcher', 'Probability']]
            st.dataframe(pred_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
