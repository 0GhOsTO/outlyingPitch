# Pitcher Anomaly Detection with Statcast Data - Midterm Report

**Video Presentation:** [[Presentation](https://www.youtube.com/watch?v=rqN7egnuWlw)]

---

## Project Overview

We are building a model that, given a single pitch based on Statcast features and context, predicts which pitcher threw it. This will be done for all pitchers with a sufficient number of innings pitched in the MLB. The model learns the distinctive characteristics of each pitcher's repertoire based on pitching metrics (release point, spin rate, movement) as well as game context (current count, position of runners, score differential).

**Use Case:** This model serves as an advanced scouting tool. By feeding pitches from another pitcher's starts into the model, we can analyze which pitcher profiles they most closely resemble and how they compare to different pitching styles. This provides insights into pitch characteristics, helps identify similar pitchers for comparative analysis, and reveals unique aspects of a pitcher's approach that distinguish them from others.

---

## Objectives

1. **Goal 1:** Given pitch characteristics and game context, predict which pitcher threw the pitch.
2. **Goal 2:** Use prediction probabilities and misclassifications to identify similar pitchers and compare pitching styles for scouting analysis.

---

## Data Collection

We gathered data from MLB Statcast using the pybaseball library for the 2025 season. The dataset includes comprehensive information for each pitch thrown:

**Kinematic Features:**
- Release speed, position (x, y, z), and extension
- Spin rate and spin axis
- Pitch movement (pfx_x, pfx_z)
- Velocity components (vx0, vy0, vz0)
- Acceleration components (ax, ay, az)
- Plate location (plate_x, plate_z)

**Contextual Features:**
- Pitch type
- Ball-strike count
- Pitcher handedness (p_throws)
- Batter handedness (stand)
- Player name

All available features are documented at https://baseballsavant.mlb.com/csv-docs.

---

## Data Processing

### Categorical Feature Encoding

To perform classification, we converted categorical features into binary features:

**Pitch Type Encoding:**
- The pitch_type field contains encodings such as "FF" (four-seam fastball) or "CH" (changeup)
- We created a separate binary feature for each pitch type
- Each pitch now has a 1 in its corresponding pitch type column and 0 in all others

**Ball-Strike Count Encoding:**
- Rather than treating balls and strikes as scaled numerical values, we recognized that each count has distinct strategic implications
- We created binary features for each possible count (e.g., count_0-0, count_1-2, count_3-2)
- This allows the model to learn the unique characteristics of pitches thrown in different counts

**Handedness Encoding:**
- p_throws and stand represent pitcher and batter handedness (L or R)
- Currently implemented with 2 binary features each (pitcher_hand_L, pitcher_hand_R, batter_hand_L, batter_hand_R)
- This approach has been identified as problematic due to perfect collinearity and will be revised in future iterations

### Data Filtering

We filtered the dataset to focus on qualified major league starting pitchers:
- Applied a minimum pitch count threshold of 2,000 pitches per pitcher
- This filtering removed relief pitchers, who are deployed in different roles and situations
- Final dataset includes 110 pitchers
- This prevents the model from learning role-specific patterns rather than pitcher-specific characteristics

### Feature Scaling

All numerical features were normalized using StandardScaler to ensure they are on comparable ranges for the neural network.

---

## Preliminary Visualizations

We created several visualizations to understand the data:

### Release Point Analysis

Scatter plot of release_pos_x vs release_pos_z showing distinct clusters for left-handed and right-handed pitchers:

![Release Point Clusters](images/release_point_clusters.png)

Individual pitcher release point plots colored by pitch type, revealing consistency and variation patterns:

![Release Point by Pitch Type](images/release_point_pitch_type.png)

### Pitch Distribution

Histogram of total pitches thrown by each pitcher, showing the distribution across all MLB pitchers. This was used to determine appropriate minimum pitch count threshold:

![Pitch Count Distribution](images/pitch_count_histogram.png)

### Feature Correlation Matrices

Initial correlation matrix showed confounding effects of mixing left-handed and right-handed pitchers. Separate correlation matrices by handedness revealed clearer relationships between features, with notable correlations between release_speed and plate_z, and spin_rate with spin_axis within handedness groups:

![Correlation Matrix by Handedness](images/correlation_matrix_handedness.png)

---

## Data Modeling Methods

### Model Architecture

We implemented a Multi-Layer Perceptron (MLP) neural network for pitcher classification:

**Network Structure:**
- Input layer: Variable size based on number of features (after one-hot encoding)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Dropout regularization: 0.3 dropout rate after each hidden layer
- Output layer: 110 neurons (one per pitcher) with softmax activation

**Training Configuration:**
- Loss function: Cross-entropy loss
- Optimizer: Adam with learning rate of 0.001
- Batch size: 64
- Training epochs: 50
- Data split: 80% training, 20% validation (stratified by pitcher)

**Implementation Details:**
- Used PyTorch framework
- Applied StandardScaler normalization to all numerical features
- Tracked training loss, validation loss, and validation accuracy across epochs

---

## Preliminary Results

### Classification Performance

The neural network achieved strong performance in pitcher identification:

**Overall Accuracy Metrics:**
- Top-1 Accuracy (Exact Match): 0.8307
- Top-3 Accuracy: 0.9607
- Top-5 Accuracy: 0.9847

These results indicate that the model correctly identifies the pitcher on the first guess 83% of the time, and includes the correct pitcher in the top 3 predictions 96% of the time.

### Per-Pitcher Analysis

Performance varied significantly across individual pitchers:
- Some pitchers achieved near-perfect classification accuracy
- Other pitchers showed poor accuracy, indicating less distinctive pitch characteristics or overlap with other pitchers' profiles

### Feature Importance Investigation

We applied SHAP (SHapley Additive exPlanations) analysis to pitchers with poor classification accuracy:
- Identified that the handedness encoding scheme was problematic
- The model appeared to learn the collinear relationship between pitcher_hand_L and pitcher_hand_R rather than learning meaningful patterns
- This confirmed our suspicion that having two binary features for a single binary characteristic creates redundancy

### Training Dynamics

Analysis of training and validation loss curves revealed:
- Both losses decreased steadily over epochs
- Validation loss tracked training loss closely, indicating minimal overfitting
- The model converged within 50 epochs

---

## Future Work

### Alternative Classification Methods

We plan to evaluate simpler classification approaches:
- Decision Trees
- Random Forests
- K-Nearest Neighbors
- XGBoost
- CatBoost
- Support Vector Machines

These methods may provide better interpretability and could reveal whether the neural network's complexity is necessary for this task.

### Unsupervised Learning Approach

We will explore unsupervised methods for anomaly detection:
- Isolation Forests for direct outlier detection
- Autoencoders to learn pitcher-specific representations
- Clustering methods to identify natural groupings of pitchers

### Feature Engineering Improvements

Based on preliminary results, we will:
- Revise handedness encoding to eliminate redundancy (use single binary feature or remove one)
- Experiment with derived features such as pitch movement magnitude
- Consider interaction terms between contextual and kinematic features

### Enhanced Evaluation

We plan to implement additional evaluation metrics:
- Confusion matrices to identify which pitchers are commonly confused
- Precision and recall per pitcher
- Analysis of misclassification patterns to identify pitcher similarities
- Top-k accuracy analysis for scouting comparisons

### Pitcher Comparison Pipeline

Once classification performance is optimized:
- Use prediction probabilities to measure similarity between pitchers
- Analyze confusion patterns to identify pitchers with similar repertoires
- Develop interpretable comparisons for scouting applications
- Create visualization tools for pitch-by-pitch comparison across pitchers

---

## Repository Contents

- `feature_exploration.ipynb`: Preliminary data visualization and exploration
- `MLP_feature_cleaning.ipynb`: Feature cleaning, Neural network implementation and training
- `README.md`: This midterm report  