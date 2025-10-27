# Pitcher Anomaly Detection with Statcast Data

**(Edits to our project plan that reference the issue will be in bold text. Also, check note at bottom of readme)**

## Project Overview
We will build a model that, given a single pitch based on Statcast features and context, predicts which pitcher threw it. This will be done for **qualified** pitchers in the MLB (**who have thrown at least 162 innings per season**), and based on those pitches assigned to that pitcher, we can then identify “outlier” pitches. This will be based on how unusual that pitch was for that pitcher in the context of pitching metrics (release point, spin rate, …), as well as game context (current count, position of runners, …). We can assign an anomaly score to this and then flag suspicious, uncharacteristic pitches.  
Use case: Recently, multiple MLB pitchers have been investigated purposefully missing pitches in connection with gambling. Our model can help identify these anomaly pitches that help encourage further human review.

---

## Objectives/Goals
- Goal 1: Identify based on a given pitch, predict the pitcher.  
- Goal 2: For a specific pitcher, quantify how “off” the pitch was and give it an anomaly scoring metric based around the pitch/game context - (by classification we have kind of already down this. I.e. logistic regression assigns a probability).  

---

## Data Collection/Cleaning Process
- Gather data from MLB savant using pybaseball. Clean data to sort by pitcher We have access to many features for each pitch, like spin axis, count it was thrown in, release point, velocity, etc. All available features are at https://baseballsavant.mlb.com/csv-docs.  
- The source of the data will be from MLB Statcast via pybaseball (statcast_pitcher per pitcher) for the last decade.  
- Per each pitch we will keep kinematic data like ( velocity, spin rate/axis, movement, release position, vertical approach angle, acceleration, plate location). Also context data like pitch type, count, zone, batter, handedness, catcher, inning/outs/runners, score differential, and game IDs.  

As for the cleaning, we will drop corrupted rows and fill the few missing values within each pitcher season. Fortunately, each pitch already contains a field for the pitcher who threw it. Each pitch_type will be mapped to a binary feature (i.e. 1 if cutter, 0 otherwise). Rare pitches will be mapped to ‘other’.  
We then normalize the numerical pitch data so they are on comparable ranges.  

---

## Modeling Process
For our first task, classifying the pitch to the pitcher, we've attempted a neural network approach that tries to directly assign any given pitch to a pitcher. This approach may not be the best, as it lacks the ability to capture differences in pitch type by pitcher (for example, a fastball from one pitcher may have characteristics that resemble a slider in another pitcher). To remedy this, one option would be to make a classifier for each pitch type and have the models train on only pitches of that pitch type.

We could use one of two approaches, or both. We could use some form of unsupervised learning for pure outlier detection. The other approach is to create binary classifiers for each pitcher using XGBoost, CatBoost, logistic regression, etc., and then select the best model. We consider a pitch “suspect” if the model determines it is unlikely that the pitcher threw it.  

**Test Plan** - We will train on 80% of the pitching data for model tuning and validation then we will have 20% of the pitching data as a test. **We will only test on data that was trained on in the same year (because the pitching style of different pitchers can change from year to year, it would be inconsistent to compare data across years. We learned this from our visualization of the data as well as looking at data visualizations for different pitchers on Savant).** We will first classify the pitch to a certain pitcher and given the actual pitcher who threw it, we can model how atypical it is. Can model probability as a simple Gaussian (bell curve) function. If the pitch has an abnormally low Z score, then we can flag the pitch as atypical and warrant further review.  

---

## Visualization Process
We can do something similar to this by comparing the vector of weights for each pitcher. Helpful for identifying which pitchers overlap. We can use a heatmap to compare the classifiers as well to identify how successful the model is for different groups.
Tools such as SHAP will help us in identifying which features are most significant in determining a classification. We've implemented this in our feature_exploration notebook, with fast-access graphs available in the outputs folder.

---

## Test Plan & Evaluation Metrics (Will be modified as we walk through)

### Data Split
- Training set → Use to teach the model
- Validation set → use to tune settings
- Test set → check the performance of the trained machine

### Baseline
- Classification → always predict the majority classification
- Regression → always predict the average value.

### Classification model training
- Train XGBoost, CatBoost, logistic regression, etc.  
- Use the validation data to stop early if it’s not improving  
- Adjust the few knobs: max_depth, learning_rate, etc…  

### Final Check
- Pick the best model between all different kinds of classifiers.  
- Compare result with the baseline.  

### Evaluation Metrics (as we walk through)
- Recall/fb-score: **this is for evaluating the performance of our binary classifiers. We specifically look at recall because we want to avoid false negatives (i.e. a pitch being marked an outlier when it isn't). But fb score is still relevant if we want to know whether or not a false negative or false positive is more significant.**
- ROC curve/AUC: **This is also for evaluation of our binary classifiers, if we ever determine the false positive rate to be a significant issue**
- Silhouette Score: (**This could be used for our unsupervised learning approach**)

---

## Potential Risks
- There might not be enough data for each” person. Therefore, we will not be able to classify every person who plays. We will be able to classify people who have already “enough” data.  
- If the model is imbalanced, it might cause the high accuracy but catches 0 outliers(We will be able to find this out by other metrics)  
- High risk on failure of predicting the “specific” case.


## Note
- The reason we do not want to train on past seasons and test on new ones is due to inconsistencies with the data measured by year. For example, there may be a pitcher that passes our threshold for qualified pitcher in one season, but not the other. So we decided to treat each year as its own monolith, comparing only pitches that were thrown in the same year.
- We will elaborate on the features that influence specific pitch types more than others (for example, if spin axis is more likely to vary in a slider pitch than another feature)
