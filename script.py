import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
import dcor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import copy

import warnings

# Suppress all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Function to generate the dataset with a fixed random seed
def generate_dataset(strength_of_noise, n_samples=10000, random_seed=None):
    # Fix the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate 7 independent noise features from standard normal distribution
    independent_features = np.random.normal(0, 1, (n_samples, 7))
    
    # Generate the two independent dependent features: 'local trigger' and 'slope'
    local_trigger = np.random.normal(0, 1, n_samples)
    slope = np.random.normal(0, 1, n_samples)
    
    # Generate the third dependent feature based on 'local trigger' and 'slope'
    third_feature = np.where(local_trigger < 0, 
                             np.random.normal(0, 1, n_samples), 
                             slope * 2 + strength_of_noise * np.random.normal(0, 1, n_samples))
    
    # Normalize the third feature (subtract mean and divide by standard deviation)
    third_feature = (third_feature - np.mean(third_feature)) / np.std(third_feature)
    
    # Stack all features together (7 independent + 3 dependent)
    dependent_features = np.column_stack((local_trigger, slope, third_feature))
    dataset = np.column_stack((independent_features, dependent_features))
    
    return dataset

# Example of generating the dataset with a specific strength of noise and random seed
# dataset = generate_dataset(strength_of_noise=1.0, random_seed=42)
# print(dataset.shape)  # Output: (10000, 10)


# Function to calculate Pearson correlation
def pearson_correlation(dataset):
    corr_matrix = np.corrcoef(dataset, rowvar=False)
    return corr_matrix


# Function to calculate Mutual Information between pairs of features
def mutual_information(dataset):
    n_features = dataset.shape[1]
    mi_scores = np.zeros((n_features, n_features))
    
    # Calculate mutual information for each pair of features
    for i in range(n_features):
        for j in range(i+1, n_features):
            mi_scores[i, j] = mutual_info_regression(dataset[:, i].reshape(-1, 1), dataset[:, j])
            mi_scores[j, i] = mi_scores[i, j]  # Symmetry
    return mi_scores


# Function to calculate Conditional Entropy (using discretization) between pairs of features
def conditional_entropy(dataset, n_bins=10):
    n_features = dataset.shape[1]
    ce_scores = np.zeros((n_features, n_features))
    
    # Discretize the features
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized_data = discretizer.fit_transform(dataset)
    
    # Calculate conditional entropy for each pair of features
    for i in range(n_features):
        for j in range(i+1, n_features):
            ce_scores[i, j] = mutual_info_score(discretized_data[:, i], discretized_data[:, j])
            ce_scores[j, i] = ce_scores[i, j]  # Symmetry
    return ce_scores


# Function to calculate Correlation of Distances (using dcor) between pairs of features
def correlation_of_distances(dataset):
    n_features = dataset.shape[1]
    dcor_scores = np.zeros((n_features, n_features))
    
    # Calculate distance correlation for each pair of features
    for i in range(n_features):
        for j in range(i+1, n_features):
            dcor_scores[i, j] = dcor.distance_correlation(dataset[:, i], dataset[:, j])
            dcor_scores[j, i] = dcor_scores[i, j]  # Symmetry
    return dcor_scores


# Function to run competitor methods on the dataset
def competitor_methods(dataset):
    pearson_corr = pearson_correlation(dataset)
    mi_scores = mutual_information(dataset)
    ce_scores = conditional_entropy(dataset)
    dcor_scores = correlation_of_distances(dataset)
    
    return {
        "Pearson Correlation": pearson_corr,
        "Mutual Information": mi_scores,
        "Conditional Entropy": ce_scores,
        "Distance Correlation": dcor_scores
    }


# Function to implement the proposed method with empirical distribution-based synthetic data generation
def proposed_method(dataset, random_seed=None):
    # Fix the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate synthetic dataset by sampling from the empirical distribution of each feature in the original dataset
    synthetic_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[1]):
        synthetic_dataset[:, i] = np.random.choice(dataset[:, i], size=dataset.shape[0], replace=True)
    
    # Create labels for original (1) and synthetic (0) data
    labels = np.concatenate([np.ones(dataset.shape[0]), np.zeros(synthetic_dataset.shape[0])])
    
    # Combine original and synthetic datasets
    combined_data = np.vstack([dataset, synthetic_dataset])
    
    # Define the Random Forest classifier and hyperparameter grid, including random seed for the classifier
    rf = RandomForestClassifier(random_state=random_seed)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(combined_data, labels)
    
    # Get the best model from the grid search
    best_rf = grid_search.best_estimator_
    
    # Predict on the training data
    predictions = best_rf.predict(combined_data)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Get feature importances
    feature_importances = best_rf.feature_importances_
    
    return accuracy, feature_importances, best_rf


def _flatten_df(df_input):
    df = copy.deepcopy(df_input)
    data = df.values
    np.fill_diagonal(data, np.nan)
    df = pd.DataFrame(data)
    flattened_df = df.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    flattened_df.rename(columns={'index': 'feature_1', 'Column': 'feature_2'}, inplace=True)
    flattened_df = flattened_df.dropna(axis=0)
    return flattened_df


def _get_statistic_of_independent_features(df_input):
    df = copy.deepcopy(df_input)
    df = df[df['feature_1'] < 7]
    df = df[df['feature_2'] < 7]
    return df["Value"].mean(), df["Value"].min(), df["Value"].max()


def _get_statistic_of_dependent_features(df_input):
    df = copy.deepcopy(df_input)
    df = df[df['feature_1'] >= 7]
    df = df[df['feature_2'] >= 7]
    return df["Value"].mean(), df["Value"].min(), df["Value"].max()


def _get_top3_features_competitors(df_input):
    df = copy.deepcopy(df_input)
    df = df.sort_values("Value", ascending=True)
    df = df.tail(6)
    result = df["feature_1"].values.tolist() + df["feature_2"].values.tolist()
    result = sorted(list(set(result)))
    return result


statistics = []
for strength_of_noise in [0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 1, 2, 10]:
    dataset = generate_dataset(strength_of_noise=strength_of_noise, random_seed=42)
    competitor_results = competitor_methods(dataset)

    print("========================================")
    print(f"strength_of_noise: {strength_of_noise}")
    print("========================================")

    print("Pearson_Correlation:")
    print(pd.DataFrame(competitor_results["Pearson Correlation"]).abs().to_string())
    print()
    print("Mutual_Information:")
    print(pd.DataFrame(competitor_results["Mutual Information"]).to_string())
    print()
    print("Conditional_Entropy:")
    print(pd.DataFrame(competitor_results["Conditional Entropy"]).to_string())
    print()
    print("Distance_Correlation:")
    print(pd.DataFrame(competitor_results["Distance Correlation"]).to_string())
    print()

    accuracy, feature_importances, best_rf = proposed_method(dataset, random_seed=42)
    print("The Proposed Method:")
    print("Accuracy:", accuracy)
    print("Feature Importances:", pd.DataFrame(feature_importances).to_string())

    for alias in ["Pearson Correlation", "Mutual Information", "Conditional Entropy", "Distance Correlation"]:
        e = {"strength_of_noise": strength_of_noise}
        df = pd.DataFrame(competitor_results[alias])
        if alias == "Pearson Correlation":
            df = df.abs()
        df = _flatten_df(df)
        e["Method"] = alias.replace(" ", "_")
        e["avg_independent_features_statistics"], e["min_independent_features_statistics"], e["max_independent_features_statistics"] = _get_statistic_of_independent_features(df)
        e["avg_dependent_features_statistics"], e["min_dependent_features_statistics"], e["max_dependent_features_statistics"] = _get_statistic_of_dependent_features(df)
        e["top3_selected_features"] = _get_top3_features_competitors(df)
        statistics.append(e)
    
    df = pd.DataFrame(feature_importances, columns=["Value"])
    df = df.sort_values("Value", ascending=True)
    df = df.reset_index()
    proposed_method_independent_features_statistics_avg = df[df["index"] < 7]["Value"].mean()
    proposed_method_dependent_features_statistics_avg = df[df["index"] >= 7]["Value"].mean()
    proposed_method_independent_features_statistics_min = df[df["index"] < 7]["Value"].min()
    proposed_method_dependent_features_statistics_min = df[df["index"] >= 7]["Value"].min()
    proposed_method_independent_features_statistics_max = df[df["index"] < 7]["Value"].max()
    proposed_method_dependent_features_statistics_max = df[df["index"] >= 7]["Value"].max()
    proposed_method_top3_selected_features = str(sorted(df.tail(3)["index"].values.tolist()))
    e = {"strength_of_noise": strength_of_noise}
    e["Method"] = "Proposed_Method"
    e["avg_independent_features_statistics"] = proposed_method_independent_features_statistics_avg
    e["avg_dependent_features_statistics"] = proposed_method_dependent_features_statistics_avg
    e["min_independent_features_statistics"] = proposed_method_independent_features_statistics_min
    e["min_dependent_features_statistics"] = proposed_method_dependent_features_statistics_min
    e["max_independent_features_statistics"] = proposed_method_independent_features_statistics_max
    e["max_dependent_features_statistics"] = proposed_method_dependent_features_statistics_max
    e["top3_selected_features"] = proposed_method_top3_selected_features
    statistics.append(e)

statistics_df = pd.DataFrame(statistics)
statistics_df["independent_vs_dependent_avg_statistics_diff_%"] = (statistics_df["avg_dependent_features_statistics"] - statistics_df["avg_independent_features_statistics"]) / statistics_df["avg_independent_features_statistics"] *100
statistics_df["independent_vs_dependent_avg_statistics_diff_%"] = statistics_df["independent_vs_dependent_avg_statistics_diff_%"].round(2)
print()
print("========================================")
print(f"Statistics")
print("========================================")
print(statistics_df.to_string())
print()
