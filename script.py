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
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import mannwhitneyu, ttest_ind
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
    return np.abs(corr_matrix)

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
        'n_estimators': [100],
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

# Helper function to flatten DataFrame
def _flatten_df(df_input):
    df = copy.deepcopy(df_input)
    data = df.values
    np.fill_diagonal(data, np.nan)
    df = pd.DataFrame(data)
    flattened_df = df.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    flattened_df.rename(columns={'index': 'feature_1', 'Column': 'feature_2'}, inplace=True)
    flattened_df = flattened_df.dropna(axis=0)
    return flattened_df

# Helper function to get statistics of independent features
def _get_statistic_of_independent_features(df_input):
    df = copy.deepcopy(df_input)
    df = df[(df['feature_1'] < 7) | (df['feature_2'] < 7)]
    return df["Value"].mean(), df["Value"].min(), df["Value"].max()

# Helper function to get statistics of dependent features
def _get_statistic_of_dependent_features(df_input):
    df = copy.deepcopy(df_input)
    df = df[df['feature_1'] >= 7]
    df = df[df['feature_2'] >= 7]
    return df["Value"].mean(), df["Value"].min(), df["Value"].max()

# Helper function to get top 3 features from competitors
def _get_top3_features_competitors(df_input):
    df = copy.deepcopy(df_input)
    df = df.sort_values("Value", ascending=True)
    df = df.tail(6)
    result = df["feature_1"].values.tolist() + df["feature_2"].values.tolist()
    result = sorted(list(set(result)))
    return result

################################################################################
# Main code to generate statistics
################################################################################

statistics = []
n=3
for strength_of_noise in [0.01, 10]: #[0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 1, 2, 10]:
    for iteration in range(n):
        start_time = datetime.datetime.now()
        dataset = generate_dataset(strength_of_noise=strength_of_noise, random_seed=42+iteration)
        competitor_results = competitor_methods(dataset)

        # Calculate statistics for each competitor method
        for alias in ["Pearson Correlation", "Mutual Information", "Conditional Entropy", "Distance Correlation"]:
            e = {"strength_of_noise": strength_of_noise}
            df = pd.DataFrame(competitor_results[alias])
            if alias == "Pearson Correlation":
                df = df.abs()
            df = _flatten_df(df)
            e["iteration"] = iteration
            e["Method"] = alias.replace(" ", "_")
            e["avg_independent_features_statistics"], e["min_independent_features_statistics"], e["max_independent_features_statistics"] = _get_statistic_of_independent_features(df)
            e["avg_dependent_features_statistics"], e["min_dependent_features_statistics"], e["max_dependent_features_statistics"] = _get_statistic_of_dependent_features(df)
            e["top3_selected_features"] = _get_top3_features_competitors(df)
            statistics.append(e)
        
        # Calculate statistics for the proposed method
        accuracy, feature_importances, best_rf = proposed_method(dataset, random_seed=42)
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
        e["iteration"] = iteration
        e["Method"] = "Proposed_Method"
        e["avg_independent_features_statistics"] = proposed_method_independent_features_statistics_avg
        e["avg_dependent_features_statistics"] = proposed_method_dependent_features_statistics_avg
        e["min_independent_features_statistics"] = proposed_method_independent_features_statistics_min
        e["min_dependent_features_statistics"] = proposed_method_dependent_features_statistics_min
        e["max_independent_features_statistics"] = proposed_method_independent_features_statistics_max
        e["max_dependent_features_statistics"] = proposed_method_dependent_features_statistics_max
        e["top3_selected_features"] = proposed_method_top3_selected_features
        statistics.append(e)

        finish_time = datetime.datetime.now()
        print(f"Time taken: {finish_time - start_time}")

# Convert statistics to DataFrame and calculate additional statistics
statistics_df = pd.DataFrame(statistics)
statistics_df["independent_vs_dependent_avg_statistics_diff_%"] = (statistics_df["avg_dependent_features_statistics"] - statistics_df["avg_independent_features_statistics"]) / statistics_df["avg_independent_features_statistics"] *100
statistics_df["independent_vs_dependent_avg_statistics_diff_%"] = statistics_df["independent_vs_dependent_avg_statistics_diff_%"].round(2)
print()
print("========================================")
print(f"Statistics")
print("========================================")
print(statistics_df.to_string())
print()


################################################################################
# Perform statistical testing to compare independent and dependent feature statistics with t-test
################################################################################
comparison_results_ttest = []

for strength_of_noise in statistics_df["strength_of_noise"].unique():
    for method in statistics_df["Method"].unique():
        subset = statistics_df[(statistics_df["strength_of_noise"] == strength_of_noise) & (statistics_df["Method"] == method)]
        
        for ind_stat, dep_stat in [("min", "min"),
                                   ("min", "avg"),
                                   ("min", "max"),
                                   ("avg", "min"),
                                   ("avg", "avg"),
                                   ("avg", "max"),
                                   ("max", "min"),
                                   ("max", "avg"),
                                   ("max", "max")]:
            
            ind_values = subset[f"{ind_stat}_independent_features_statistics"].replace(0, np.nan).dropna()
            dep_values = subset[f"{dep_stat}_dependent_features_statistics"].replace(0, np.nan).dropna()
            
            if not ind_values.empty and not dep_values.empty:
                # Apply log transformation
                log_ind_values = np.log(ind_values)
                log_dep_values = np.log(dep_values)
                
                t_stat, p_value = stats.ttest_ind(log_ind_values, log_dep_values, alternative='less')
                avg_ind_stat_value = ind_values.mean()
                avg_dep_stat_value = dep_values.mean()
                comparison_results_ttest.append({
                    "strength_of_noise": strength_of_noise,
                    "Method": method,
                    "ind_stat": ind_stat,
                    "dep_stat": dep_stat,
                    #"t_stat": t_stat,
                    "p_value": p_value,
                    "avg_ind_stat_value": avg_ind_stat_value,
                    "avg_dep_stat_value": avg_dep_stat_value
                })

comparison_df_ttest = pd.DataFrame(comparison_results_ttest)
comparison_df_ttest = comparison_df_ttest.sort_values(by=["Method", "strength_of_noise", "ind_stat", "dep_stat"])
print()
print("========================================")
print(f"t-test; Comparison Results - Independent vs Dependent Feature Statistics")
print("========================================")
print(comparison_df_ttest.to_string())
print()
print("========================================")
print(f"t-test; Comparison Results - Independent vs Dependent Feature Statistics SHORT")
print("========================================")
print(comparison_df_ttest[(comparison_df_ttest["ind_stat"] == "max") & (comparison_df_ttest["dep_stat"] == "min")].to_string())
print()

################################################################################
# Perform statistical testing to compare independent and dependent feature statistics with Mann-Whitney U test
################################################################################
comparison_results_mannwhitneyu = []

for strength_of_noise in statistics_df["strength_of_noise"].unique():
    for method in statistics_df["Method"].unique():
        subset = statistics_df[(statistics_df["strength_of_noise"] == strength_of_noise) & (statistics_df["Method"] == method)]
        
        for ind_stat, dep_stat in [("min", "min"),
                                   ("min", "avg"),
                                   ("min", "max"),
                                   ("avg", "min"),
                                   ("avg", "avg"),
                                   ("avg", "max"),
                                   ("max", "min"),
                                   ("max", "avg"),
                                   ("max", "max")]:
            
            ind_values = subset[f"{ind_stat}_independent_features_statistics"].replace(0, np.nan).dropna()
            dep_values = subset[f"{dep_stat}_dependent_features_statistics"].replace(0, np.nan).dropna()
            
            if not ind_values.empty and not dep_values.empty:
                u_stat, p_value = stats.mannwhitneyu(ind_values, dep_values, alternative='less')
                avg_ind_stat_value = ind_values.mean()
                avg_dep_stat_value = dep_values.mean()
                comparison_results_mannwhitneyu.append({
                    "strength_of_noise": strength_of_noise,
                    "Method": method,
                    "ind_stat": ind_stat,
                    "dep_stat": dep_stat,
                    #"u_stat": u_stat,
                    "p_value": p_value,
                    "avg_ind_stat_value": avg_ind_stat_value,
                    "avg_dep_stat_value": avg_dep_stat_value
                })

comparison_df_mannwhitneyu = pd.DataFrame(comparison_results_mannwhitneyu)
comparison_df_mannwhitneyu = comparison_df_mannwhitneyu.sort_values(by=["Method", "strength_of_noise", "ind_stat", "dep_stat"])
print()
print("========================================")
print(f"Mann-Whitney U test; Comparison Results - Independent vs Dependent Feature Statistics")
print("========================================")
print(comparison_df_mannwhitneyu.to_string())
print()
print("========================================")
print(f"Mann-Whitney U test; Comparison Results - Independent vs Dependent Feature Statistics SHORT")
print("========================================")
print(comparison_df_mannwhitneyu[(comparison_df_mannwhitneyu["ind_stat"] == "max") & (comparison_df_mannwhitneyu["dep_stat"] == "min")].to_string())
print()


################################################################################
# Lognormal test results of statistics distribution and confidence intervals calculation
################################################################################
lognormal_results = []

for strength_of_noise in statistics_df["strength_of_noise"].unique():
    for method in statistics_df["Method"].unique():
        subset = statistics_df[(statistics_df["strength_of_noise"] == strength_of_noise) & (statistics_df["Method"] == method)]
        
        # Check if the distribution of dependent statistics is lognormal
        dependent_stats_avg = subset["avg_dependent_features_statistics"].replace(0, np.nan).dropna()
        dependent_stats_min = subset["min_dependent_features_statistics"].replace(0, np.nan).dropna()
        dependent_stats_max = subset["max_dependent_features_statistics"].replace(0, np.nan).dropna()
        
        # Check if the distribution of independent statistics is lognormal
        independent_stats_avg = subset["avg_independent_features_statistics"].replace(0, np.nan).dropna()
        independent_stats_min = subset["min_independent_features_statistics"].replace(0, np.nan).dropna()
        independent_stats_max = subset["max_independent_features_statistics"].replace(0, np.nan).dropna()
        
        # Fit lognormal distribution and perform KS test for average statistics
        if not dependent_stats_avg.empty:
            shape_avg, loc_avg, scale_avg = stats.lognorm.fit(dependent_stats_avg, floc=0)
            kstest_result_avg = stats.kstest(dependent_stats_avg, 'lognorm', args=(shape_avg, loc_avg, scale_avg))
            ci_avg = stats.lognorm.interval(0.95, shape_avg, loc=loc_avg, scale=scale_avg)
        else:
            kstest_result_avg = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_avg = (0, 0)
        
        if not independent_stats_avg.empty:
            shape_ind_avg, loc_ind_avg, scale_ind_avg = stats.lognorm.fit(independent_stats_avg, floc=0)
            kstest_result_ind_avg = stats.kstest(independent_stats_avg, 'lognorm', args=(shape_ind_avg, loc_ind_avg, scale_ind_avg))
            ci_ind_avg = stats.lognorm.interval(0.95, shape_ind_avg, loc=loc_ind_avg, scale=scale_ind_avg)
        else:
            kstest_result_ind_avg = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_ind_avg = (0, 0)
        
        # Fit lognormal distribution and perform KS test for min statistics
        if not dependent_stats_min.empty:
            shape_min, loc_min, scale_min = stats.lognorm.fit(dependent_stats_min, floc=0)
            kstest_result_min = stats.kstest(dependent_stats_min, 'lognorm', args=(shape_min, loc_min, scale_min))
            ci_min = stats.lognorm.interval(0.95, shape_min, loc=loc_min, scale=scale_min)
        else:
            kstest_result_min = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_min = (0, 0)
        
        if not independent_stats_min.empty:
            shape_ind_min, loc_ind_min, scale_ind_min = stats.lognorm.fit(independent_stats_min, floc=0)
            kstest_result_ind_min = stats.kstest(independent_stats_min, 'lognorm', args=(shape_ind_min, loc_ind_min, scale_ind_min))
            ci_ind_min = stats.lognorm.interval(0.95, shape_ind_min, loc=loc_ind_min, scale=scale_ind_min)
        else:
            kstest_result_ind_min = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_ind_min = (0, 0)
        
        # Fit lognormal distribution and perform KS test for max statistics
        if not dependent_stats_max.empty:
            shape_max, loc_max, scale_max = stats.lognorm.fit(dependent_stats_max, floc=0)
            kstest_result_max = stats.kstest(dependent_stats_max, 'lognorm', args=(shape_max, loc_max, scale_max))
            ci_max = stats.lognorm.interval(0.95, shape_max, loc=loc_max, scale=scale_max)
        else:
            kstest_result_max = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_max = (0, 0)
        
        if not independent_stats_max.empty:
            shape_ind_max, loc_ind_max, scale_ind_max = stats.lognorm.fit(independent_stats_max, floc=0)
            kstest_result_ind_max = stats.kstest(independent_stats_max, 'lognorm', args=(shape_ind_max, loc_ind_max, scale_ind_max))
            ci_ind_max = stats.lognorm.interval(0.95, shape_ind_max, loc=loc_ind_max, scale=scale_ind_max)
        else:
            kstest_result_ind_max = stats.kstest([0], 'lognorm', args=(0, 0, 1))  # Default result for empty data
            ci_ind_max = (0, 0)
        
        lognormal_results.append({
            "strength_of_noise": strength_of_noise,
            "Method": method,
            "dependent_avg_statistic": kstest_result_avg.statistic,
            "dependent_avg_pvalue": kstest_result_avg.pvalue,
            "dependent_avg_is_lognormal": kstest_result_avg.pvalue > 0.05,
            "dependent_avg_shape": shape_avg,
            "dependent_avg_loc": loc_avg,
            "dependent_avg_scale": scale_avg,
            "dependent_avg_ci_lower": ci_avg[0],
            "dependent_avg_ci_upper": ci_avg[1],
            "dependent_min_statistic": kstest_result_min.statistic,
            "dependent_min_pvalue": kstest_result_min.pvalue,
            "dependent_min_is_lognormal": kstest_result_min.pvalue > 0.05,
            "dependent_min_shape": shape_min,
            "dependent_min_loc": loc_min,
            "dependent_min_scale": scale_min,
            "dependent_min_ci_lower": ci_min[0],
            "dependent_min_ci_upper": ci_min[1],
            "dependent_max_statistic": kstest_result_max.statistic,
            "dependent_max_pvalue": kstest_result_max.pvalue,
            "dependent_max_is_lognormal": kstest_result_max.pvalue > 0.05,
            "dependent_max_shape": shape_max,
            "dependent_max_loc": loc_max,
            "dependent_max_scale": scale_max,
            "dependent_max_ci_lower": ci_max[0],
            "dependent_max_ci_upper": ci_max[1],
            "ind_avg_statistic": kstest_result_ind_avg.statistic,
            "ind_avg_pvalue": kstest_result_ind_avg.pvalue,
            "ind_avg_is_lognormal": kstest_result_ind_avg.pvalue > 0.05,
            "ind_avg_shape": shape_ind_avg,
            "ind_avg_loc": loc_ind_avg,
            "ind_avg_scale": scale_ind_avg,
            "ind_avg_ci_lower": ci_ind_avg[0],
            "ind_avg_ci_upper": ci_ind_avg[1],
            "ind_min_statistic": kstest_result_ind_min.statistic,
            "ind_min_pvalue": kstest_result_ind_min.pvalue,
            "ind_min_is_lognormal": kstest_result_ind_min.pvalue > 0.05,
            "ind_min_shape": shape_ind_min,
            "ind_min_loc": loc_ind_min,
            "ind_min_scale": scale_ind_min,
            "ind_min_ci_lower": ci_ind_min[0],
            "ind_min_ci_upper": ci_ind_min[1],
            "ind_max_statistic": kstest_result_ind_max.statistic,
            "ind_max_pvalue": kstest_result_ind_max.pvalue,
            "ind_max_is_lognormal": kstest_result_ind_max.pvalue > 0.05,
            "ind_max_shape": shape_ind_max,
            "ind_max_loc": loc_ind_max,
            "ind_max_scale": scale_ind_max,
            "ind_max_ci_lower": ci_ind_max[0],
            "ind_max_ci_upper": ci_ind_max[1]
        })
        
        # Plot QQ-plots and scatter plots for each statistic
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        # QQ plots for dependent statistics
        stats.probplot(dependent_stats_avg, dist="lognorm", sparams=(shape_avg, loc_avg, scale_avg), plot=axes[0, 0])
        axes[0, 0].set_title(f'QQ Plot - Avg Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        stats.probplot(dependent_stats_min, dist="lognorm", sparams=(shape_min, loc_min, scale_min), plot=axes[0, 1])
        axes[0, 1].set_title(f'QQ Plot - Min Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        stats.probplot(dependent_stats_max, dist="lognorm", sparams=(shape_max, loc_max, scale_max), plot=axes[0, 2])
        axes[0, 2].set_title(f'QQ Plot - Max Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        # QQ plots for independent statistics
        stats.probplot(independent_stats_avg, dist="lognorm", sparams=(shape_ind_avg, loc_ind_avg, scale_ind_avg), plot=axes[1, 0])
        axes[1, 0].set_title(f'QQ Plot - Avg Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        stats.probplot(independent_stats_min, dist="lognorm", sparams=(shape_ind_min, loc_ind_min, scale_ind_min), plot=axes[1, 1])
        axes[1, 1].set_title(f'QQ Plot - Min Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        stats.probplot(independent_stats_max, dist="lognorm", sparams=(shape_ind_max, loc_ind_max, scale_ind_max), plot=axes[1, 2])
        axes[1, 2].set_title(f'QQ Plot - Max Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        # Scatter plots with lognormal distribution for dependent statistics
        x_avg = np.linspace(dependent_stats_avg.min(), dependent_stats_avg.max(), 100)
        axes[2, 0].scatter(dependent_stats_avg, np.zeros_like(dependent_stats_avg), alpha=0.5)
        axes[2, 0].plot(x_avg, stats.lognorm.pdf(x_avg, shape_avg, loc=loc_avg, scale=scale_avg), 'r-', lw=2)
        axes[2, 0].set_title(f'Scatter Plot - Avg Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        x_min = np.linspace(dependent_stats_min.min(), dependent_stats_min.max(), 100)
        axes[2, 1].scatter(dependent_stats_min, np.zeros_like(dependent_stats_min), alpha=0.5)
        axes[2, 1].plot(x_min, stats.lognorm.pdf(x_min, shape_min, loc=loc_min, scale=scale_min), 'r-', lw=2)
        axes[2, 1].set_title(f'Scatter Plot - Min Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        x_max = np.linspace(dependent_stats_max.min(), dependent_stats_max.max(), 100)
        axes[2, 2].scatter(dependent_stats_max, np.zeros_like(dependent_stats_max), alpha=0.5)
        axes[2, 2].plot(x_max, stats.lognorm.pdf(x_max, shape_max, loc=loc_max, scale=scale_max), 'r-', lw=2)
        axes[2, 2].set_title(f'Scatter Plot - Max Dependent Stats\n{method} - Noise: {strength_of_noise}')
        
        # Scatter plots with lognormal distribution for independent statistics
        x_ind_avg = np.linspace(independent_stats_avg.min(), independent_stats_avg.max(), 100)
        axes[3, 0].scatter(independent_stats_avg, np.zeros_like(independent_stats_avg), alpha=0.5)
        axes[3, 0].plot(x_ind_avg, stats.lognorm.pdf(x_ind_avg, shape_ind_avg, loc=loc_ind_avg, scale=scale_ind_avg), 'r-', lw=2)
        axes[3, 0].set_title(f'Scatter Plot - Avg Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        x_ind_min = np.linspace(independent_stats_min.min(), independent_stats_min.max(), 100)
        axes[3, 1].scatter(independent_stats_min, np.zeros_like(independent_stats_min), alpha=0.5)
        axes[3, 1].plot(x_ind_min, stats.lognorm.pdf(x_ind_min, shape_ind_min, loc=loc_ind_min, scale=scale_ind_min), 'r-', lw=2)
        axes[3, 1].set_title(f'Scatter Plot - Min Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        x_ind_max = np.linspace(independent_stats_max.min(), independent_stats_max.max(), 100)
        axes[3, 2].scatter(independent_stats_max, np.zeros_like(independent_stats_max), alpha=0.5)
        axes[3, 2].plot(x_ind_max, stats.lognorm.pdf(x_ind_max, shape_ind_max, loc=loc_ind_max, scale=scale_ind_max), 'r-', lw=2)
        axes[3, 2].set_title(f'Scatter Plot - Max Independent Stats\n{method} - Noise: {strength_of_noise}')
        
        plt.tight_layout()
        plt.show()

lognormal_df = pd.DataFrame(lognormal_results)

# Split into six tables using .loc to avoid SettingWithCopyWarning
avg_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'dependent_avg_statistic', 'dependent_avg_pvalue', 'dependent_avg_is_lognormal', 'dependent_avg_shape', 'dependent_avg_loc', 'dependent_avg_scale', 'dependent_avg_ci_lower', 'dependent_avg_ci_upper']].copy()
min_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'dependent_min_statistic', 'dependent_min_pvalue', 'dependent_min_is_lognormal', 'dependent_min_shape', 'dependent_min_loc', 'dependent_min_scale', 'dependent_min_ci_lower', 'dependent_min_ci_upper']].copy()
max_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'dependent_max_statistic', 'dependent_max_pvalue', 'dependent_max_is_lognormal', 'dependent_max_shape', 'dependent_max_loc', 'dependent_max_scale', 'dependent_max_ci_lower', 'dependent_max_ci_upper']].copy()
ind_avg_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'ind_avg_statistic', 'ind_avg_pvalue', 'ind_avg_is_lognormal', 'ind_avg_shape', 'ind_avg_loc', 'ind_avg_scale', 'ind_avg_ci_lower', 'ind_avg_ci_upper']].copy()
ind_min_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'ind_min_statistic', 'ind_min_pvalue', 'ind_min_is_lognormal', 'ind_min_shape', 'ind_min_loc', 'ind_min_scale', 'ind_min_ci_lower', 'ind_min_ci_upper']].copy()
ind_max_df = lognormal_df.loc[:, ['strength_of_noise', 'Method', 'ind_max_statistic', 'ind_max_pvalue', 'ind_max_is_lognormal', 'ind_max_shape', 'ind_max_loc', 'ind_max_scale', 'ind_max_ci_lower', 'ind_max_ci_upper']].copy()

# Rename columns for clarity
avg_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']
min_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']
max_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']
ind_avg_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']
ind_min_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']
ind_max_df.columns = ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'shape', 'loc', 'scale', 'ci_lower', 'ci_upper']

# Add a new column to identify the type
avg_df.loc[:, 'type'] = 'avg_dependent'
min_df.loc[:, 'type'] = 'min_dependent'
max_df.loc[:, 'type'] = 'max_dependent'
ind_avg_df.loc[:, 'type'] = 'avg_independent'
ind_min_df.loc[:, 'type'] = 'min_independent'
ind_max_df.loc[:, 'type'] = 'max_independent'

# Split the tables into three
stat_pvalue_df = pd.concat([avg_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']],
                            min_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']],
                            max_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']],
                            ind_avg_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']],
                            ind_min_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']],
                            ind_max_df.loc[:, ['strength_of_noise', 'Method', 'statistic', 'pvalue', 'is_lognormal', 'type']]])
stat_pvalue_df = stat_pvalue_df.sort_values(by=["Method", "strength_of_noise", "type"])

shape_loc_scale_df = pd.concat([avg_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']],
                                min_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']],
                                max_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']],
                                ind_avg_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']],
                                ind_min_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']],
                                ind_max_df.loc[:, ['strength_of_noise', 'Method', 'shape', 'loc', 'scale', 'type']]])
shape_loc_scale_df = shape_loc_scale_df.sort_values(by=["Method", "strength_of_noise", "type"])
ci_df = pd.concat([avg_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']],
                   min_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']],
                   max_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']],
                   ind_avg_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']],
                   ind_min_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']],
                   ind_max_df.loc[:, ['strength_of_noise', 'Method', 'ci_lower', 'ci_upper', 'type']]])
ci_df = ci_df.sort_values(by=["Method", "strength_of_noise", "type"])
print()
print("========================================")
print(f"Lognormal Test Results - Statistics and P-values")
print("========================================")
print(stat_pvalue_df.to_string())
print()
print("========================================")
print(f"Lognormal Test Results - Shape, Loc, and Scale")
print("========================================")
print(shape_loc_scale_df.to_string())
print()
print("========================================")
print(f"Lognormal Test Results - Confidence Intervals")
print("========================================")
print(ci_df.to_string())
print()
print("========================================")
print(f"Lognormal Test Results - Confidence Intervals SHORT")
print("========================================")
print(ci_df[(ci_df["type"] == "min_dependent") | (ci_df["type"] == "max_independent")].to_string())
print()


################################################################################
# Create a figure for each method and strength of noise for comaping confidence intervals
################################################################################
methods = ci_df['Method'].unique()
strengths_of_noise = ci_df['strength_of_noise'].unique()

for method in methods:
    for strength in strengths_of_noise:
        method_strength_df = ci_df[(ci_df['Method'] == method) & (ci_df['strength_of_noise'] == strength)]
        
        if not method_strength_df.empty:
            # Original charts for avg, min, max
            fig, ax = plt.subplots(figsize=(10, 6))
            for dep_type in ['avg', 'min', 'max']:
                dep_df = method_strength_df[method_strength_df['type'].str.contains(dep_type)]
                if not dep_df.empty:
                    ax.errorbar(dep_df['type'], dep_df['ci_lower'], 
                                yerr=(dep_df['ci_upper'] - dep_df['ci_lower']) / 2, 
                                fmt='o', label=f'{method} - {strength} - {dep_type}')
            
            ax.set_xlabel('Type (avg, min, max)')
            ax.set_ylabel('Confidence Interval')
            ax.set_title(f'Confidence Intervals for {method} at Noise Strength {strength}')
            ax.legend()
            ax.grid(True)
            plt.show()
            
            # Additional charts for max_independent and min_dependent
            fig, ax = plt.subplots(figsize=(10, 6))
            for dep_type in ['max_independent', 'min_dependent']:
                dep_df = method_strength_df[method_strength_df['type'] == dep_type]
                if not dep_df.empty:
                    ax.errorbar(dep_df['type'], dep_df['ci_lower'], 
                                yerr=(dep_df['ci_upper'] - dep_df['ci_lower']) / 2, 
                                fmt='o', label=f'{method} - {strength} - {dep_type}')
            
            ax.set_xlabel('Type (max_independent, min_dependent)')
            ax.set_ylabel('Confidence Interval')
            ax.set_title(f'Confidence Intervals for {method} at Noise Strength {strength} (Additional)')
            ax.legend()
            ax.grid(True)
            plt.show()


