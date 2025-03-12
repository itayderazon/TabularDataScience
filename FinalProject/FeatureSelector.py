import numpy as np
import pandas as pd
import warnings
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
warnings.filterwarnings('ignore')



class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A hybrid feature selection method combining tree-based feature importance and clustering.
    """
    
    def __init__(self, task_type='classification', n_clusters=None, importance_threshold=0.01):
        self.task_type = task_type
        self.n_clusters = n_clusters
        self.importance_threshold = importance_threshold
        
        # Initialize the appropriate RandomForest model
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.selected_features_ = None
        self.feature_importance_ = None
    
    def fit(self, X, y, min_threshold=0.5, max_threshold=0.95):

    # Compute feature importances
        self.model.fit(X, y)
        importance = self.model.feature_importances_
        self.feature_importance_ = pd.Series(importance, index=X.columns)
    
    # Select features above the importance threshold
        important_features = self.feature_importance_[self.feature_importance_ > self.importance_threshold]
    
    # If no important features, return empty selection
        if len(important_features) == 0:
            self.selected_features_ = []
            self.feature_clusters_ = pd.Series()
            return self
    
    # Compute correlation matrix for important features
        corr_matrix = X[important_features.index].corr().abs()
    
    # Get upper triangle of correlation matrix (excluding diagonal)
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append(corr_matrix.iloc[i, j])
    
    # Sort correlation values
        corr_values = sorted(corr_values)
    
    # Determine the adaptive threshold using multiple methods
        thresholds = []
    
    # Method 1: Gap statistic - find the largest gap
        if len(corr_values) > 1:
            gaps = [corr_values[i+1] - corr_values[i] for i in range(len(corr_values)-1)]
            max_gap_idx = np.argmax(gaps)
        # Only consider gaps in high correlation range (above 0.5)
            if corr_values[max_gap_idx] >= min_threshold:
                gap_threshold = (corr_values[max_gap_idx] + corr_values[max_gap_idx+1]) / 2
                thresholds.append(gap_threshold)
    
    # Method 2: Elbow method - find point of maximum curvature
        if len(corr_values) > 2:
        # Calculate curvature (approximate second derivative)
            curvature = []
            for i in range(1, len(corr_values)-1):
                y1, y2, y3 = corr_values[i-1], corr_values[i], corr_values[i+1]
                curvature.append(y1 + y3 - 2*y2)
        
        # Find maximum curvature point
            if curvature:
                max_curve_idx = np.argmax(curvature) + 1  # +1 because we start at index 1
                if corr_values[max_curve_idx] >= min_threshold:
                    elbow_threshold = corr_values[max_curve_idx]
                    thresholds.append(elbow_threshold)
    
    # Method 3: Use KDE to find the valley in distribution
        if len(corr_values) > 10:  # Need enough points for KDE
            try:

            
            # Only consider higher correlations for density estimation
                high_corrs = [c for c in corr_values if c >= min_threshold]
                if len(high_corrs) > 5:  # Need enough points
                    kde = gaussian_kde(high_corrs)
                    x = np.linspace(min_threshold, max_threshold, 100)
                    y = kde(x)
                
                # Find peaks (maxima)
                    peaks, _ = find_peaks(y)
                
                # Find valleys (minima)
                    valleys, _ = find_peaks(-y)
                
                    if len(valleys) > 0:
                    # Choose the most significant valley
                        valley_depths = [min(y[peaks[i]], y[peaks[i+1]]) - y[valleys[i]] 
                                        for i in range(len(valleys)) 
                                        if i < len(peaks)-1 and valleys[i] > peaks[i] and valleys[i] < peaks[i+1]]
                    
                        if valley_depths:
                            best_valley_idx = np.argmax(valley_depths)
                            kde_threshold = x[valleys[best_valley_idx]]
                            thresholds.append(kde_threshold)
            except (ImportError, ValueError) as e:
            # Skip this method if there's an error
                pass
    
    # Select final threshold
        if thresholds:
        # If multiple thresholds found, take the average
            adaptive_threshold = sum(thresholds) / len(thresholds)
        # Keep it within reasonable bounds
            adaptive_threshold = max(min(adaptive_threshold, max_threshold), min_threshold)
        else:
        # Fallback to a default if no threshold found
            adaptive_threshold = 0.85
    
        print(f"Adaptive correlation threshold: {adaptive_threshold:.4f}")
        self.correlation_threshold_ = adaptive_threshold
    
    # Helper function to create connected components based on high correlations
        def find_correlation_groups(corr_matrix, threshold):
        # Initialize each feature in its own group
            n_features = corr_matrix.shape[0]
            feature_names = corr_matrix.index
        
        # Use Union-Find data structure for efficient connected components
            parent = list(range(n_features))
        
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
        
            def union(x, y):
                parent[find(x)] = find(y)
        
        # For each pair of highly correlated features, merge their groups
            for i in range(n_features):
                for j in range(i+1, n_features):
                    if corr_matrix.iloc[i, j] >= threshold:
                        union(i, j)
        
        # Collect final groups
            groups = {}
            for i in range(n_features):
                root = find(i)
                if root not in groups:
                    groups[root] = []
                groups[root].append(feature_names[i])
        
            return list(groups.values())
    
    # Find correlation groups using the adaptive threshold
        correlation_groups = find_correlation_groups(corr_matrix, adaptive_threshold)
    
    # Rest of your existing code to handle the groups
    # Create cluster assignments
        cluster_id = 0
        feature_clusters = {}
        for group in correlation_groups:
            for feature in group:
                feature_clusters[feature] = cluster_id
            cluster_id += 1
    
    # Convert to Series for easier handling
        self.feature_clusters_ = pd.Series(feature_clusters)
    
    # Select highest importance feature from each group
        selected_features = []
        for group in correlation_groups:
            best_feature = important_features[group].idxmax()
            selected_features.append(best_feature)
    
    # simply take the top n_clusters by importance
        if self.n_clusters is not None and len(selected_features) > self.n_clusters:
        # Sort by importance and take top n_clusters
            selected_features = [f for f in important_features.sort_values(ascending=False).index 
                                if f in selected_features][:self.n_clusters]
    
        self.selected_features_ = selected_features
        return self
    
    def transform(self, X):
        """
        Transform X to retain only the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Call 'fit' before calling transform.")
        return X[self.selected_features_]
    
    def fit_transform(self, X, y):
        """
        Fit to the data, then transform it.
        """
        return self.fit(X, y).transform(X)
