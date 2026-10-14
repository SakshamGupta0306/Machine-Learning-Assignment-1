import numpy as np
from common import load_data, stratified_split, get_accuracy

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.priors = None
        self.means = None
        self.vars = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        
        # Compute smoothing term: 1e-9 * max variance of features in training set
        epsilon = self.var_smoothing * np.max(np.var(X, axis=0))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = len(X_c) / len(X)
            self.means[idx, :] = np.mean(X_c, axis=0)
            self.vars[idx, :] = np.var(X_c, axis=0) + epsilon

    def predict(self, X):
        # Vectorized log-likelihood computation using broadcasting
        # X shape: (n_samples, n_features)
        # means/vars shape: (n_classes, n_features)
        
        # Reshape to (n_samples, n_classes, n_features)
        X_expanded = X[:, np.newaxis, :]
        means_expanded = self.means[np.newaxis, :, :]
        vars_expanded = self.vars[np.newaxis, :, :]
        
        # Log Gaussian PDF calculation
        log_priors = np.log(self.priors)
        log_probs = -0.5 * np.sum(
            np.log(2 * np.pi * vars_expanded) + ((X_expanded - means_expanded) ** 2) / vars_expanded, 
            axis=2
        )
        
        # Add log prior to log likelihoods
        total_log_prob = log_probs + log_priors
        
        # Return class with highest log probability
        return self.classes[np.argmax(total_log_prob, axis=1)]

def run_part5_sanity_checks():
    print("--- Part 5 Sanity Check ---")
    
    # 1D feature, class 0 {0,2}, class 1 {10,12}. x=3 -> class 0, x=9 -> class 1[cite: 8]
    X_s = np.array([[0], [2], [10], [12]])
    y_s = np.array([0, 0, 1, 1])
    
    gnb_check = GaussianNB(var_smoothing=1e-9)
    gnb_check.fit(X_s, y_s)
    
    X_test_s = np.array([[3], [9]])
    preds = gnb_check.predict(X_test_s)
    
    print(f"Prediction for x=3: {preds[0]} (Expected: 0)")
    print(f"Prediction for x=9: {preds[1]} (Expected: 1)\n")

if __name__ == "__main__":
    run_part5_sanity_checks()
    
    X, y_species, y_family, species_to_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    
    X_train, y_train = X[train_idx], y_species[train_idx]
    X_test, y_test = X[test_idx], y_species[test_idx]
    y_family_test = y_family[test_idx]

    print("=== Gaussian Naive Bayes ===")
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    
    train_acc = get_accuracy(y_train, gnb.predict(X_train)) * 100
    preds_test = gnb.predict(X_test)
    test_acc = get_accuracy(y_test, preds_test) * 100
    family_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in preds_test])) * 100
    
    print(f"  Train Acc: {train_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Derived Family Acc: {family_acc:.2f}%")
    
    # Correlation analysis for Part 5 Task 2
    # corr_matrix = np.corrcoef(X_train, rowvar=False)
    # avg_corr = np.mean(np.abs(corr_matrix) - np.eye(X_train.shape[1]))
    # print(f"\n  [Analysis] Average absolute feature correlation: {avg_corr:.3f}")
    
    # Correlation analysis for Part 5 Task 2
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    
    # Temporarily set numpy print options for clean matrix formatting
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print("\n  [Analysis] 22x22 Feature Correlation Matrix:")
    print(corr_matrix)
    
    # Reset print options back to default
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
    
    avg_corr = np.mean(np.abs(corr_matrix) - np.eye(X_train.shape[1]))
    print(f"\n  [Analysis] Average absolute feature correlation: {avg_corr:.3f}")