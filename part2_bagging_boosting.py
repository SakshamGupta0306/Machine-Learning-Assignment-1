import numpy as np
from common import load_data, stratified_split, get_accuracy, get_species_to_family
from part1_decision_tree import DecisionTree

class Bagging:
    def __init__(self, n_estimators=25, criterion="gini", seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        self.n_classes = np.max(y) + 1
        # One fresh RandomState per ensemble model
        rng = np.random.RandomState(self.seed)
        N = len(X)
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample of N rows with replacement
            indices = rng.randint(0, N, size=N)
            tree = DecisionTree(criterion=self.criterion)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_classes).argmax(), axis=0, arr=all_preds)

class AdaBoostSAMME:
    def __init__(self, n_estimators=50, criterion="gini", max_depth=3):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.models = [] # Stores tuples of (alpha, tree)
        self.K = None

    def fit(self, X, y):
        self.n_classes = np.max(y) + 1
        N = len(X)
        self.K = np.max(y) + 1
        w = np.ones(N) / N
        self.models = []

        for t in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion, max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)
            
            preds = tree.predict(X)
            
            # Weighted error
            misclassified = (preds != y)
            err = np.sum(w[misclassified])
            err = np.clip(err, 1e-10, 1 - 1e-10)
            
            if err >= 1.0 - (1.0 / self.K):
                break # Model is no better than random guessing
                
            # Amount of say (SAMME alpha)
            alpha = np.log((1.0 - err) / err) + np.log(self.K - 1)
            
            # Update weights
            w = w * np.exp(alpha * misclassified)
            w = w / np.sum(w) # Normalise
            
            self.models.append((alpha, tree))

    def predict(self, X):
        # Aggregate weighted votes
        scores = np.zeros((len(X), self.K))
        for alpha, tree in self.models:
            preds = tree.predict(X)
            # Add alpha to the predicted class score for each sample
            scores[np.arange(len(X)), preds] += alpha
            
        # Tie-breaking defaults to smallest class label due to argmax behavior
        return np.argmax(scores, axis=1)

if __name__ == "__main__":
    # Note: load_data now returns 3 variables
    X, y_species, y_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    X_train, y_train, X_test, y_test = X[train_idx], y_species[train_idx], X[test_idx], y_species[test_idx]
    
    # Build mapping from train only
    species_to_family = get_species_to_family(y_train, y_family[train_idx])
    y_family_test = y_family[test_idx]
    
    print("=== Bagging (T=25, max_depth=None) ===")
    for criterion in ["gini", "entropy"]:
        bagging = Bagging(criterion=criterion)
        bagging.fit(X_train, y_train)
        
        train_acc = get_accuracy(y_train, bagging.predict(X_train)) * 100
        preds_test = bagging.predict(X_test)
        test_acc = get_accuracy(y_test, preds_test) * 100
        family_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in preds_test])) * 100
        
        print(f"Criterion: {criterion.capitalize()}")
        print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Derived Family Acc: {family_acc:.2f}%")

    print("\n=== AdaBoost SAMME (T=50) ===")
    for criterion in ["gini", "entropy"]:
        # Standard requirement (max_depth=3)
        adaboost = AdaBoostSAMME(criterion=criterion, max_depth=3)
        adaboost.fit(X_train, y_train)
        
        train_acc = get_accuracy(y_train, adaboost.predict(X_train)) * 100
        preds_test = adaboost.predict(X_test)
        test_acc = get_accuracy(y_test, preds_test) * 100
        family_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in preds_test])) * 100
        
        # Stumps analysis requirement (max_depth=1)
        stump_boost = AdaBoostSAMME(criterion=criterion, max_depth=1)
        stump_boost.fit(X_train, y_train)
        
        stump_train_acc = get_accuracy(y_train, stump_boost.predict(X_train)) * 100
        stump_preds_test = stump_boost.predict(X_test)
        stump_test_acc = get_accuracy(y_test, stump_preds_test) * 100
        stump_family_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in stump_preds_test])) * 100
        
        print(f"Criterion: {criterion.capitalize()}")
        print(f"  Depth=3 -> Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Derived Family Acc: {family_acc:.2f}%")
        print(f"  Depth=1 -> Train Acc: {stump_train_acc:.2f}% | Test Acc: {stump_test_acc:.2f}% | Derived Family Acc: {stump_family_acc:.2f}%\n")