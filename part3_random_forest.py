import numpy as np
from common import load_data, stratified_split, get_accuracy
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging

class RandomForest:
    def __init__(self, n_estimators=50, criterion="gini", max_features=4, seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        # One fresh RandomState per ensemble model for ALL its randomness
        rng = np.random.RandomState(self.seed)
        N = len(X)
        self.trees = []
        
        for _ in range(self.n_estimators):
            indices = rng.randint(0, N, size=N)
            # Pass the SAME rng object to the tree so it handles feature subsetting from the same state
            tree = DecisionTree(
                criterion=self.criterion, 
                max_features=self.max_features, 
                rng=rng
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote with tie-breaking to smallest class label
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=10).argmax(), 
            axis=0, 
            arr=all_preds
        )
        return final_preds

def pairwise_agreement(trees, X):
    """Calculates the average pairwise agreement of test predictions (Extra task)."""
    n_trees = len(trees)
    all_preds = np.array([tree.predict(X) for tree in trees])
    agreements = []
    
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            agree_fraction = np.mean(all_preds[i] == all_preds[j])
            agreements.append(agree_fraction)
            
    return np.mean(agreements)

if __name__ == "__main__":
    X, y_species, y_family, species_to_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    X_train, y_train = X[train_idx], y_species[train_idx]
    X_test, y_test = X[test_idx], y_species[test_idx]
    y_family_test = y_family[test_idx]

    print("=== Random Forest (T=50, m=4, max_depth=None) ===")
    for criterion in ["gini", "entropy"]:
        rf = RandomForest(criterion=criterion, max_features=4)
        rf.fit(X_train, y_train)
        
        train_acc = get_accuracy(y_train, rf.predict(X_train)) * 100
        preds_test = rf.predict(X_test)
        test_acc = get_accuracy(y_test, preds_test) * 100
        family_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in preds_test])) * 100
        
        print(f"Criterion: {criterion.capitalize()}")
        print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Derived Family Acc: {family_acc:.2f}%")
        
        # Extra: Compare agreement with Bagging
        bagging = Bagging(n_estimators=50, criterion=criterion)
        bagging.fit(X_train, y_train)
        
        rf_agreement = pairwise_agreement(rf.trees, X_test) * 100
        bag_agreement = pairwise_agreement(bagging.trees, X_test) * 100
        
        print(f"  [Pairwise Agreement on Test Set] RF: {rf_agreement:.2f}% vs Bagging: {bag_agreement:.2f}%\n")