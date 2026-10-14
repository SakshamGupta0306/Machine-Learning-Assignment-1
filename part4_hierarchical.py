import numpy as np
from common import load_data, stratified_split, get_accuracy
from part1_decision_tree import DecisionTree

def compute_confusion_matrix(y_true, y_pred, n_classes):
    """Computes a confusion matrix using pure NumPy."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2)
    return cm.reshape(n_classes, n_classes)

class HierarchicalClassifier:
    def __init__(self, criterion="gini"):
        self.criterion = criterion
        self.router = DecisionTree(criterion=criterion)
        self.experts = {}
        self.families = None

    def fit(self, X, y_species, y_family):
        self.families = np.unique(y_family)
        
        # 1. Train Router on true Family
        self.router.fit(X, y_family)
        
        # 2. Train Experts on true Family partitions
        for fam in self.families:
            mask = (y_family == fam)
            expert = DecisionTree(criterion=self.criterion)
            # Expert trains to predict Species on its family's subset
            expert.fit(X[mask], y_species[mask])
            self.experts[fam] = expert

    def predict_family(self, X):
        return self.router.predict(X)

    def predict(self, X):
        pred_families = self.predict_family(X)
        final_species = np.zeros(len(X), dtype=int)
        
        # Route grouped samples to their respective experts to avoid per-sample loops
        for fam in self.families:
            mask = (pred_families == fam)
            if np.any(mask):
                final_species[mask] = self.experts[fam].predict(X[mask])
                
        return final_species

    def oracle_predict(self, X, true_families):
        """Predicts species assuming a perfect router."""
        oracle_species = np.zeros(len(X), dtype=int)
        for fam in self.families:
            mask = (true_families == fam)
            if np.any(mask):
                oracle_species[mask] = self.experts[fam].predict(X[mask])
        return oracle_species

if __name__ == "__main__":
    X, y_species, y_family, species_to_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    
    X_train, y_species_train, y_family_train = X[train_idx], y_species[train_idx], y_family[train_idx]
    X_test, y_species_test, y_family_test = X[test_idx], y_species[test_idx], y_family[test_idx]
    
    n_species = len(np.unique(y_species))
    n_families = len(np.unique(y_family))

    print("=== Hierarchical Decision Tree ===")
    for criterion in ["gini", "entropy"]:
        print(f"\n--- Criterion: {criterion.capitalize()} ---")
        clf = HierarchicalClassifier(criterion=criterion)
        clf.fit(X_train, y_species_train, y_family_train)
        
        # Router Predictions
        fam_pred_train = clf.predict_family(X_train)
        fam_pred_test = clf.predict_family(X_test)
        
        # Final Species Predictions
        spec_pred_train = clf.predict(X_train)
        spec_pred_test = clf.predict(X_test)
        
        # Oracle Predictions
        oracle_spec_test = clf.oracle_predict(X_test, y_family_test)
        
        # Accuracies
        A_router_train = get_accuracy(y_family_train, fam_pred_train)
        A_router_test = get_accuracy(y_family_test, fam_pred_test)
        A_overall_train = get_accuracy(y_species_train, spec_pred_train)
        A_overall_test = get_accuracy(y_species_test, spec_pred_test)
        A_oracle = get_accuracy(y_species_test, oracle_spec_test)
        
        print(f"Router Acc (Train):  {A_router_train * 100:.2f}%")
        print(f"Router Acc (Test):   {A_router_test * 100:.2f}%")
        print(f"Overall Acc (Train): {A_overall_train * 100:.2f}%")
        print(f"Overall Acc (Test):  {A_overall_test * 100:.2f}%")
        print(f"Oracle Acc (Test):   {A_oracle * 100:.2f}%")
        
        # Failure Analysis
        p_A = np.mean((fam_pred_test == y_family_test) & (spec_pred_test != y_species_test))
        p_B = np.mean(fam_pred_test != y_family_test)
        
        print(f"\nFailure Fractions:")
        print(f"  p_A (Router Right, Expert Wrong): {p_A * 100:.2f}%")
        print(f"  p_B (Router Wrong, Species Wrong): {p_B * 100:.2f}%")
        
        # Numerical Check
        check_val = A_router_test - p_A
        print(f"Numerical Check (A_overall == A_router - p_A): {A_overall_test:.4f} == {check_val:.4f}")
        
        # Confusion Matrices
        print("\nRouter Confusion Matrix (4x4):")
        print(compute_confusion_matrix(y_family_test, fam_pred_test, n_families))
        
        print("\nFinal Species Confusion Matrix (10x10):")
        print(compute_confusion_matrix(y_species_test, spec_pred_test, n_species))
        
        # Per-family test accuracy
        print("\nPer-Family Overall Accuracy (Test Set):")
        family_names = {
            0: "Bufonidae", 
            1: "Dendrobatidae", 
            2: "Hylidae", 
            3: "Leptodactylidae"
        }
        for fam in clf.families:
            mask = (y_family_test == fam)
            fam_acc = get_accuracy(y_species_test[mask], spec_pred_test[mask])
            print(f"  {family_names[fam]}: {fam_acc * 100:.2f}%")
        print("-" * 50)