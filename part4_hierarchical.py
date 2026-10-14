import numpy as np
from common import load_data, stratified_split, get_accuracy, get_species_to_family
from part1_decision_tree import DecisionTree

def compute_confusion_matrix(y_true, y_pred, n_classes):
    """Computes a confusion matrix using pure NumPy."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2)
    return cm.reshape(n_classes, n_classes)

class HierarchicalClassifier:
    def __init__(self, router, experts, int_to_family_name):
        self.router = router
        self.experts = experts
        self.int_to_family_name = int_to_family_name

    def predict_family(self, X):
        return self.router.predict(X)

    def predict(self, X):
        pred_families = self.predict_family(X)
        final_species = np.zeros(len(X), dtype=int)
        
        # Route grouped samples to their respective experts
        for fam_int, fam_name in self.int_to_family_name.items():
            mask = (pred_families == fam_int)
            if np.any(mask):
                final_species[mask] = self.experts[fam_name].predict(X[mask])
        return final_species

    def oracle_predict(self, X, true_families):
        """Predicts species assuming a perfect router."""
        oracle_species = np.zeros(len(X), dtype=int)
        for fam_int, fam_name in self.int_to_family_name.items():
            mask = (true_families == fam_int)
            if np.any(mask):
                oracle_species[mask] = self.experts[fam_name].predict(X[mask])
        return oracle_species

if __name__ == "__main__":
    X, y_species, y_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    
    X_train, y_species_train, y_family_train = X[train_idx], y_species[train_idx], y_family[train_idx]
    X_test, y_species_test, y_family_test = X[test_idx], y_species[test_idx], y_family[test_idx]
    
    # Generate the mapping strictly from training data to avoid leakage
    species_to_family = get_species_to_family(y_species_train, y_family_train)
    int_to_fam = {0: "Bufonidae", 1: "Dendrobatidae", 2: "Hylidae", 3: "Leptodactylidae"}
    
    n_species = len(np.unique(y_species))
    n_families = len(np.unique(y_family))

    print("=== Hierarchical Decision Tree ===")
    for criterion in ["gini", "entropy"]:
        print(f"\n--- Criterion: {criterion.capitalize()} ---")
        
        # 1. Train Flat Model for Comparison
        flat_tree = DecisionTree(criterion=criterion)
        flat_tree.fit(X_train, y_species_train)
        flat_preds = flat_tree.predict(X_test)
        flat_acc = get_accuracy(y_species_test, flat_preds)
        flat_fam_acc = get_accuracy(y_family_test, np.array([species_to_family[p] for p in flat_preds]))
        
        # 2. Train Hierarchical External Models
        router = DecisionTree(criterion=criterion)
        router.fit(X_train, y_family_train)
        
        experts = {}
        for fam_int, fam_name in int_to_fam.items():
            mask = (y_family_train == fam_int)
            expert = DecisionTree(criterion=criterion)
            expert.fit(X_train[mask], y_species_train[mask])
            experts[fam_name] = expert
            
        clf = HierarchicalClassifier(router, experts, int_to_fam)
        
        # 3. Predictions
        fam_pred_train = clf.predict_family(X_train)
        fam_pred_test = clf.predict_family(X_test)
        
        spec_pred_train = clf.predict(X_train)
        spec_pred_test = clf.predict(X_test)
        oracle_spec_test = clf.oracle_predict(X_test, y_family_test)
        
        A_router_train = get_accuracy(y_family_train, fam_pred_train)
        A_router_test = get_accuracy(y_family_test, fam_pred_test)
        
        A_overall_train = get_accuracy(y_species_train, spec_pred_train)
        A_overall_test = get_accuracy(y_species_test, spec_pred_test)
        A_oracle = get_accuracy(y_species_test, oracle_spec_test)
        
        # 4. Failure Analysis
        p_A = np.mean((fam_pred_test == y_family_test) & (spec_pred_test != y_species_test))
        p_B = np.mean(fam_pred_test != y_family_test)
        
        print(f"Hierarchical Router Acc (Train):  {A_router_train * 100:.2f}%")
        print(f"Hierarchical Overall Acc (Train): {A_overall_train * 100:.2f}%\n")
        print(f"Hierarchical Overall Acc (Test): {A_overall_test * 100:.2f}% vs Flat Species Acc: {flat_acc * 100:.2f}%")
        print(f"Hierarchical Router Acc (Test):  {A_router_test * 100:.2f}% vs Flat Family Acc: {flat_fam_acc * 100:.2f}%")
        
        print(f"\nOracle Acc (Test):   {A_oracle * 100:.2f}%")
        
        print(f"\nFailure Fractions:")
        print(f"  p_A (Router Right, Expert Wrong): {p_A * 100:.2f}%")
        print(f"  p_B (Router Wrong, Species Wrong): {p_B * 100:.2f}%")
        
        # Strict Numerical Check
        assert np.isclose(A_overall_test, A_router_test - p_A)
        print("Numerical Check (A_overall == A_router - p_A): PASS")
        
        # 5. Matrices and Summaries
        print("\nRouter Confusion Matrix (4x4):")
        print(compute_confusion_matrix(y_family_test, fam_pred_test, n_families))
        
        print("\nFinal Species Confusion Matrix (10x10):")
        print(compute_confusion_matrix(y_species_test, spec_pred_test, n_species))
        
        print("\nPer-Family Overall Accuracy (Test Set):")
        for fam_int, fam_name in int_to_fam.items():
            mask = (y_family_test == fam_int)
            fam_acc = get_accuracy(y_species_test[mask], spec_pred_test[mask])
            print(f"  {fam_name}: {fam_acc * 100:.2f}%")
        print("-" * 50)