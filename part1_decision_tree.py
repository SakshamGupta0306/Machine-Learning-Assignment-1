import numpy as np
from common import load_data, stratified_split, get_accuracy, get_species_to_family

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, gain=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain

class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, max_features=None, rng=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self.root = None
        self.n_classes = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        
        self.n_classes = np.max(y) + 1
        self.root = self._build_tree(X, y, sample_weight, depth=0)

    def _impurity(self, class_weights):
        p = class_weights / (np.sum(class_weights) + 1e-15)
        p = p[p > 0]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        else: # entropy
            return -np.sum(p * np.log2(p))

    def _build_tree(self, X, y, w, depth):
        total_weight = np.sum(w)
        class_weights = np.bincount(y, weights=w, minlength=self.n_classes)
        
        # Tie-breaking in leaf: smallest class label (first occurrence of max weight)
        majority_class = np.argmax(class_weights)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           np.count_nonzero(class_weights) <= 1:
            return Node(value=majority_class)

        best_gain = 0.0
        best_split = None

        n_features = X.shape[1]
        features = np.arange(n_features)
        if self.max_features is not None:
            if self.rng is None:
                raise ValueError("rng must be provided if max_features is set.")
            features = self.rng.choice(n_features, self.max_features, replace=False)
            features = np.sort(features)
        parent_impurity = self._impurity(class_weights)

        for f in features:
            X_f = X[:, f]
            sort_idx = np.argsort(X_f)
            X_sort, y_sort, w_sort = X_f[sort_idx], y[sort_idx], w[sort_idx]
            
            # Find distinct values and cut points
            distinct_mask = X_sort[1:] != X_sort[:-1]
            if not np.any(distinct_mask):
                continue
                
            cut_indices = np.where(distinct_mask)[0]
            candidate_thresholds = (X_sort[cut_indices] + X_sort[cut_indices + 1]) / 2.0

            # Vectorized class weights accumulation
            y_one_hot = np.zeros((len(y), self.n_classes))
            y_one_hot[np.arange(len(y)), y_sort] = 1
            w_one_hot = y_one_hot * w_sort[:, None]
            
            left_class_weights = np.cumsum(w_one_hot, axis=0)[cut_indices]
            right_class_weights = class_weights - left_class_weights
            
            W_L = np.sum(left_class_weights, axis=1)
            W_R = np.sum(right_class_weights, axis=1)
            
            # Calculate impurity for all splits
            if self.criterion == "gini":
                p_L = left_class_weights / (W_L[:, None] + 1e-15)
                imp_L = 1.0 - np.sum(p_L ** 2, axis=1)
                p_R = right_class_weights / (W_R[:, None] + 1e-15)
                imp_R = 1.0 - np.sum(p_R ** 2, axis=1)
            else: # entropy
                # Suppress divide by zero warnings for vectorized log2
                with np.errstate(divide='ignore', invalid='ignore'):
                    p_L = left_class_weights / (W_L[:, None] + 1e-15)
                    log2_p_L = np.nan_to_num(np.log2(p_L), neginf=0.0)
                    imp_L = -np.sum(p_L * log2_p_L, axis=1)
                    
                    p_R = right_class_weights / (W_R[:, None] + 1e-15)
                    log2_p_R = np.nan_to_num(np.log2(p_R), neginf=0.0)
                    imp_R = -np.sum(p_R * log2_p_R, axis=1)

            gain = parent_impurity - (W_L / total_weight) * imp_L - (W_R / total_weight) * imp_R

            # Tie breaking handled by finding max gain and picking the first instance (lowest threshold)
            max_gain_idx = np.argmax(gain)
            max_gain = gain[max_gain_idx]

            # If gain ties with a previous feature, we implicitly prefer the lower feature index 
            # because we only update if max_gain > best_gain
            if max_gain > best_gain:
                best_gain = max_gain
                best_split = (f, candidate_thresholds[max_gain_idx], cut_indices[max_gain_idx])

        if best_gain <= 0.0:
            return Node(value=majority_class)

        # Apply the best split
        f, thresh, split_idx = best_split
        sort_idx = np.argsort(X[:, f])
        
        left_mask = np.zeros(len(X), dtype=bool)
        left_mask[sort_idx[:split_idx + 1]] = True
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], w[right_mask], depth + 1)

        return Node(feature_idx=f, threshold=thresh, left=left_node, right=right_node, gain=best_gain)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

def run_part1_sanity_checks():
    print("--- Part 1 Sanity Checks ---")
    tree = DecisionTree(criterion="gini")
    tree.n_classes = 2
    g, h = tree._impurity(np.array([3, 3])), DecisionTree(criterion="entropy")._impurity(np.array([3, 3]))
    print(f"(a) Gini: {g:.2f} | Entropy: {h:.2f}")

    X_b, y_b = np.array([[1], [2], [3], [4]]), np.array([0, 0, 1, 1])
    tree_b = DecisionTree(criterion="entropy")
    tree_b.fit(X_b, y_b)
    print(f"(b) Acc: {get_accuracy(y_b, tree_b.predict(X_b))*100:.0f}%, Threshold: {tree_b.root.threshold}, Gain: {tree_b.root.gain:.4f}")

    X_c, y_c, w_c = np.array([[1], [1], [1]]), np.array([0, 0, 1]), np.array([0.1, 0.1, 0.8])
    tree_c = DecisionTree()
    tree_c.fit(X_c, y_c, sample_weight=w_c)
    print(f"(c) Leaf prediction: {tree_c.predict(X_c)[0]}")
    
    # New check (d)
    tree_d = DecisionTree()
    tree_d.fit(np.random.rand(5, 3), np.zeros(5, int))
    print(f"(d) Pure node prediction: {tree_d.predict(np.random.rand(1, 3))[0]} (Expected: 0)\n")


if __name__ == "__main__":
    run_part1_sanity_checks()
    
    # 1. Load Data & Split
    X, y_species, y_family = load_data()
    train_idx, test_idx = stratified_split(y_species, test_frac=0.2, seed=42)
    X_train, y_train, X_test, y_test = X[train_idx], y_species[train_idx], X[test_idx], y_species[test_idx]
    
    # Build mapping from train only
    species_to_family = get_species_to_family(y_train, y_family[train_idx])
    y_family_test = y_family[test_idx]

    print(f"Total rows: {len(X)}")
    print(f"Total features: {X.shape[1]}")
    print(f"Train split: {len(X_train)} rows, Test split: {len(X_test)} rows\n")

    # Task 1 & 2: Train flat tree (Unconstrained) and evaluate
    print("--- Single Decision Tree (Unconstrained) ---")
    for criterion in ["gini", "entropy"]:
        tree = DecisionTree(criterion=criterion, min_samples_split=2)
        tree.fit(X_train, y_train)
        
        preds_train = tree.predict(X_train)
        preds_test = tree.predict(X_test)
        
        train_acc = get_accuracy(y_train, preds_train) * 100
        test_acc = get_accuracy(y_test, preds_test) * 100
        
        # Derived Family Accuracy
        preds_family_test = np.array([species_to_family[p] for p in preds_test])
        family_acc = get_accuracy(y_family_test, preds_family_test) * 100
        
        print(f"Criterion: {criterion.capitalize()}")
        print(f"  Train Acc (Species):  {train_acc:.2f}%")
        print(f"  Test Acc (Species):   {test_acc:.2f}%")
        print(f"  Derived Family Acc:   {family_acc:.2f}%\n")

    # Task 3: Depth Sweep
    print("--- Depth Sweep Analysis ---")
    depths = [1, 2, 3, 5, 8, 12, None]
    for criterion in ["gini", "entropy"]:
        print(f"{criterion.capitalize()} Criterion:")
        for d in depths:
            tree = DecisionTree(criterion=criterion, max_depth=d)
            tree.fit(X_train, y_train)
            train_acc = get_accuracy(y_train, tree.predict(X_train)) * 100
            test_acc = get_accuracy(y_test, tree.predict(X_test)) * 100
            d_str = str(d) if d is not None else "None"
            print(f"  Depth {d_str:>4}: Train={train_acc:>6.2f}%, Test={test_acc:>6.2f}%")
        print()