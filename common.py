import numpy as np
import pandas as pd

def stratified_split(labels, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test

def load_data(filepath="Frogs_MFCCs.csv"):
    df = pd.read_csv(filepath)
    
    # Dynamically extract the 22 MFCCs features to ignore spacing inconsistencies
    feature_cols = [col for col in df.columns if col.startswith("MFCCs_")]
    X = df[feature_cols].values
    
    # Extract raw string labels
    species_raw = df['Species'].values
    family_raw = df['Family'].values
    
    # Sort classes alphabetically so integer mapping aligns with "smallest class label" tie-breaking
    species_classes = np.sort(np.unique(species_raw))
    family_classes = np.sort(np.unique(family_raw))
    
    # Map strings to integers
    species_to_int = {s: i for i, s in enumerate(species_classes)}
    family_to_int = {f: i for i, f in enumerate(family_classes)}
    
    y_species = np.array([species_to_int[s] for s in species_raw])
    y_family = np.array([family_to_int[f] for f in family_raw])
    
    # Build the Species -> Family mapping for derived family accuracy
    species_to_family = {}
    for s_idx, f_idx in zip(y_species, y_family):
        species_to_family[s_idx] = f_idx
        
    return X, y_species, y_family, species_to_family

def get_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)