"""
Train a random forest classifier on the CSMAE output features and evaluate it.
"""
from typing import Dict, Optional, Literal, Union
import pickle
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt
from eval_utils import get_ben_report, read_data

def random_forest(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    out_folder: Path,
    criterion: Literal["gini", "entropy", "log_loss"] = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 10,
    min_samples_leaf: int = 10,
    max_features: Union[Literal["sqrt", "log2"], int, float] = 'sqrt',
    bootstrap: bool = False,
    n_jobs: int = -1,
    classes: int = 19,
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> Dict[str, object]:
    """Train random forest classifiers and evaluate them on the test set.

    Args:
        train_file (Path): File containing training features and labels
        val_file (Path): File containing validation features and labels
        test_file (Path): File containing test features and labels
        out_folder (Path): Folder to save model and evaluation results
        n_estimators (int, optional): _description_. Defaults to 10.
    For the rest of the arguments, see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Returns:
        Dict[str, object]: _description_
    """
    # Read data
    train_keys, Y_train, X_train = read_data(train_file, classes)
    assert len(train_keys) == len(Y_train) == len(X_train), "train dimensions mismatch"
    val_keys, Y_val, X_val = read_data(val_file, classes)
    assert len(val_keys) == len(Y_val) == len(X_val), "val dimensions mismatch"
    test_keys, Y_test, X_test = read_data(test_file, classes)
    assert len(test_keys) == len(Y_test) == len(X_test), "test dimensions mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "feature dimensions mismatch"
    assert Y_train.shape[1] == Y_val.shape[1] == Y_test.shape[1], "label dimensions mismatch"
    n_features = X_train.shape[1]
    n_classes = Y_train.shape[1]
    if verbose:
        print(f"Loaded {len(train_keys)} training, {len(val_keys)} validation, and {len(test_keys)} test samples")
        print(f"Feature dimension: {n_features}, label dimension: {n_classes}")
        print(f"criterion: {criterion}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, max_features: {max_features}, bootstrap: {bootstrap}, n_jobs: {n_jobs}, random_state: {random_state}")
        print("Training random forest classifier with n_estimators=10...")
    
    # Training the base model with n_estimators=10
    model10 = RandomForestClassifier(
        n_estimators=10,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=10000, # The more, the better. A question of memory.,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    ).fit(X_train, Y_train)
    
    # Testing the base model
    Y_test_scores10 = np.array(model10.predict_proba(X_test))[:,:,1].T # select class "true" and transpose to get shape (n_samples, n_classes)
    assert Y_test_scores10.shape == Y_test.shape, f"Y_test_scores10.shape {Y_test_scores10.shape} must equal Y_test.shape {Y_test.shape}"
    
    # Training + validation for different n_estimators
    best_model = None
    scores = dict()
    best_n_estimators = 0
    pb = tqdm([1, 2, 5, 10, 15, 20, 25, 30, 40, 50], desc="Validate n_estimators")
    for n_estimators in pb:
        pb.set_postfix(n_estimators=n_estimators)
        # Training
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=None,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        ).fit(X_train, Y_train)
        # Validation
        Y_pred = model.predict(X_val)
        score = hamming_loss(Y_val, Y_pred) # other metrics lead to the same result (larger n_estimators = better), but take longer
        scores[n_estimators] = score
        if best_model is None or score < scores[best_n_estimators]:
            best_model = model
            best_n_estimators = n_estimators
    
    # Testing the best model
    assert best_model is not None
    Y_test_scores_best = np.array(best_model.predict_proba(X_test))[:,:,1].T
    assert Y_test_scores_best.shape == Y_test.shape, f"Y_test_scores_best.shape {Y_test_scores_best.shape} must equal Y_test.shape {Y_test.shape}"
    
    # Save models
    out_folder.mkdir(parents=True, exist_ok=True)
    model_file = out_folder / "model10.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model10, f)
    if verbose:
        print(f"Model saved to {model_file}")
    model_file = out_folder / "model_best.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    if verbose:
        print(f"Model saved to {model_file}")
    
    # Plot hyperparameter validation
    fig, ax = plt.subplots()
    x = list(scores.keys())
    y = list(scores.values())
    ax.plot(x, y)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Hamming loss")
    fig.savefig(str(out_folder / "n_estimators_validation.svg"))
    with open(out_folder / "n_estimators_validation.json", "w") as f:
        json.dump(scores, f)
    
    # Report for the base model
    report10, main_metrics10 = get_ben_report(Y_test, Y_test_scores10, decision_threshold=0.5)
    if verbose:
        print(report10)
        print(main_metrics10)
    report10.to_csv(out_folder / "report10.csv")
    with open(out_folder / "summary10.txt", "w") as f:
        f.write(main_metrics10)
    
    # Report for the best model
    report_best, main_metrics_best = get_ben_report(Y_test, Y_test_scores_best, decision_threshold=0.5)
    if verbose:
        print(report_best)
        print(main_metrics_best)
    report_best.to_csv(out_folder / "report_best.csv")
    with open(out_folder / "summary_best.txt", "w") as f:
        f.write(f"n_estimators={best_n_estimators}\n{main_metrics_best}")
    
    return {
        "report10": report10,
        "main_metrics10": main_metrics10,
        "report_best": report_best,
        "main_metrics_best": main_metrics_best,
    }

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("train_file", type=Path)
    parser.add_argument("val_file", type=Path)
    parser.add_argument("test_file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--classes", type=int, default=19)
    args = parser.parse_args()
    random_forest(
        args.train_file,
        args.val_file,
        args.test_file,
        args.out,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        n_jobs=args.n_jobs,
        classes=args.classes,
        verbose=not args.silent,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()