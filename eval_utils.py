from typing import List, Literal, Tuple
from sklearn.metrics import classification_report, fbeta_score, hamming_loss, average_precision_score
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from run_model import BEN19_LABELS, BEN43_LABELS

def get_ben_report(
    Y_test: np.ndarray,
    Y_score: np.ndarray,
    averages: List[Literal['micro', 'macro', 'samples', 'weighted']] = ["micro", "macro", "weighted", "samples"],
    decision_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, str]:
    """Generate a classification report for BEN19 or BEN43 labels.

    Args:
        Y_test (np.ndarray): _description_
        Y_score (np.ndarray): Output logits (not binary predictions!)
        averages (List[Literal[&#39;micro&#39;, &#39;macro&#39;, &#39;samples&#39;, &#39;weighted&#39;]], optional): Average methods for averaged scores. Defaults to ["micro", "macro", "weighted", "samples"].

    Returns:
        Tuple[pd.DataFrame, str]: classification report and summary string
    """
    Y_pred = Y_score > decision_threshold
    assert Y_test.shape == Y_pred.shape, f"Y_test.shape {Y_test.shape} must equal Y_pred.shape {Y_pred.shape}"
    assert Y_test.shape[-1] in [19, 43], f"Y_test.shape[-1] {Y_test.shape[-1]} must be 19 or 43"
    labels = BEN43_LABELS if Y_test.shape[-1] == 43 else BEN19_LABELS
    
    report = classification_report(Y_test, Y_pred, output_dict=True, target_names=list(labels), zero_division=0)
    report = pd.DataFrame(report).T
    f2_scores = [
        *fbeta_score(Y_test, Y_pred, average=None, beta=2), # type: ignore
        *[fbeta_score(Y_test, Y_pred, average=avg, beta=2) for avg in averages],
    ]
    report["f2-score"] = f2_scores
    ap_scores = [
        *average_precision_score(Y_test, Y_score, average=None), # type: ignore
        *[average_precision_score(Y_test, Y_score, average=avg) for avg in averages],
    ]
    report["ap-score"] = ap_scores
    
    out = f"Recall macro: {report['recall']['macro avg']:.2%}"
    out += f"\nRecall sample: {report['recall']['samples avg']:.2%}"
    out += f"\nRecall micro: {report['recall']['micro avg']:.2%}"
    out += f"\nF2 macro: {report['f2-score']['macro avg']:.2%}"
    out += f"\nF2 sample: {report['f2-score']['samples avg']:.2%}"
    out += f"\nF2 micro: {report['f2-score']['micro avg']:.2%}"
    out += f"\nmAP macro: {report['ap-score']['macro avg']:.2%}"
    out += f"\nmAP sample: {report['ap-score']['samples avg']:.2%}"
    out += f"\nmAP micro: {report['ap-score']['micro avg']:.2%}"
    out += f"\nHamming loss: {hamming_loss(Y_test, Y_pred):.2%}"
    
    # reorder columns
    return report[["precision", "recall", "f1-score", "f2-score", "ap-score", "support"]], out

def read_data(file: Path, classes: int):
    """Reads data (keys, X, Y) from a pickle file.
    Can adapt to the following formats: (keys, Y, X) or (keys, Y19, Y43, X) or (keys, Y43, Y19, X).

    Args:
        file (Path): Pickle file to read.
        classes (int): Number of classes. 19 or 43.
            If there are four values in the pickle file, the correct Y is selected based on this number.
            If there are only three values, this number is ignored.

    Raises:
        ValueError: If the number of elements in the pickle file is not 3 or 4 or the number of classes is not part of it.

    Returns:
        3 np.ndarrays: keys, Y, X
    """
    with open(file, "rb") as f:
        vals = pickle.load(f)
        if len(vals) == 3:
            keys, Y, X = vals
        elif len(vals) == 4:
            keys, _, _, X = vals
            if vals[1].shape[1] == classes:
                Y = vals[1]
            elif vals[2].shape[1] == classes:
                Y = vals[2]
            else:
                raise ValueError(f"Could not find labels in file {file}")
        else:
            raise ValueError(f"Invalid number of elements in file {file}")
    return keys, Y, X