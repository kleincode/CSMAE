from pathlib import Path
from mlp_classifier import mlp_classifier
from random_forest import random_forest

def main():
    for classes in [19, 43]:
        for model_type in ["CECD", "CESD", "SECD", "SESD"]:
            out_folder = Path(f"experiments/CSMAE-{model_type}/ben{classes}_linear_probing")
            if out_folder.is_dir():
                print(f"Skipping {out_folder}")
                continue
            model_folder = Path(f"features_out/CSMAE-{model_type}")
            train_file = model_folder / f"train_{model_type}_{classes}.pkl"
            val_file = model_folder / f"val_{model_type}_{classes}.pkl"
            test_file = model_folder / f"test_{model_type}_{classes}.pkl"
            mlp_classifier(
                train_file,
                val_file,
                test_file,
                out_folder,
                n_layers=1
            )
    
    for classes in [19, 43]:
        for model_type in ["CECD", "CESD", "SECD", "SESD"]:
            out_folder = Path(f"experiments/CSMAE-{model_type}/ben{classes}_random_forest")
            if out_folder.is_dir():
                print(f"Skipping {out_folder}")
                continue
            model_folder = Path(f"features_out/CSMAE-{model_type}")
            train_file = model_folder / f"train_{model_type}_{classes}.pkl"
            val_file = model_folder / f"val_{model_type}_{classes}.pkl"
            test_file = model_folder / f"test_{model_type}_{classes}.pkl"
            random_forest(
                train_file,
                val_file,
                test_file,
                out_folder,
            )
    
    for classes in [19, 43]:
        for model_type in ["CECD", "CESD", "SECD", "SESD"]:
            out_folder = Path(f"experiments/CSMAE-{model_type}/ben{classes}_mlp4")
            if out_folder.is_dir():
                print(f"Skipping {out_folder}")
                continue
            model_folder = Path(f"features_out/CSMAE-{model_type}")
            train_file = model_folder / f"train_{model_type}_{classes}.pkl"
            val_file = model_folder / f"val_{model_type}_{classes}.pkl"
            test_file = model_folder / f"test_{model_type}_{classes}.pkl"
            mlp_classifier(
                train_file,
                val_file,
                test_file,
                out_folder,
                n_layers=4
            )
    
    print("Done!")


if __name__ == "__main__":
    main()
