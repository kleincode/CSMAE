# CSMAE - Evaluation
This is a fork of the original CSMAE repository by Hackstein et al. As part of my course work for the 'Hot Topics in Computer Vision' seminar, I evaluated the models proposed and trained by the original authors on the BigEarthNet dataset (only!) for Sentinel-1 imagery.

## Requirements
Here's what I did to get the model running on BEN:
1. Download and unzip `BigEarthNet-S1` from [bigearth.net](https://bigearth.net/) into `csmae_data/BigEarthNet-S1-v1.0`
2. Install BigEarthNet-encoder package to create an lmdb file, the format used by the scripts in the repository:
```bash
pip install bigearthnet-encoder
cd csmae_data
ben_encoder write-s1-lmdb-with-lbls ./BigEarthNet-S1-v1.0
mv S1_lmdb.db BigEarthNetEncoded.lmdb
```
3. Recreate the Conda environment provided by the authors. I used WSL2.
```bash
conda env create --name csmae --file environment.yaml
```
4. I received the original model weights from the authors and put them into `trained_models/CSMAE-CECD`, `trained_models/CSMAE-CESD`, etc.

## Running the models
`run_model.py` can be used to run any of the four models on a given data subset (as defined by a CSV) and export the image features to a Pickle file, for example:
```bash
python run_model.py --model CSMAE-CESD --csv BigEarthNet-MM_19-classes_models/splits/train.csv --output features_out/cesd/train.pkl
```
For all arguments, see `run_model.py`. The generated Pickle file contains a tuple `(keys, labels19, labels43, outputs)` where `keys` is a list of length `N` of all the sample keys, `labels19` is a `N x 19` boolean array containing the BEN19 labels, `labels43` is a `N x 43` boolean array containing the BEN43 labels, and `outputs` is a `N x 768` array containing the encoder arrays/embeddings. These output pickle files can then be read by all classifier scripts and the UMAP visualizer.

## Visualization
`visualize_ben.ipynb` contains basic visualizations and statistics about the BEN19 dataset. `visualize_umap.ipynb` visualizes the CSMAE feature vectors using UMAP. The output images are available under `imgs`.

## Training neural classifiers (linear, MLP)
Use `mlp_classifier.py` to train a neural classifier on the feature vectors using PyTorch.

Example usage to train a linear classifier:
```bash
python mlp_classifier.py features_out/cesd/train.pkl features_out/cesd/val.pkl features_out/cesd/test.pkl --out experiments/CSMAE-CESD/ben19_linear_probing --n_layers 1 --classes 19
```

Example usage to train the small MLP with four layers:
```bash
python mlp_classifier.py features_out/cesd/train.pkl features_out/cesd/val.pkl features_out/cesd/test.pkl --out experiments/CSMAE-CESD/ben19_mlp4 --n_layers 4 --classes 19
```
The results are in the `experiments` folder and contain a loss curve (SVG), a full classification report and a summary file.

## Training random forests
Use `random_forest.py` to train random forests on the feature vectors using sklearn.

Example usage:
```bash
python random_forest.py features_out/cesd/train.pkl features_out/cesd/val.pkl features_out/cesd/test.pkl --out experiments/CSMAE-CESD/ben19_random_forest --classes 19
```

The script first trains a base random forest model with the following fixed hyperparameters for comparability to other students' works:
```
n_estimators=10	<-- The more, the better but uses linearly more time and memory
criterion='gini'
max_depth=None
min_samples_split=10
min_samples_leaf=10
max_features='sqrt'
max_leaf_nodes=10000	<-- The more, the better. A question of memory.
bootstrap=False
n_jobs=-1			<-- For multi-core processing (-1 … use all cores)
```
Next, the hyperparameter `n_estimators` is optimized on the validation set in the range between 1 and 50. In all experiments, the maximum value for `n_estimators` (50) was optimal. The script saves the models, the classification reports and summaries for both the reference and the optimal model to the output folder as well as the hyperparameter optimization curve (n_estimators vs validation Hamming loss). Results are in the `experiments` folder.

## Training baseline models
`train_baseline.py` was used to train baseline models (resnet50, resnet18) directly on the supervised scene classification task on BEN19.

Example usage:
```bash
python train_baseline.py --epochs 10 --out_folder trained_models/resnet18_10e --architecture resnet18
```
Checkpoints, training and validation losses, validation metrics, classification report and summary file are saved to the output folder. Everything except the checkpoints is available under `trained_models/resnet**_10e`.

---

Original README:

# Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing

![Alt text](csmae.png?raw=true "Model: Cross-Sensor Masked Autoencoders")

This repository contains the code of the paper [Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing](https://arxiv.org/abs/2401.07782). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein), [Gencer Sumbul](https://people.epfl.ch/gencer.sumbul?lang=en), [Kai Norman Clasen](https://rsim.berlin/team/members/kai-norman-clasen) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

```bibtex
@ARTICLE{hackstein2024exploring,
    author={Hackstein, Jakob and Sumbul, Gencer and Clasen, Kai Norman and Demir, Begüm},
    title={Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing},
    url={https://arxiv.org/abs/2401.07782},
    eprint={2401.07782},
    archivePrefix={arXiv},
    year={2024},
}
```

## Training CSMAE models

1. First, set up a python (conda) environment based on the `environment.yaml` file. 

2. Training and model parameters can be adjusted via yaml-files. The paper introduces four different CSMAE variants, termed _CECD_, _CESD_, _SECD_ and _SESD_, and each variant has a pre-defined `csmae_<variant>.yaml` file already. You can also modify them according to your needs and configure new CSMAE models. To do so, check the explanations for relevant parameters in existing `csmae.yaml` files.

3. Independent on your yaml-file, two entries have to be completed:

    - The training progress is tracked on [Weights & Biases](https://wandb.ai/). To this end, the `wandb.entity` and `wandb.project` fields have to be entered in the `wandb` attribute.

    - For training, [BigEarthNet-MM](https://bigearth.net/) is required. The dataloader requires the LMDB format which is explained [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). Finally, the `data.root_dir` should point to the directory containing the LMDB file and `data.split_dir` should point to the directory containing CSV-file splits of the dataset.

4. Then, pre-training can be started by running `train.py` with two flags required by Hydra:
    ```bash
    python train.py --config-path ./ --config-name csmae_<variant>.yaml
    ```

5. Checkpoints of trained models with config files are stored under `./trained_models`.

## Evaluation of CSMAE models

To compute image retrieval results, run the `retrieval.py` script. The two required flags are
- name of the folder, which contains the model checkpoint to be evaluated
- the GPU device number used for inference.

For instance, a model stored under `./trained_models/abcd1234/` can be evaluated with

```bash
python retrieval.py abcd1234 0
```

## Model Weights

We share model weights for the best-performing CSMAE variants [here](https://tubcloud.tu-berlin.de/s/E4RcbGjzSrjBq7R). To load weights into a backbone, see the following code snippet.

```python
import torch
from src.csmae_backbone import CSMAEBackbone
from omegaconf import OmegaConf

csmae_variant = 'sesd'
cfg = OmegaConf.load(f'./checkpoints/{csmae_variant}/cfg.yaml')
model = CSMAEBackbone(**cfg.kwargs)

state_dict = torch.load(f'./checkpoints/{csmae_variant}/weights.ckpt', map_location="cpu")['state_dict']
for k in list(state_dict.keys()):
    if "backbone" in k:
        state_dict[k.replace("backbone.", "")] = state_dict[k]
    del state_dict[k]

model.load_state_dict(state_dict, strict=True)
```

## Acknowledgement

This work is supported by the European Research Council
(ERC) through the ERC-2017-STG BigEarth Project under
Grant 759764 and by the European Space Agency (ESA)
through the Demonstrator Precursor Digital Assistant Interface
For Digital Twin Earth (DA4DTE) Project and by the German Ministry for
Economic Affairs and Climate Action through the AI-Cube
Project under Grant 50EE2012B.

The code for pre-training is inspired by [solo-learn](https://github.com/vturrisi/solo-learn) and the code for dataloading partly stems from [ConfigILM](https://github.com/lhackel-tub/ConfigILM).


## Authors
**Jakob Hackstein**
https://rsim.berlin/team/members/jakob-hackstein

**Gencer Sumbul**
https://people.epfl.ch/gencer.sumbul?lang=en

**Kai Norman Clasen**
https://rsim.berlin/team/members/kai-norman-clasen 

**Begüm Demir**
https://rsim.berlin/team/members/begum-demir

For questions, requests and concerns, please contact [Jakob Hackstein via mail](mailto:hackstein@tu-berlin.de)

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2024 Jakob Hackstein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

