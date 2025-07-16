# OmicsTweezer
Implementation of "OmicsTweezer: a data distribution-independent cell deconvolution model for multi-omics data" (Cell Genomics) 

<p align="center">
  <img src="./Graphical%20Abstract.png" alt="model" width="60%">
</p>

## Installation/Requirements

### 1. Create a new conda environmenty.

```bash
conda create -n OmicsTweezer python=3.12
conda activate OmicsTweezer
```

### 2. Install the required packages.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## How to run the code
```bash
conda activate OmicsTweezer
git clone https://github.com/Fry2024/OmicsTweezer.git
```
```python
## load single cell reference and bulk/spatial/proteomics data
import scanpy as sc
sc_reference = sc.read("sc_reference.h5ad") ## Replace "sc_reference.h5ad" with your file path
target_bulk = sc.read("target_bulk.h5ad") ## Replace "target_bulk.h5ad" with your file path
sc_reference.obs["CellType"] = sc_reference.obs.cell_type ## Make sure obs contains CellType
#Make sure var.index is the gene name
##
```



```python
## run OmicsTweezer
from OmicsTweezer import train_predict
predict = train_predict.train_predict(sc_reference, target_bulk,num=3000, scale=True, ot_weight=1)
```

```python
## num: Number of samples
## scale：Whether to regularize
## ot_weight：ratio of OT loss 
```


## Example (See "tutorial.ipynb" for details)
```
# Unzip "Batch_effect_simulation_dataset.zip" in the Data folder 
```

```python
import scanpy as sc
sc_reference = sc.read("Data/Batch_effect_simulation_dataset/sc_reference.h5ad")
target_bulk = sc.read("Data/Batch_effect_simulation_dataset/target_bulk.h5ad")
sc_reference.obs["CellType"] = sc_reference.obs.cell_type
```
```python
##sc_reference
AnnData object with n_obs × n_vars = 5000 × 1969
    obs: 'cell', 'cell_type', 'batch', 'umap_x', 'umap_y', 'CellType'
    var: 'gene'
##target_bulk
AnnData object with n_obs × n_vars = 100 × 1969
    obs: 'Group1', 'Group2', 'Group3', 'Group4', 'Group5'
#Make sure var.index is the gene name
```
```python
from OmicsTweezer import train_predict
predict = train_predict.train_predict(sc_reference, target_bulk,num=3000, scale=True, ot_weight=1)
```

```
predict.head(5)
```
```
      Group1	  Group2	  Group3	  Group4	 Group5
0	0.298285	0.413822	0.104297	0.076381	0.107216
1	0.130346	0.344667	0.252189	0.127101	0.145697
2	0.086282	0.123843	0.232533	0.383752	0.173591
3	0.251280	0.112248	0.116560	0.271230	0.248682
4	0.097646	0.137434	0.532700	0.139232	0.092987


