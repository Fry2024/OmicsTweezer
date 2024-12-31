# OmicsTweezer
Implementation of "OmicsTweezer: a data distribution-independent cell deconvolution model for multi-omics data"

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
git clone gihub
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


## Example
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
    Group1	  Group2	  Group3	  Group4	Group5
0	0.215530	0.419044	0.140187	0.102696	0.122542
1	0.139136	0.307193	0.258386	0.125336	0.169949
2	0.128368	0.133724	0.218429	0.368799	0.150679
3	0.272953	0.098638	0.148167	0.256565	0.223677
4	0.100419	0.132774	0.518788	0.142555	0.105465

```
