import os
import json
from functools import reduce

import anndata
import pandas as pd
import numpy as np
import random
import torch

from .utils import ProcessInputData
from .simulation import generate_simulated_data
from OmicsTweezer import model


ARCHITECTURES = {
    "m256": (
        [256, 128, 64, 32],
        [0, 0, 0, 0],
    ),
    "m512": (
        [512, 256, 128, 64],
        [0, 0.3, 0.2, 0.1],
    ),
    "m1024": (
        [1024, 512, 256, 128],
        [0, 0.6, 0.3, 0.1],
    ),
}

ARCHITECTURE_ORDER = ["m256", "m512", "m1024"]


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, CPU, and CUDA random seeds."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Helps identify unsupported nondeterministic operations without crashing.
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Seed is fixed, seed is {seed}")


def _resolve_simudata(necessary_data, sparse, seed):
    """Return the simulated pseudobulk for an object or a supported path."""
    if isinstance(necessary_data, str):
        postfix = necessary_data.rsplit(".", 1)[-1].lower()

        if postfix == "txt":
            return generate_simulated_data(
                sc_data=necessary_data,
                samplenum=5000,
                sparse=sparse,
                random_state=seed,
            )

        if postfix == "h5ad":
            return anndata.read_h5ad(necessary_data)

        if postfix == "pth":
            raise ValueError("Do not accept a model as input.")

        raise ValueError("Unsupported input file. Expected .txt or .h5ad.")

    if isinstance(necessary_data, pd.DataFrame):
        print("DataFrame")
        return necessary_data

    if isinstance(necessary_data, anndata.AnnData):
        print("AnnData")
        return necessary_data

    raise TypeError(
        "`necessary_data` must be an AnnData object, "
        "pandas DataFrame, or supported file path."
    )


def prepare_inputs(necessary_data, real_bulk, sep="\t", sparse=True, seed=42):
    """
    Turn a simulated pseudobulk and a target bulk into the matrices the models
    consume.

    Returns the processed training matrix and labels, the processed target
    matrix, the gene names, the cell types and the sample names.
    """
    simudata = _resolve_simudata(necessary_data, sparse, seed)

    if not isinstance(real_bulk, anndata.AnnData):
        raise TypeError("`real_bulk` must be an AnnData object.")

    # AnnData.to_df() safely handles dense and sparse expression matrices.
    real_bulk_df = real_bulk.to_df()

    (
        train_x,
        train_y,
        test_x,
        genename,
        celltypes,
        samplename,
    ) = ProcessInputData(
        simudata,
        real_bulk_df,
        sep=sep,
        datatype="counts",
    )

    print(
        "training data shape is ",
        train_x.shape,
        "\ntest data shape is ",
        test_x.shape,
    )

    return train_x, train_y, test_x, genename, celltypes, samplename


def _build_adatas(train_x, train_y, test_x, test_y):
    """Wrap the processed matrices as the AnnData pair the model expects."""
    if not isinstance(test_y, pd.DataFrame):
        test_y = pd.DataFrame(test_y)

    cell_types = np.asarray(test_y.columns, dtype=str)

    if train_y.shape[1] != len(cell_types):
        raise ValueError(
            f"Training labels contain {train_y.shape[1]} cell types, "
            f"but target metadata contain {len(cell_types)} columns."
        )

    train_y_df = pd.DataFrame(
        train_y,
        columns=cell_types,
    )

    train_adata = anndata.AnnData(
        X=np.asarray(train_x, dtype=np.float32),
        obs=train_y_df.copy(),
    )

    test_adata = anndata.AnnData(
        X=np.asarray(test_x, dtype=np.float32),
        obs=test_y.copy(),
    )

    train_adata.uns["cell_types"] = cell_types
    test_adata.uns["cell_types"] = cell_types

    return train_adata, test_adata, cell_types


def train(
    train_x,
    train_y,
    test_x,
    test_y,
    ot_weight,
    batch_size=128,
    epochs=128,
    device=None,
    seed=42,
    model_dir=None,
):
    """
    Train the m256, m512 and m1024 models and return them keyed by
    architecture.

    When `model_dir` is given every model is written there as
    `omicstweezer_<architecture>.pt`, together with a `models.json` manifest,
    so inference can be repeated later without training again.
    """
    train_adata, test_adata, cell_types = _build_adatas(
        train_x, train_y, test_x, test_y
    )

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)

    trained = {}
    saved_paths = {}

    for key in ARCHITECTURE_ORDER:
        print(f"Training {key} with run seed {seed}")

        model_da = model.OmicsTweezer(
            architectures=ARCHITECTURES[key],
            epochs=epochs,
            batch_size=batch_size,
            target_type="simulated",
            learning_rate=0.0001,
            device=device,
            seed=seed
        )

        print(train_adata)
        print(test_adata)
        print("OT weight:", ot_weight)

        model_da.train(
            train_adata,
            test_adata,
            ot_weight,
        )

        print(f"{key} model training finished!")

        trained[key] = model_da

        if model_dir is not None:
            path = os.path.join(model_dir, f"omicstweezer_{key}.pt")
            model_da.save(path)
            saved_paths[key] = path

    if model_dir is not None:
        manifest = {
            "architectures": ARCHITECTURE_ORDER,
            "paths": saved_paths,
            "cell_types": list(np.asarray(cell_types, dtype=str)),
            "ot_weight": ot_weight,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
        }
        with open(os.path.join(model_dir, "models.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    return trained


def load_models(model_dir, device=None):
    """Load the ensemble written by `train` into a dict keyed by architecture."""
    manifest_path = os.path.join(model_dir, "models.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        paths = manifest["paths"]
    else:
        paths = {
            key: os.path.join(model_dir, f"omicstweezer_{key}.pt")
            for key in ARCHITECTURE_ORDER
        }

    loaded = {}
    for key in ARCHITECTURE_ORDER:
        path = paths[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        loaded[key] = model.OmicsTweezer.load(path, device=device)

    return loaded


def predict(
    models=None,
    model_dir=None,
    target_data=None,
    device=None,
    batch_size=None,
    output_dir=None,
    prefix="omicstweezer",
):
    """
    Predict cell-type proportions with the ensemble and average them.

    Pass `models` as returned by `train`, or `model_dir` to load a previously
    saved ensemble. `target_data` selects what to predict: leave it out to use
    the target each model was trained against, or pass an AnnData, DataFrame
    or array to score other samples.

    Returns a dict with:
        "average"      the mean of the three models, samples x cell types
        "per_model"    each architecture's own predictions, keyed by name
        "ground_truth" the labels carried by the loader, keyed by name

    When `output_dir` is given the average and the per-model tables are also
    written there as tab-separated files.
    """
    if models is None and model_dir is None:
        raise ValueError("Provide either `models` or `model_dir`.")

    if models is None:
        models = load_models(model_dir, device=device)

    per_model = {}
    ground_truth = {}

    for key in ARCHITECTURE_ORDER:
        if key not in models:
            raise KeyError(f"No model for architecture {key}.")

        predictions, truth = models[key].prediction(
            target_data=target_data,
            batch_size=batch_size,
        )

        per_model[key] = predictions
        ground_truth[key] = truth

    ordered = [per_model[key] for key in ARCHITECTURE_ORDER]
    average = reduce(lambda a, b: a + b, ordered) / len(ordered)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        for key in ARCHITECTURE_ORDER:
            path = os.path.join(output_dir, f"{prefix}_proportions_{key}.tsv")
            per_model[key].to_csv(path, sep="\t")
            print(f"Wrote {key} proportions to {path}")

        average_path = os.path.join(output_dir, f"{prefix}_proportions.tsv")
        average.to_csv(average_path, sep="\t")
        print(f"Wrote averaged proportions to {average_path}")

    return {
        "average": average,
        "per_model": per_model,
        "ground_truth": ground_truth,
    }


def test(
    train_x,
    train_y,
    test_x,
    test_y,
    ot_weight,
    batch_size=128,
    epochs=128,
    device=None,
    seed=42,
    model_dir=None,
    output_dir=None,
):
    """
    Train the ensemble and predict its training target.

    Returns the dict produced by `predict`.
    """
    models = train(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        ot_weight=ot_weight,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        seed=seed,
        model_dir=model_dir,
    )

    return predict(models=models, output_dir=output_dir)


def mian(
    necessary_data,
    real_bulk,
    ot_weight=1,
    sep="\t",
    sparse=True,
    batch_size=128,
    epochs=128,
    device=None,
    seed=42,
    model_dir=None,
    output_dir=None,
):
    """
    Train OmicsTweezer and predict cell-type proportions.

    Parameters
    ----------
    necessary_data
        Simulated pseudobulk AnnData object or input path.
    real_bulk
        Target bulk AnnData object.
    ot_weight
        Weight for the domain adaptation loss.
    batch_size
        Training batch size.
    epochs
        Number of training epochs.
    device
        None automatically selects CUDA when available, otherwise CPU.
        Use "cpu", "cuda", or "cuda:0" to select explicitly.
    model_dir
        When given, the trained models are written there for later reuse.
    output_dir
        When given, the averaged and per-model proportions are written there.

    Returns the dict produced by `predict`.
    """
    seed = int(seed)
    set_seed(seed)

    print("begin")
    print(f"Requested OmicsTweezer seed: {seed}")

    (
        train_x,
        train_y,
        test_x,
        genename,
        celltypes,
        samplename,
    ) = prepare_inputs(
        necessary_data,
        real_bulk,
        sep=sep,
        sparse=sparse,
        seed=seed,
    )

    return test(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=real_bulk.obs,
        ot_weight=ot_weight,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        seed=seed,
        model_dir=model_dir,
        output_dir=output_dir,
    )