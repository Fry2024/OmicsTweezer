import anndata
import pandas as pd
import numpy as np
import random
import torch

from .utils import ProcessInputData
from .simulation import generate_simulated_data
from OmicsTweezer import model


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





def mian(
    necessary_data,
    real_bulk,
    ot_weight=1,
    sep="\t",
    sparse=True,
    batch_size=128,
    epochs=128,
    device=None,
    seed=42
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
    """
    seed = int(seed)
    set_seed(seed)

    print("begin")
    print(f"Requested OmicsTweezer seed: {seed}")

    if isinstance(necessary_data, str):
        postfix = necessary_data.rsplit(".", 1)[-1].lower()

        if postfix == "txt":
            simudata = generate_simulated_data(
                sc_data=necessary_data,
                samplenum=5000,
                sparse=sparse,
                random_state=seed,
            )

        elif postfix == "h5ad":
            simudata = anndata.read_h5ad(necessary_data)

        elif postfix == "pth":
            raise ValueError("Do not accept a model as input.")

        else:
            raise ValueError(
                "Unsupported input file. Expected .txt or .h5ad."
            )

    elif isinstance(necessary_data, pd.DataFrame):
        simudata = necessary_data
        print("DataFrame")

    elif isinstance(necessary_data, anndata.AnnData):
        simudata = necessary_data
        print("AnnData")

    else:
        raise TypeError(
            "`necessary_data` must be an AnnData object, "
            "pandas DataFrame, or supported file path."
        )

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

    pred, ground_truth = test(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=real_bulk.obs,
        ot_weight=ot_weight,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        seed=seed
    )

    return pred, ground_truth


def test(
    train_x,
    train_y,
    test_x,
    test_y,
    ot_weight,
    batch_size=128,
    epochs=128,
    device=None,
    seed=42
):
    architectures = {
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

    if not isinstance(test_y, pd.DataFrame):
        test_y = pd.DataFrame(test_y)

    cell_types = np.asarray(test_y.columns, dtype=str)

    if train_y.shape[1] != len(cell_types):
        raise ValueError(
            f"Training labels contain {train_y.shape[1]} cell types, "
            f"but target metadata contain {len(cell_types)} columns."
        )

    # Construct this only once instead of recreating it inside each model loop.
    train_y_df = pd.DataFrame(
        train_y,
        columns=cell_types,
    )

    final_predictions = []
    final_ground_truth = []

    for key in ["m256", "m512", "m1024"]:
        print(f"Training {key} with run seed {seed}")

        model_da = model.OmicsTweezer(
            architectures=architectures[key],
            epochs=epochs,
            batch_size=batch_size,
            target_type="simulated",
            learning_rate=0.0001,
            device=device,
            seed=seed
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

        print(train_adata)
        print(test_adata)
        print("OT weight:", ot_weight)

        model_da.train(
            train_adata,
            test_adata,
            ot_weight,
        )

        print(f"{key} model training finished!")

        predictions, ground_truth = model_da.prediction()

        final_predictions.append(predictions)
        final_ground_truth.append(ground_truth)

    return final_predictions, final_ground_truth