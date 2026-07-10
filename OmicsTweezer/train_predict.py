from OmicsTweezer import simulation
from OmicsTweezer import deconvolution
from sklearn.preprocessing import MinMaxScaler


def train_predict(
    train_reference,
    test_bulk,
    num=3000,
    scale=True,
    ot_weight=1,
    device=None
):
    """
    Train OmicsTweezer and predict target cell-type proportions.

    Parameters
    ----------
    device : None, str, or torch.device
        None automatically uses CUDA when available, otherwise CPU.
        Use "cpu", "cuda", or "cuda:0" to select explicitly.
    """

    # Generate simulated training bulk samples.
    train_bulk = simulation.generate_simulated_data(
        train_reference,
        samplenum=num,
        sparse=False,
    )

    # Work on copies so the original objects are not modified.
    train_bulk = train_bulk.copy()
    test_bulk_copy = test_bulk.copy()

    if scale:
        scaler = MinMaxScaler()

        # Fit only on training data, then apply the same transformation
        # to target data.
        train_bulk.X = scaler.fit_transform(train_bulk.X)
        test_bulk_copy.X = scaler.transform(test_bulk_copy.X)

    # OmicsTweezer expects cell-type columns in target obs.
    # These are placeholders for real target data.
    if test_bulk_copy.n_obs > train_bulk.n_obs:
        raise ValueError(
            "The target contains more samples than the simulated training data. "
            "Increase `num`."
        )

    test_bulk_copy.obs = train_bulk.obs.iloc[
        :test_bulk_copy.n_obs
    ].copy()

    predict_output, ground_truth = deconvolution.mian(
        necessary_data=train_bulk,
        real_bulk=test_bulk_copy,
        ot_weight=ot_weight,
        sep="\t",
        batch_size=128,
        epochs=30,
        device=device,
    )

    return (
        predict_output[0]
        + predict_output[1]
        + predict_output[2]
    ) / 3