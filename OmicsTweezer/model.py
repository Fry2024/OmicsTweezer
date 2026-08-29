import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import torch


def resolve_device(device=None):
    """
    Select the computation device.

    device=None:
        Use CUDA when available, otherwise CPU.

    device="cpu":
        Force CPU.

    device="cuda" or "cuda:0":
        Request CUDA. Falls back to CPU when CUDA is unavailable.
    """
    if device is None:
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    requested_device = torch.device(device)

    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA was requested but is unavailable. Falling back to CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")

        if requested_device.index is not None:
            if requested_device.index >= torch.cuda.device_count():
                warnings.warn(
                    f"CUDA device {requested_device.index} does not exist. "
                    "Falling back to cuda:0.",
                    RuntimeWarning,
                )
                return torch.device("cuda:0")

    return requested_device


def set_seed(seed=42):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    print(f"Seed is fixed, seed is {seed}")


def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out


class OmicsTweezer(object):
    def __init__(
        self,
        architectures,
        epochs,
        batch_size,
        target_type,
        learning_rate,
        device=None,
        seed=42):
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.target_type = target_type
        self.learning_rate = learning_rate

        self.celltype_num = None
        self.labels = None
        self.used_features = None

        self.seed = int(seed)
        self.outdir = "./result_data/"

        self.architectures_dim = architectures[0]
        self.architectures_drop = architectures[1]

        self.device = resolve_device(device)

        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(self.device)}")

    def OmicsTweezer_model(self, celltype_num):
        feature_num = len(self.used_features)


        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, self.architectures_dim[0], self.architectures_drop[0]), 
                                        EncoderBlock(self.architectures_dim[0], self.architectures_dim[1], self.architectures_drop[1]),  
                                     )

        self.predictor_da = nn.Sequential(EncoderBlock(self.architectures_dim[1], self.architectures_dim[2],self.architectures_drop[2]), 
                                          EncoderBlock(self.architectures_dim[2], self.architectures_dim[3], self.architectures_drop[3]), 
                                          nn.Linear(self.architectures_dim[3], celltype_num), 
                                          nn.Softmax(dim=1))
        



        model_da = nn.ModuleList([])
        print("no initial weight")

        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)

        return model_da

    def prepare_dataloader(
        self,
        source_data,
        target_data,
        batch_size,
    ):
        # ---------------------------------------------------------------
        # Source data
        # ---------------------------------------------------------------
        source_ratios = [
            source_data.obs[ctype]
            for ctype in source_data.uns["cell_types"]
        ]

        self.source_data_x = np.asarray(
            source_data.X,
            dtype=np.float32,
        )

        self.source_data_y = np.asarray(
            source_ratios,
            dtype=np.float32,
        ).T

        tr_data = torch.from_numpy(self.source_data_x)
        tr_labels = torch.from_numpy(self.source_data_y)

        source_dataset = Data.TensorDataset(
            tr_data,
            tr_labels,
        )

        source_generator = torch.Generator()
        source_generator.manual_seed(self.seed)

        self.train_source_loader = Data.DataLoader(
            dataset=source_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=source_generator,
        )

        # ---------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------
        self.labels = source_data.uns["cell_types"]
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # ---------------------------------------------------------------
        # Target data
        # ---------------------------------------------------------------
        self.target_data_x = np.asarray(
            target_data.X,
            dtype=np.float32,
        )

        if self.target_type == "simulated":
            target_ratios = [
                target_data.obs[ctype]
                for ctype in self.labels
            ]

            self.target_data_y = np.asarray(
                target_ratios,
                dtype=np.float32,
            ).T

        elif self.target_type == "real":
            self.target_data_y = np.random.rand(
                target_data.shape[0],
                self.celltype_num,
            ).astype(np.float32)

        else:
            raise ValueError(
                "`target_type` must be 'simulated' or 'real'."
            )

        te_data = torch.from_numpy(self.target_data_x)
        te_labels = torch.from_numpy(self.target_data_y)

        target_dataset = Data.TensorDataset(
            te_data,
            te_labels,
        )

        target_generator = torch.Generator()
        target_generator.manual_seed(self.seed + 1)

        self.train_target_loader = Data.DataLoader(
            dataset=target_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=target_generator,
        )

        self.test_target_loader = Data.DataLoader(
            dataset=target_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def train(self, source_data, target_data, loss_weight):
        set_seed(self.seed)
        # Prepare data loaders.
        self.prepare_dataloader(
            source_data,
            target_data,
            self.batch_size,
        )

        # Build model and move all contained modules to the selected device.
        self.model_da = self.OmicsTweezer_model(
            self.celltype_num
        ).to(self.device)

        self.loss_weight = loss_weight

        optimizer_da1 = torch.optim.Adam(
            [
                {"params": self.encoder_da.parameters()},
                {"params": self.predictor_da.parameters()},
            ],
            lr=self.learning_rate,
        )

        metric_logger = defaultdict(list)

        non_blocking = self.device.type == "cuda"

        for epoch in range(self.num_epochs):
            self.model_da.train()

            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch = 0.0

            for batch_idx, (source_x, source_y) in enumerate(
                self.train_source_loader
            ):
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                # Move all tensors to the same device as the model.
                source_x = source_x.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )

                source_y = source_y.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )

                target_x = target_x.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )

                embedding_source = self.encoder_da(source_x)
                embedding_target = self.encoder_da(target_x)

                frac_pred = self.predictor_da(embedding_source)

                pred_loss = L1_loss(frac_pred, source_y)
                pred_loss_epoch += pred_loss.item()

                w_distance = (
                    embedding_source.mean()
                    - embedding_target.mean()
                )

                loss = (
                    pred_loss
                    + self.loss_weight * torch.abs(w_distance)
                )

                optimizer_da1.zero_grad()
                loss.backward()
                optimizer_da1.step()

            pred_loss_epoch /= batch_idx + 1
            metric_logger["pred_loss"].append(pred_loss_epoch)

            print(
                "============= "
                f"Epoch {epoch + 1:02d}/{self.num_epochs:02d} "
                "in training ============="
            )
            print(f"pred_loss={pred_loss_epoch:f}")

        self.metric_logger = metric_logger
        return metric_logger

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path):
        """
        Write the trained weights and everything needed to rebuild the model to
        a single file.

        Stores the encoder and predictor state dicts together with the
        architecture, the cell-type labels, the gene order the model was fitted
        on, and the constructor settings.
        """
        if not hasattr(self, "encoder_da") or not hasattr(self, "predictor_da"):
            raise RuntimeError("Nothing to save: the model has not been built.")

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "encoder_state_dict": {
                k: v.detach().cpu() for k, v in self.encoder_da.state_dict().items()
            },
            "predictor_state_dict": {
                k: v.detach().cpu() for k, v in self.predictor_da.state_dict().items()
            },
            "architectures_dim": list(self.architectures_dim),
            "architectures_drop": list(self.architectures_drop),
            "celltype_num": int(self.celltype_num),
            "labels": list(np.asarray(self.labels, dtype=str)),
            "used_features": list(self.used_features),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "target_type": self.target_type,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "loss_weight": getattr(self, "loss_weight", None),
            "pred_loss": list(getattr(self, "metric_logger", {}).get("pred_loss", [])),
        }

        torch.save(checkpoint, path)
        print(f"Saved OmicsTweezer model to {path}")
        return path

    @classmethod
    def load(cls, path, device=None, map_location="cpu"):
        """
        Rebuild a trained model from a file written by `save`.

        Returns an instance whose weights are restored and which is ready for
        `predict`, with no training required.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        obj = cls(
            architectures=(
                checkpoint["architectures_dim"],
                checkpoint["architectures_drop"],
            ),
            epochs=checkpoint["num_epochs"],
            batch_size=checkpoint["batch_size"],
            target_type=checkpoint["target_type"],
            learning_rate=checkpoint["learning_rate"],
            device=device,
            seed=checkpoint["seed"],
        )

        obj.labels = np.asarray(checkpoint["labels"], dtype=str)
        obj.celltype_num = int(checkpoint["celltype_num"])
        obj.used_features = list(checkpoint["used_features"])
        obj.loss_weight = checkpoint.get("loss_weight", None)

        obj.model_da = obj.OmicsTweezer_model(obj.celltype_num).to(obj.device)
        obj.encoder_da.load_state_dict(checkpoint["encoder_state_dict"])
        obj.predictor_da.load_state_dict(checkpoint["predictor_state_dict"])
        obj.encoder_da.to(obj.device)
        obj.predictor_da.to(obj.device)

        print(f"Loaded OmicsTweezer model from {path}")
        return obj

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def _target_loader_from(self, target_data, batch_size=None):
        """
        Build an unshuffled loader over arbitrary target data, aligning its
        genes to the order the model was fitted on.
        """
        batch_size = self.batch_size if batch_size is None else batch_size

        if hasattr(target_data, "to_df"):
            frame = target_data.to_df()
        elif isinstance(target_data, pd.DataFrame):
            frame = target_data
        else:
            array = np.asarray(target_data, dtype=np.float32)
            if array.shape[1] != len(self.used_features):
                raise ValueError(
                    f"Target has {array.shape[1]} columns but the model was "
                    f"fitted on {len(self.used_features)} genes. Pass an "
                    f"AnnData or a DataFrame so the genes can be aligned."
                )
            frame = pd.DataFrame(array, columns=self.used_features)

        missing = [g for g in self.used_features if g not in frame.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} of the model's genes are absent from the "
                f"target, e.g. {missing[:5]}."
            )

        frame = frame.loc[:, self.used_features]
        x = np.asarray(frame.to_numpy(), dtype=np.float32)

        dataset = Data.TensorDataset(
            torch.from_numpy(x),
            torch.zeros((x.shape[0], self.celltype_num), dtype=torch.float32),
        )

        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return loader, list(frame.index)

    def prediction(self, target_data=None, batch_size=None):
        """
        Predict cell-type proportions.

        With no argument the target supplied during training is used. Passing
        an AnnData or DataFrame predicts that data instead, which is what a
        reloaded model needs.

        Returns the predicted proportions and the ground truth carried by the
        loader; for new data the ground truth is a zero frame.
        """
        if not hasattr(self, "model_da"):
            raise RuntimeError(
                "No model available. Call train(), or load() a saved model."
            )

        if target_data is None:
            if not hasattr(self, "test_target_loader"):
                raise RuntimeError(
                    "This model carries no target loader, which is the case "
                    "after load(). Pass target_data to predict on."
                )
            loader = self.test_target_loader
            index = None
        else:
            loader, index = self._target_loader_from(target_data, batch_size)

        self.model_da.eval()

        preds = None
        gt = None

        non_blocking = self.device.type == "cuda"

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(
                loader
            ):
                x = x.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )

                logits = self.predictor_da(
                    self.encoder_da(x)
                )

                logits = logits.cpu().numpy()
                frac = y.numpy()

                preds = (
                    logits
                    if preds is None
                    else np.concatenate((preds, logits), axis=0)
                )

                gt = (
                    frac
                    if gt is None
                    else np.concatenate((gt, frac), axis=0)
                )

        target_preds = pd.DataFrame(
            preds,
            columns=self.labels,
        )

        ground_truth = pd.DataFrame(
            gt,
            columns=self.labels,
        )

        if index is not None:
            target_preds.index = index
            ground_truth.index = index

        return target_preds, ground_truth