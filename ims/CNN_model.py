

"""Lightweight CNN model wrapper for GC-IMS datasets.

Provides a CNNModel class that integrates with ims.Dataset objects.

Functionality:
- build simple Conv2D model compatible with ims.Spectrum data
- fit / evaluate / save / load model
- plot training history (accuracy/loss)
- generate class-wise saliency maps and average maps
- compute PCA on saliency maps and export PC1 loadings maps

This module keeps external dependencies minimal dependencies, 
like tensorflow and tf-keras-vis need to be installed for the package to properly work
"""
from typing import Optional, Sequence, Dict
import os
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2  
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from sklearn.decomposition import PCA


class CNNModel:
    """Minimal CNN wrapper for GC-IMS `ims.Dataset`.

    Contract (inputs/outputs):
    - Input: "ims.Dataset" where each element is an "ims.Spectrum"
    - Labels: "dataset.labels" (list-like)
    - Output: trained Keras model saved to disk, saliency and PCA images

    Basic usage:
    >>> import ims
    >>> from ims.CNN_model import CNNModel
    >>> ds = ims.Dataset.read_mea("data", subfolders=True)
    >>> model = CNNModel(dataset=ds)
    >>> model.fit(epochs=10)
    >>> model.save("my_model.h5")
    """

    def __init__(self, dataset, name: Optional[str] = None):
        self.dataset = dataset
        self.name = name 
        self.model = None
        self.history = None
        self.prepared = False

    def prep_data(self):
        """Convert dataset to X, Y arrays suitable for Keras."""
        X = []
        for i in self.dataset.data:
            arr = np.array(i.values, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            X.append(arr)

        X = np.stack(X)
        Y = np.array(self.dataset.labels) 
        y_enc = LabelEncoder().fit_transform(Y)
        y_cat = to_categorical(y_enc)

        self.X = X
        self.Y = y_cat
        self.y_raw = Y
        self.prepared = True
        return X, y_cat

    def build(self, input_shape: Optional[Sequence[int]] = None, num_classes: Optional[int] = None):
        """Builds a simple CNN.

        If input_shape or num_classes are not given they are inferred from the dataset
        after calling "prep_data()".
        """
        if tf is None:
            raise RuntimeError("TensorFlow is required to build the model")

        if not self.prepared:
            print("prepare data with the prep_data() method before building the model")

        input_shape = input_shape or tuple(self.X.shape[1:])
        num_classes = num_classes or self.Y.shape[1]

        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), padding="same", input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(48, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(layers.Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def fit(self, epochs=20, batch_size=8, validation_split=0.2, **fit_kwargs):
        """Train the model on the dataset.

        Stores training history in `self.history`.
        """
        if self.model is None:
            self.build()
            print("build() method not called, model is built automatically")

        self.history = self.model.fit(
            self.X,
            self.Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            **fit_kwargs,
        )
        return self.history

    def evaluate(self, X=None, Y=None):
        X = X if X is not None else getattr(self, "X")
        Y = Y if Y is not None else getattr(self, "Y")
        return self.model.evaluate(X, Y)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history from self.history."""
        if self.history is None:
            print("No training history found. Train the model first using fit().")
            return
            
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history.get("accuracy", []), label="train_acc")
        plt.plot(self.history.history.get("val_accuracy", []), label="val_acc", linestyle="--")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history.get("loss", []), label="train_loss")
        plt.plot(self.history.history.get("val_loss", []), label="val_loss", linestyle="--")
        plt.title("Loss")
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def generate_saliency_maps(self, output_dir: str = "saliency_maps", smooth_samples: int = 25, smooth_noise: float = 0.05):
        """Generate and save saliency maps per sample and compute average per class.

        Returns a dict mapping class label -> list of saliency maps (2D arrays).
        """
        os.makedirs(output_dir, exist_ok=True)
        saliency = Saliency(self.model)

        maps_by_class: Dict[str, list] = {c: [] for c in sorted(set(self.y_raw))}

        for i in range(len(self.X)):
            x_input = np.expand_dims(self.X[i], axis=0)
            # infer class index by model prediction on sample or from labels
            class_index = int(np.argmax(self.Y[i]))
            score = CategoricalScore([class_index])
            sal_map = saliency(score, x_input, smooth_samples=smooth_samples, smooth_noise=smooth_noise)[0]

            label = self.y_raw[i]
            maps_by_class[label].append(sal_map)

            # save individual map
            fname = os.path.join(output_dir, f"saliency_{label}_{i}.png")
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            im = ax.imshow(sal_map, cmap="RdBu_r", origin="lower", aspect="auto")
            ax.set_title(f"{label} #{i}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            plt.close()

        # save average maps per class
        avg_maps = {}
        for label, maps in maps_by_class.items():
            if len(maps) == 0:
                continue
            avg = np.mean(np.stack(maps), axis=0)
            avg_maps[label] = avg
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            im = ax.imshow(avg, cmap="RdBu_r", origin="lower", aspect="auto")
            ax.set_title(f"Average saliency: {label}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"average_saliency_{label}.png"), bbox_inches="tight")
            plt.close()

        return maps_by_class, avg_maps

    def saliency_pca_loadings(self, maps_by_class: Dict[str, list], output_dir: str = "saliency_maps"):
        """Compute PCA per-class on flattened saliency maps and save PC1 loadings visualization."""
        os.makedirs(output_dir, exist_ok=True)
        pc1_maps = {}
        for label, maps in maps_by_class.items():
            if len(maps) == 0:
                continue
            flat = [m.flatten() for m in maps]
            M = np.stack(flat)
            pca = PCA()
            pca.fit(M)
            pc1 = pca.components_[0].reshape(maps[0].shape)
            # normalize for display
            mx = np.max(np.abs(pc1))
            if mx > 0:
                pc1n = pc1 / mx
            else:
                pc1n = pc1

            pc1_maps[label] = pc1n
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            im = ax.imshow(pc1n, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-1, vmax=1)
            ax.set_title(f"PC1 loadings: {label}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pc1_loadings_{label}.png"), bbox_inches="tight")
            plt.close()

        return pc1_maps
