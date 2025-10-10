"""Lightweight CNN model wrapper for GC-IMS data.

Provides a CNN classification model for GC-IMS data. Labels need to be provided via subfolder read in,
manual assignment, or automatic labelling--> see ims.Dataset.read_mea() class for details.


Functionality:
- builds a simple Conv2D model compatible with ims.Dataset
- fit / evaluate / save / load model
- plot training history (accuracy/loss)
- generate class-wise saliency maps and average maps
- compute PCA on saliency maps and export PC1 loadings maps

This module uses external dependencies that need to be installed manually prior to use:
- TensorFlow/Keras >=(TF 2.11)
- tf-keras-vis tested on (v0.8.7)
"""
from typing import Optional, Sequence, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2  
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import pandas as pd

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from sklearn.decomposition import PCA


class CNNModel:
    """Minimal CNN wrapper for GC-IMS `ims.Dataset`.

    Contract (inputs/outputs):
    - Input: `ims.Dataset` where each element is an `ims.Spectrum` with `.values` matrix
    - Labels: `dataset.labels` (list-like)
    - Output: trained Keras model saved to disk, saliency and PCA images

    Parameters
    ----------
    dataset : ims.Dataset
        Dataset containing GC-IMS spectra with labels
    name : str, optional
        Optional name identifier for the model

    Attributes
    ----------
    dataset : ims.Dataset
        Input dataset
    name : str or None
        Model name identifier
    model : tensorflow.keras.Model or None
        Compiled Keras model
    history : tensorflow.keras.callbacks.History or None
        Training history after fitting
    prepared : bool
        Flag indicating if data has been prepared for training
    X : np.ndarray
        Prepared feature arrays (after prep_data())
    Y : np.ndarray
        One-hot encoded labels (after prep_data())
    y_raw : np.ndarray
        Original string labels (after prep_data())
    validation_split : float or None
        Validation split ratio used during training
    train_indices : list or None
        Indices of samples used for training
    val_indices : list or None
        Indices of samples used for validation

    Examples
    --------
    >>> import ims
    >>> from ims.CNN_model import CNNModel
    >>> ds = ims.Dataset.read_mea("data")
    >>> model = CNNModel(ds)
    >>> model.fit(epochs=10, validation_split=0.2)
    >>> model.save("my_model.h5")
    >>> # Generate validation report
    >>> report, sample_info = model.val_report(val_info=True)
    >>> print(sample_info.head())  # Show sample split information
    """

    def __init__(self, dataset, name: Optional[str] = None):
        self.dataset = dataset
        self.name = name 
        self.model = None
        self.history = None
        self.prepared = False
        self.validation_split = None
        self.train_indices = None
        self.val_indices = None

    def prep_data(self):
        """Convert dataset to X, y_cat arrays.

        Transforms the preproccessed dataset into numpy arrays, and one-hot encoded labels.
        Expands 2D spectrum arrays to 3D (adding channel dimension).
        IMPORTANT: Make sure that you perfrom proper preprocessing on the dataset!
        Call methods like ims.Dataset.interp_riprel().cut_dt().cut_rt().normalization() or .scaling()
        for preprocessing.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X : Feature arrays with shape (n_samples, height, width, channels)
            y_cat : One-hot encoded labels with shape (n_samples, n_classes)

        Notes
        -----
        Sets instance attributes X, Y, y_raw and prepared flag.
        Must be called before build() or fit().
        """
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
        """Build a simple CNN architecture.

        Creates a sequential CNN with 2 convolutional blocks followed by dense layers.
        Architecture: Conv2D(8) -> BatchNorm -> ReLU -> MaxPool -> Conv2D(32) -> 
        BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense(48) -> Dense(32) -> Dense(n_classes)

        Parameters
        ----------
        input_shape : tuple of int, optional
            Shape of input tensors (height, width, channels).
            If None, inferred from prepared data.
        num_classes : int, optional
            Number of output classes for classification.
            If None, inferred from prepared data labels.

        Returns
        -------
        tensorflow.keras.Model
            Compiled Keras model ready for training

        Raises
        ------
        RuntimeError
            If TensorFlow is not available
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

        Trains the CNN using prepared data with specified hyperparameters.
        Automatically builds model if not already done.

        Parameters
        ----------
        epochs : int, default=20
            Number of training epochs
        batch_size : int, default=8
            Batch size for training
        validation_split : float, default=0.2
            Fraction of data to use for validation (0-1)
        **fit_kwargs
            Additional keyword arguments passed to model.fit()

        Returns
        -------
        tensorflow.keras.callbacks.History
            Training history object containing loss and metric values

        Notes
        -----
        Stores training history in self.history attribute.
        """
        if self.model is None:
            self.build()
            print("build() method not called, model is built automatically")

        # Store validation split and calculate indices
        self.validation_split = validation_split
        n_samples = len(self.X)
        n_val = int(n_samples * validation_split)
        
        # Keras uses the last validation_split fraction of data for validation
        self.train_indices = list(range(n_samples - n_val))
        self.val_indices = list(range(n_samples - n_val, n_samples))

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
        """Evaluate model performance on given data.

        Parameters
        ----------
        X : np.ndarray, optional
            Feature arrays. If None, uses self.X from prepared data.
        Y : np.ndarray, optional
            Target labels. If None, uses self.Y from prepared data.

        Returns
        -------
        list
            Evaluation metrics [loss, accuracy, ...]
        """
        X = X if X is not None else getattr(self, "X")
        Y = Y if Y is not None else getattr(self, "Y")
        return self.model.evaluate(X, Y)

    def val_report(self, val_info=True):
        """Generate validation report with classification metrics and sample split information.

        Creates a detailed validation report including:
        - Classification report (precision, recall, f1-score) for validation set
        - Optional sample split information showing which samples went to train/validation

        Parameters
        ----------
        val_info : bool, default=True
            If True, returns DataFrame with sample split information.
            If False, only returns classification report.

        Returns
        -------
        dict or tuple
            If val_info=False: Classification report dictionary
            If val_info=True: Tuple of (classification_report_dict, samples_dataframe)

        Raises
        ------
        ValueError
            If model hasn't been trained yet or validation split wasn't used

        Notes
        -----
        Requires that fit() has been called with validation_split > 0.
        The classification report is computed on the validation set only.

        Examples
        --------
        >>> model.fit(epochs=10, validation_split=0.2)
        >>> # Get only classification report
        >>> report = model.val_report(val_info=False)
        >>> # Get both report and sample info
        >>> report, sample_df = model.val_report(val_info=True)
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating validation report. Call fit() first.")
        
        if self.val_indices is None or len(self.val_indices) == 0:
            raise ValueError("No validation data found. Make sure fit() was called with validation_split > 0.")

        # Get validation data
        X_val = self.X[self.val_indices]
        Y_val = self.Y[self.val_indices]
        y_val_raw = self.y_raw[self.val_indices]

        # Get predictions on validation set
        val_predictions = self.model.predict(X_val, verbose=0)
        val_pred_indices = np.argmax(val_predictions, axis=1)
        
        # Convert indices back to labels
        unique_labels = sorted(set(self.y_raw))
        val_pred_labels = [unique_labels[idx] for idx in val_pred_indices]

        # Generate classification report
        class_report = classification_report(
            y_val_raw, 
            val_pred_labels, 
            output_dict=True,
            zero_division=0
        )

        # Print formatted classification report
        print("Validation Classification Report:")
        print("=" * 50)
        print(classification_report(y_val_raw, val_pred_labels, zero_division=0))
        print(f"\nValidation samples: {len(self.val_indices)}")
        print(f"Training samples: {len(self.train_indices)}")
        print(f"Validation split: {self.validation_split:.1%}")

        if not val_info:
            return class_report

        # Create sample information DataFrame
        sample_data = []
        
        # Add training samples
        for idx in self.train_indices:
            sample_data.append({
                'sample_index': idx,
                'sample_name': self.dataset.data[idx].name if hasattr(self.dataset.data[idx], 'name') else f"sample_{idx}",
                'true_label': self.y_raw[idx],
                'split': 'train',
                'predicted_label': None,  # No predictions for training in this context
                'prediction_probability': None
            })
        
        # Add validation samples with predictions
        for i, idx in enumerate(self.val_indices):
            pred_probs = val_predictions[i]
            max_prob = np.max(pred_probs)
            
            sample_data.append({
                'sample_index': idx,
                'sample_name': self.dataset.data[idx].name if hasattr(self.dataset.data[idx], 'name') else f"sample_{idx}",
                'true_label': y_val_raw[i],
                'split': 'validation',
                'predicted_label': val_pred_labels[i],
                'prediction_probability': max_prob
            })

        samples_df = pd.DataFrame(sample_data)
        
        # Print sample split summary
        print(f"\nSample Split Summary:")
        print(f"Total samples: {len(self.dataset.data)}")
        print(f"Training samples: {len(self.train_indices)}")
        print(f"Validation samples: {len(self.val_indices)}")
        print(f"\nValidation samples by class:")
        val_class_counts = samples_df[samples_df['split'] == 'validation']['true_label'].value_counts().sort_index()
        for label, count in val_class_counts.items():
            print(f"  {label}: {count}")

        return class_report, samples_df

    def save(self, path: str):
        """Save the trained model to disk.

        Parameters
        ----------
        path : str
            File path where to save the model (e.g., "model.h5")

        Notes
        -----
        Creates directory if it doesn't exist.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history from self.history.

        Creates a 2-subplot figure showing accuracy and loss curves
        for both training and validation data over epochs.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot image. If None, only displays plot.

        Notes
        -----
        Requires that fit() has been called to generate training history.
        """
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

    def salmap(self, smooth_samples: int = 25, smooth_noise: float = 0.05, save_single_salmap: bool = False, **kwargs):
        """Generate and save saliency maps per sample and compute average per class.

        Creates gradient-based saliency maps showing which input regions
        most influence model predictions. Optionally saves individual maps and 
        always computes class-averaged maps.

        Parameters
        ----------
        smooth_samples : int, default=25
            Number of samples for SmoothGrad noise averaging
        smooth_noise : float, default=0.05
            Standard deviation of noise added for SmoothGrad
        save_single_salmap : bool, default=False
            Whether to save individual saliency maps for each spectrum
        **kwargs
            Additional keyword arguments:
            - output_dir : str, default="saliency_maps"
              Directory to save individual saliency maps (only used if save_single_salmap=True)

        Returns
        -------
        tuple[Dict[str, list], Dict[str, np.ndarray]]
            maps_by_class : Dictionary mapping class labels to lists of saliency maps
            avg_maps : Dictionary mapping class labels to averaged saliency maps

        Notes
        -----
        Uses tf-keras-vis library for gradient computation.
        If save_single_salmap=True, saves individual maps as "salmap_{label}_{index}.png"
        Use export_salmaps() method to save average maps with custom format/dpi.
        """
        saliency = Saliency(self.model)
        maps_by_class: Dict[str, list] = {c: [] for c in sorted(set(self.y_raw))}

        # Create progress bar for saliency map generation
        total_samples = len(self.X)
        pbar = tqdm(total=total_samples, desc="Generating saliency maps", unit="maps")
        
        for i in range(total_samples):
            x_input = np.expand_dims(self.X[i], axis=0)
            class_index = int(np.argmax(self.Y[i]))
            score = CategoricalScore([class_index])
            sal_map = saliency(score, x_input, smooth_samples=smooth_samples, smooth_noise=smooth_noise)[0]

            label = self.y_raw[i]
            maps_by_class[label].append(sal_map)

            # optionally save individual map
            if save_single_salmap:
                output_dir = kwargs.get('output_dir', 'saliency_maps')
                os.makedirs(output_dir, exist_ok=True)
                fname = os.path.join(output_dir, f"salmap_{label}_{i}.png")
                fig, ax = self._plot_salmap(sal_map, f"{label} #{i}", i)
                fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.2)
                plt.close(fig)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Current": f"{label} (#{i})"})
        
        pbar.close()

        # compute average maps per class and display them
        avg_maps = {}
        for label, maps in maps_by_class.items():
            if len(maps) == 0:
                continue
            avg = np.mean(np.stack(maps), axis=0)
            avg_maps[label] = avg
            # display average map
            fig, ax = self._plot_salmap(avg, f"Average salmap: {label}")
            plt.show()

        return maps_by_class, avg_maps

    def _plot_salmap(self, sal_map, title, spectrum_idx=0, vmin=None, vmax=None, width=6, height=6):
        """Internal method to plot saliency maps with consistent styling.
        
        Parameters
        ----------
        sal_map : np.ndarray
            2D saliency map to plot
        title : str
            Plot title
        spectrum_idx : int, default=0
            Index of spectrum to use for axis extents
        vmin : float, optional
            Minimum value for colormap scaling. If None, uses automatic scaling.
        vmax : float, optional
            Maximum value for colormap scaling. If None, uses automatic scaling.
        width : int, default=6
            Figure width in inches
        height : int, default=6
            Figure height in inches
            
        Returns
        -------
        tuple
            (matplotlib.figure.Figure, matplotlib.pyplot.axes)
        """
        fig, ax = plt.subplots(figsize=(width, height))
        
        # use first spectrum for consistent axis mapping
        spectrum = self.dataset.data[spectrum_idx]
        
        im = ax.imshow(
            sal_map,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        
            extent=(
                min(spectrum.drift_time),
                max(spectrum.drift_time),
                min(spectrum.ret_time),
                max(spectrum.ret_time),
            ),
        )
        
        plt.colorbar(im, ax=ax).set_label("Saliency [arbitrary units]")
        ax.set_title(title)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.set_xlabel(spectrum._drift_time_label)
        ax.set_ylabel("Retention time [s]")
        
        plt.tight_layout()
        return fig, ax

    def PCA_salmaps(self, maps_by_class: Dict[str, list]):
        """Compute PCA per-class on flattened saliency maps and display PC1 loadings visualization.

        Performs Principal Component Analysis on saliency maps within each class
        to identify common spatial patterns. Visualizes the first principal component
        loadings which captures the highest variance saliency features.

        Parameters
        ----------
        maps_by_class : Dict[str, list]
            Dictionary mapping class labels to lists of 2D saliency maps,
            typically from salmap()

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping class labels to normalized PC1 loadings arrays

        Notes
        -----
        For each class, flattens all saliency maps, fits PCA, and reshapes 
        the first principal component back to 2D for visualization.
        Use export_salmaps() method to save PC1 loadings.
        """
        pc1_maps = {}
        for label, maps in maps_by_class.items():
            if len(maps) == 0:
                continue
            # stack and reshape maps to 2D array (n_samples, height*width)
            flat_maps = np.reshape(np.stack(maps), (len(maps), -1))

            
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(flat_maps)

            # reshape PC1 back to 2D image
            pc1_map = pc1.reshape(maps[0].shape[0], maps[0].shape[1])

            # normalize to [0, 1] range for visualization
            pc1_map = (pc1_map - np.min(pc1_map)) / (np.max(pc1_map) - np.min(pc1_map))

            pc1_maps[label] = pc1_map

            
            fig, ax = self._plot_PCA_salmap(pc1_map, f"PC1 loadings: {label}")

        return pc1_maps

    def _plot_PCA_salmap(self, pc1_map, title, vmin=None, vmax=None, width=6, height=6):
        """Internal method to plot PCA saliency maps (PC1 loadings) with consistent styling.
        
        Parameters
        ----------
        pc1_map : np.ndarray
            2D PC1 loadings map to plot
        title : str
            Plot title
        vmin : float, optional
            Minimum value for colormap scaling. If None, uses automatic scaling.
        vmax : float, optional
            Maximum value for colormap scaling. If None, uses automatic scaling.
        width : int, default=6
            Figure width in inches
        height : int, default=6
            Figure height in inches
            
        Returns
        -------
        tuple
            (matplotlib.figure.Figure, matplotlib.pyplot.axes)
        """
        fig, ax = plt.subplots(figsize=(width, height))
        
        im = ax.imshow(
            pc1_map,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        
        plt.colorbar(im, ax=ax).set_label("PC1 Loadings [arbitrary units]")
        ax.set_title(title)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.set_xlabel("Drift time [ms]")
        ax.set_ylabel("Retention time [s]")
        
        plt.tight_layout()
        return fig, ax

    def export_salmaps(self, avg_maps: Dict[str, np.ndarray] = None, pc1_maps: Dict[str, np.ndarray] = None, path: str = None, dpi: int = 300, file_format: str = "png", **kwargs):
        """Export saliency maps and/or PC1 loadings as image files.

        Parameters
        ----------
        avg_maps : Dict[str, np.ndarray], optional
            Dictionary of average saliency maps from salmap() method
        pc1_maps : Dict[str, np.ndarray], optional
            Dictionary of PC1 loadings from PCA_salmaps() method
        path : str, optional
            Directory to save images, by default current working directory
        dpi : int, default=300
            Resolution for saved images
        file_format : str, default="png"
            Image format (png, jpg, svg, etc.)
        **kwargs
            Additional arguments:
            - vmin : float, optional
              Minimum value for colormap scaling. If not provided, uses automatic scaling.
            - vmax : float, optional
              Maximum value for colormap scaling. If not provided, uses automatic scaling.
            - Other arguments passed to matplotlib savefig

        Notes
        -----
        At least one of avg_maps or pc1_maps must be provided.
        Saves average saliency maps as "average_salmap_{label}.{format}"
        and PC1 loadings as "pc1_loadings_{label}.{format}".
        By default, uses automatic colormap scaling for optimal contrast.

        Examples
        --------
        >>> maps_by_class, avg_maps = model.salmap()
        >>> pc1_maps = model.PCA_salmaps(maps_by_class)
        >>> # Export with automatic scaling (default)
        >>> model.export_salmaps(avg_maps, pc1_maps)
        >>> # Export with custom colormap range
        >>> model.export_salmaps(avg_maps, pc1_maps, dpi=600, file_format="svg", vmin=0.2, vmax=0.8)
        >>> # Export only average maps
        >>> model.export_salmaps(avg_maps=avg_maps)
        >>> # Export only PC1 loadings with manual scaling
        >>> model.export_salmaps(pc1_maps=pc1_maps, vmin=0, vmax=1)
        """
        if avg_maps is None and pc1_maps is None:
            raise ValueError("At least one of avg_maps or pc1_maps must be provided")
        
        if path is None:
            path = os.getcwd()
        
        os.makedirs(path, exist_ok=True)
        
        # Extract plot parameters from kwargs (None means automatic scaling)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        
        # Export average saliency maps
        if avg_maps is not None:
            for label, avg_map in avg_maps.items():
                fig, ax = self._plot_salmap(avg_map, f"Average salmap: {label}", vmin=vmin, vmax=vmax)
                fig.savefig(
                    f"{path}/avg_salmap_{label}.{file_format}",
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.2,
                    **kwargs
                )
                plt.close(fig)
        
        # Export PC1 loadings
        if pc1_maps is not None:
            for label, pc1_map in pc1_maps.items():
                fig, ax = self._plot_PCA_salmap(pc1_map, f"PC1 loadings: {label}", vmin=vmin, vmax=vmax)
                fig.savefig(
                    f"{path}/pc1_loadings_{label}.{file_format}",
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.2,
                    **kwargs
                )
                plt.close(fig)

    @classmethod
    def load(cls, path: str, dataset=None):
        """Load a saved model from disk.

        Parameters
        ----------
        path : str
            File path to the saved model (e.g., "model.h5")
        dataset : ims.Dataset, optional
            Dataset to associate with the loaded model for predictions.
            If None, only the model is loaded without dataset context.

        Returns
        -------
        CNNModel
            New CNNModel instance with loaded model

        Raises
        ------
        FileNotFoundError
            If the model file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Create new instance
        instance = cls(dataset=dataset)
        
        # Load the trained model
        instance.model = tf.keras.models.load_model(path)
        
        # If dataset is provided, prepare it for consistency
        if dataset is not None:
            instance.prep_data()
        
        print(f"Model loaded successfully from {path}")
        return instance

    def predict(self, dataset, return_probabilities=False):
        """Predict class labels for a dataset.

        Parameters
        ----------
        dataset : ims.Dataset
            Dataset to predict (required - must be different from training data)
        return_probabilities : bool, default=False
            If True, returns class probabilities instead of predicted labels

        Returns
        -------
        np.ndarray or tuple
            If return_probabilities=False: array of predicted class labels
            If return_probabilities=True: tuple of (predicted_labels, probabilities)

        Raises
        ------
        ValueError
            If model is not trained or dataset dimensions don't match training data
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        if dataset is None:
            raise ValueError("A dataset must be provided for prediction")
        
        # Prepare prediction data
        X_pred = []
        for spectrum in dataset.data:
            arr = np.array(spectrum.values, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            X_pred.append(arr)
        
        X_pred = np.stack(X_pred)
        
        # Check if dimensions and size of predict sample matches training data
        if hasattr(self, 'X') and X_pred.shape[1:] != self.X.shape[1:]:
            raise ValueError(
                f"Prediction dataset shape {X_pred.shape[1:]} doesn't match "
                f"training data shape {self.X.shape[1:]}. "
                f"Ensure same preprocessing (cut_dt, cut_rt, binning) is applied."
            )
        
        # Make predictions
        probabilities = self.model.predict(X_pred)
        predicted_indices = np.argmax(probabilities, axis=1)
        
        # Convert indices back to class labels if we have the original labels
        if hasattr(self, 'y_raw'):
            # Get unique labels in the same order as during training
            unique_labels = sorted(set(self.y_raw))
            predicted_labels = np.array([unique_labels[idx] for idx in predicted_indices])
        else:
            # Return indices if no label mapping available
            predicted_labels = predicted_indices
            print("Warning: No labels available, class indices are returned instead.")
        
        if return_probabilities:
            return predicted_labels, probabilities
        else:
            return predicted_labels
