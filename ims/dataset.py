from ims import Spectrum
import numpy as np
import os
from glob import glob
from copy import deepcopy
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtwalign import dtw
from sklearn.utils import resample
from sklearn.model_selection import (
    ShuffleSplit,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    StratifiedShuffleSplit,
)


class Dataset:
    """
    Dataset class coordinates many GC-IMS spectra
    (instances of ims.Spectrum class) with labels, file
    and sample names.

    ims.Spectrum methods are applied to all spectra. It also contains
    additional functionality and methods that require multiple spectra
    at a time such as alignments and calculating means. Most operations
    are done inplace for memory efficiency.


    Parameters
    ----------
    data : list
        Lists instances of `ims.Spectrum`.

    name : str
        Name of the dataset.

    files : list
        Lists one file name per spectrum. Must be unique.

    samples : list
        Lists sample names. A sample can have multiple files in
        case of repeat determination. Needed to calculate means.

    labels : list or numpy.ndarray
        Classification or regression labels.

    Attributes
    ----------
    preprocessing : list
        Keeps track of applied preprocessing steps.

    weights : numpy.ndarray of shape (n_samples, n_features)
        Stores the weights from scaling when the method is called.
        Needed to correct the loadings in PCA automatically.

    train_index : list
        Keeps the indices from train_test_split method.
        Used for plot annotations in PLS_DA and PLSR classes.

    test_index : list
        Keeps the indices from train_test_split method.
        Used for plot annotations in PLS_DA and PLSR classes.

    Example
    -------
    >>> import ims
    >>> ds = ims.Dataset.read_mea("IMS_data")
    >>> print(ds)
    Dataset: IMS_data, 58 Spectra
    """

    def __init__(self, data, name=None, files=None, samples=None, labels=None):
        self.data = data
        self.name = name
        self.files = files
        self.samples = samples
        self.labels = labels
        self.preprocessing = []

    def __repr__(self):
        return f"Dataset: {self.name}, {len(self)} Spectra"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]

        if isinstance(key, slice):
            return Dataset(
                self.data[key],
                self.name,
                self.files[key],
                self.samples[key],
                self.labels[key],
            )

        if isinstance(key, list) or isinstance(key, np.ndarray):
            return Dataset(
                [self.data[i] for i in key],
                self.name,
                [self.files[i] for i in key],
                [self.samples[i] for i in key],
                [self.labels[i] for i in key],
            )

    def __delitem__(self, key):
        del self.data[key]
        del self.files[key]
        del self.samples[key]
        del self.labels[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        """Concatenates two ims.Datasets instances"""
        ds = Dataset(
            self.data + other.data,
            f"{self.name} {other.name}",
            self.files + other.files,
            self.samples + other.samples,
            self.labels + other.labels,
        )
        ds.preprocessing = self.preprocessing + other.preprocessing
        return ds

    def copy(self):
        """
        Uses deepcopy from the copy module in the standard library.
        Most operations happen inplace. Use this method if you do not
        want to change the original variable.

        Returns
        -------
        Dataset
            deepcopy of self.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> new_variable = ds.copy()
        """
        return deepcopy(self)

    @property
    def timestamps(self):
        """
        Property of timestamps when each spectrum in dataset was recorded.

        Returns
        -------
        list
            List of Python datetime objects.
        """
        return [spectrum.time for spectrum in self]

    @property
    def sample_indices(self):
        """
        This property returns information about all spectra indices
        for each sample in the dataset.
        Useful to select or remove specific samples or files.

        Returns
        -------
        dict
            Sample names as keys, lists with indices of spectra as values.
        """
        u_samples = np.unique(self.samples)
        indices = []
        for i in u_samples:
            index = np.where(np.array(self.samples) == i)
            indices.append(index)

        indices = [list(i[0]) for i in indices]
        indices = dict(zip(u_samples, indices))
        return indices

    @staticmethod
    def _measurements(path, subfolders):
        """
        Lists paths to every file in folder.
        Optionally generates label and sample names by splitting file paths.
        """
        if subfolders:
            files = []
            samples = []
            labels = []
            paths = [os.path.normpath(i) for i in glob(f"{path}/*/*/*")]
            name = os.path.split(path)[1]
            for filedir in paths:
                file_name = os.path.split(filedir)[1]
                files.append(file_name)
                sample_name = filedir.split(os.sep)[-2]
                samples.append(sample_name)
                label = filedir.split(os.sep)[-3]
                labels.append(label)
        else:
            paths = [os.path.normpath(i) for i in glob(f"{path}/*")]
            name = os.path.split(path)[1]
            files = [os.path.split(i)[1] for i in paths]
            samples = []
            labels = []

        return (paths, name, files, samples, labels)

    @classmethod
    def read_mea(cls, path, subfolders=False, sequential=False, compr="wavecompr", **kwargs):
        """
        Reads all mea files from G.A.S Dortmund instruments in the
        given directory and combines them into a dataset.
        Much faster than reading csv files and therefore preferred.

        If subfolders=True expects the following folder structure
        for each label and sample:

        * Data
            * Group A
                * Sample A
                    * File A
                    * File B
                * Sample B
                    * File A
                    * File B

        Labels can then be auto-generated from directory names.
        Otherwise labels and sample names need to be added from other sources
        for all methods to work.
        
        If sequential=True files are compressed with the chosen compr method ("wavecompr" or "binning").
        Compression is done sequentially which is more memory efficient than comnpression of the 
        full dataset afterwards. Espically useful for big datatsets or systems with lower RAM specifications.
        Valid kwargs for the compression methods can be found in the respective definitions.  
        
        Parameters
        ----------
        path : str
            Absolute or relative file path.

        subfolders : bool, optional
            Uses subdirectory names as labels, 
            by default False

        sequential : bool, optional
            Sequential compression of each file, 
            by default False

        compr : str, optional
            Compression method "binning" or "wavecompr" are valid, 
            by default "wavecompr"

        Returns
        -------
        Dataset

        Raises
        ------
        ValueError
            If compression method is not supported
        
        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data", subfolders=True)
        >>> print(ds)
        Dataset: IMS_data, 58 Spectra
        """
        paths, name, files, samples, labels = Dataset._measurements(path, subfolders)
        if sequential:
            if compr == "wavecompr":
                data = [Spectrum.read_mea(i).wavecompr(**kwargs) for i in paths]
                
            elif compr == "binning":
                data = [Spectrum.read_mea(i).binning(**kwargs) for i in paths] 
                
            else:
                 raise ValueError(f"Unsupported compression algorithm: {compr}! Use wavecompr or binning instead.")
            
        else:
            data = [Spectrum.read_mea(i) for i in paths]
        
        dataset = cls(data, name, files, samples, labels)
        if sequential:
            dataset.preprocessing.append(f"{compr}")
        
        return dataset

    @classmethod
    def read_zip(cls, path, subfolders=False):
        """
        Reads zipped csv and json files from G.A.S Dortmund mea2zip converting tool.
        Present for backwards compatibility. Reading mea files is much faster and saves
        the manual extra step of converting.

        If subfolders=True expects the following folder structure
        for each label and sample:

        * Data
            * Group A
                * Sample A
                    * File A
                    * File B
                * Sample B
                    * File A
                    * File B

        Labels can then be auto-generated from directory names.
        Otherwise labels and sample names need to be added from other sources
        for all methods to work.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Dataset

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_zip("IMS_data", subfolders=True)
        >>> print(ds)
        Dataset: IMS_data, 58 Spectra
        """
        paths, name, files, samples, labels = Dataset._measurements(path, subfolders)
        data = [Spectrum.read_zip(i) for i in paths]
        return cls(data, name, files, samples, labels)

    @classmethod
    def read_csv(cls, path, subfolders=False):
        """
        Reads generic csv files. The first row must be
        the drift time values and the first column must be
        the retention time values. Values inbetween are the
        intensity matrix.
        Uses the time when the file was created as timestamp.

        If subfolders=True expects the following folder structure
        for each label and sample:

        * Data
            * Group A
                * Sample A
                    * File A
                    * File B
                * Sample B
                    * File A
                    * File B

        Labels can then be auto-generated from directory names.
        Otherwise labels and sample names need to be added from other sources
        for all methods to work.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Dataset

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_csv("IMS_data", subfolders=True)
        >>> print(ds)
        Dataset: IMS_data, 58 Spectra
        """
        paths, name, files, samples, labels = Dataset._measurements(path, subfolders)
        data = [Spectrum.read_csv(i) for i in paths]
        return cls(data, name, files, samples, labels)

    @classmethod
    def read_hdf5(cls, path):
        """
        Reads hdf5 files exported by the Dataset.to_hdf5 method.
        Convenient way to store preprocessed spectra.
        Especially useful for larger datasets as preprocessing
        requires more time.
        Preferred to csv because of faster read and write speeds.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Dataset

        Example
        -------
        >>> import ims
        >>> sample = ims.Dataset.read_mea("IMS_data")
        >>> sample.to_hdf5("IMS_data_hdf5")
        >>> sample = ims.Dataset.read_hdf5("IMS_data_hdf5")
        """
        with h5py.File(path, "r") as f:
            labels = [i.decode() for i in f["dataset"]["labels"]]
            samples = [i.decode() for i in f["dataset"]["samples"]]
            files = [i.decode() for i in f["dataset"]["files"]]
            preprocessing = [i.decode() for i in f["dataset"]["preprocessing"]]

            data = []
            for key in f.keys():
                if key == "dataset":
                    continue
                values = np.array(f[key]["values"])
                ret_time = np.array(f[key]["ret_time"])
                drift_time = np.array(f[key]["drift_time"])
                name = str(f[key].attrs["name"])
                time = datetime.strptime(f[key].attrs["time"], "%Y-%m-%dT%H:%M:%S")
                drift_time_label = str(f[key].attrs["drift_time_label"])
                spectrum = Spectrum(name, values, ret_time, drift_time, time)
                spectrum._drift_time_label = drift_time_label
                data.append(spectrum)

            name = os.path.split("Test.hdf5")[1]
            name = name.split(".")[0]

        dataset = cls(data, name, files, samples, labels)
        dataset.preprocessing = preprocessing
        return dataset

    def to_hdf5(self, name=None, path=None):
        """
        Exports the dataset as hdf5 file.
        It contains one group per spectrum and one with labels etc.
        Use ims.Dataset.read_hdf5 to read the file and construct a dataset.

        Parameters
        ----------
        name : str, optional
            Name of the hdf5 file. File extension is not needed.
            If not set, uses the dataset name attribute,
            by default None.

        path : str, otional
            Path to save the file. If not set uses the current working
            directory, by default None.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> ds.to_hdf5()
        >>> ds = ims.Dataset.read_hdf5("IMS_data.hdf5")
        """
        if name is None:
            name = self.name

        if path is None:
            path = os.getcwd()

        with h5py.File(f"{path}/{name}.hdf5", "w-") as f:
            data = f.create_group("dataset")
            data.create_dataset("labels", data=self.labels)
            data.create_dataset("samples", data=self.samples)
            data.create_dataset("files", data=self.files)
            data.create_dataset("preprocessing", data=self.preprocessing)

            for sample in self:
                grp = f.create_group(sample.name)
                grp.attrs["name"] = sample.name
                grp.create_dataset("values", data=sample.values)
                grp.create_dataset("ret_time", data=sample.ret_time)
                grp.create_dataset("drift_time", data=sample.drift_time)
                grp.attrs["time"] = datetime.strftime(sample.time, "%Y-%m-%dT%H:%M:%S")
                grp.attrs["drift_time_label"] = sample._drift_time_label

    def select(self, labels=None, samples=None, files=None):
        """
        Selects all spectra of specified labels, samples, or files.
        Must set at least one of the parameters.

        Parameters
        ----------
        labels : list of str, optional
            List of label names to keep, by default None

        samples : list of str, optional
            List of sample names to keep, by default None

        files : list of str, optional
            List of file names to keep, by default None

        Returns
        -------
        Dataset
            Contains only matching spectra.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> group_a = ds.select(labels=["GroupA", "GroupB"], samples=["SampleA"])
        """
        if labels is None and samples is None and files is None:
            raise ValueError("Must provide at least one of labels, samples, or files.")

        indices = set()

        if labels is not None:
            for label in labels:
                if label not in self.labels:
                    raise ValueError(f"Label '{label}' does not exist in the dataset.")
                indices.update(i for i, j in enumerate(self.labels) if j == label)

        if samples is not None:
            for sample in samples:
                if sample not in self.samples:
                    raise ValueError(f"Sample '{sample}' does not exist in the dataset.")
                indices.update(i for i, j in enumerate(self.samples) if j == sample)

        if files is not None:
            for file in files:
                if file not in self.files:
                    raise ValueError(f"File '{file}' does not exist in the dataset.")
                indices.update(i for i, j in enumerate(self.files) if j == file)

        data = []
        files = []
        labels = []
        samples = []

        for i in indices:
            data.append(self.data[i])
            files.append(self.files[i])
            labels.append(self.labels[i])
            samples.append(self.samples[i])

        return Dataset(
            data=data,
            name=self.name,
            files=files,
            samples=samples,
            labels=labels,
        )

    def drop(self, labels=None, samples=None, files=None):
        """
        Removes all spectra of specified labels, samples, or files from the dataset.
        Must set at least one of the parameters.

        Parameters
        ----------
        labels : str or list of str, optional
            List of label names to remove, by default None

        samples : str or list of str, optional
            List of sample names to remove, by default None

        files : str or list of str, optional
            List of file names to remove, by default None

        Returns
        -------
        Dataset
            Contains only spectra that do not match the specified criteria.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> ds = ds.drop(labels=["GroupA"], files=[FileA, FileB])
        """
        if labels is None and samples is None and files is None:
            raise ValueError("Must provide at least one of labels, samples, or files.")

        indices = set(range(len(self.data)))

        if labels is not None:
            for label in labels:
                if label not in self.labels:
                    raise ValueError(f"Label '{label}' does not exist in the dataset.")
                indices -= {i for i, j in enumerate(self.labels) if j == label}

        if samples is not None:
            for sample in samples:
                if sample not in self.samples:
                    raise ValueError(f"Sample '{sample}' does not exist in the dataset.")
                indices -= {i for i, j in enumerate(self.samples) if j == sample}

        if files is not None:
            for file in files:
                if file not in self.files:
                    raise ValueError(f"File '{file}' does not exist in the dataset.")
                indices -= {i for i, j in enumerate(self.files) if j == file}

        data = []
        files = []
        labels = []
        samples = []

        for i in indices:
            data.append(self.data[i])
            files.append(self.files[i])
            labels.append(self.labels[i])
            samples.append(self.samples[i])

        return Dataset(
            data=data,
            name=self.name,
            files=files,
            samples=samples,
            labels=labels,
        )

    def normalization(self):
        """
        Normalizes each spectrum in the dataset by scaling its intensity values to the range [0, 1].

        This method ensures that the intensity values of each spectrum are individually normalized 
        between 0 and 1 based on their unique minimum and maximum values. This standardizes the 
        dataset, facilitating consistent comparison and analysis across different spectra.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> ds.normalization()

        Returns
        -------
        self : Dataset
            The dataset with normalized spectra.
        """

        self.data = [Spectrum.normalization(i) for i in self.data]
        self.preprocessing.append("normalization")
        
        return self

    def add_spectrum(self, spectrum, sample, label):
        """
        Adds a ims.Spectrum to the dataset.
        Sample name and label must be provided because they are
        not stored in the ims.Spectrum class.

        Parameters
        ----------
        spectrum : ims.Spectrum
            GC-IMS spectrum to add to the dataset.

        sample : str
            The sample name is added to the sample attribute.
            Necessary because sample names are not stored in ims.Spectrum class.

        label : various
            Classification or regression label is added to the label attribute.
            Necessary because labels are not stored in ims.Spectrum class.

        Returns
        -------
        Dataset
            With Spectrum added.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> ds.add_spectrum(sample, "sample_name", "class_label")
        """
        self.data.append(spectrum)
        self.files.append(spectrum.name)
        self.samples.append(sample)
        self.labels.append(label)
        return self

    def groupby(self, key="label"):
        """
        Groups dataset by label or sample.

        Parameters
        ----------
        key : str, optional
            "label" or "sample" are valid keys, by default "label"

        Returns
        -------
        list
            List of one ims.Dataset instance per group or sample.
        """
        if key != "label" and key != "sample":
            raise ValueError('Only "label" or "sample" are valid keys!')

        result = []
        if key == "label":
            for group in np.unique(self.labels):
                result.append(self.select(labels=group))
            return result

        if key == "sample":
            for sample in np.unique(self.samples):
                result.append(self.select(samples=sample))
            return result

    def plot(self, index=0, **kwargs):
        """
        Plots the spectrum of selected index and adds the label to the title.

        Parameters
        ----------
        index : int, optional
            Index of spectrum to plot, by default 0

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        ax = self[index].plot(**kwargs)
        plt.title(f"{self[index].name}; {self.labels[index]}")
        return ax

    def train_test_split(self, test_size=0.2, stratify=False, random_state=None):
        """
        Splits the dataset in train and test sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to be used for validation.
            Should be between 0.0 and 1.0,
            by default 0.2

        stratify : bool, optional
            Wheter to stratify output or not.
            Preserves the percentage of samples from each class in each split,
            by default False.

        random_state : int, optional
            Controls the randomness. Pass an int for reproducible output,
            by default 1

        Returns
        -------
        tuple of numpy.ndarray
            X_train, X_test, y_train, y_test

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_Data")
        >>> X_train, X_test, y_train, y_test = ds.train_test_split()
        """

        if stratify:
            s = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train, test = next(s.split(self.data, y=self.labels))
        else:
            s = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train, test = next(s.split(self.data))

        # set attributes for annotations of plots with sample names
        self.train_index = train
        self.test_index = test

        X_train, y_train = self[train].get_xy()
        X_test, y_test = self[test].get_xy()
        return X_train, X_test, y_train, y_test

    def kfold_split(self, n_splits=5, shuffle=True, random_state=None, stratify=False):
        """
        K-Folds cross-validator (sklearn.model_selection.KFold).
        Splits the dataset into k consecutive folds and provides
        train and test data.

        If stratify is True uses StratifiedKfold instead.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds. Must be at least 2,
            by default 5.

        shuffle : bool, optional
            Whether to shuffle the data before splitting,
            by default True.

        random_state : int, optional
            When shuffle is True random_state affects the order of the indices.
            Pass an int for reproducible splits,
            by default None.

        stratify : bool, optional
            Wheter to stratify output or not.
            Preserves the percentage of samples from each class in each split,
            by default False.

        Yields
        ------
        tuple
            (X_train, X_test, y_train, y_test) per iteration

        Example
        -------
        >>> import ims
        >>> from sklearn.metrics import accuracy_score
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> model = ims.PLS_DA(ds)
        >>> accuracy = []
        >>> for X_train, X_test, y_train, y_test in ds.kfold_split():
        >>>     model.fit(X_train, y_train)
        >>>     y_pred = model.predict(X_test)
        >>>     accuracy.append(accuracy_score(y_test, y_pred))
        """
        if stratify:
            kf = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            kf = KFold(n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in kf.split(self, self.labels):
            train_data = self[train_index]
            test_data = self[test_index]
            X_train, y_train = train_data.get_xy()
            X_test, y_test = test_data.get_xy()
            yield X_train, X_test, y_train, y_test

    def shuffle_split(self, n_splits=5, test_size=0.2, random_state=None):
        """
        Shuffled splits for montecarlo cross-validation. Randomly selects
        a fraction of the dataset, without replacements, per split
        (sklearn.model_selection.ShuffleSplit).

        Parameters
        ----------
        n_splits : int, optional
            Number of re-shuffling and splitting iterations,
            by default 5.

        test_size : float, optional
            Proportion of the dataset to include in the test split,
            by default 0.2.

        random_state : int, optional
            Controls randomness. Pass an int for reproducible output,
            by default None.

        Yields
        -------
        tuple
            (X_train, X_test, y_train, y_test) per iteration

        Example
        -------
        >>> import ims
        >>> from sklearn.metrics import accuracy_score
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> model = ims.PLS_DA(ds)
        >>> accuracy = []
        >>> for X_train, X_test, y_train, y_test in ds.shuffle_split():
        >>>     model.fit(X_train, y_train)
        >>>     y_pred = model.predict(X_test)
        >>>     accuracy.append(accuracy_score(y_test, y_pred))
        """
        rs = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        for train_index, test_index in rs.split(self, self.labels):
            train_data = self[train_index]
            test_data = self[test_index]
            X_train, y_train = train_data.get_xy()
            X_test, y_test = test_data.get_xy()
            yield X_train, X_test, y_train, y_test

    def bootstrap(self, n_bootstraps=5, n_samples=None, random_state=None):
        """
        Iteratively resamples dataset with replacement. Samples can
        be included multiple times or not at all in the training data.
        Uses all samples that are not present in the training data as test data.

        Parameters
        ----------
        n_bootstraps : int, optional
            Number of iterations, by default 5.

        n_samples : int, optional
            Number of samples to draw per iteration. Is set to
            the lenghth of the dataset if None,
            by default None.

        random_state : int, optional
            Controls randomness, pass an int for reproducible output,
            by default None.

        Yields
        -------
        tuple
            (X_train, X_test, y_train, y_test) per iteration

        Example
        -------
        >>> import ims
        >>> from sklearn.metrics import accuracy_score
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> model = ims.PLS_DA(ds)
        >>> accuracy = []
        >>> for X_train, X_test, y_train, y_test in ds.bootstrap():
        >>>     model.fit(X_train, y_train)
        >>>     y_pred = model.predict(X_test)
        >>>     accuracy.append(accuracy_score(y_test, y_pred))
        """
        for _ in range(n_bootstraps):
            train_data, train_labels = resample(
                self.data, self.labels, n_samples=n_samples, random_state=random_state
            )

            test_data = []
            test_labels = []
            for i, j in enumerate(self.data):
                if j not in train_data:
                    test_data.append(j)
                    test_labels.append(self.labels[i])
            X_train, y_train = Dataset(train_data, labels=train_labels).get_xy()
            X_test, y_test = Dataset(test_data, labels=test_labels).get_xy()
            yield X_train, X_test, y_train, y_test

    def leave_one_out(self):
        """
        Leave-One-Out cross-validator.
        Provides train test splits and uses each sample once as test set
        while the remaining data is used for training.

        Yields
        -------
        tuple
            X_train, X_test, y_train, y_test

        Example
        -------
        >>> import ims
        >>> from sklearn.metrics import accuracy_score
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> model = ims.PLS_DA(ds)
        >>> accuracy = []
        >>> for X_train, X_test, y_train, y_test in ds.leave_one_out():
        >>>     model.fit(X_train, y_train)
        >>>     y_pred = model.predict(X_test, y_test)
        >>>     accuracy.append(accuracy_score(y_test, y_pred))
        """
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(self):
            train_data = self[train_index]
            test_data = self[test_index]
            X_train, y_train = train_data.get_xy()
            X_test, y_test = test_data.get_xy()
            yield X_train, X_test, y_train, y_test

    def mean(self):
        """
        Calculates means for each sample, in case of repeat determinations.
        Automatically determines which file belongs to which sample.
        Sample names are used for mean spectra and file names are no longer needed.

        Returns
        -------
        Dataset
            With mean spectra.
        """
        indices = self.sample_indices
        u_samples = np.unique(self.samples)

        labels = []
        grouped_data = []
        for i in u_samples:
            label = self.labels[indices[i][0]]
            labels.append(label)

            data = []
            index = indices[i]
            for j in index:
                data.append(self.data[j])
            grouped_data.append(data)

        means = []
        for i in grouped_data:
            means.append(sum(i) / len(i))

        for i, j in zip(means, u_samples):
            i.name = j

        self.data = means
        self.samples = list(u_samples)
        self.labels = labels
        self.preprocessing.append("mean()")
        return self

    def asymcorr(self, lam=1e7, p=1e-3, niter=20):
        """
        Retention time baseline correction using asymmetric least squares.

        Parameters
        ----------
        lam : float, optional
            Controls smoothness. Larger numbers return smoother curves,
            by default 1e7

        p : float, optional
            Controls asymmetry, by default 1e-3

        niter : int, optional
            Number of iterations during optimization,
            by default 20

        Returns
        -------
        Dataset
        """
        self.data = [Spectrum.asymcorr(i, lam, p, niter) for i in self.data]
        self.preprocessing.append("asymcorr")
        return self

    def savgol(self, window_length=10, polyorder=2, direction="both"):
        """
        Applys a Savitzky-Golay filter to intensity values.
        Can be applied in the drift time, retention time or both directions.

        Parameters
        ----------
        window_length : int, optional
            The length of the filter window, by default 10

        polyorder : int, optional
            The order of the polynomial used to fit the samples, by default 2

        direction : str, optional
            The direction in which to apply the filter.
            Can be 'drift time', 'retention time' or 'both'.
            By default 'both'

        Returns
        -------
        Dataset
        """
        self.data = [
            Spectrum.savgol(i, window_length, polyorder, direction) for i in self.data
        ]
        self.preprocessing.append("savgol")
        return self

    def tophat(self, size=15):
        """
        Applies white tophat filter on data matrix as a baseline correction.
        Size parameter is the diameter of the circular structuring element.
        (Slow with large size values.)

        Parameters
        ----------
        size : int, optional
            Size of structuring element, by default 15.

        Returns
        -------
        Dataset
        """
        self.data = [Spectrum.tophat(i, size) for i in self.data]
        self.preprocessing.append("tophat")
        return self

    def sub_first_rows(self, n=1):
        """
        Subtracts first row from every row in spectrum.
        Effective and simple baseline correction
        if RIP tailing is a concern but can hide small peaks.

        Returns
        -------
        Dataset
        """
        self.data = [Spectrum.sub_first_rows(i, n) for i in self.data]
        self.preprocessing.append("sub_first_row")
        return self

    def interp_riprel(self, rip_position=None):
        """
        Interpolates all spectra to common RIP relative drift time coordinate.
        Alignment along drift time coordinate.
        
        Parameters
        ----------
        rip_position : int or float, optional
            Position of the RIP in ms. If None the maximum value is used,
            by default None.

        Returns
        -------
        Dataset
            With RIP relative spectra.
        """
        dt_riprel = []
        interp_fn = []
        for i in self.data:
            dt = i.drift_time
            if rip_position is None:
                rip = np.median(np.argmax(i.values, axis=1)).astype("int32")
                rip_ms = np.mean(dt[rip])
            else:
                rip_ms = rip_position
            riprel = dt / rip_ms
            f = interp1d(riprel, i.values, axis=1, kind="cubic")
            dt_riprel.append(riprel)
            interp_fn.append(f)

        start = max([i[0] for i in dt_riprel])
        end = min([i[-1] for i in dt_riprel])
        interv = np.median([(i[-1] - i[0]) / len(i) for i in dt_riprel])
        new_dt = np.arange(start, end, interv)

        for i, f in zip(self.data, interp_fn):
            i.values[:, : len(new_dt)]
            i.values = f(new_dt)
            i.drift_time = new_dt
            i._drift_time_label = "Drift time RIP relative"

        self.preprocessing.append("interp_riprel()")
        return self
    
    def align_ret_time(self, reference="mean"):
        """
        Retention time alignment based on dymanic time warping.

        Parameters
        ----------
        reference : str, int or Spectrum, optional
            Reference intensity values and retention time.
            If "mean" is used, calculates the mean from all samples in dataset.
            An integer is used to index the dataset and select a Spectrum.
            If a Spectrum is given, uses this external sample as reference,
            by default "mean".
        """        
        if isinstance(reference, str) and reference == "mean":
            reference_ret_time = np.mean(
                np.vstack([sample.ret_time for sample in self.data]),
                axis=0
                )
            X, _ = self.get_xy(flatten=False)
            reference_values = np.mean(X, axis=0)
            
        elif isinstance(reference, int):
            reference_ret_time = self.data[reference].ret_time
            reference_values = self.data[reference].values
            
        elif isinstance(reference, Spectrum):
            reference_ret_time = reference.ret_time
            reference_values = reference.values

        for sample in self.data:
            res = dtw(sample.values, reference_values)
            warping_path = res.get_warping_path(target="query")
            sample.values = sample.values[warping_path, :]
            sample.ret_time = reference_ret_time
            
        return self

    def rip_scaling(self):
        """
        Scales values relative to global maximum.
        Can be useful to directly compare spectra from
        instruments with different sensitivity.

        Returns
        -------
        Dataset
            With scaled values.
        """
        self.data = [Spectrum.rip_scaling(i) for i in self.data]
        self.preprocessing.append("rip_scaling")
        return self

    def resample(self, n=2):
        """
        Resamples each spectrum by calculating means of every n rows.
        If the length of the retention time is not divisible by n
        it and the data matrix get cropped by the remainder at the long end.

        Parameters
        ----------
        n : int, optional
            Number of rows to mean,
            by default 2.

        Returns
        -------
        Dataset
            Resampled values.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_Data")
        >>> print(ds[0].shape)
        (4082, 3150)
        >>> ds.resample(2)
        >>> print(ds[0].shape)
        (2041, 3150)
        """
        self.data = [Spectrum.resample(i, n) for i in self.data]
        self.preprocessing.append(f"resample({n})")
        return self

    def binning(self, n=2):
        """
        Downsamples each spectrum by binning the array with factor n.
        Similar to Spectrum.resampling but works on both dimensions
        simultaneously.
        If the dimensions are not divisible by the binning factor
        shortens it by the remainder at the long end.
        Very effective data reduction because a factor n=2 already
        reduces the number of features to a quarter.

        Parameters
        ----------
        n : int, optional
            Binning factor, by default 2.

        Returns
        -------
        Dataset
            Downsampled data matrix.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_Data")
        >>> print(ds[0].shape)
        (4082, 3150)
        >>> ds.binning(2)
        >>> print(ds[0].shape)
        (2041, 1575)
        """
        self.data = [Spectrum.binning(i, n) for i in self.data]
        self.preprocessing.append(f"binning({n})")
        return self
    
    def wavecompr(self, direction="ret_time", wavelet="db3", level=3):
        """
        Data reduction by wavelet compression.
        Can be applied to drift time, retention time or both axis.

        Parameters
        ----------
        direction : str, optional
            The direction in which to apply the filter.
            Can be 'drift time', 'retention time' or 'both'.
            By default 'ret_time'.

        wavelet : str, optional
            Wavelet object or name string,
            by default "db3".

        level : int, optional
            Decomposition level (must be >= 0),
            by default 3.

        Returns
        -------
        Dataset

        Raises
        ------
        ValueError
            When direction is neither 'ret_time', 'drift_time' or 'both'.
        """        
        self.data = [Spectrum.wavecompr(i, direction, wavelet, level) for i in self.data]
        self.preprocessing.append(f"wavecompr")
        return self

    def cut_dt(self, start, stop=None):
        """
        Cuts data along drift time coordinate.
        Range in between start and stop is kept.
        If stop is not given uses the end of the array instead.
        Combination with RIP relative drift time
        values makes it easier to cut the RIP away and focus
        on the peak area.

        Parameters
        ----------
        start : int or float
            Start value on drift time coordinate.

        stop : int or float, optional
            Stop value on drift time coordinate.
            If None uses the end of the array,
            by default None.

        Returns
        -------
        Dataset
            New drift time range.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> print(ds[0].shape)
        (4082, 3150)
        >>> ds.interp_riprel().cut_dt(1.05, 2)
        >>> print(ds[0].shape)
        (4082, 1005)
        """
        self.data = [Spectrum.cut_dt(i, start, stop) for i in self.data]
        self.preprocessing.append(f"cut_dt({start}, {stop})")
        return self

    def cut_rt(self, start, stop=None):
        """
        Cuts data along retention time coordinate.
        Range in between start and stop is kept.
        If stop is not given uses the end of the array instead.

        Parameters
        ----------
        start : int or float
            Start value on retention time coordinate.

        stop : int or float, optional
            Stop value on retention time coordinate.
            If None uses the end of the array,
            by default None.

        Returns
        -------
        Dataset
            New retention time range.

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> print(ds[0].shape)
        (4082, 3150)
        >>> sample.cut_rt(80, 500)
        >>> print(ds[0].shape)
        (2857, 3150)
        """
        self.data = [Spectrum.cut_rt(i, start, stop) for i in self.data]
        self.preprocessing.append(f"cut_rt({start}, {stop})")
        return self

    def export_plots(self, folder_name=None, file_format="jpg", **kwargs):
        """
        Saves a figure per spectrum as image file. See the docs for
        matplotlib savefig function for supported file formats and kwargs
        (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).

        Creates a new folder for the plots in the current working directory.

        Parameters
        ----------
        folder_name : str, optional
            New directory to save the images to.

        file_format : str, optional
            See matplotlib savefig docs for information
            about supported formats,
            by default 'jpeg'

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> ds.export_plots("IMS_data_plots")
        """
        if folder_name is None:
            folder_name = self.name.join("_plots")

        os.mkdir(folder_name)

        for i in self.data:
            i.export_plot(path=folder_name, file_format=file_format, **kwargs)

    def get_xy(self, flatten=True):
        """
        Returns features (X) and labels (y) as numpy arrays.

        Parameters
        ----------
        flatten : bool, optional
            Flattens 3D datasets to 2D, by default True

        Returns
        -------
        tuple
            (X, y)

        Example
        -------
        >>> import ims
        >>> ds = ims.Dataset.read_mea("IMS_data")
        >>> X, y = ds.get_xy()
        """
        X = [i.values for i in self.data]
        X = np.stack(X)
        y = np.array(self.labels)

        if flatten:
            a, b, c = X.shape
            X = X.reshape(a, b * c)

        return (X, y)

    def scaling(self, method="pareto", mean_centering=True):
        """
        Scales and mean centeres features according to selected method.

        Parameters
        ----------
        method : str, optional
            "pareto", "auto" or "var" are valid,
            by default "pareto".

        mean_centering : bool, optional
            If true center the data before scaling,
            by default True.

        Returns
        -------
        Dataset

        Raises
        ------
        ValueError
            If scaling method is not supported.
        """
        X = [i.values for i in self.data]
        X = np.stack(X)
        a, b, c = X.shape
        X = X.reshape(a, b * c)

        if method == "auto":
            weights = 1 / np.std(X, 0)
        elif method == "pareto":
            weights = 1 / np.sqrt(np.std(X, 0))
        elif method == "var":
            weights = 1 / np.var(X, 0)
        else:
            raise ValueError(f"{method} is not a supported method!")

        weights = np.nan_to_num(weights, posinf=0, neginf=0)

        if mean_centering:
            X = (X - np.mean(X, 0)) * weights
        else:
            X = X * weights

        for i, j in enumerate(self.data):
            j.values = X[i, :].reshape(b, c)

        self.weights = weights
        self.preprocessing.append(f"scaling({method})")
        return self
