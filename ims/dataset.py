from ims import Spectrum
import numpy as np
import dask
from dask import delayed
import os
from glob import glob
import h5py
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder


# TODO: generalize for classification as well as regression tasks


class Dataset:

    def __init__(self, data, name, files, samples, groups, _path):
        """
        DataSet class to coordinate many Spectrum instances
        as dask delayed objects with group and sample names available
        as attributes.
        Maps Spectrum methods to all spectra in DataSet and contains
        methods that require multiple spectra.

        Use one of the read_... methods as alternative constructor.

        Parameters
        ----------
        data : list
            lists instances of Spectrum

        name : str
            Uses the folder name if alternative constructor is used.

        files : list
            Lists file names of every file that
            was originally in the dataset.

        samples : list
            Lists sample names, one entry per spectrum.

        groups : list
            Lists group names, one entry per spectrum.
        """
        self.data = data
        self.name = name
        self.files = files
        self.samples = samples
        self.groups = groups
        self.preprocessing = []
        self._path = _path

    def __repr__(self):
        return f'Dataset: {self.name}, {len(self)} Spectra'

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @property
    def sample_indices(self):
        """
        Property method. Gives information about where each
        sample is in the dataset.

        Returns
        -------
        dict
            Sample names as keys,
            lists with indices of spectra as values
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
        Optionally generates group and sample names by splitting file paths.
        """
        if subfolders:
            files = []
            samples = []
            groups = []
            paths = glob(f'{path}/*/*/*')
            name = os.path.split(path)[1]
            for filedir in paths:
                filedir = os.path.normpath(filedir)
                file_name = os.path.split(filedir)
                files.append(file_name)
                sample_name = path.split(os.sep)[-2]
                samples.append(sample_name)
                group = path.split(os.sep)[-3]
                groups.append(group)
        else:
            paths = [os.path.normpath(i) for i in glob(f'{path}/*')]
            name = os.path.split(path)[1]
            files = [os.path.split(i)[1] for i in paths]
            samples = []
            groups = []

        return (paths, name, files, samples, groups)

    @classmethod
    def read_mea(cls, path, subfolders=False):
        """
        Reads all GAS mea files in directory.

        If subfolders=True expects the following folder structure
        for each group and sample:

        Data
        |--> Group A
            |--> Sample A
                |--> file a
                |--> file b

        Labels are auto-generated from directory names.

        Parameters
        ----------
        path : str
            Directory with the data.

        subfolders : bool, optional
            Uses subdirectory names as labels,
            by default True

        Returns
        -------
        Dataset
        """
        paths, name, files, samples, groups = Dataset._measurements(
            path, subfolders
        )
        data = [
            delayed(Spectrum.read_mea)(i, subfolders) for i in paths
        ]
        return cls(data, name, files, samples, groups, path)

    @classmethod
    def read_zip(cls, path, subfolders=True):
        """
        Reads all zip archives from GAS mea to zip tool in directory.

        If subfolders=True expects the following folder structure
        for each group and sample:

        Data
        |--> Group A
            |--> Sample A
                |--> file a
                |--> file b

        Labels are auto-generated from directory names.

        Parameters
        ----------
        path : str
            Directory with the data.

        subfolders : bool, optional
            Uses subdirectory names as labels,
            by default True

        Returns
        -------
        Dataset
        """
        paths, name, files, samples, groups = Dataset._measurements(
            path, subfolders
        )
        data = [
            delayed(Spectrum.read_zip)(i, subfolders) for i in paths
        ]
        return cls(data, name, files, samples, groups, path)

    @classmethod
    def read_hdf5(cls, path):
        """
        Reads all hdf5 files generated by Spectrum.to_hdf5
        method in input directory.

        (Preferred over read_zip because it is much faster.)

        Parameters
        ----------
        path : str
            Directory with the data.

        Returns
        -------
        Dataset
            data, samples and groups attributes are not
            ordered but correctly associated.
        """
        paths = glob(f'{path}/*.hdf5')
        name = os.path.split(path)[1]

        data = [
            delayed(Spectrum.read_hdf5)(i) for i in paths
        ]

        samples = []
        groups = []
        files = []
        for i in paths:
            with h5py.File(i, 'r') as f:
                samples.append(str(f.attrs['sample']))
                groups.append(str(f.attrs['group']))
                files.append(str(f.attrs['name']))

        return cls(data, name, files, samples, groups, path)

    def to_npy(self, folder_name):
        """
        Exports values of all spectra as npy file.
        Makes a target directory with a data folder
        and a npy file with group labels.

        Use the load_npy function from ml submodule to read
        the data.

        Parameters
        ----------
        folder_name : str
            Name of new target directory.
        """
        os.mkdir(folder_name)
        os.mkdir(f'{folder_name}/data')
        le = LabelEncoder()
        labels = le.fit_transform(self.groups)
        np.save(f'{folder_name}/labels.npy', labels)
        np.save(f'{folder_name}/label_names.npy', self.groups)
        np.save(f'{folder_name}/samples.npy', self.groups)
        exports = []
        for i, j in enumerate(self.data):
            exports.append(
                delayed(np.save)(f'{folder_name}/data/{i}', j.values)
                )
        dask.compute(exports)

    def to_hdf5(self, folder_name):
        """
        Exports all spectra as hdf5 files.
        Saves them to new folder.

        Parameters
        ----------
        folder_name : str
            Name of new target directory.
        """
        os.mkdir(folder_name)
        exports = [
            delayed(Spectrum.to_hdf5)(i, path=folder_name)
            for i in self.data
        ]
        dask.compute(exports)
  
    def compute(self):
        """
        Calls dask.compute on data attribute.

        Returns
        -------
        Dataset
            With computed data.
        """
        self.data = list(*dask.compute(self.data))
        return self

    def persist(self):
        """
        Calls dask.persist on data attribute.

        Returns
        -------
        Dataset
            Persists data in memory.
        """
        self.data = list(*dask.persist(self.data))
        return self

    def visualize(self):
        """
        Calls dask.visualize on data attribute.

        Returns
        -------
        graphviz figure
        """
        return dask.visualize(self.data)

    def select(self, group=None, sample=None):
        """
        Selects all spectra of specified group or sample.
        Must give at least one argument.

        Parameters
        ----------
        group : str, optional
            Group name to keep, by default None

        sample : str, optional
            Sample name to keep, by default None

        Returns
        -------
        Dataset
            Contains only matching spectra.
        """
        if group is None and sample is None:
            raise ValueError("Must give either group or sample value.")
        
        if group is not None:
            name = group
            indices = []
            for i, j in enumerate(self.groups):
                if j == group:
                    indices.append(i)
        if sample is not None:
            name = sample
            indices = []
            for i, j in enumerate(self.samples):
                if j == sample:
                    indices.append(i)

        result = []
        files = []
        groups = []
        samples = []
        for i in indices:
            result.append(self.data[i])
            files.append(self.files[i])
            groups.append(self.groups[i])
            samples.append(self.samples[i])

        return Dataset(
            data=result,
            name=name,
            files=files,
            samples=samples,
            groups=groups,
            _path=self._path
        )

    # TODO: Write groupby method that returns multiple DataSet instances
    # def groupby(self):
    #     indices = self.sample_indices
    #     u_samples = np.unique(self.samples)

    #     g_groups = []
    #     g_samples = []
    #     g_files = []
    #     g_data = []
    #     for i in u_samples:
    #         idx = indices[i]
    #         groups = []
    #         samples = []
    #         files = []
    #         data = []
    #         for j in idx:
    #             groups.append(self.groups[j])
    #             samples.append(self.samples[j])
    #             files.append(self.files[j])
    #             data.append(self.data[j])

    #         g_groups.append(groups)
    #         g_samples.append(samples)
    #         g_files.append(files)
    #         g_data.append(data)

    #     self.groups = g_groups
    #     self.samples = g_samples
    #     self.files = g_files
    #     self.data = g_data
    #     return self

    def mean(self):
        """
        Calculates means for each sample,
        in case of repeat determinations.
        Autmatically groups by sample.

        Returns
        -------
        Dataset
            With mean spectra.
        """
        indices = self.sample_indices
        u_samples = np.unique(self.samples)

        groups = []
        grouped_data = []
        for i in u_samples:
            group = self.groups[indices[i][0]]
            groups.append(group)

            data = []
            index = indices[i]
            for j in index:
                data.append(self.data[j])
            grouped_data.append(data)

        means = []
        for i in grouped_data:
            means.append(delayed(Spectrum.mean)(i))

        self.data = means
        self.samples = list(u_samples)
        self.groups = groups
        self.preprocessing.append('mean')
        return self

    # TODO: rewrite method so it is no longer a massive bottleneck
    def riprel(self):
        """
        Interpolates all spectra to common RIP relative
        drift time coordinate.
        Alignment along drift time coordinate.

        (Slow and inefficient!)

        Returns
        -------
        Dataset
            With RIP relative spectra.
        """
        self.data = dask.compute(self.data)[0]

        dt_riprel = []
        interp_fn = []
        for i in self.data:
            dt = i.drift_time
            rip = np.median(np.argmax(i.values, axis=1)).astype('int32')
            rip_ms = np.mean(dt[rip])
            riprel = dt / rip_ms
            f = interp1d(riprel, i.values, axis=1, kind='cubic')
            dt_riprel.append(riprel)
            interp_fn.append(f)

        start = max([i[0] for i in dt_riprel])
        end = min([i[-1] for i in dt_riprel])
        interv = np.median([(i[-1]-i[0]) / len(i) for i in dt_riprel])
        new_dt = np.arange(start, end, interv)

        for i, f in zip(self.data, interp_fn):
            i.values[:, :len(new_dt)]
            i.values = f(new_dt)
            i.drift_time = new_dt
            i.drift_time_label = 'Drift Time RIP relative'

        return self

    def rip_scaling(self):
        """
        Scales values relative to global maximum for each spectrum.

        Returns
        -------
        Dataset
            With scaled values.
        """
        self.data = [
            delayed(Spectrum.rip_scaling)(i) for i in self.data
            ]
        self.preprocessing.append('rip_scaling')
        return self
    
    def resample(self, n):
        """
        Resamples spectrum by calculating means of every n rows.
        (Retention time coordinate needs to be divisible by n)
        

        Parameters
        ---------
        n : int

        Returns
        -------
        GCIMS-DataSet
            Resampled values array for each spectrum

        """       
        self.data = [
            delayed(Spectrum.resample)(i, n) for i in self.data
            ]
        self.preprocessing.append(f'resample({n})')
        return self

    def cut_dt(self, start, stop):
        """
        Cuts spectra on drift time coordinate.
        Specifiy coordinate values not index directly.

        Parameters
        ----------
        start : int/float
            start value on drift time coordinate
        
        stop : int/float
            stop value on drift time coordinate

        Returns
        -------
        Dataset
            With cut spectra.
        """
        self.data = [
            delayed(Spectrum.cut_dt)(i, start, stop) for i in self.data
        ]
        self.preprocessing.append(f'cut_dt({start, stop})')
        return self

    def cut_rt(self, start, stop):
        """
        Cuts spectra on retention time coordinate.
        Specifiy coordinate values not index directly.

        Parameters
        ----------
        start : int/float
            start value on retention time coordinate
        
        stop : int/float
            stop value on retention time coordinate

        Returns
        -------
        Dataset
            With cut spectra.
        """
        self.data = [
            delayed(Spectrum.cut_rt)(i, start, stop) for i in self.data
        ]
        self.preprocessing.append(f'cut_rt({start, stop})')
        return self

    def tophat(self, size=15):
        """
        Applies white tophat filter on data.
        Baseline correction.

        (Slower with larger size.)

        Parameters
        ----------
        size : int, optional
            Size of structuring element, by default 15

        Returns
        -------
        Dataset
            With tophat applied.
        """
        self.data = [
            delayed(Spectrum.tophat)(i, size) for i in self.data
        ]
        self.preprocessing.append('tophat')
        return self

    def sub_first_row(self):
        """
        Subtracts first row from every row in spectrum.
        Baseline correction.

        Returns
        -------
        Dataset
            With corrected baseline.
        """
        self.data = [
            delayed(Spectrum.sub_first_row)(i) for i in self.data
        ]
        self.preprocessing.append('sub_first_line')
        return self

    def export_plots(self, folder_name, file_format='jpeg', **kwargs):
        """
        Exports a static plot for each spectrum to disk.
        Replicates group folders.

        Kaleido is required:
        $ conda install -c plotly python-kaleido

        Parameters
        ----------
        folder_name : str, optional
            New directory to save the images.

        file_format : str, optional
            See plotly kaleido documentation for supported formats
            https://plotly.com/python/static-image-export/,
            by default 'jpeg'
        """
        group_names = np.unique(self.groups)
        sample_names = np.unique(self.samples)
        sample_indices = self.sample_indices
        os.mkdir(folder_name)
        for group in group_names:
            os.mkdir(f'{folder_name}/{group}')

        exports = []
        for i in sample_names:
            indices = sample_indices[i]
            for j in indices:
                group = self.groups[j]
                fig = delayed(Spectrum.export_plot)(
                    self.data[j], path=f'{folder_name}/{group}',
                    file_format=file_format, **kwargs
                )
                exports.append(fig)
        dask.compute(exports)

    def export_images(self, folder_name, file_format='jpeg'):
        """
        Exports spectrum as grayscale image for classification in Orange 3.
        (Not a plot!)

        Parameters
        ----------
        folder_name : str, optional
            New directory to save the images

        file_format : str, optional
            See imageio docs for supported formats:
            https://imageio.readthedocs.io/en/stable/formats.html,
            by default 'jpeg'
        """
        group_names = np.unique(self.groups)
        sample_names = np.unique(self.samples)
        sample_indices = self.sample_indices
        os.mkdir(folder_name)
        for group in group_names:
            os.mkdir(f'{folder_name}/{group}')

        exports = []
        for i in sample_names:
            indices = sample_indices[i]
            for j in indices:
                group = self.groups[j]
                fig = delayed(Spectrum.export_image)(
                    self.data[j], path=f'{folder_name}/{group}',
                    file_format=file_format
                )
                exports.append(fig)
        dask.compute(exports)

    def get_xy(self, flatten=True):
        """
        Returns X and y for machine learning as
        numpy arrays.

        Parameters
        ----------
        flatten : bool, optional
            Flattens 3D datasets to 2D,by default True

        Returns
        -------
        tuple
            (X, y) as np.arrays
        """
        X = [i.values for i in self.data]
        X = np.stack(X)
        y = np.array(self.groups)

        if flatten:
            a, b, c = X.shape
            X = X.reshape(a, b*c)

        return (X, y)
