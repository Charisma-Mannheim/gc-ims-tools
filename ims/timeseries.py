from mymodule.ims import GCIMS_Spectrum

import numpy as np
import pandas as pd
import datetime
import dask
from dask import delayed

import os
from glob import glob
import h5py


class TimeSeries:

    def __init__(self, name, data, time):
        self.name = name
        self.data = data
        self.time = time

    def __repr__(self):
       return f'''
TimeSeries:
{self.name}, {len(self)} Spectra
{self.data.index[0]} : {self.data.index[-1]}\n
'''

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        else:
            return TimeSeries(self.name, self.data[key])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def read_zip(cls, path):
        """
        Reads all zip archives from GAS mea to zip tool in directory.
        Time coordinate is generated from Timestamp meta attribute.

        Parameters
        ----------
        path : str
            Input directory.

        Returns
        -------
        TimeSeries
            Alternative constructor.
        """
        paths = glob(f'{path}/*')
        name = os.path.split(path)[1]
        data = [
            delayed(GCIMS_Spectrum.read_zip)(i) for i in paths
        ]
        meta_attr = [
            GCIMS_Spectrum._read_meta_attr_zip(i) for i in paths
        ]
        datetime = [
            GCIMS_Spectrum._get_datetime(i) for i in meta_attr
        ]
        data = pd.Series(data, datetime)
        return cls(name, data)

    @classmethod
    def read_mea(cls, path):
        """
        Reads all GAS mea files in directory.
        Time coordinate is generated from Timestamp meta attribute.

        Parameters
        ----------
        path : str
            Input directory.

        Returns
        -------
        TimeSeries
            Alternative constructor.
        """
        paths = glob(f'{path}/*')
        name = os.path.split(path)[1]
        meta_attr = [
            GCIMS_Spectrum._read_meta_attr_mea(i) for i in paths
        ]
        timeline = [
            GCIMS_Spectrum._get_datetime(i) for i in meta_attr
        ]

        start = timeline[0]
        hours = [i - start for i in timeline]

        data = [
            delayed(GCIMS_Spectrum.read_mea)(i, time=j) for i, j in zip(paths, hours)
        ]
    
        
        # data = [
        #     GCIMS_Spectrum.set_time(i, j) for i, j in zip(data, timeline)
        # ]

        # new_data = []
        # for i, j in zip(data, timeline):
        #     new_data.append(i._set_time(j))

        data = pd.Series(data, timeline)

        return cls(name, data, hours)

    @classmethod
    def read_hdf5(cls, path):
        """
        Reads hdf5 files produced by GCIMS_Spectrum.to_hdf5 method.
        Alternative contstructor.

        Parameters
        ----------
        path : str
            Input directory.

        Returns
        -------
        TimeSeries
        """
        paths = glob(f'{path}/*')
        name = os.path.split(path)[1]
        data = [
            delayed(GCIMS_Spectrum.read_hdf5)(i) for i in paths
        ]

        meta_attr = []
        for i in paths:
            with h5py.File(i, 'r') as f:
                meta_keys = list(f['values'].attrs.keys())
                meta_values = list(f['values'].attrs.values())
                meta = dict(zip(meta_keys, meta_values))
                meta_attr.append(meta)

        datetime = [
            GCIMS_Spectrum._get_datetime(i) for i in meta_attr
        ]
        data = pd.Series(data, datetime)
        return cls(name, data)

    def compute(self):
        """
        Calls dask.compute on data.values attribute.

        Returns
        -------
        TimeSeries
            With computed data.
        """
        values = dask.compute(list(self.data.values))[0]
        self.data = pd.Series(values, self.data.index)
        return self

    def visualize(self):
        """
        Calls dask.visualize on data attribute.

        Returns
        -------
        graphviz figure
        """
        return dask.visualize(list(self.data.values))

    def find_gaps(self):
        pass

    def plot_time(self):
        pass

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
        np.save(f'{folder_name}/labels.npy', self.data.index)
        exports = []
        for i, j in enumerate(self.data):
            exports.append(delayed(np.save)(f'{folder_name}/data/{i}', j.values))
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
            delayed(GCIMS_Spectrum.to_hdf5)(i, path=folder_name)
            for i in self.data
        ]
        dask.compute(exports)

    # def agg(self, key, func):
    #     data = list(self[key])
    #     if method == 'mean':
    #         mean = delayed(GCIMS_Spectrum.mean)(data)
    #         self[key].data = mean
    #     return self

    def cut_dt(self, cut_range):
        """
        Cuts spectra on drift time coordinate.
        Specifiy coordinate values not index directly.

        Parameters
        ----------
        cut_range : tuple
            (start, stop) range in between is kept.

        Returns
        -------
        TimeSeries
            With cut spectra.
        """
        self.data.values = [
            delayed(GCIMS_Spectrum.cut_dt)(i, cut_range) for i in self.data.values
        ]
    
    def cut_rt(self, cut_range):
        """
        Cuts spectra on retention time coordinate.
        Specifiy coordinate values not index directly.

        Parameters
        ----------
        cut_range : tuple
            (start, stop) range in between is kept.

        Returns
        -------
        TimeSeries
            With cut spectra.
        """  
        self.data.values = [
            delayed(GCIMS_Spectrum.cut_rt)(i, cut_range) for i in self.data.values
        ]
    
    def align(self):
        pass

    def export_plots(self, folder_name, file_format='jpeg', **kwargs):
        """
        Exports a static plot for each spectrum to disk.
        
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
        os.mkdir(folder_name)
        exports = [
            delayed(GCIMS_Spectrum.export_plot)(
                i, folder_name, file_format, **kwargs) for i in self.data.values
        ]
        dask.compute(exports)

    def export_images(self, folder_name, file_format='jpeg', **kwargs):
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
        os.mkdir(folder_name)
        exports = [
            delayed(GCIMS_Spectrum.export_image, **kwargs)(
                i, folder_name, file_format) for i in self.data.values
        ]
        dask.compute(exports)

    def line_chart(self):  # plot total intensity vs time or just one peak
        pass

    def interpolate(self):
        pass

    def seasonal_adjustment(self):
        pass

    def granger_causality(self):
        pass

    def var(self):  # vector auto regression
        pass
