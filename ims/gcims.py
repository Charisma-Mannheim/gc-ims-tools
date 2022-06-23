import os
import re
import json
import h5py
from array import array
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
from time import ctime
from skimage.morphology import white_tophat, disk
from zipfile import ZipFile


class Spectrum:
    """
    Represents one GCIMS-Spectrum with the data matrix,
    retention and drift time coordinates.
    Sample or file name and timestamp are included unique identifiers.
    
    This class contains all methods that can be applied on a per spectrum basis,
    like I/O, plotting and some preprocessing tools. Methods that return a Spectrum change the instance inplace. Use the copy method.


    Parameters
    ----------
    name : str
        File or sample name as a unique identifier.
        Reader methods set this attribute to the file name without extension.

    values : numpy.array
        Intensity matrix.
        
    ret_time : numpy.array
        Retention time coordinate.

    drift_time : numpy.array
        Drift time coordinate.

    time : datetime object
        Timestamp when the spectrum was recorded.
        
    Example
    -------
    >>> import ims
    >>> sample = ims.Spectrum.read_mea("sample.mea")
    >>> sample.plot()
    """
    def __init__(self, name, values, ret_time, drift_time, time):
        self.name = name
        self.values = values
        self.ret_time = ret_time
        self.drift_time = drift_time
        self.time = time
        self._drift_time_label = 'Drift time [ms]'
        
    def __repr__(self):
        return f"GC-IMS Spectrum: {self.name}"
    
    # add, radd and truediv are implemented to calculate means easily
    def __add__(self, other):
        values = self.values + other.values
        ret_time = self.ret_time + other.ret_time
        drift_time = self.drift_time + other.drift_time
        x =  Spectrum(self.name, values, ret_time, drift_time,
                      self.time)
        x._drift_time_label = self._drift_time_label
        return x

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise NotImplementedError()

    def __truediv__(self, other):
        if isinstance(other, int):
            values = self.values / other
            ret_time = self.ret_time / other
            drift_time = self.drift_time / other
            x = Spectrum(self.name, values, ret_time,
                         drift_time, self.time)
            x._drift_time_label = self._drift_time_label
            return x
        else:
            raise NotImplementedError()

    @property
    def shape(self):
        """
        Shape property of the data matrix.
        Equivalent to `ims.Spectrum.values.shape`.
        """
        return self.values.shape

    def copy(self):
        """
        Uses deepcopy from the copy module in the standard library.
        Most operations happen inplace. Use this method if you do not
        want to change the original variable.

        Returns
        -------
        Spectrum
            deepcopy of self.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> new_variable = sample.copy()
        """
        return deepcopy(self)

    @classmethod
    def read_zip(cls, path):
        """
        Reads zipped csv and json files from G.A.S Dortmund mea2zip converting tool.
        Present for backwards compatibility.
        Reading mea files is much faster and saves the manual extra step of converting.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Spectrum

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_zip("sample.mea")
        >>> print(sample)
        GC-IMS Spectrum: sample
        """
        with ZipFile(path) as myzip:
            with myzip.open('csv_data.csv', 'r') as mycsv:
                values = pd.read_csv(mycsv, header=None)
            with myzip.open('meta_attributes.json', 'r') as myjson:
                meta_attr = json.load(myjson)

        values = np.array(values)
        values = np.delete(values, -1, axis=1)

        ret_time = np.arange(meta_attr['Chunks count'])\
            * (meta_attr['Chunk averages'] + 1)\
            * meta_attr['Chunk trigger repetition [ms]']\
            / 1000
        drift_time = np.arange(meta_attr['Chunk sample count'])\
            / meta_attr["Chunk sample rate [kHz]"]

        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        time = datetime.strptime(meta_attr["Timestamp"],
                                 "%Y-%m-%dT%H:%M:%S")

        return cls(name, values, ret_time, drift_time, time)

    @classmethod
    def read_mea(cls, path):
        """
        Reads mea files from G.A.S Dortmund instruments.
        Alternative constructor for ims.Spectrum class.
        Much faster than reading csv files and therefore preferred.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Spectrum

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> print(sample)
        GC-IMS Spectrum: sample
        """
        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        name = name.split('.')[0]

        with open(path, "rb") as f:
            content = f.read()
            i = content.index(0)
            meta_attr = content[:i-1]
            meta_attr = meta_attr.decode("windows-1252")
            data = content[i+1:]
            data = array('h', data)

        meta_attr = meta_attr.split("\n")

        key_re = re.compile("^.*?(?==)")
        value_re = re.compile("(?<==)(.*?)(?=\[|$)")
        # unit_re = re.compile("\[(.*?)\]")

        for i in meta_attr:
            key = key_re.search(i).group(0).strip()
            value = value_re.search(i).group(0).strip()
            if "Chunks count" in key:
                chunks_count = int(value)
            elif "Chunk averages" in key:
                chunk_averages = int(value)
            elif "Chunk sample count" in key:
                chunk_sample_count = int(value)
            elif "Chunk sample rate" in key:
                chunk_sample_rate = int(value)
            elif "Chunk trigger repetition" in key:
                chunk_trigger_repetition = int(value)
            elif "Timestamp" in key:
                timestamp = datetime.strptime(value, '"%Y-%m-%dT%H:%M:%S"')

        data = np.array(data)
        data = data.reshape(chunks_count, chunk_sample_count)

        ret_time = np.arange(chunks_count) * (chunk_averages + 1)\
            * chunk_trigger_repetition / 1000

        drift_time = np.arange(chunk_sample_count) / chunk_sample_rate

        return cls(name, data, ret_time, drift_time, timestamp)

    @classmethod
    def read_csv(cls, path):
        """
        Reads generic csv files. The first row must be
        the drift time values and the first column must be
        the retention time values. Values inbetween are the
        intensity matrix.
        Uses the time when the file was created as timestamp.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Spectrum
        
        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_csv("sample.csv")
        >>> print(sample)
        GC-IMS Spectrum: sample
        """
        name = os.path.split(path)[1]
        name = name.split('.')[0]
        df = pd.read_csv(path)
        values = df.values
        ret_time = df.index
        drift_time= df.columns
        timestamp = os.path.getctime(path)
        timestamp = ctime(timestamp)
        timestamp = datetime.strptime(timestamp,
                                      "%a %b  %d %H:%M:%S %Y")
        return cls(name, values, ret_time, drift_time, timestamp)

    @classmethod
    def read_hdf5(cls, path):
        """
        Reads hdf5 files exported by the to_hdf5 method.
        Convenient way to store preprocessed spectra.
        Especially useful for larger datasets as preprocessing
        requires more time.
        Preferred to csv because of very fast read and write speeds.

        Parameters
        ----------
        path : str
            Absolute or relative file path.

        Returns
        -------
        Spectrum

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> sample.to_hdf5()
        >>> sample = ims.Spectrum.read_hdf5("sample.hdf5")
        """     
        with h5py.File(path, 'r') as f:
            values = np.array(f['values'])
            ret_time = np.array(f['ret_time'])
            drift_time = np.array(f['drift_time'])
            name = str(f.attrs['name'])
            time = datetime.strptime(f.attrs['time'], "%Y-%m-%dT%H:%M:%S")
            drift_time_label = str(f.attrs["drift_time_label"])

        spectrum = cls(name, values, ret_time, drift_time, time)
        spectrum._drift_time_label = drift_time_label
        return spectrum

    def to_hdf5(self, path=None):
        """
        Exports spectrum as hdf5 file.
        Useful to save preprocessed spectra, especially for larger datasets.
        Preferred to csv format because of very fast read and write speeds.

        Parameters
        ----------
        path : str, optional
            Directory to export files to,
            by default the current working directory.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> sample.to_hdf5()
        >>> sample = ims.Spectrum.read_hdf5("sample.hdf5")
        """
        if path is None:
            path = os.getcwd()
        
        with h5py.File(f'{path}/{self.name}.hdf5', 'w-') as f:
            f.create_dataset('values', data=self.values)
            f.create_dataset('ret_time', data=self.ret_time)
            f.create_dataset('drift_time', data=self.drift_time)
            f.attrs['name'] = self.name
            f.attrs['time'] = datetime.strftime(self.time,
                                                "%Y-%m-%dT%H:%M:%S")
            f.attrs['drift_time_label'] = self._drift_time_label

    def tophat(self, size=15):
        """
        Applies white tophat filter on data matrix as a baseline correction.
        Size parameter is the diameter of the circular structuring element.
        (Slow with large size values.)

        Parameters
        ----------
        size : int, optional
            Size of structuring element, by default 15

        Returns
        -------
        Spectrum
        """
        self.values = white_tophat(self.values, disk(size))
        return self

    def sub_first_rows(self, n=1):
        """
        Subtracts first n rows from every row in spectrum.
        Effective and simple baseline correction
        if RIP tailing is a concern but can hide small peaks.

        Returns
        -------
        Spectrum
        """
        fl = self.values[0:n-1, :].mean(axis=0)
        self.values = self.values - fl
        # self.values[self.values < 0] = 0
        return self

    def riprel(self):
        """
        Replaces drift time coordinate with RIP relative values.
        Useful to cut away the RIP because it´s position is set to 1.

        Does not interpolate the data matrix to a completly artificial
        axis like ims.Dataset.interp_riprel.

        Returns
        -------
        Spectrum
            RIP relative drift time coordinate
            otherwise unchanged.
        """
        rip_index = np.argmax(self.values, axis=1)[0]
        rip_ms = self.drift_time[rip_index]
        dt_riprel = self.drift_time / rip_ms
        self.drift_time = dt_riprel
        self._drift_time_label = 'Drift time RIP relative'
        return self

    def rip_scaling(self):
        """
        Scales values relative to global maximum.
        Can be useful to directly compare spectra from
        instruments with different sensitivity.

        Returns
        -------
        Spectrum
            With scaled values.
        """
        m = np.max(self.values)
        self.values = self.values / m
        return self

    def resample(self, n=2):
        """
        Resamples spectrum by calculating means of every n rows.
        If the length of the retention time is not divisible by n
        it and the data matrix get cropped by the remainder at the long end.

        Parameters
        ----------
        n : int, optional
            Number of rows to mean,
            by default 2.

        Returns
        -------
        Spectrum
            Resampled values.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> print(sample.shape)
        (4082, 3150)
        >>> sample.resample(2)
        >>> print(sample.shape)
        (2041, 3150)
        """
        a, _ = self.values.shape
        rest = a % n
        if rest != 0:
            self.values = self.values[:a-rest, :]
            self.ret_time = self.ret_time[:a-rest]

        self.values = (self.values[0::n, :] + self.values[1::n, :]) / n
        self.ret_time = self.ret_time[::n]
        return self

    def binning(self, n=2):
        """
        Downsamples spectrum by binning the array with factor n.
        Similar to ims.Spectrum.resampling but works on both dimensions
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
        Spectrum
            Downsampled data matrix.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> print(sample.shape)
        (4082, 3150)
        >>> sample.binning(2)
        >>> print(sample.shape)
        (2041, 1575)
        """
        a, b = self.values.shape
        rest0 = a % n
        rest1 = b % n

        if rest0 != 0:
            self.values = self.values[:a-rest0, :]
            self.ret_time = self.ret_time[:a-rest0]
        if rest1 != 0:
            self.values = self.values[:, :b-rest1]
            self.drift_time = self.drift_time[:b-rest1]

        new_dims = (a // n, b // n)

        shape = (new_dims[0], a // new_dims[0],
                 new_dims[1], b // new_dims[1])

        self.values = self.values.reshape(shape).mean(axis=(-1, 1))
        self.ret_time = self.ret_time[::n]
        self.drift_time = self.drift_time[::n]
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
        Spectrum
            New drift time range.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> print(sample.shape)
        (4082, 3150)
        >>> sample.riprel().cut_dt(1.05, 2)
        >>> print(sample.shape)
        (4082, 1005)
        """
        if stop is None:
            stop = len(self.drift_time)

        idx_start = np.abs(self.drift_time - start).argmin()
        idx_stop = np.abs(self.drift_time - stop).argmin()
        self.drift_time = self.drift_time[idx_start:idx_stop]
        self.values = self.values[:, idx_start:idx_stop]
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
        Spectrum
            New retention time range.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> print(sample.shape)
        (4082, 3150)
        >>> sample.cut_rt(80, 500)
        >>> print(sample.shape)
        (2857, 3150)
        """
        if stop is None:
            stop = len(self.ret_time)

        idx_start = np.abs(self.ret_time - start).argmin()
        idx_stop = np.abs(self.ret_time - stop).argmin()
        self.ret_time = self.ret_time[idx_start:idx_stop]
        self.values = self.values[idx_start:idx_stop, :]
        return self

    def plot(self, vmin=30, vmax=400, width=9, height=10):
        """
        Plots Spectrum using matplotlibs imshow.
        Use %matplotlib widget in IPython or %matplotlib notebook
        in jupyter notebooks to get an interactive plot widget.
        Returning the figure is needed to make the export plot utilities possible.

        Parameters
        ----------
        vmin : int, optional
            Minimum of color range, by default 30.

        vmax : int, optional
            Maximum of color range, by default 300.

        width : int, optional
            Width in inches, by default 9.

        height : int, optional
            Height in inches, by default 10.

        Returns
        -------
        tuple
            (matplotlib.figure.Figure, matplotlib.pyplot.axes)
        
        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> fig, ax = sample.plot()
        """        

        fig, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            self.values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=(min(self.drift_time), max(self.drift_time),
                    min(self.ret_time), max(self.ret_time))
            )

        plt.colorbar().set_label("Intensities [arbitrary units]")
        plt.title(self.name)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.xlabel(self._drift_time_label)
        plt.ylabel("Retention time [s]")

        return fig, ax

    def export_plot(self, path=None, dpi=300,
                    file_format='jpg', **kwargs):
        """
        Saves the figure as image file. See the docs for
        matplotlib savefig function for supported file formats and kwargs
        (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).

        Parameters
        ----------
        path : str, optional
            Directory to save the image,
            by default current working directory.

        file_format : str, optional
            See matplotlib savefig docs for information about supported formats,
            by default 'jpg'.

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> sample.export_plot()
        """
        if path is None:
            path = os.getcwd()

        fig, _ = self.plot(**kwargs)
        fig.savefig(f'{path}/{self.name}.{file_format}', dpi=dpi,
                    bbox_inches="tight", pad_inches=0.2)
