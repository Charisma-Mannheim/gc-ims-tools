import os
import re
import h5py
from array import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from datetime import date, datetime

import json
from zipfile import ZipFile
import pandas as pd

class Spectrum:

    def __init__(self, name, values, ret_time, drift_time, time):
        """
        Represents on GCIMS-Spectrum including the data matrix,
        retention and drift time coordinates and meta attributes.

        Contains all methods that can be applied per spectrum.

        Use one of the read_... methods as constructor.

        Parameters
        ----------
        name : str
            File or Sample name.

        values : np.array
            Data array stored as numpy array.
            
        ret_time : np.array
            Retention time as numpy array.

        drift_time : np.array
            Drift time as numpy array

        meta_attr : dict
            Meta attributes from GAS software.
        """
        self.name = name
        self.values = values
        self.ret_time = ret_time
        self.drift_time = drift_time
        self.time = time
        self._drift_time_label = 'Drift Time [ms]'
        
    def __repr__(self):
        return f"GC-IMS Spectrum: {self.name}"
    
    # add, radd and truediv are implemented to calculate means easily
    def __add__(self, other):
        values = self.values + other.values
        ret_time = self.ret_time + other.ret_time
        drift_time = self.drift_time + other.drift_time
        return Spectrum(self.name, values, ret_time, drift_time,
                        self.meta_attr)

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
            return Spectrum(self.name, values, ret_time,
                            drift_time, self.meta_attr)
        else:
            raise NotImplementedError()

    @property
    def shape(self):
        return self.values.shape

    @classmethod
    def read_zip(cls, path):
        """
        Reads zip files from GAS mea to csv converter.
        """
        with ZipFile(path) as myzip:
            with myzip.open('csv_data.csv', 'r') as mycsv:
                values = pd.read_csv(mycsv, header=None)
            with myzip.open('meta_attributes.json', 'r') as myjson:
                meta_attr = json.load(myjson)

        values = np.array(values)
        values = np.delete(values, -1, axis=1)

        ret_time, drift_time = Spectrum.calc_coordinates(meta_attr)

        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        time = datetime.strptime(meta_attr["Timestamp"],
                                      "%Y-%m-%dT%H:%M:%S")

        return cls(name, values, ret_time, drift_time, time)

    @staticmethod
    def read_meta_attr(path):
        '''
        Reads and formats just the meta attributes from GAS mea file.
        '''
        with open(path, "rb") as f:
            data = f.read()
            meta_attr = []
            for i in data:
                if i == 0:
                    break
                meta_attr.append(chr(i))

        meta_attr = "".join(meta_attr)
        meta_attr = meta_attr.split('\n')[:-1]

        # regex matching can be improved to ignore leading and trailing whitespace
        # insted of calling str.strip so often
        key_re = re.compile("^.*?(?==)")
        value_re = re.compile("(?<==)(.*?)(?=\[|$)")
        unit_re = re.compile("\[(.*?)\]")

        keys = []
        values = []
        for i in meta_attr:
            value = value_re.search(i).group(0).strip()
            if '"' in value:
                value = value.strip('"')
            else:
                value = int(value)
            values.append(value)
            key_name = key_re.search(i).group(0).strip()
            unit = unit_re.search(i)
            if unit is None:
                unit = ""
            else:
                unit = unit.group(0).strip()
            key = " ".join((key_name, unit)).strip()
            keys.append(key)

        meta_attr = dict(zip(keys, values))
        meta_attr["Timestamp"] = datetime.strptime(meta_attr["Timestamp"],
                                                   "%Y-%m-%dT%H:%M:%S")
        return meta_attr

    @staticmethod
    def calc_coordinates(meta_attr):
        '''
        Calculates retention and drift time coordinates from meta attributes.
        '''
        ret_time = np.arange(meta_attr['Chunks count'])\
            * (meta_attr['Chunk averages'] + 1)\
            * meta_attr['Chunk trigger repetition [ms]']\
            / 1000
        drift_time = np.arange(meta_attr['Chunk sample count'])\
            / meta_attr["Chunk sample rate [kHz]"]
        return (ret_time, drift_time)

    @classmethod
    def read_mea(cls, path):
        """
        Reads mea files from GAS.

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
            Directory to the file.

        subfolders : bool, optional
            Uses subdirectory names as labels,
            by default False

        Returns
        -------
        Dataset
        """
        meta_attr = Spectrum.read_meta_attr(path)
        ret_time, drift_time = Spectrum.calc_coordinates(meta_attr)
        time = meta_attr["Timestamp"]

        # get sample and group names from folder names
        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        name = name.split('.')[0]

        with open(path, mode='rb') as f:
            data = f.read()
            
            # meta attributes are separated by a null byte
            # add 1 to exclude it
            start = data.index(0) + 1

            # read the remaining data
            # values are stored as signed short int in two bytes
            # Chunks count is the size of the retention time
            # Chunks sample count is the size of the drift time
            values = data[start:]
            values = array('h', values)
            values = np.array(values).reshape(
                meta_attr['Chunks count'],
                meta_attr['Chunk sample count']
                )

        return cls(name, values, ret_time, drift_time, time)

    @classmethod
    def read_hdf5(cls, path):
        """
        Reads hdf5 files exported by the to_hdf5 method.
        Labels are attached to the file, so a folder strucutre is
        not necessary.

        Parameters
        ----------
        path : str
            Directory of the file.

        Returns
        -------
        Spectrum
        """     
        with h5py.File(path, 'r') as f:
            values = np.array(f['values'])
            ret_time = np.array(f['ret_time'])
            drift_time = np.array(f['drift_time'])
            name = str(f.attrs['name'])
            time = datetime.strptime(f.attrs['time'], "%Y-%m-%dT%H:%M:%S")
        return cls(name, values, ret_time, drift_time, time)

    def to_hdf5(self, path=os.getcwd()):
        """
        Exports spectrum including all coordinates
        and labels as hdf5 file.

        Use read_hdf5 method on those files.

        Parameters
        ----------
        path : str, optional
            Directory to export files to,
            by default os.getcwd()
        """

        with h5py.File(f'{path}/{self.name}.hdf5', 'w-') as f:
            f.create_dataset('values', data=self.values)
            f.create_dataset('ret_time', data=self.ret_time)
            f.create_dataset('drift_time', data=self.drift_time)

            f.attrs['name'] = self.name
            f.attrs['time'] = datetime.strftime(self.time, "%Y-%m-%dT%H:%M:%S")

    def riprel(self):
        """
        Replaces drift time coordinate with RIP relative values.

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
        self._drift_time_label = 'Drift Time RIP relative'
        return self

    def rip_scaling(self):
        """
        Scales values relative to global maximum.

        Returns
        -------
        Spectrum
            With scaled values.
        """
        m = np.max(self.values)
        self.values = self.values / m
        return self

    def resample(self, n):
        """
        Resamples spectrum by calculating means of every n rows.
        (Retention time coordinate needs to be divisible by n)

        Parameters
        ----------
        n : int
            Number of rows to mean

        Returns
        -------
        GCIMS-Spectrum
            Resampled values array
        """
        self.values = (self.values[0::n, :] + self.values[1::n, :]) / n
        self.ret_time = self.ret_time[::n]
        return self

    def rebin(self, n):
        """
        Downsamples spectrum by binning the array.
        If the dims are not devisible by the binning factor
        shortens the dim by the remainder at the long end. 

        Parameters
        ----------
        n : int
            Binning factor.

        Returns
        -------
        Spectrum
            Downsampled
        """
        a, b = self.values.shape
        rest0 = a % n
        rest1 = b % n
        
        if rest0 != 0:
            self.values = self.values[:a-rest0, :]
        if rest1 != 0:
            self.values = self.values[:, :b-rest1]
            
        new_dims = (a // n, b // n)
        
        shape = (new_dims[0], a // new_dims[0],
                 new_dims[1], b // new_dims[1])
        
        self.values = self.values.reshape(shape).mean(axis=(-1, 1))
        self.ret_time = self.ret_time[::n]
        self.drift_time = self.drift_time[::n]
        return self

    def cut_dt(self, start, stop):
        """
        Cuts data along drift time coordinate.
        Range in between start and stop is kept.

        Parameters
        ----------
        start : int/float
            start value on drift time coordinate
        
        stop : int/float
            stop value on drift time coordinate

        Returns
        -------
        Spectrum
            With cut drift time
        """
        idx_start = np.abs(self.drift_time - start).argmin()
        idx_stop = np.abs(self.drift_time - stop).argmin()
        self.drift_time = self.drift_time[idx_start:idx_stop]
        self.values = self.values[:, idx_start:idx_stop]
        return self

    def cut_rt(self, start, stop):
        """
        Cuts data along retention time coordinate.
        Range in between start and stop is kept.

        Parameters
        ----------
        start : int/float
            start value on retention time coordinate
        
        stop : int/float
            stop value on retention time coordinate

        Returns
        -------
        Spectrum
            With cut retention time
        """
        idx_start = np.abs(self.ret_time - start).argmin()
        idx_stop = np.abs(self.ret_time - stop).argmin()
        self.ret_time = self.ret_time[idx_start:idx_stop]
        self.values = self.values[idx_start:idx_stop, :]
        return self

    def wavelet_compression(self):
        pass

    def plot(self, vmin=30, vmax=300, width=9, height=10):
        """
        ims.Spectrum.plot
        -----------------

        Plots Spectrum using pyplot.imshow.
        Use %matplotlib widget in IPython or %matplotlib notebook
        in jupyter notebooks to get an interactive plot.
        
        Disable the autoshow function in IPython with a semicolon
        otherwise plots may show up twice.

        Parameters
        ----------
        vmin : int, optional
            min of color range, by default 30

        vmax : int, optional
            max of color range, by default 300

        width : int, optional
            width in inches, by default 9

        height : int, optional
            height in inches, by default 10

        Returns
        -------
        matplotlib.pyplot.Figure
        """        

        fig, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            self.values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax
            )

        plt.colorbar().set_label("Intensities [arbitrary units]")
        plt.title(self.name, fontsize=16)

        # TODO: axis labels display weird numbers
        xlocs, _ = plt.xticks()
        ylocs, _ = plt.yticks()

        rt_ticks = [round(self.ret_time[int(i)]) for i in ylocs[1:-1]]
        dt_ticks = [round(self.drift_time[int(i)], 1) for i in xlocs[1:-1]]

        plt.xticks(xlocs[1:-1], dt_ticks)
        plt.yticks(ylocs[1:-1], rt_ticks)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.xlabel(self._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)
        return fig

    # TODO: Write compare spectra plot method
    def compare(self, other):
        pass

    def export_plot(self, path=os.getcwd(), dpi=300,
                    file_format='jpg', **kwargs):
        """
        ims.Spectrum.export_plot
        ------------------------

        Exports plot to disk.

        Parameters
        ----------
        path : str, optional
            Directory to save the image,
            by default current working directory

        file_format : str, optional
            by default 'jpg'
        """
        fig = self.plot(**kwargs)
        fig.savefig(f'{path}/{self.name}.{file_format}', dpi=dpi, quality=95,
                    bbox_inches="tight", pad_inches=0.5)

    # TODO: Add automated peak finding and integrating features
    # def find_peaks(self):
    #     return peak_local_max(self.values, min_distance=5, threshold_abs=100)

    # def integrate_peak(self, dt_range, rt_range):
    #     values = self.values[dt_range[0]:dt_range[1],
    #                          rt_range[0]:rt_range[1]]

    # def signal_to_noise(self):
    #     m = self.values.mean(axis=None)
    #     sd = self.values.std(axis=None)
    #     return np.where(sd == 0, 0, m/sd)
