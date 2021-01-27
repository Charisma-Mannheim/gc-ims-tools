import numpy as np
import pandas as pd
import xarray as xr

import os
import json
import csv
import h5py
from datetime import datetime
from zipfile import ZipFile
from itertools import islice
from array import array

import plotly.express as px
import datashader as ds
import imageio

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import simps
from skimage.morphology import white_tophat, disk, local_maxima
from skimage.feature import peak_local_max


class GCIMS_Spectrum:

    def __init__(self, name, values, sample, group,
                 ret_time, drift_time, meta_attr, time):
        """
        Represents on GCIMS-Spectrum including retention
        and drift time coordinates, meta attributes and labels.

        Contains all methods that can be applied per spectrum.

        Use one of the read_... methods as constructor.

        Parameters
        ----------
        name : str
            File or Sample name.

        values : np.array
            Data array stored as numpy array.

        sample : str
            Sample label.

        group : str
            Group label

        ret_time : np.array
            Retention time as numpy array.

        drift_time : np.array
            Drift time as numpy array

        meta_attr : dict
            Meta attributes from GAS software.
            
        time: str
            Used by TimeSeries class. Elapsed time.
        """
        self.name = name
        self.values = values
        self.sample = sample
        self.group = group
        self.ret_time = ret_time
        self.drift_time = drift_time
        self.drift_time_label = 'Drift Time [ms]'
        self.meta_attr = meta_attr
        self.time = time

    def __repr__(self):
        return f"GC-IMS Spectrum: {self.name}"

    def __add__(self, other):
        name = self.sample
        mean_values = (self.values + other.values) / 2
        mean_ret_time = (self.ret_time + other.ret_time) / 2
        mean_drift_time = (self.drift_time + other.drift_time) / 2

        return GCIMS_Spectrum(name, mean_values, self.sample, self.group,
                              mean_ret_time, mean_drift_time, self.meta_attr)

    
    @staticmethod
    def set_time(x, time):
        y = x
        y.time = time
        return y

    @staticmethod
    def _get_datetime(meta_attr):
        """
        Returns Timestamp meta attributes entry as
        datetime.datetime object. Needed to generate
        the time coordinate in TimeSeries class.
        """
        return datetime.strptime(meta_attr['Timestamp'], '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def _read_meta_attr_zip(path):
        """
        Reads only the json file with meta attributes
        from zip archive
        """
        with ZipFile(path) as myzip:
            with myzip.open('meta_attributes.json', 'r') as myjson:
                meta_attr = json.load(myjson)
        return meta_attr

    @staticmethod
    def _read_meta_attr_mea(path):
        '''
        Reads and formats just the meta attributes from GAS mea file.
        '''
        with open(path, mode='rb') as f:
            data = f.read()
            meta_attr = []
            for i in data:
                if i == 0:
                    break
                meta_attr.append(chr(i))
                
            meta_attr = ''.join(meta_attr)
            meta_attr = meta_attr.split('\n')
            meta_attr.pop(-1)
            names = []
            nums = []
            units = []
            for i in meta_attr:
                name, value = i.split('=')
                name = name.strip()
                value = value.strip()
                if '[' in value:
                    num, unit = value.split(' ')
                else:
                    num = value
                    unit = ''
                nums.append(num)
                units.append(unit)
                names.append(name)

            keys = [*zip(names, units)]
            keys = [' '.join(i).strip() for i in keys]
            attr = [i.strip('"') if '"' in i else int(i) for i in nums]
            meta_attr = dict(zip(keys, attr))
        return meta_attr

    @staticmethod
    def _calc_ret_and_drift_coords(meta_attr):
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
    def read_mea(cls, path, subfolders=False, time=None):
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
        GCIMS_DataSet
        """
        with open(path, mode='rb') as f:
            data = f.read()

            # read unitl first null byte for meta attributes
            meta_attr = []
            for i in data:
                if i == 0:
                    break
                meta_attr.append(chr(i))

            # keep track of bytelength to start reading at this point later
            byte_len_meta = len(meta_attr)
            
            meta_attr = GCIMS_Spectrum._read_meta_attr_mea(path)

            # read the remaining data
            # values are stored as signed short int in two bytes
            # Chunks count is the size of the retention time
            # Chunks sample count is the size of the drift time
            values = []
            for i in islice(data, byte_len_meta + 1, None):
                values.append(i)

            values = bytearray(values)
            values = array('h', values)
            values = np.array(values)

            values = values.reshape((meta_attr['Chunks count'],
                                     meta_attr['Chunk sample count']))

        ret_time, drift_time = GCIMS_Spectrum._calc_ret_and_drift_coords(meta_attr)

        # get sample and group names from folder names
        name = os.path.split(path)[1]
        if subfolders:
            sample = os.path.split(os.path.split(path)[0])[1]
            group = os.path.split(os.path.split(os.path.split(path)[0])[0])[1]
        else:
            sample = ''
            group = ''
    
        return cls(name, values, sample, group,
                   ret_time, drift_time, meta_attr, time)

    @classmethod
    def read_zip(cls, path, subfolders=False, time=None):
        """
        Reads zip archive from GAS mea to zip tool.

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
        GCIMS_DataSet
        """
        with ZipFile(path) as myzip:
            with myzip.open('csv_data.csv', 'r') as mycsv:
                values = pd.read_csv(mycsv, header=None)
            with myzip.open('meta_attributes.json', 'r') as myjson:
                meta_attr = json.load(myjson)

        values = np.array(values)
        values = np.delete(values, -1, axis=1)

        ret_time, drift_time = GCIMS_Spectrum._calc_ret_and_drift_coords(meta_attr)

        name = os.path.split(path)[1]
        if subfolders:
            sample = os.path.split(os.path.split(path)[0])[1]
            group = os.path.split(os.path.split(os.path.split(path)[0])[0])[1]
        else:
            sample = ''
            group = ''
        return cls(name, values, sample, group,
                   ret_time, drift_time, meta_attr, time)

    @classmethod
    def read_hdf5(cls, path, time=None):
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
        GCIMS_Spectrum
        """     
        with h5py.File(path, 'r') as f:
            values = np.array(f['values'])
            ret_time = np.array(f['ret_time'])
            drift_time = np.array(f['drift_time'])
            group = str(f.attrs['group'])
            name = str(f.attrs['name'])
            sample = str(f.attrs['sample'])
            time = str(f.attrs['time'])
            meta_keys = list(f['values'].attrs.keys())
            meta_values = list(f['values'].attrs.values())
            meta_attr = dict(zip(meta_keys, meta_values))
        return cls(
            name, values, sample, group,
            ret_time, drift_time, meta_attr, time
            )

    def to_xarray(self):
        """
        Constructs an xarray DataArray.

        Returns
        -------
        xarray.DataArray
            Coordinates are ret_time and drift_time.
        """
        return xr.DataArray(
            data=self.values,
            coords=(self.ret_time, self.drift_time),
            dims=('ret_time', 'drift_time'),
            name=self.name,
            attrs=self.meta_attr
        )

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
            values = f.create_dataset('values', data=self.values)
            ret_time = f.create_dataset('ret_time', data=self.ret_time)
            drift_time = f.create_dataset('drift_time', data=self.drift_time)

            f.attrs['group'] = self.group
            f.attrs['sample'] = self.sample
            f.attrs['name'] = self.name
            f.attrs['time'] = self.time

            for i in self.meta_attr:
                values.attrs[i] = self.meta_attr[i]

    @classmethod
    def mean(cls, spectra_list):
        """
        Calculates means from all spectra in list.
        Mainly needed for the mean implementation in
        GCIMS_DataSet class.

        Parameters
        ----------
        spectra_list : list
            List with all spectra to use.

        Returns
        -------
        GCIMS_Spectrum
            With mean values.
        """
        name = spectra_list[0].sample
        sample = spectra_list[0].sample
        group = spectra_list[0].group
        meta_attr = spectra_list[0].meta_attr

        length = len(spectra_list)
        values = np.array(sum([i.values for i in spectra_list])) / length
        ret_time = np.array(sum([i.ret_time for i in spectra_list])) / length
        drift_time = np.array(sum([i.drift_time for i in spectra_list]))\
            / length
        
        time = None

        return cls(name, values, sample, group, ret_time,
                   drift_time, meta_attr, time)

    def riprel(self):
        """
        Replaces drift time coordinate with RIP relative values.

        Returns
        -------
        GCIMS_Spectrum
            RIP relative drift time coordinate
            otherwise unchanged.
        """
        rip_index = np.argmax(self.values, axis=1)[0]
        rip_ms = self.drift_time[rip_index]
        dt_riprel = self.drift_time / rip_ms
        self.drift_time = dt_riprel
        self.drift_time_label = 'Drift Time RIP relative'
        return self

    def rip_scaling(self):
        """
        Scales values relative to global maximum.

        Returns
        -------
        GCIMS_Spectrum
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
        GCIMS_Spectrum
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
        GCIMS_Spectrum
            With cut retention time
        """
        idx_start = np.abs(self.ret_time - start).argmin()
        idx_stop = np.abs(self.ret_time - stop).argmin()
        self.ret_time = self.ret_time[idx_start:idx_stop]
        self.values = self.values[idx_start:idx_stop, :]
        return self

    # TODO: add features for data reduction
    def wavelet_compression(self):
        pass

    def tophat(self, size=15):
        """
        Applies white tophat filter on values.
        Baseline correction.

        (Slower with larger size.)

        Parameters
        ----------
        size : int, optional
            Size of structuring element, by default 15

        Returns
        -------
        GCIMS_Spectrum
            With tophat applied.
        """      
        self.values = white_tophat(self.values, disk(size))
        return self

    def sub_first_row(self):
        """
        Subtracts first row from every row in spectrum.
        Baseline correction.

        Returns
        -------
        GCIMS_Spectrum
            With corrected baseline.
        """
        fl = self.values[0, :]
        self.values = self.values - fl
        self.values[self.values < 0] = 0
        return self

    def plot(self, datashader=True, range_color=(40,250),
             width=600, height=700):
        """
        Plots GCIMS_Spectrum as plotly image.
        Optionally plots datashader representation of data
        instead of actual values, to speed up the function.

        Parameters
        ----------
        datashader : bool, optional
            If True plots datashader representation,
            by default True

        range_color : tuple, optional
            (low, high) to map the colorscale,
            by default (40,250)

        width : int, optional
            plot width in pixels,
            by default 600

        height : int, optional
            plot height in pixels,
            by default 700

        Returns
        -------
        plotly express Figure
            Interactive plot.
        """
        img = xr.DataArray(
            data=self.values,
            coords=[self.ret_time, self.drift_time],
            dims=['Retention Time [s]', self.drift_time_label]
        )
        
        if self.time is None:
            title = self.name
        else:
            title = f'{self.name}   /   {self.time} h'
        
        if datashader:
            cvs = ds.Canvas()
            agg = cvs.raster(img)
            agg.name = 'Value'
        else:
            agg = img
            agg.name = 'Value'

        fig = px.imshow(
            img=agg,
            origin='lower',
            width=width,
            height=height,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            range_color=range_color
        )
        fig.update_layout(
            title=title,
            font=dict(size=16),
            margin=dict(l=50, r=20, t=100, b=20),
            hoverlabel=dict(
                bgcolor='white',
                font_size=16
            )
        )
        return fig

    def export_plot(self, path=os.getcwd(), file_format='jpeg', **kwargs):
        """
        Exports static plot to disk.

        Kaleido is required:
        $ conda install -c plotly python-kaleido

        Parameters
        ----------
        path : str, optional
            Directory to save the image,
            by default os.getcwd()

        file_format : str, optional
            See plotly kaleido documentation for supported formats
            https://plotly.com/python/static-image-export/,
            by default 'jpeg'
        """
        fig = self.plot(**kwargs)
        fig.write_image(f'{path}/{self.name}.{file_format}')

    def export_image(self, path=os.getcwd(), file_format='jpeg'):
        """
        Exports spectrum as grayscale image for classification in Orange 3.
        (Not a plot!)

        Parameters
        ----------
        path : str, optional
            Directory to save the image,
            by default os.getcwd()

        file_format : str, optional
            See imageio docs for supported formats:
            https://imageio.readthedocs.io/en/stable/formats.html,
            by default 'jpeg'
        """
        imageio.imwrite(uri=f'{path}/{self.name}.{file_format}',
                        im=self.values)

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
