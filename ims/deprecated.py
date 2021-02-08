from ims_module.ims import Spectrum

import numpy as np
import pandas as pd
import xarray as xr
import json
from zipfile import ZipFile

import os

import plotly.express as px
import datashader as ds

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import simps
from skimage.morphology import white_tophat, disk, local_maxima
from skimage.feature import peak_local_max



class Spectrum_Old(Spectrum):
    
    def __init__(self, name, values, sample, group,
                 ret_time, drift_time, meta_attr, time):

        super(self).__init__(name, values, sample, group,
                 ret_time, drift_time, meta_attr, time)
    
    
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
        Dataset
        """
        with ZipFile(path) as myzip:
            with myzip.open('csv_data.csv', 'r') as mycsv:
                values = pd.read_csv(mycsv, header=None)
            with myzip.open('meta_attributes.json', 'r') as myjson:
                meta_attr = json.load(myjson)

        values = np.array(values)
        values = np.delete(values, -1, axis=1)

        ret_time, drift_time = Spectrum._calc_ret_and_drift_coords(meta_attr)

        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        if subfolders:
            sample = path.split(os.sep)[-2]
            group = path.split(os.sep)[-3]
        else:
            sample = ''
            group = ''

        return cls(name, values, sample, group,
                   ret_time, drift_time, meta_attr, time)


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
        Spectrum
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
        Spectrum
            With corrected baseline.
        """
        fl = self.values[0, :]
        self.values = self.values - fl
        self.values[self.values < 0] = 0
        return self
    
    def plotly_plot(self, datashader=True, range_color=(40,250),
             width=600, height=700):
        """
        Plots Spectrum as plotly image.
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
    
    
    def export_image(self, path=os.getcwd(), file_format='jpeg'):
        """
        Exports spectrum as grayscale image for classification in Orange 3.
        (Not a plot!)

        Parameters
        ----------
        path : str, optional
            Directory to save the image,
            by default current working directory

        file_format : str, optional
            See imageio docs for supported formats:
            https://imageio.readthedocs.io/en/stable/formats.html,
            by default 'jpg'
        """
        imageio.imwrite(uri=f'{path}/{self.name}.{file_format}',
                        im=self.values)