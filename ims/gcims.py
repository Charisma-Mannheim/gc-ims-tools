import os
import re
import h5py
from datetime import datetime
from array import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class Spectrum:

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
        self._drift_time_label = 'Drift Time [ms]'
        self.meta_attr = meta_attr
        self.time = datetime.strptime(self.meta_attr['Timestamp'],
                                      '%Y-%m-%dT%H:%M:%S')
        
    def __repr__(self):
        return f"GC-IMS Spectrum: {self.name}"

    # def __add__(self, other):
    #     name = self.sample
    #     mean_values = (self.values + other.values) / 2
    #     mean_ret_time = (self.ret_time + other.ret_time) / 2
    #     mean_drift_time = (self.drift_time + other.drift_time) / 2

    #     return Spectrum(name, mean_values, self.sample, self.group,
    #                           mean_ret_time, mean_drift_time, self.meta_attr)
    
    
    def __add__(self, other):
        values = self.values + other.values
        ret_time = self.ret_time + other.ret_time
        drift_time = self.drift_time + other.drift_time
        return Spectrum(self.name, values, self.sample, self.group,
                        ret_time, drift_time, self.meta_attr, self.time)
        
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise NotImplementedError()
    

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
        Dataset
        """
        meta_attr = Spectrum.read_meta_attr(path)
        ret_time, drift_time = Spectrum._calc_ret_and_drift_coords(meta_attr)

        # get sample and group names from folder names
        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        name = name.split('.')[0]
        if subfolders:
            sample = path.split(os.sep)[-2]
            group = path.split(os.sep)[-3]
        else:
            sample = ''
            group = ''
        
        with open(path, mode='rb') as f:
            data = f.read()
            #find first null byte
            start = 0
            for i, j in enumerate(data):
                start = i
                if j == 0:
                    break

            # read the remaining data
            # values are stored as signed short int in two bytes
            # Chunks count is the size of the retention time
            # Chunks sample count is the size of the drift time
            values = data[start + 1:]
            values = bytearray(values)
            values = array('h', values)
            values = np.array(values)
            values = values.reshape((meta_attr['Chunks count'],
                                     meta_attr['Chunk sample count']))

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
        Spectrum
        """     
        with h5py.File(path, 'r') as f:
            values = np.array(f['values'])
            ret_time = np.array(f['ret_time'])
            drift_time = np.array(f['drift_time'])
            group = str(f.attrs['group'])
            name = str(f.attrs['name'])
            sample = str(f.attrs['sample'])
            time = datetime.strptime(str(f.attrs['time']), '%Y-%m-%dT%H:%M:%S')
            meta_keys = list(f['values'].attrs.keys())
            meta_values = list(f['values'].attrs.values())
            meta_attr = dict(zip(meta_keys, meta_values))
        return cls(
            name, values, sample, group,
            ret_time, drift_time, meta_attr, time
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
            f.attrs['time'] = datetime.strftime(self.time, '%Y-%m-%dT%H:%M:%S')

            for i in self.meta_attr:
                values.attrs[i] = self.meta_attr[i]

    @classmethod
    def mean(cls, spectra_list):
        """
        Calculates means from all spectra in list.
        Mainly needed for the mean implementation in
        Dataset class.

        Parameters
        ----------
        spectra_list : list
            List with all spectra to use.

        Returns
        -------
        Spectrum
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

    # TODO: add features for data reduction
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
        matplotlib.figure.Figure
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

        plt.colorbar()
        plt.title(self.name, fontsize=16)

        # FIXME: Axes values do not work when riprel dt axis
        # because of round
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
        
        plt.show()
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
