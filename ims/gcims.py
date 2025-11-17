import os
import re
import json
import h5py
import pywt
from array import array
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
from time import ctime
from skimage.morphology import white_tophat, disk
from skimage.measure import label, regionprops
from skimage import measure
from zipfile import ZipFile
from ims.utils import asymcorr
from findpeaks import findpeaks
from scipy.signal import savgol_filter
from scipy import ndimage as ndi
from skimage.segmentation import watershed


class Spectrum:
    """
    Represents one GCIMS-Spectrum with the data matrix,
    retention and drift time coordinates.
    Sample or file name and timestamp are included unique identifiers.

    This class contains all methods that can be applied on a per spectrum basis,
    like I/O, plotting and some preprocessing tools.
    Methods that return a Spectrum change the instance inplace. Use the copy method.


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

    def __init__(self, name, values, ret_time, drift_time, time, meta_attr):
        self.name = name
        self.values = values
        self.ret_time = ret_time
        self.drift_time = drift_time
        self.time = time
        self.peak_table = None
        self._drift_time_label = "Drift time [ms]"
        self.meta_attr = meta_attr

    def __repr__(self):
        return f"GC-IMS Spectrum: {self.name}"

    # add, radd and truediv are implemented to calculate means easily
    def __add__(self, other):
        values = self.values + other.values
        ret_time = self.ret_time + other.ret_time
        drift_time = self.drift_time + other.drift_time
        x = Spectrum(self.name, values, ret_time, drift_time, self.time, self.meta_attr)
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
            x = Spectrum(self.name, values, ret_time, drift_time, self.time, self.meta_attr)
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
            with myzip.open("csv_data.csv", "r") as mycsv:
                values = pd.read_csv(mycsv, header=None)
            with myzip.open("meta_attributes.json", "r") as myjson:
                meta_attr = json.load(myjson)

        values = np.array(values)
        values = np.delete(values, -1, axis=1)

        ret_time = (
            np.arange(meta_attr["Chunks count"])
            * (meta_attr["Chunk averages"] + 1)
            * meta_attr["Chunk trigger repetition [ms]"]
            / 1000
        )
        drift_time = (
            np.arange(meta_attr["Chunk sample count"])
            / meta_attr["Chunk sample rate [kHz]"]
        )

        path = os.path.normpath(path)
        name = os.path.split(path)[1]
        time = datetime.strptime(meta_attr["Timestamp"], "%Y-%m-%dT%H:%M:%S")

        return cls(name, values, ret_time, drift_time, time, meta_attr)

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
        name = name.split(".")[0]

        with open(path, "rb") as f:
            content = f.read()
            i = content.index(0)
            meta_attr = content[: i - 1]
            meta_attr = meta_attr.decode("windows-1252")
            data = content[i + 1 :]
            data = array("h", data)

        meta_attr = meta_attr.split("\n")

        key_re = re.compile(r"^.*?(?==)")
        value_re = re.compile(r"(?<==)(.*?)(?=\[|$)")
        # unit_re = re.compile(r"\[(.*?)\]")

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

        ret_time = (
            np.arange(chunks_count)
            * (chunk_averages + 1)
            * chunk_trigger_repetition
            / 1000
        )

        drift_time = np.arange(chunk_sample_count) / chunk_sample_rate

        return cls(name, data, ret_time, drift_time, timestamp, meta_attr)

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
        name = name.split(".")[0]
        df = pd.read_csv(path, index_col = [0])
        values = df.values
        ret_time = np.array(df.index)
        drift_time = df.columns.to_numpy().astype(float)
        timestamp = os.path.getctime(path)
        timestamp = ctime(timestamp)
        timestamp = datetime.strptime(timestamp, "%a %b  %d %H:%M:%S %Y")
        return cls(name, values, ret_time, drift_time, timestamp, meta_attr={})

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
        with h5py.File(path, "r") as f:
            values = np.array(f["values"])
            ret_time = np.array(f["ret_time"])
            drift_time = np.array(f["drift_time"])
            name = str(f.attrs["name"])
            time = datetime.strptime(f.attrs["time"], "%Y-%m-%dT%H:%M:%S")
            drift_time_label = str(f.attrs["drift_time_label"])
            if "meta_attr" in f.attrs:
                meta_attr = json.loads(f.attrs["meta_attr"])
            else:
                meta_attr = {}

        spectrum = cls(name, values, ret_time, drift_time, time, meta_attr)
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

        with h5py.File(f"{path}/{self.name}.hdf5", "w-") as f:
            f.create_dataset("values", data=self.values)
            f.create_dataset("ret_time", data=self.ret_time)
            f.create_dataset("drift_time", data=self.drift_time)
            f.attrs["name"] = self.name
            f.attrs["time"] = datetime.strftime(self.time, "%Y-%m-%dT%H:%M:%S")
            f.attrs["drift_time_label"] = self._drift_time_label
            f.attrs["meta_attr"] = json.dumps(self.meta_attr)
    
    def normalization(self):
        """
        Normalize a single spectrum by scaling its values to the range [0, 1].

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> sample.normalization()

        Returns
        -------
        self : Spectrum
            The spectrum with normalized values.
        """

        # Find the minimum and maximum values of the intensity values
        min_value = np.min(self.values)
        max_value = np.max(self.values)

        # Normalize the intensity values to the range [0, 1]
        if max_value > min_value:
            self.values = (self.values - min_value) / (max_value - min_value)
        # Raise Error
        else:
            raise ValueError(f"Spectrum {self.name} cannot be normalized because it has constant or zero values.")

        return self

    def find_peaks(self, limit=None, denoise="fastnl", window=30, verbose=0):
        """
        Automated GC-IMS peak detection based on persistent homology.

        Parameters
        ----------
        spectrum : ims.Spectrum
            GC-IMS spectrum to use.

        limit : float
            Values > limit are active search areas to detect regions of interest (ROI).
            If None limit is selected by the minimum persistence score,
            by default None.

        denoise : string, (default : 'fastnl', None to disable)
            Filtering method to remove noise:
                * None
                * 'fastnl'
                * 'bilateral'
                * 'lee'
                * 'lee_enhanced'
                * 'kuan'
                * 'frost'
                * 'median'
                * 'mean'

        window : int, (default : 30)
            Denoising window. Increasing the window size may removes noise better but may also removes details of image in certain denoising methods.

        verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Returns
        -------
        pandas.DataFrame
            Peak table with drift and retention times,
            the correspondig x and y indices,
            the maximum intensity of the peak,
            birth and death levels and scores.

        References
        ----------
        Taskesen, E. (2020). findpeaks is for the detection of peaks and valleys in a 1D vector and 2D array (image). (Version 2.3.1) [Computer software]. https://erdogant.github.io/findpeaks
        """
        if limit is None:
            fp = findpeaks(
                method="topology",
                limit=0,
                scale=True,
                denoise=denoise,
                window=window,
                verbose=verbose,
            )
            fp.fit(self.values)
            limit = fp.results["persistence"]["score"].min()

        # actual peak detection
        fp = findpeaks(
            method="topology",
            limit=limit,
            scale=True,
            denoise=denoise,
            window=window,
            verbose=verbose,
        )
        fp.fit(self.values)

        # reindex to ensure consistent numbering and start at 1
        df = fp.results["persistence"].reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "peak number"

        df["x"] = df["x"].astype("int")
        df["y"] = df["y"].astype("int")

        # add the drift and retention time values for all indices
        if hasattr(self, "rip_ms"):
            df.insert(0, "riprel_dt", self.drift_time[df["x"].values])
            df.insert(3, "abs_dt", self.drift_time[df["x"].values] * self.rip_ms)
        else:
            df.insert(0, "abs_dt", self.drift_time[df["x"].values])
        df.insert(2, "ret_time", self.ret_time[df["y"].values])
        df.insert(0, "compound", "")
        df.insert(1, "intensity", self.values[df["y"].values, df["x"].values])

        self.peak_table = df
        return self

    def detect_peaks(self, threshold_rel=0.5, peak_size=10):
        """
        Fast peak detection using simple thresholding and connected components.
        Returns a labeled mask and a list of peak outlines. Make sure to cut out the RIP
        before using this method to avoid incorrect thresholding.

        Parameters
        ----------
        threshold_rel : float, default=0.5
            Relative threshold for peak detection. Decrease to be more sensitive, increase to to detect more intense peaks only.
        
        peak_size : int, default=10
            Minimum pixel peak size (number of connected pixels) required for a region to be considered a peak.
            Peaks with fewer pixels than this threshold will be filtered out as noise.

        Returns
        -------
        self : Spectrum
            The spectrum with updated peaklist attribute containing peak information

        Example
        -------
        >>> import ims
        >>> sample = ims.Spectrum.read_mea("sample.mea")
        >>> sample.riprel().cut_dt().cut_rt()
        >>> sample.detect_peaks()
        >>> sample.plot_thresholding()
        """
        # Thresholding of 2d array
        threshold = threshold_rel * np.max(self.values)
        binary_mask = self.values > threshold

        # Label connected regions (peaks)
        labeled_mask = label(binary_mask)

        # Get outlines and peak information
        outlines = []
        peaks_data = []
        peak_counter = 1  # Counter for consistent peak labeling after filtering
        
        for region_label in range(1, labeled_mask.max() + 1):
            mask = labeled_mask == region_label
            
            # Get coordinates of the peak (indices where mask is True)
            coords = np.argwhere(mask)
            
            # Filter out small peaks based on peak_size
            if len(coords) < peak_size:
                continue
            
            contours = measure.find_contours(mask, 0.5)
            outlines.append(contours)
            
            # Find peak maximum within this region
            region_values = self.values * mask
            max_idx = np.unravel_index(np.argmax(region_values), region_values.shape)
            max_intensity = self.values[max_idx]
            
            # Calculate peak volume (background-subtracted)
            peak_volume = np.sum(self.values[mask] - threshold)
            
            # Get drift and retention time coordinates
            max_dt_idx = max_idx[1]  # x (drift time)
            max_rt_idx = max_idx[0]  # y (retention time)
            
            peak_entry = {
                'peak_label': peak_counter,
                'intensity': max_intensity,
                'volume': peak_volume,
                'ret_time': self.ret_time[max_rt_idx],
                'x': max_dt_idx,
                'y': max_rt_idx,
                'n_pixels': len(coords)
            }
            
           
            if hasattr(self, "rip_ms"):
                peak_entry['riprel_dt'] = self.drift_time[max_dt_idx]
                peak_entry['abs_dt'] = self.drift_time[max_dt_idx] * self.rip_ms
            else:
                peak_entry['abs_dt'] = self.drift_time[max_dt_idx]
                print("Warning! Make sure that you are using a region without the RIP, for peak detection to function properly.")
            peaks_data.append(peak_entry)
            peak_counter += 1  

        # create peak_list
        df = pd.DataFrame(peaks_data)
        
        if len(df) > 0:
            # Reorder columns to match find_peaks style
            if hasattr(self, "rip_ms"):
                column_order = ['peak_label', 'intensity', 'volume', 'riprel_dt', 'abs_dt', 'ret_time', 'x', 'y', 'n_pixels']
            else:
                column_order = ['peak_label', 'intensity', 'volume', 'abs_dt', 'ret_time', 'x', 'y', 'n_pixels']
            
            df = df[column_order]
            df.index = df.index + 1
            df.index.name = "peak number"
        
        # Store results in instanced dataframes
        self.peaklist = df
        print(f"Found {len(self.peaklist)} peaks")
        self.peaklist_mask = labeled_mask
        self.peaklist_outlines = outlines
        
        return self
    
    def plot_thresholding(self, outline_color="yellow", linewidth=2, annotate=False, **kwargs):
        """
        Plots GC-IMS spectrum with peak outlines from detect_peaks method.
        Must be called after detect_peaks method.

        Parameters
        ----------
        outline_color : str, default="yellow"
            Color for the peak outlines
        linewidth : float, default=2
            Width of the outline lines
        annotate : bool, default=False
            If True, display peak labels on the plot
        **kwargs
            Additional keyword arguments passed to the plot() method:
            - vmin : int, default=30
              Minimum intensity for the colormap
            - vmax : int, default=400
              Maximum intensity for the colormap
            - width : int, default=6
              Figure width in inches
            - height : int, default=6
              Figure height in inches

        Returns
        -------
        matplotlib.pyplot.axes
            The axes object with the plotted spectrum and peak outlines

        Example
        -------
        >>> sample.detect_peaks()
        >>> sample.plot_thresholding(annotate=True)
        >>> plt.show()
        """
        if not hasattr(self, 'peaklist') or self.peaklist is None:
            raise ValueError("Call 'detect_peaks' method first.")
        
        # Plot spectrum
        fig, ax = self.plot(**kwargs)
        
        # Overlay peak outlines using stored results
        for peak_contours in self.peaklist_outlines:
            for contour in peak_contours:
                y, x = contour.T
                ax.plot(
                    self.drift_time[x.astype(int)], 
                    self.ret_time[y.astype(int)], 
                    color=outline_color, 
                    linewidth=linewidth
                )
        
        # Add peak labels if requested
        if annotate:
            for i, row in self.peaklist.iterrows():
                if "riprel_dt" in self.peaklist.columns:
                    x = row["riprel_dt"]
                else:
                    x = row["abs_dt"]
                y = row["ret_time"]
                label = str(int(row["peak_label"]))
                ax.text(x, y, label, c="yellow", fontsize=12, ha='left', va='center')
        
        ax.set_title(f"{self.name} - {len(self.peaklist)} peaks detected")
        
        return ax

    def plot_peaks(self):
        """
        Plots GC-IMS spectrum with peak labels from findpeaks method.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if self.peak_table is None:
            raise ValueError("Call 'find_peaks' method first.")

        # call ims.Spectrum.plot method
        _, ax = self.plot()

        # iterate over peak table and add labels
        for i, row in self.peak_table.iterrows():
            if "riprel_dt" in self.peak_table.columns:
                x = row["riprel_dt"]
            else:
                x = row["abs_dt"]
            y = row["ret_time"]
            label = f"{i} {row['compound']}"
            ax.text(x, y, label, c="yellow", fontsize=12)

        return ax

    def plot_persistence(self):
        """
        Persistance plot of birth vs death levels from findpeak method.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if self.peak_table is None:
            raise ValueError("Call 'find_peaks' method first.")
        
        x = self.peak_table["birth_level"].values
        y = self.peak_table["death_level"].values

        ax = plt.subplot()
        ax.plot(x, y, ".", c="tab:blue")

        # draw line across plot
        X = np.c_[x, y]
        ax.plot([np.min(X), np.max(X)], [np.min(X), np.max(X)], "--", c="tab:grey")

        # set axis limits and labels
        ax.set_xlim((np.min(X), np.max(X)))
        ax.set_ylim((np.min(X), np.max(X)))
        ax.set_xlabel("Birth level")
        ax.set_ylabel("Death level")

        # add peak labels
        for i, row in self.peak_table.iterrows():
            x = row["birth_level"]
            y = row["death_level"]
            label = f"{i} {row['compound']}"
            ax.text(x, y, label)

        return ax

    def watershed_segmentation(self, threshold):
        """
        Finds boundaries for overlapping peaks using watershed segmentation.
        Requires peak_table for starting coordinates.

        Parameters
        ----------
        threshold : int
            Threshold is used to binarize the intensity values to calculate the distances.

        Returns
        -------
        numpy.ndarray
            Labels array with same shape as intensity values.
        """
        if self.peak_table is None:
            raise ValueError("Call 'find_peaks' method first.")
        
        # Binarize intensity values
        image = np.copy(self.values) >= threshold

        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(image)

        # Use x and y coordinates from the peak table
        coords = self.peak_table[["y", "x"]].values

        # Watershed segmentation according to scikit-image tutorial
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        return labels

    def calc_reduced_mobility(self, T = 318.15, p = 1013.25, Ud = 2132, L = 5.3):
        """
        Calculates the reduced mobility values for the drift times denoted in the peak table.
        The formula for the calculation of the reduced mobility values originates from
        Ahrens, A., Zimmermann, S. Towards a hand-held, fast, and sensitive gas chromatograph-ion mobility spectrometer
        for detecting volatile compounds. Anal Bioanal Chem 413, 1009–1016 (2021). https://doi.org/10.1007/s00216-020-03059-9

        Parameters
        ----------
        T : float, optional
            Temperature of the IMS cell, by default 318.15
        p : float, optional
            pressure in the IMS cell, by default 1013.25
        Ud : int, optional
            Drift voltage, by default 2132
        L : float, optional
            Length of the drift tube, by default 5.3

        Returns
        -------
        pandas.DataFrame
            returns the original dataframe from the find_peaks method,
            but adds a column for the reduced mobility
        """
        if self.peak_table is None:
            raise ValueError("Call 'find_peaks' method first.")
        T0 = 273.15
        p0 = 1013.15

        dt = self.peak_table["abs_dt"].values
        K0 = (L**2 * T0 * p) / (dt * 10**-3 * Ud * T * p0)
        self.peak_table["mobility"] = K0
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
        Spectrum
        """
        for i in range(self.values.shape[1]):
            y = self.values[:, i]
            self.values[:, i] = asymcorr(y, lam=lam, p=p, niter=niter)

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
        Spectrum
        """
        if direction == "drift_time":
            axis = 1
        elif direction == "ret_time":
            axis = 0
        elif direction == "both":
            self.values = savgol_filter(self.values, window_length, polyorder, axis=0)
            axis = 1
        else:
            raise ValueError(
                "Only 'drift_time', 'ret_time' or 'both' are valid options!"
            )

        self.values = savgol_filter(self.values, window_length, polyorder, axis=axis)
        return self

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
        fl = self.values[0 : n - 1, :].mean(axis=0)
        self.values = self.values - fl
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
        self._drift_time_label = "Drift time RIP relative"
        self.rip_ms = rip_ms
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
            self.values = self.values[: a - rest, :]
            self.ret_time = self.ret_time[: a - rest]

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
            self.values = self.values[: a - rest0, :]
            self.ret_time = self.ret_time[: a - rest0]
        if rest1 != 0:
            self.values = self.values[:, : b - rest1]
            self.drift_time = self.drift_time[: b - rest1]

        new_dims = (a // n, b // n)

        shape = (new_dims[0], a // new_dims[0], new_dims[1], b // new_dims[1])

        self.values = self.values.reshape(shape).mean(axis=(-1, 1))
        self.ret_time = self.ret_time[::n]
        self.drift_time = self.drift_time[::n]
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
        Spectrum

        Raises
        ------
        ValueError
            When direction is neither 'ret_time', 'drift_time' or 'both'.
        """
        if direction == "ret_time":
            coef_values = pywt.wavedec(
                self.values,
                wavelet=wavelet,
                level=level,
                axis=0
                )
            self.values = coef_values[0]
            self.ret_time = np.linspace(
                self.ret_time[0],
                stop=self.ret_time[-1],
                num=self.values.shape[0]
                )

        elif direction == "drift_time":
            coef_values = pywt.wavedec(
                self.values,
                wavelet=wavelet,
                level=level,
                axis=1
                )
            self.values = coef_values[0]
            self.drift_time = np.linspace(
                self.drift_time[0],
                stop=self.drift_time[-1],
                num=self.values.shape[1]
                )

        elif direction == "both":
            coef_values = pywt.wavedec2(
                self.values,
                wavelet=wavelet,
                level=level,
            )
            self.values = coef_values[0]
            self.ret_time = np.linspace(
                self.ret_time[0],
                stop=self.ret_time[-1],
                num=self.values.shape[0]
                )
            self.drift_time = np.linspace(
                self.drift_time[0],
                stop=self.drift_time[-1],
                num=self.values.shape[1]
                )

        else:
            raise ValueError("Direction must be 'ret_time', 'drift_time or 'both'!")

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

    def plot(self, vmin=30, vmax=400, width=6, height=6):
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
            extent=(
                min(self.drift_time),
                max(self.drift_time),
                min(self.ret_time),
                max(self.ret_time),
            ),
        )

        plt.colorbar().set_label("Intensities [arbitrary units]")
        plt.title(self.name)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.xlabel(self._drift_time_label)
        plt.ylabel("Retention time [s]")

        return fig, ax

    def export_plot(self, path=None, dpi=300, file_format="jpg", **kwargs):
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
        fig.savefig(
            f"{path}/{self.name}.{file_format}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
