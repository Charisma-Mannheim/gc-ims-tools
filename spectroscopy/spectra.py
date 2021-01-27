import numpy as np
import pandas as pd
import os
from glob import glob
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.preprocessing import LabelEncoder

# TODO: write a pure numpy version, or at least get rid of the multiindex

class Spectra:

    def __init__(self, data, name=None, dim=None, groups=None,
                 samples=None, files=None):
        """
        DataSet class for spectroscopic data. Stores spectra in
        pandas DataFrame, where each row represents one spectrum.

        Contains IO, preprocessing and utility methods.

        Parameters
        ----------
        data : pd.DataFrame
            Each row represents one spectrum, column names are
            the coordinate labels, index needs to be a multiindex
            with group, sample and file layers for the mean method.

        name : str, optional
            Name appears as title in plots,
            by default None

        dim : str, optional
            Dimension name for axis labels in plots,
            e.g. wavenumbers,
            by default None

        groups : list, optional
            Lists all group labels for easy access,
            by default None

        samples : list, optional
            Lists all sample labels,
            by default None

        files : list, optional
            Lists all file names loaded in class,
            by default None
        """
        self.data = data
        self.name = name
        self.dim = dim
        self.groups = groups
        self.samples = samples
        self.files = files

    def __repr__(self):
        return f'{self.name}, {len(self.data.index)} Spectra'

    @classmethod
    def read_csv(cls, path, subfolders=True, **kwargs):
        """
        Reads csv files and combines them in a pandas DataFrame.

        Parameters
        ----------
        path : str
            Path to input folder.

        subfolders : boolean, optional
            Generates sample and group labels from subfolder names,
            by default True.

        kwargs:
            See docs for pandas.read_csv for valid arguments.
        """
        files = []
        samples = []
        groups = []
        name = os.path.split(path)[1]
        if subfolders:
            filedirs = glob(f'{path}/*/*/*')
            for filedir in filedirs:
                head, file_name = os.path.split(filedir)
                files.append(file_name)
                head, sample_name = os.path.split(head)
                samples.append(sample_name)
                group = os.path.split(head)[1]
                groups.append(group)
        else:
            filedirs = glob(f'{path}/*')

        datalist = []
        for i in filedirs:
            x = pd.read_csv(i, header=1, index_col=0, **kwargs)
            datalist.append(x)

        index = pd.MultiIndex.from_tuples(
            zip(groups, samples, files),
            names=['group', 'sample', 'file']
            )
        data = pd.concat(datalist, axis=1)
        data = data.transpose()
        data.index = index
        dim = data.columns.name
        return cls(data, name, dim, groups, samples, files)

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
        np.save(f'{folder_name}/sample_names.npy', self.samples)
        for i, j in enumerate(self.data):
            np.save(f'{folder_name}/data/{i}.npy', j)

    # TODO: add color by group or label feature
    def plot(self, line_width=0.5,
             yaxis_title='Absorbance Units'):
        """
        Plots all rows in DataFrame as lineplot.
        Column names are used as x axis.
        Index is used as hovertool labels.

        Parameters
        ----------
        df : pd.DataFrame
            Each row represents one spectrum,
            column names are x axis (i.e. wavenumbers)

        line_width : float, optional
            by default 0.5

        xaxis_title : str, optional
            by default df.columns.name

        yaxis_title : str, optional
            by default 'Absorbance Units'

        title : str, optional
            by default None

        Returns
        -------
        plotly figure
        """
        fig = go.Figure()
        for idx, row in self.data.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=row.index, y=row.values,
                    name='',
                    hovertemplate=str(idx),
                    line=dict(
                        color='royalblue',
                        width=line_width
                    )
                )
            )
        fig.update_layout(
            showlegend=False,
            title=self.name,
            xaxis_title=self.dim,
            yaxis_title=yaxis_title,
            autosize=False,
            width=700,
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(size=16),
            hoverlabel=dict(
                bgcolor='white',
                font_size=18
            )
        )
        return fig

    def mean(self):
        """
        Calculates mean spectra by using a pandas groupby operation.
        Requires MultiIndex with group, sample and file layers.

        Returns
        -------
        Spectra
            With mean spectra and file layer dropped from index.
        """
        new_index = self.data.index.droplevel(2).drop_duplicates()
        new_index.rename(['group', 'sample'], inplace=True)
        self.samples = list(new_index.get_level_values('sample'))
        self.groups = list(new_index.get_level_values('group'))
        self.data = self.data.groupby('sample').mean()
        self.data.index = new_index
        return self

    def cut(self, start, stop):
        """
        Cuts spectra along x axis.

        Parameters
        ----------
        start : int/float
            start value on dimension coordinate
        
        stop : int/float
            stop value on dimension coordinate

        Returns
        -------
        Spectra
            With all spectra cut.
        """
        idx_start = np.abs(self.data.columns - start).argmin()
        idx_stop = np.abs(self.data.columns - stop).argmin()
        self.data = self.data.iloc[:, idx_start:stop]
        return self

    def rubberband(self):
        """
        Performs rubberband baseline subtraction for each row in dataframe.
        Column names are used as x axis.

        Adapted from:
        https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

        Returns
        -------
        Spectra
            With baseline corrected.
        """
        value_list = []
        x = np.array(self.data.columns)

        for i in self.data.index:
            y = self.data.loc[i].values
            # Find the convex hull
            v = ConvexHull(np.array(list(zip(x, y)))).vertices
            # Rotate convex hull vertices until they start from the lowest one
            v = np.roll(v, -v.argmin())
            # Leave only the ascending part
            v = v[:v.argmax()]
            # Create baseline using linear interpolation between vertices
            baseline = np.interp(x, x[v], y[v])
            new_y = y - baseline
            value_list.append(new_y)

        new_values = np.stack(value_list)
        self.data = pd.DataFrame(new_values, index=self.data.index,
                                 columns=self.data.columns)
        return self

    def savgol(self, window_length=3, polyorder=2, deriv=0):
        """
        Applies savitzky golay filter on all spectra.
        Also used to calculate derivatives.

        Parameters
        ----------
        window_length : int, optional
            The length of the filter window (i.e. the number of coefficients).
            window_length must be a positive odd integer,
            by default 3

        polyorder : int, optional
            The order of the polynomial used to fit the samples.
            polyorder must be less than window_length,
            by default 2

        deriv : int, optional
            The order of the derivative to compute.
            This must be a nonnegative integer,
            by default 0

        Returns
        -------
        Spectra
            With smoothing applied.
        """
        new_values = savgol_filter(
            self.data.values, window_length=window_length,
            polyorder=polyorder,
            deriv=deriv
            )
        self.data = pd.DataFrame(data=new_values, index=self.data.index,
                                 columns=self.data.columns)
        return self

    def msc(self, reference=None):
        """
        Performs multiplicative scatter correction.

        Adapted from:
        https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

        Parameters
        ----------
        reference : numpy array, optional
            Reference spectrum.
            If not given estimate from mean,
            by default None

        Returns
        -------
        Spectra
            With scatter correction applied.
        """
        data = self.data.values
        # mean centre correction
        for i in range(data.shape[0]):
            data[i, :] -= data[i, :].mean()
        # Get the reference spectrum. If not given, estimate it from the mean
        if reference is None:
            # Calculate mean
            ref = np.mean(data, axis=0)
        else:
            ref = reference
        # Define a new array and populate it with the corrected data
        data_msc = np.zeros_like(data)
        for i in range(data.shape[0]):
            # Run regression
            fit = np.polyfit(ref, data[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (data[i, :] - fit[0][1]) / fit[0][0]

        self.data = pd.DataFrame(data_msc, index=self.data.index,
                                 columns=self.data.columns)
        return self

    def get_xy(self):
        """
        Returns X and y for machine learning as
        numpy arrays.

        Returns
        -------
        tuple
            (X, y) as np.arrays
        """
        X = self.data.values
        y = np.array(self.groups)
        return (X, y)
