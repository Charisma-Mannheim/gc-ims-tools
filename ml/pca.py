from ims_module.ims import GCIMS_DataSet
from ims_module.spectroscopy import Spectra
from ims_module.ml import BaseModel
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import datashader as ds


class PCA_Model(BaseModel):

    def __init__(self, dataset, scaling_method=None, **kwargs):
        """
        Wrapper class for scikit learn PCA.
        Adds plots for explained variance ratio,
        loadings and scatter plots of components.

        Parameters
        ----------
        dataset : varies
            GCIMS_DataSet or Spectra

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None
        """
        super().__init__(dataset, scaling_method)
        self.pca = PCA(**kwargs).fit(self.X)
        self.pc = self.pca.transform(self.X)
        if self.scaling_method is not None and scaling_method != 'standard':
            self.loadings = self.pca.components_ / self.weights
        else:
            self.loadings = self.pca.components_

    def __repr__(self):
        return f'''
PCA:
{self.dataset.name},
{self.scaling_method} scaling
'''

    def expl_var_plot(self):
        pass


    def expl_var_ratio_plot(self):
        """
        Plots explained variance ratio per principal component
        and cumulatively.

        Returns
        -------
        [type]
            [description]
        """
        x = [*range(1, self.pca.n_components_ + 1)]
        y = self.pca.explained_variance_ratio_
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.cumsum(y) * 100,
                name='cumulative',
                hovertemplate='%{y} %'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y * 100,
                name='per PC',
                hovertemplate='%{y} %'
            )
        )
        fig.update_layout(
            hovermode='x'
        )
        fig.update_layout(
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio [%]',
            autosize=False,
            width=700,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=16),
            hoverlabel=dict(
                font_size=16,
                bgcolor='white'
                )
        )
        return fig

    def scatter_plot(self, PCs=(1, 2)):
        """
        Interactive scatter plot of two or three principal components.

        Parameters
        ----------
        PCs : tuple, optional
            The components to plot, by default (1, 2).

        Returns
        -------
        plotly express scatter or scatter_3d plot
        """
        pc_dataframe = pd.DataFrame(
            data=self.pc,
            columns=[f'PC{x}' for x in range(1, self.pca.n_components_ + 1)]
        )
        pc_dataframe['Sample'] = self.dataset.samples
        pc_dataframe['Group'] = self.dataset.groups

        if len(PCs) == 2:
            fig = px.scatter(
                data_frame=pc_dataframe,
                x=f'PC{PCs[0]}',
                y=f'PC{PCs[1]}',
                color='Group',
                hover_name='Sample',
                symbol='Group',
                width=700,
                height=500
                )
            fig.update_traces(
                marker=dict(
                    size=12,
                    opacity=0.7,
                    line=dict(width=1)
                    )
            )
            fig.update_layout(
                title=f'PC{PCs[0]} vs PC{PCs[1]}',
                autosize=False,
                margin=dict(l=30, r=20, t=60, b=30),
                font=dict(size=16),
                hoverlabel=dict(font_size=16)
            )

        elif len(PCs) == 3:
            fig = px.scatter_3d(
                data_frame=pc_dataframe,
                x=f'PC{PCs[0]}',
                y=f'PC{PCs[1]}',
                z=f'PC{PCs[2]}',
                color='Group',
                hover_name='Sample',
                symbol='Group',
                width=700,
                height=500
            )
            fig.update_traces(
                marker=dict(
                    size=5,
                    opacity=0.7,
                    line=dict(width=1)
                    )
            )
            fig.update_layout(
                title=f'PC{PCs[0]} vs PC{PCs[1]} vs PC{PCs[2]}',
                autosize=False,
                margin=dict(l=30, r=20, t=50, b=20),
                font=dict(size=14),
                hoverlabel=dict(font_size=16)
            )
        else:
            raise ValueError('You can only plot 2 or 3 principal components!')

        return fig

    def loadings_plot(self, PC=1, range_color=(-0.005, 0.005),
                      datashader=True):
        """
        Plots loadings in original coordinates.

        Parameters
        ----------
        PC : int, optional
            principal component to plot, by default 1

        range_color : tuple, optional
            min and max to anchor the colormap,
            by default (-0.005, 0.005)
            (works for IMS data without RIP)

        datashader : boolean, optional
            if True plots datashader raster instead of original data,
            by default True
            (only relevant for IMS data)

        Returns
        -------
        plotly image
        """
        if isinstance(self.dataset, GCIMS_DataSet):
            loadings = self.loadings.reshape(
                (self.pca.n_components_,
                 self.dataset[0].values.shape[0],
                 self.dataset[0].values.shape[1])
            )
            dt = np.stack(
                [self.dataset[i].drift_time for i in
                 range(len(self.dataset.data))]
                )
            dt_mean = dt.mean(axis=0)
            rt = np.stack(
                [self.dataset[i].ret_time for i in
                 range(len(self.dataset.data))]
                )
            rt_mean = rt.mean(axis=0)

            loadings_ds = xr.Dataset(
                {'Loadings': (('PC', 'Retention Time [s]',
                               'Drift Time [RIPrel]'), loadings)},
                coords={
                    'PC': [
                        f'PC{idx}' for idx in range(1, loadings.shape[0] + 1)
                    ],
                    'Retention Time [s]': rt_mean,
                    self.dataset[0].drift_time_label: dt_mean}
                )

            sample = loadings_ds.sel(PC=f'PC{PC}')

            if datashader:
                cvs = ds.Canvas()
                agg = cvs.raster(sample.Loadings)
            else:
                agg = sample.Loadings

            fig = px.imshow(
                img=agg,
                origin='lower',
                width=600,
                height=700,
                aspect='auto',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                range_color=range_color
            )
            fig.update_layout(
                title=loadings_ds.PC.values[PC-1],
                font=dict(size=16),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=16
                )
            )
            return fig

        elif isinstance(self.dataset, Spectra):
            
            # TODO: Plot Loadings one at a time next
            # to the corresponding PC and the original data
            
            loadings_df = pd.DataFrame(
                self.loadings,
                index=[f'PC{i}' for i in range(1, self.loadings.shape[0] + 1)],
                columns=self.dataset.data.columns
                )

            fig = go.Figure()
            for idx, row in loadings_df.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=row.index, y=row.values,
                        name='',
                        hovertemplate=str(idx),
                        line=dict(
                            color='royalblue',
                            width=0.3
                        )
                    )
                )
            fig.update_layout(
                showlegend=False,
                title=f'PCA Loadings {self.dataset.name}',
                xaxis_title=self.dataset.dim,
                yaxis_title='Loadings',
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
