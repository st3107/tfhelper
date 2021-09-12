import math
import typing
from pathlib import Path

import matplotlib.pyplot as plt
from pkg_resources import resource_filename

import numpy as np
import xarray as xr
from xarray.plot.facetgrid import FacetGrid
from tensorflow.keras.utils import Sequence

EXAMPLE_DIR = resource_filename("tfhelper", "data/nc_files")


class MySequence(Sequence):

    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            threshold: float = 0.95,
            normalize: bool = True,
            use_threshold: bool = True
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.normalize = normalize
        self.filenames = []
        self._glob_files()
        self.shuffle()

    def _glob_files(self) -> None:
        _data_dir = Path(self.data_dir)
        self.filenames = list(map(str, _data_dir.glob("[!.]*.nc")))
        return

    def shuffle(self) -> None:
        np.random.shuffle(self.filenames)
        return

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        nc_files = self.filenames[slice(idx * self.batch_size, (idx + 1) * self.batch_size)]
        x, y = [], []
        for f in nc_files:
            ds = xr.load_dataset(f)
            sx = ds["G"].values
            if self.normalize:
                smin, smax = sx.min(), sx.max()
                sx = np.divide(sx - smin, (smax - smin))
            max_ = ds["fraction"].values.max()
            if self.use_threshold:
                if max_ >= self.threshold:
                    sy = np.array([1., 0.])
                else:
                    sy = np.array([0., 1.])
            else:
                sy = np.array([max_, 1. - max_])
            x.append(sx)
            y.append(sy)
        return np.stack(x).astype("float32"), np.stack(y).astype("float32")

    def visualize(self, idx, **kwargs) -> None:
        x, y = self.__getitem__(idx)
        n = x.shape[0]
        da = xr.DataArray(x, dims=["sample", "x"])
        dim0 = da.dims[0]
        if n > 1:
            kwargs.setdefault("col", dim0)
            col_wrap = math.ceil(math.sqrt(n))
            kwargs.setdefault("col_wrap", col_wrap)
            facet: FacetGrid = da.plot(**kwargs)
            axes: np.ndarray = facet.axes.flatten()
            for i in range(n):
                axes[i].legend(["y = {}".format(y[i])])
        else:
            _, ax = plt.subplots()
            ax: plt.Axes
            kwargs.setdefault("ax", ax)
            da.plot(**kwargs)
            ax.set_title("{} = {}".format(da.dims[0], 0))
            ax.legend(["y = {}".format(y[0])])
        return
