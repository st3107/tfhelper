import math
import typing
from pathlib import Path

import matplotlib.pyplot as plt
from pkg_resources import resource_filename

import numpy as np
import tqdm
import xarray as xr
from xarray.plot.facetgrid import FacetGrid
import tensorflow as tf
from tensorflow.keras.utils import Sequence

EXAMPLE_DIR = resource_filename("tfhelper", "data/nc_files")


class MySequence(Sequence):

    def __init__(
            self,
            filenames: typing.List[str],
            batch_size: int,
            threshold: float = 0.95,
            normalize: bool = True,
            use_threshold: bool = True,
            all_in_mem: bool = False,
            verbose: int = 0
    ):
        self.batch_size = batch_size
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.normalize = normalize
        self.verbose = verbose
        self.filenames = filenames
        self.all_in_mem = all_in_mem
        self.x_data = None
        self.y_data = None
        if self.all_in_mem:
            self.x_data, self.y_data = self.load_files(self.filenames)

    def load_files(self, nc_files: typing.List[str]) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        x, y = [], []
        if self.verbose > 0:
            nc_files = tqdm.tqdm(nc_files)
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
        _x, _y = np.stack(x), np.stack(y)
        return tf.constant(_x, dtype=tf.float16), tf.constant(_y, dtype=tf.float16)

    def __len__(self) -> int:
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx: int) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        if self.all_in_mem:
            return self.x_data[s], self.y_data[s]
        nc_files = self.filenames[s]
        return self.load_files(nc_files)

    def visualize(self, idx, **kwargs) -> None:
        x, y = self.__getitem__(idx)
        x, y = x.numpy(), y.numpy()
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


def create_seqs(
        data_dir: str,
        batch_size: int,
        fractions: typing.Sequence[float] = (0.8, 0.2),
        pattern: str = r"[!.]*.nc",
        **kwargs
) -> typing.List[Sequence]:
    # get all files
    _data_dir = Path(data_dir)
    filenames = list(map(str, _data_dir.glob(pattern)))
    np.random.shuffle(filenames)
    n = len(filenames)
    # get slices
    ss = [0]
    ss.extend([round(f * n) for f in fractions])
    m = len(ss)
    for i in range(1, m):
        ss[i] += ss[i-1]
    # make sequences
    mss = []
    for i in range(1, m):
        ms = MySequence(filenames[ss[i-1]:ss[i]], batch_size, **kwargs)
        mss.append(ms)
    return mss


def concat_ncs(files: typing.Sequence[str]) -> xr.Dataset:
    dss = []
    for f in files:
        ds = xr.load_dataset(f)
        ds = ds[["G", "structure", "fraction"]]
        dss.append(ds)
    return xr.concat(dss, "mixture")
