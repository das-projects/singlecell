import functools
import gc
import operator
import os
import shutil
import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Union, Optional  # Meta+
from typing import Iterable, Sequence, Mapping, Literal  # Generic ABCs
from typing import Tuple, List, Dict  # Generic

import jax
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import csv


class SCData:
    def __init__(
            self,
            scdata: Optional[Union[pa.Table, pd.DataFrame, np.ndarray]] = None,
            obs: Optional[Union[pa.Array, pa.Table, Mapping[str, Iterable[Any]]]] = None,
            var: Optional[Union[pa.Array, pa.Table, Mapping[str, Iterable[Any]]]] = None,
            uns: Optional[Union[pa.DictionaryArray, pd.Categorical, Mapping[str, Any]]] = None,
            obsm: Optional[Union[pa.DictionaryArray, np.ndarray]] = None,
            varm: Optional[Union[pa.DictionaryArray, np.ndarray]] = None,
            dtype: Optional[pa.DataType] = None,
            source: Optional[Union[Path, List[Path]]] = None,
            filetype: Optional[str] = None,
            data_dir: Optional[PathLike] = None,
            platform: Literal['cpu', 'gpu', 'tpu'] = 'cpu',
    ):
        # Generate unique file directory, if not provided
        self.data_dir = './data' if data_dir is None else data_dir
        dir_id = uuid.uuid4().hex[:6].upper() if data_dir is None else None
        # Folder with compressed parquet format raw data
        self.data_raw = self.data_dir + '/raw/' + dir_id if data_dir is None else self.data_dir + '/raw'
        self.parquet_file = self.data_raw + '/data.arrow'
        # Folder with arrow format for fast in-memory streaming data processing
        self.data_interim = self.data_dir + '/interim/' + dir_id if data_dir is None else self.data_dir + '/interim'
        self.arrow_file = self.data_interim + '/data.arrow'
        # Folder with processed data, graphs, models, etc
        self.data_processed = self.data_dir + '/processed/' + dir_id if data_dir is None else self.data_dir + '/processed'

        # Set platform between 'cpu', 'gpu' or 'tpu'
        self.use_cores = True
        self.set_jax_platform(platform)

        # Miscellaneous Initialization
        self.dtype = pa.float32() if dtype is None else dtype
        self.source = source
        self.batch_size = 10000
        self.filetype = "parquet"

        # Convert data into on-disk source using pyarrow.dataset
        # This allows us to read/stream/view depending on the size
        # Stream when file size is too big to fit in memory, with batch size dependent on available resources
        if scdata is not None:  # If data matrix is provided
            self.dataset = self.convert_to_on_disk(scdata, obs, var, uns, obsm, varm)
        elif self.source is not None:  # If source directory for external data is provided
            self.dataset = self.read_data(source, filetype)

        self.update_attributes()
        if self.filetype == "parquet":
            self.parquet_to_arrow()
        self.memory_mapped_dataset = pa.memory_map(self.arrow_file, 'r')

    def to_memory(self):
        return pa.ipc.RecordBatchFileReader(self.memory_mapped_dataset)

    def convert_to_on_disk(self, scdata, obs, var, uns, obsm, varm):
        # Ensure unique names in obs and var, if they exist
        obs = self.make_unique(obs) if obs is not None else None
        var = self.make_unique(var) if var is not None else None

        # Read scdata
        # If Numpy array convert to pandas
        if isinstance(scdata, np.ndarray):
            # Assert shape constraints
            assert scdata.shape.__len__() == 2, "Single Cell Data matrix must be 2 dimensional"
            if obs is not None:
                assert scdata.shape[0] == len(obs), "Length of row names must match the rows of the data matrix"
            if var is not None:
                assert scdata.shape[1] == len(var), "Length of column names must match the columns of the data " \
                                                    "matrix "
            # Convert to Pandas: Check if this can be done by zero-copy
            obs = pa.array(["cell{0}".format(x) for x in np.arange(scdata.shape[0])]) if (obs is None) else obs
            var = pa.array(["gene{0}".format(x) for x in np.arange(scdata.shape[1])]) if (var is None) else var
            scdata = pd.DataFrame(scdata, columns=var, index=obs)

        # Convert Pandas to pyarrow.Table
        if isinstance(scdata, pd.DataFrame):
            # Conversion is zero-copy whenever possible, if not then all available threads are used
            scdata = pa.Table.from_pandas(df=scdata, preserve_index=True, nthreads=self.use_cores)

        # Select data part of the pyarrow.Table
        if isinstance(scdata, (pa.Table, pa.RecordBatch)):
            # Select the appropriate subset of data
            scdata = self.ensure_scdata_format(scdata, obs, var, obsm, varm, uns)

        # Save data to raw data dir
        ds.write_dataset(scdata, base_dir=self.data_raw, format="parquet")

        # On disk dataset access
        dataset = ds.dataset(self.data_raw, format="parquet")
        self.filetype = "parquet"

        return dataset

    def ensure_scdata_format(self, scdata: Optional[pa.Table, pa.RecordBatch], obs, var,
                             obsm: Optional[pa.DictionaryArray, pd.Categorical, np.ndarray],
                             varm: Optional[pa.DictionaryArray, pd.Categorical, np.ndarray],
                             uns: Optional[pa.DictionaryArray, pd.Categorical, np.ndarray]):
        # Read obs and var
        self.obs = self.format_obs(obs, scdata)
        self.var = self.format_var(var, scdata)

        # Read varm and obsm
        self.obsm = self.format_dict(obsm, self.obs)
        self.varm = self.format_dict(varm, self.var)

        # Read unstructured annotations
        self.uns = self.format_dict(uns)

        # Cast scdata in the SCData format
        scdata = pa.concat_tables(self.obs, scdata.cast(self.scdata_schema).select(self.var))

        return scdata

    @property
    def scdata_schema(self):
        return pa.schema(self.fields, self.metadata)

    @property
    def fields(self):
        return [pa.field(item, self.dtype, metadata=self.varm[item] if self.varm is not None else None) for item in
                self.var]

    @property
    def metadata(self):
        return {b'uns': self.uns, b'obsm': self.obsm}

    def format_obs(self, obs, scdata):
        obs_schema = pa.schema([pa.field('__index_level_0__', pa.string(), metadata=None)])
        if obs in None:
            if any([name == '__index_level_0__' for name in scdata.schema.names]):
                return scdata.read_all().column('__index_level_0__') \
                    if isinstance(scdata, pa.ipc.RecordBatchFileReader) else scdata.column('__index_level_0__')
            else:
                return pa.table([["cell{0}".format(x) for x in np.arange(self.n_obs)]], schema=obs_schema)
        elif isinstance(obs, pa.Array):
            obs = self.make_unique(obs)
            return pa.Table.from_arrays(arrays=obs, schema=obs_schema)
        elif isinstance(obs, (pa.Table, pa.RecordBatch)):
            assert obs.shape[1] == 1, "Row names must be a single column"
            assert obs.shape[0] == self.n_obs, "Length of row names must match the rows of the data matrix"
            obs = self.make_unique(obs)
            return pa.Table.from_arrays(arrays=obs, schema=obs_schema)

    def format_var(self, var, mmap):
        return self.make_unique([name for name in mmap.read_all().schema.names if name != '__index_level_0__']) if (
                var is None) else self.make_unique(var)

    @staticmethod
    def format_dict(dict_array, array=None):
        array = range(len(dict_array.flatten())) if array is None else array
        if isinstance(dict_array, pa.DictionaryArray):
            return dict_array
        elif isinstance(dict_array, pd.Categorical):
            return pa.DictionaryArray.from_arrays(dict_array.categories, dict_array.codes)
        elif isinstance(dict_array, np.ndarray):
            assert len(array) == len(dict_array.flatten()), "Annotation length does not match the shape of data"
            return pa.DictionaryArray.from_arrays(array, dict_array.flatten())

    def update_attributes(self):
        self.obs = self.dataset.to_table(columns=['__index_level_0__'])
        self.var = self.make_unique([name for name in self.dataset.schema.names if name != '__index_level_0__'])
        self.uns = self.dataset.schema.metadata[b'uns']
        self.obsm = self.dataset.schema.metadata[b'obsm']
        self.varm = [self.dataset.schema.field(name).metadata for name in self.var]

    @staticmethod
    def make_unique(value):
        if isinstance(value, (pa.Table, pa.Array, pa.RecordBatch)):
            value = functools.reduce(operator.iconcat, value.to_pandas().values.tolist(), [])
        counter = {item: 1 for item in value}
        if len(counter) == len(value):
            return value
        unique_list = []
        for id, item in enumerate(value):
            if item in value[:id]:
                unique_list += [str(item) + '-{0}'.format(counter[item])]
                counter[item] += 1
            else:
                unique_list += [item]
        return unique_list

    def read_data(self, source, filetype):
        _, _, filenames = next(os.walk(source))
        supported_filetypes = ["parquet", "arrow", "csv", "h5"]

        if filetype is None:
            filetypes = [type for type in supported_filetypes if any([file.endswith(type) for file in filenames])]
            filetype_is_supported = False if filetypes is None else True
            filetype = filetypes[0] if filetype_is_supported else None
        else:
            filetype_is_supported = any([filetype == supported_filetype for supported_filetype in supported_filetypes])

        # List of files with a given filetype
        def data_files(ftype):
            return [file for file in filenames if file.endswith(ftype)]

        # Move file to a given location
        def move_file(file, location):
            return shutil.move(file, location)

        if filetype_is_supported:
            # Move files to raw or interim if parquet or arrow, respectively and return dataset object
            if filetype == "parquet":
                for file in data_files(filetype):
                    move_file(file, self.data_raw)
                self.filetype = "parquet"
                return ds.dataset(self.data_raw, format=filetype)
            elif filetype == "arrow":
                for file in data_files(filetype):
                    move_file(file, self.data_interim)
                self.arrow_to_parquet()
                self.filetype = "arrow"
                return ds.dataset(self.data_raw, format=filetype)
            elif filetype == "csv":
                self.filetype = "parquet"
                return self.read_csv(data_files(filetype))
            elif filetype == "h5":
                self.filetype = "parquet"
                return self.read_hdf5(data_files(filetype))
        else:
            print("No supported filetype at {0}".format(source))

    def parquet_to_arrow(self):
        batches = self.dataset.to_batches(batch_size=self.batch_size)
        gc.collect()

        sink = pa.OSFile(self.arrow_file, 'wb')
        writer = pa.RecordBatchFileWriter(sink, self.dataset.schema)
        for _, batch in enumerate(batches):
            writer.write_batch(batch)
        writer.close()
        self.filetype = "arrow"

    def arrow_to_parquet(self):
        dataset = ds.dataset(self.data_interim, format="arrow")
        batches = dataset.to_batches(batch_size=self.batch_size)
        gc.collect()

        writer = pq.ParquetWriter(self.parquet_file, dataset.schema)
        for _, batch in enumerate(batches):
            writer.write_table(batch)
        writer.close()

    def read_csv(self, filenames, delimiter=','):
        global parquet_writer
        for file in filenames:
            csv_reader = csv.open_csv(file, read_options=csv.ReadOptions(use_threads=True),
                                      parse_options=csv.ParseOptions(delimiter=delimiter),
                                      convert_options=csv.ConvertOptions(column_types=self.dtype))
            parquet_writer = pq.ParquetWriter(self.parquet_file, csv_reader.schema)

            nrow = 0
            for batch in csv_reader:
                batch_df = batch.to_pandas()
                nrow += batch_df.shape[0]
                parquet_writer.write_table(pa.Table.from_pandas(df=batch_df))

        parquet_writer.close()
        return ds.dataset(self.parquet_file, format="parquet")

    def read_hdf5(self, filenames, chunksize=10000):
        global parquet_writer, parquet_schema
        for file in filenames:
            stream = pd.read_hdf(file, chunksize=chunksize)

            for i, chunk in enumerate(stream):
                if i == 0:
                    # Infer schema and open parquet file on first chunk
                    parquet_schema = pa.Table.from_pandas(df=chunk).schema
                    parquet_writer = pq.ParquetWriter(self.parquet_file, parquet_schema)

                table = pa.Table.from_pandas(df=chunk, Schema_schema=parquet_schema)
                parquet_writer.write_table(table)

        parquet_writer.close()
        return ds.dataset(self.parquet_file, format="parquet")

    @property
    def __sizeof__(self) -> int:
        return self.memory_mapped_dataset.size()

    @property
    def __repr__(self) -> str:
        backed_at = f" backed at {str(self.data_dir)!r}"
        descr = f"SCData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __eq__(self, other):
        """Equality testing"""
        if (
                self.obs == other.obs |
                self.var == other.var |
                self.uns == other.uns |
                self.obsm == other.obsm |
                self.varm == other.varm |
                self.dtype == other.dtype
        ):
            current_batches = self.to_memory()
            other_batches = other.to_memory()
            booleans = [current_batches.get_batch(i).equals(other_batches.get_batch(i)) for i in
                        range(current_batches.num_record_batches)]
            return True if all(booleans) else False
        else:
            return False

    def __delitem__(self, columns):
        batches = self.to_memory().read_all().drop(columns)
        self.update_scdata(batches)

    def __getitem__(self, offset=0, length=None) -> pa.Table:
        """Returns a zero copy slice of the object."""
        return self.to_memory().read_all().slice(offset, length)

    def __len__(self) -> int:
        return self.to_memory().read_all().shape[0]

    def __contains__(self, key: Any):
        raise AttributeError(
            "SCData has no attribute __contains__, don’t check `in scdata`."
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of scdata Table (:attr:`n_obs`, :attr:`n_vars`)."""
        return self.to_memory().read_all().shape

    @property
    def scdata(self) -> pd.DataFrame:
        return self.to_memory().read_all().select(self.var).to_pandas()

    @scdata.setter
    def scdata(self, value: Optional[Union[pa.Table, pd.DataFrame, np.ndarray]]):
        self.dataset = self.convert_to_on_disk(value, obs=None, var=None, uns=None, obsm=None, varm=None)
        self.update_attributes()
        self.parquet_to_arrow()
        self.filetype = "arrow"
        self.memory_mapped_dataset = pa.memory_map(self.arrow_file, 'r')

    @scdata.deleter
    def scdata(self):
        self.dataset = self.convert_to_on_disk(scdata=None, obs=None, var=None, uns=None, obsm=None, varm=None)
        self.update_attributes()
        self.parquet_to_arrow()
        self.filetype = "parquet"
        self.memory_mapped_dataset = None

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.to_memory().read_all().shape[0]

    @property
    def n_vars(self) -> int:
        """Number of variables/features."""
        return self.to_memory().read_all().shape[1]

    @property
    def obs(self) -> pd.DataFrame:
        """One-dimensional annotation of observations (`pd.DataFrame`)."""
        return self.obs.to_pandas()

    @obs.setter
    def obs(self, value: Optional[Union[pa.Table, pd.DataFrame, pa.array]]):
        self.obs = self.format_obs(value, self.to_memory())
        batches = self.to_memory().read_all().set_column(0, pa.field('__index_level_0__', self.dtype), self.obs)
        self.update_scdata(batches)

    @obs.deleter
    def obs(self):
        self.obs = None
        batches = self.to_memory().read_all().remove_column(0)
        self.update_scdata(batches)

    @property
    def var(self) -> pd.DataFrame:
        """One-dimensional annotation of variables/ features (`pd.DataFrame`)."""
        return self.var

    @var.setter
    def var(self, value: Optional[Union[pa.Array, pa.Table, Mapping[str, Iterable[Any]]]]):
        self.var = self.format_var(value, self.to_memory())
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @var.deleter
    def var(self):
        self.var = self.format_var(None, self.to_memory())
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @property
    def uns(self) -> Optional[pa.DictionaryArray]:
        """Unstructured annotation (ordered dictionary)."""
        return self.uns.to_pandas()

    @uns.setter
    def uns(self, value):
        self.uns = self.format_dict(value)
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @uns.deleter
    def uns(self):
        self.uns = None
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @property
    def obsm(self) -> Optional[pa.DictionaryArray]:
        """\
        Multi-dimensional annotation of observations
        (mutable structured :class:`~numpy.ndarray`).

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length `n_obs`.
        Is sliced with `scdata` and `obs` but behaves otherwise like a :term:`mapping`.
        """
        return self.obsm.to_pandas()

    @obsm.setter
    def obsm(self, value):
        self.obsm = self.format_dict(value)
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @obsm.deleter
    def obsm(self):
        self.obsm = None
        batches = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(batches)

    @property
    def varm(self) -> Optional[pa.DictionaryArray]:
        """\
        Multi-dimensional annotation of variables/features
        (mutable structured :class:`~numpy.ndarray`).

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length `n_vars`.
        Is sliced with `scdata` and `var` but behaves otherwise like a :term:`mapping`.
        """
        return self.varm.to_pandas()

    @varm.setter
    def varm(self, value):
        self.varm = self.format_dict(value)
        scdata = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(scdata)

    @varm.deleter
    def varm(self):
        self.varm = None
        scdata = self.to_memory().read_all().replace_schema_metadata(self.scdata_schema)
        self.update_scdata(scdata)

    def update_scdata(self, scdata):  # TODO: Only accepts one file for now
        ds.write_dataset(scdata, self.data_interim, format="arrow")
        dataset = ds.dataset(self.data_interim, format="arrow")
        self.filetype = "arrow"
        self.memory_mapped_dataset = pa.memory_map(dataset.files[0], 'r')

    def transpose(self, in_place: bool = True):
        """\
        Transpose whole object.

        Data matrix is transposed, observations and variables are interchanged.
        """
        scdata = self.to_memory().read_all().select(self.var).to_pandas().transpose()
        obs = self.format_obs(pa.table(self.var), self.to_memory())
        var = self.obs
        if not in_place:
            return SCData(scdata, obs, var, self.uns, self.obsm, self.varm)
        else:
            scdata: pa.Table = pa.Table.from_pandas(df=scdata, preserve_index=True, nthreads=self.use_cores)
            scdata = self.ensure_scdata_format(scdata, obs, var, self.obsm, self.varm, self.uns)
            self.update_scdata(scdata)

    T = property(transpose)

    def to_pandas(self) -> pd.DataFrame:
        """\
        Generate shallow :class:`~pandas.DataFrame`.

        The scdata matrix :attr:`X` is returned as
        :class:`~pandas.DataFrame`, where :attr:`obs_names` initializes the
        index, and :attr:`var_names` the columns.

        * No annotations are maintained in the returned object.
        * The scdata matrix is densified in case it is sparse.

        Params
        ------
        layer : str
            Key for `.layers`.
        """
        return self.scdata

    def obs_vector(self) -> pd.DataFrame:
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`obs`.

        Made for convenience, not performance.
        Intentionally permissive about arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`var_names` or :attr:`obs`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional nd array, with values for each obs in the same order
        as :attr:`obs_names`.
        """
        return self.obs.to_pandas()

    def var_vector(self) -> pd.DataFrame:
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`obs`.

        Made for convenience, not performance. Intentionally permissive about
        arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`obs_names` or :attr:`var`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional nd array, with values for each var in the same order
        as :attr:`var_names`.
        """
        return pd.DataFrame(self.var)

    def copy(self, filename: Optional[PathLike] = None) -> "SCData":
        """Full copy including on disk."""
        return SCData(self.scdata, self.obs, self.var, self.uns, self.obsm, self.varm)

    def concatenate(
            self,
            others: Iterable["SCData"],
            join: str = "inner",
            batch_key: str = "batch",
            batch_categories: Sequence[Any] = None,
            uns_merge: Optional[str] = None,
            index_unique: Optional[str] = "-",
            fill_value=None,
    ) -> "SCData":
        """\
        Concatenate along the observations axis.

        The :attr:`uns`, :attr:`varm` and :attr:`obsm` attributes are ignored.

        Currently, this works only in `'memory'` mode.

        If promote==False, a zero-copy concatenation will be performed. The schemas
        of all the Tables must be the same (except the metadata), otherwise an
        exception will be raised. The result Table will share the metadata with the
        first table.

        If promote==True, any null type arrays will be casted to the type of other
        arrays in the column of the same name. If a table is missing a particular
        field, null values of the appropriate type will be generated to take the
        place of the missing field. The new schema will share the metadata with the
        first table. Each field in the new schema will share the metadata with the
        first table which has the field defined. Note that type promotions may
        involve additional allocations on the given ``memory_pool``.

        Parameters
        ----------
        others
            AnnData matrices to concatenate with. Each matrix is referred to as
            a “batch”.
        join
            Use intersection (`'inner'`) or union (`'outer'`) of variables.
        batch_key
            Add the batch annotation to :attr:`obs` using this key.
        batch_categories
            Use these as categories for the batch annotation. By default, use increasing numbers.
        uns_merge
            Strategy to use for merging entries of uns. These strategies are applied recusivley.
            Currently implemented strategies include:

            * `None`: The default. The concatenated object will just have an empty dict for `uns`.
            * `"same"`: Only entries which have the same value in all AnnData objects are kept.
            * `"unique"`: Only entries which have one unique value in all AnnData objects are kept.
            * `"first"`: The first non-missing value is used.
            * `"only"`: A value is included if only one of the AnnData objects has a value at this
              path.
        index_unique
            Make the index unique by joining the existing index names with the
            batch category, using `index_unique='-'`, for instance. Provide
            `None` to keep existing indices.
        fill_value
            Scalar value to fill newly missing values in arrays with. Note: only applies to arrays
            and sparse matrices (not dataframes) and will only be used if `join="outer"`.

            .. note::
                If not provided, the default value is `0` for sparse matrices and `np.nan`
                for numpy arrays. See the examples below for more information.

        Returns
        -------
        :class:`~anndata.AnnData`
            The concatenated :class:`~anndata.AnnData`, where `adata.obs[batch_key]`
            stores a categorical variable labeling the batch.

        Notes
        -----

        .. warning::

           If you use `join='outer'` this fills 0s for sparse scdata when
           variables are absent in a batch. Use this with care. Dense scdata is
           filled with `NaN`. See the examples.

        Examples
        --------
        Joining on intersection of variables.
        """
        if join == 'inner':
            return SCData(pa.concat_tables([self.to_memory().read_all()].extend(others), bool_promote=False))
        elif join == 'outer':
            return SCData(pa.concat_tables([self.to_memory().read_all()].extend(others), bool_promote=True))
        else:
            return SCData(pa.concat_tables([self.to_memory().read_all()].extend(others), bool_promote=False))

    def filter_cells(
            self,
            min_counts: Optional[int] = None,
            max_counts: Optional[int] = None,
            min_genes: Optional[int] = None,
            max_genes: Optional[int] = None,
            inplace: bool = True,
            copy: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass

    def filter_genes(
            self,
            min_counts: Optional[int] = None,
            min_cells: Optional[int] = None,
            max_counts: Optional[int] = None,
            max_cells: Optional[int] = None,
            inplace: bool = True,
            copy: bool = False,
    ) -> Union["SCData", None, Tuple[np.ndarray, np.ndarray]]:
        pass

    def log1p(self, base: Optional[int], in_place: bool = True):
        return self.apply(np.log1p, in_place)

    def sqrt(self, in_place: bool = True) -> Optional["SCData"]:
        pass

    def apply(self, function, axis=None, in_place: bool = True) -> Optional[pa.RecordBatchFileReader, "SCData"]:
        scdata = self.to_pandas().apply(function, axis=axis, result_type='broadcast')
        if in_place:
            scdata = self.ensure_scdata_format(pa.Table.from_pandas(scdata), self.obs, self.var, self.uns, self.obsm,
                                               self.varm)
            self.filetype = "arrow"
            with open(self.arrow_file, 'bw') as f:
                writer = pa.RecordBatchFileWriter(f, scdata.schema)
                writer.write(scdata)
                writer.close()
            self.memory_mapped_dataset = pa.memory_map(self.arrow_file, 'r')
            return self.to_memory()
        else:
            return SCData(scdata, self.obs, self.var, self.uns, self.obsm, self.varm)

    @staticmethod
    def transform(functions, table):
        if isinstance(functions, Iterable):
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    for function in functions:
                        table[i, j] = function(table[i, j])
        else:
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    table[i, j] = functions(table[i, j])
        return table

    def subsample(
            self,
            fraction: Optional[float] = None,
            n_obs: Optional[int] = None,
            random_state=0,
            copy: bool = False,
    ) -> "SCData":
        pass

    def highly_variable_genes(
            self,
            n_top_genes: Optional[int] = None,
            min_disp: Optional[float] = 0.5,
            max_disp: Optional[float] = np.inf,
            min_mean: Optional[float] = 0.0125,
            max_mean: Optional[float] = 3,
            span: Optional[float] = 0.3,
            n_bins: int = 20,
            flavor: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',
            subset: bool = False,
            inplace: bool = True,
            batch_key: Optional[str] = None,
            check_values: bool = True,
    ) -> Optional[pd.DataFrame]:
        pass

    def normalize_total(
            self,
            target_sum: Optional[float] = None,
            exclude_highly_expressed: bool = False,
            max_fraction: float = 0.05,
            key_added: Optional[str] = None,
            inplace: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        pass

    def normalize_per_cell(
            self,
            counts_per_cell_after: Optional[float] = None,
            counts_per_cell: Optional[np.ndarray] = None,
            key_n_counts: str = 'n_counts',
            copy: bool = False,
            min_counts: int = 1,
    ) -> Optional["SCData"]:
        pass

    def regress_out(
            self,
            keys: Union[str, Sequence[str]],
            n_jobs: Optional[int] = None,
            copy: bool = False,
    ) -> Optional["SCData"]:
        pass

    def scale(
            self,
            zero_center: bool = True,
            max_value: Optional[float] = None,
            copy: bool = False,
            obsm: Optional[str] = None,
    ) -> Optional["SCData"]:
        pass

    def pca(
            self,
            n_comps: Optional[int] = None,
            zero_center: Optional[bool] = True,
            svd_solver: str = 'arpack',
            random_state: int = 0,
            return_info: bool = False,
            use_highly_variable: Optional[bool] = None,
            dtype: str = 'float32',
            copy: bool = False,
    ) -> Union["SCData", np.ndarray]:
        pass

    @staticmethod
    def set_jax_platform(platform=None):
        """
        Changes platform to CPU, GPU, or TPU. This utility only takes
        effect at the beginning of your program.

        :param str platform: either 'cpu', 'gpu', or 'tpu'.
        """
        if platform is None:
            platform = os.getenv('JAX_PLATFORM_NAME', 'cpu')
        jax.config.update('jax_platform_name', platform)
