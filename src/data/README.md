SCData: In Memory Columnar Single Cell Data Structure
=====================================================

SCData is a runtime column-oriented memory format based on the Apache Arrow framework, which is optimized for analytical
processing performance for Single Cell data on modern hardware like CPUs and GPUs. It supports zero-copy,
streaming/chunk oriented data layer designed for moving and accessing large data sets at maximum speeds without
serialization overhead.

SCData's internals which are based on Arrow, which is written in C++ and provides the essential in-memory analytics
infrastructure for improving the basic limitations of the Pandas framework on which AnnData is built on:

1) **Internals too far from the metal**

   All memory in SCData is on a per column basis, arranged in contiguous memory buffers optimized for random access and
   scan performance. Reduces CPU or GPU cache misses when looping over data in a table column, even with strings or
   non-numeric types. Data is stored and operated on in two formats, namely Apache Arrow and Apache Parquet. Parquet is
   a compressed storage format meant for transferring data and long term storage. While Arrow is the in-memory format
   meant for efficient computations.

2) **No support for memory mapped datasets**

   Perhaps the single biggest memory management problem with pandas is the requirement that data must be loaded
   completely into RAM to be processed. Using Arrow's serialization it is possible to memory map huge, bigger than RAM
   data sets and evaluate pandas-style algorithms on them in-place without loading them into memory.

3) **Poor performance in database and file ingest/export**

   Efficient memory layout and rich metadata make it an ideal container for inbound data from databases and columnar
   storage formats like Apache Parquet.

4) **Warty missing data support**

   All missing data is represented as a packed bit array, separate from the rest of the data. This makes missing data
   handling simple and consistent across all data types. This also allows analytics on the null bits (AND-ing bitmaps,
   or counting set bits) using fast bit-wise built-in hardware operators and SIMD.

5) **Lack of transparency into memory use, RAM management**

   In Pandas, all memory is owned either by NumPy or the Python interpreter, and it can be difficult to measure exactly
   how much memory is sed by a given pandas DataFrame. It is not unusual for a line of code to double or triple the
   memory footprint of a process due to temporary allocations, sometimes causing a MemoryError. In SCData memory is
   either immutable or copy-on-write. At any given time, you know if another array references a buffer that you can see.
   This enables us to avoid defensive copying.

6) **Weak support for categorical data**

   Unlike Pandas, categorical data is a first-class citizen in SCData, with efficient and consistent representation both
   in-memory and on the wire or in shared memory. It is possible to even share categories between multiple arrays.

7) **Complex groupby operations awkward and slow**

   Easier parallelization of groupby operations. However, it is difficult or even impossible in certain scenarios to
   fully parallelize a df.groupby(...).apply(f)

8) **Appending data to a DataFrame is tedious and very costly**

   In Pandas, all the data in a column from a DataFrame must reside in the same NumPy array. This is a restrictive
   requirement, and frequently results in memory-doubling and additional computation to concatenate Series and DataFrame
   objects. Table Columns in SCData can be chunked, so that appending to a table is a zero-copy operation, requiring no
   non-trivial computation or memory allocation. By designing up front for streaming, chunked tables, appending to
   existing in-memory tables is computationally inexpensive relative to pandas. This also allows for implementing
   algorithms for processing larger than memory datasets.

9) **Limited, non-extensible metadata**

   The metadata representation is decoupled from the details of computations. Thereby allowing for user-defined types
   without having to construct support for new metadata, dynamic dispatch rules to operator implementations in analytics
   and metadata preservation through operations.

10) **Eager evaluation model, no query planning. "Slow", limited multi-core algorithms for large data set**

