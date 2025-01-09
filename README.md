# Graph pattern matching

The project is a pattern matching tool in the domain of graphs.
Graph *A* is considered a pattern in graph *B*, when *A* 
is a (optionally induced) subgraph, minor, or topological minor of *B*.

Besides being able to find certain patterns in graphs, 
the application provides a graphical user interface 
that can be used to manipulate them.
As inputs, the program accepts graphs stored in `Graph6` format as well as images, 
which are then processed to produce graph drawings corresponding to shapes visible in them.
That way we may investigate how pattern matching in graphs relates to pattern matching in images.


## CLI 


### Options

#### Required Options
- `pattern`
  - **Description**: Kind of pattern to be searched for.
  - **Validation**: Must be one of the predefined patterns in the application.

#### File Options
- `searchSpaceFilePath`
  - **Description**: Path to the search space graph file.
  - **Validation**: Must be an existing file.
- `patternFilePath`
  - **Description**: Path to the pattern graph file.
  - **Validation**: Must be an existing file.

### Flags
- `-i`, `--induced`
  - **Description**: Changes the pattern we are looking for to be induced.



#### CUDA (optional)
The program by default tries to compile all implemented algorithms including CUDA algorithms. That means the default *build* process requires CUDA library (see the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and download cuda from [here](https://developer.nvidia.com/cuda-downloads)) and [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus) to *run* the CUDA algorithms. 

Note that if you don't have CUDA capable GPU you can still download and install the CUDA library in order to run default compilation process. However to reduce the binary footprint and make the installation easier for users that may don't want to install the CUDA library, the `WITH_CUDA` flag has been provided (see the [Building](#building) section).

### Building 

After downloading the source code make sure that all required dependencies are present.
Those are provided as `git submodules` and can be fetched by calling
```
git submodule update --init --recursive
```
in the project's root directory.

Once all dependencies are present, the program can be built using standard `CMake` commands.

Avaiable cmake flags:
- `WITH_CUDA`: builds the program with CUDA algorithms. When this option is disabled, the CUDA library is no longer required

#### 
