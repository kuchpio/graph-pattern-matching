# Graph Pattern Matching
The project is a pattern matching tool in the domain of graphs.Graph *A* is considered a pattern in graph *B*, when *A* is a (optionally induced) subgraph, minor, or topological minor of *B*.Besides being able to find certain patterns in graphs, the application provides a graphical user interface that can be used to manipulate them.As inputs, the program accepts graphs stored in `Graph6` format as well as images, which are then processed to produce graph drawings corresponding to shapes visible in them.This enables us to investigate how pattern matching in graphs relates to pattern matching in images.
## Installation
### Virtual Environment Setup
**Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```
**Activate the virtual environment**:

On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

     
On Windows:
     ```bash
     .\\venv\\Scripts\\activate
     ```
     
**Install the required libraries**:
   The requirements file is located in `./image/edge_detection/requirements.txt`:
   ```bash
   pip install -r ./image/requirements.txt
   ```
---
## CLI
### Options
#### Required Options
**`pattern`** 
**Description**: Kind of pattern to be searched for. 
**Validation**: Must be one of the predefined patterns in the application.
#### File Options
**`searchSpaceFilePath`** 
**Description**: Path to the search space graph file. 
**Validation**: Must be an existing file.
**`patternFilePath`** 
**Description**: Path to the pattern graph file. 
**Validation**: Must be an existing file.
### Flags
**`-i`, `--induced`** 
**Description**: Changes the pattern we are looking for to be induced.

### CUDA (Optional)
The program by default tries to compile all implemented algorithms, including CUDA algorithms. 
This means the default *build* process requires the CUDA library (see the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)) and a [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) to *run* the CUDA algorithms.
If you do not have a CUDA-capable GPU, you can still download and install the CUDA library to complete the default compilation process. 
However, to reduce the binary footprint and simplify installation for users who do not wish to install the CUDA library, the `WITH_CUDA` flag has been provided (see the [Building](#building) section).

### Building
After downloading the source code, ensure that all required dependencies are present. 
These are provided as `git submodules` and can be fetched by calling:
```bash
git submodule update --init --recursive
```
in the project's root directory.
Once all dependencies are present, the program can be built using standard `CMake` commands.
#### Available CMake Flags:
**`WITH_CUDA`**: Builds the program with CUDA algorithms. 
  When this option is disabled, the CUDA library is no longer required.
