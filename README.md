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

## Building and installation

After downloading the source code make sure that all required dependencies are present.
Those are provided as `git submodules` and can be fetched by calling
```
git submodule update --init --recursive
```
in the project's root directory.

Once all dependencies are present, the program can be built using standard `CMake` commands.

### CUDA (optional)
The program by default tries to compile all implemented algorithms including CUDA algorithms. That means the default *build* process requires CUDA library (see the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and download cuda from [here](https://developer.nvidia.com/cuda-downloads)) and [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus) to *run* the CUDA algorithms. 

Note that if you don't have CUDA capable GPU you can still download and install the CUDA library in order to run default compilation process. However to reduce the binary footprint and make the installation easier for users that may don't want to install the CUDA library, the `WITH_CUDA` flag has been provided (see the [Building](#building) section).

### Virtual Environment Setup

The program uses python scripts to convert images to graphs.
Therefore an additional setup is needed before running the program.
After building the application, navigate to `image/edge_detection/` directory in the build tree and complete following steps.

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
The requirements file is located in `./requirements.txt`:
```bash
pip install -r ./requirements.txt
```
---
After completing all the steps, the main application located in directory `app/` is ready to be used.

## Usage

The program by default works as a graphical user interface application, 
but when command line arguments are provided on launch the application will start without the graphical user interface.

### Command line interface

#### Options

##### Required Options
- `pattern`
  - **Description**: Kind of pattern to be searched for.
  - **Validation**: Must be one of the predefined patterns in the application.

##### File Options
- `searchSpaceFilePath`
  - **Description**: Path to the search space graph file.
  - **Validation**: Must be an existing file.
- `patternFilePath`
  - **Description**: Path to the pattern graph file.
  - **Validation**: Must be an existing file.

#### Flags
- `-i`, `--induced`
  - **Description**: Changes the pattern we are looking for to be induced.

### Graphical user interface

The application consists of two panels corresponding to search space graph (left) and pattern graph (right).
Each panel provides a menu at the top with three tabs `File`, `Edit`, and `View` 
as well as a canvas at the borrom that enables drawing of those graphs.
Below the panels the user can configure and begin the matching process.

#### File

The `File` tab allows the users to open files in *Graph6* format and images. 
To convert provided image to a graph the user should pass the required number of generated vertices and press `Load` button.
Finally, the `Save` button can be used to save a graph that the user was working on.

#### Edit

The `Edit` tab allows the users to change the graph's structure. 
Availible operations:
 * `Delete` (deletes all selected vertices),
 * `Connect` (connects all pairs of selected vertices),
 * `Disconnect` (connects all pairs of selected vertices),
 * `Contract` (contracts all pairs of selected vertices that are connected with edges),
 * `Subdivide` (subdivides all edges that have both ends selected).

To add a vertex the user should double click on an empty space of the graph canvas.
Two vertices can be connected by dragging the mouse from the first one to the second one while pressing the right mouse button.
To select or deselect a vertex the user should click on a vertex.
To select all vertices in a rectangular area the user can select such area while holding the left mouse button.
To extend current selection the user should hold Ctrl button.

#### View

The `View` tab allows the users to change the positioning of graph's vertices.
Selecting the `Automatic vertex positioning` checkbox will begin a process of automatic vertex positioning.
To prevent selected vertices from moving during the process the user can anchor them using the `Anchor` button.
To free anchored vertices the `Free` button should be used.
After a matching between two graphs is found the `Align` button can be used to move vertices to positions of their counterparts in the other graph.
Vertices can also be manually moved by dragging them.

#### Match

On the bottom of the window the user can select the type of matching and then start the process using the `Match` button.
During the matching process the `Match` button will change it's label to `Stop` and pressing it will cancel the process.
The `Settings` button opens a configuration window.

#### Configuration

Configuration window allows the user to fine-tune functionality of the application as well as extend it using a shared library.
The library should export a function
```cpp
core::IPatternMatcher* GetPatternMatcher();
```
where `core::IPatternMatcher` is defined in the `core` module. 
After succesfully loading such library a new button next to the `Match` button will appear.
