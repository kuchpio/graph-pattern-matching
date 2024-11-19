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

### Building 

After downloading the source code make sure that all required dependencies are present.
Those are provided as `git submodules` and can be fetched by calling
```
git submodule update --init --recursive
```
in the project's root directory.

Once all dependencies are present, the program can be built using standard `CMake` commands.
