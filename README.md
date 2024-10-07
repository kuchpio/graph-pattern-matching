# Graph pattern matching

Desktop application revolving around finding patterns in graphs.

### Definition

We will say that graph `A` is a pattern in graph `B`, when `A` is a subgraph, induced subgraph, minor, topological minor, or induced minor of `B`.

### Algorithms
Algorithms that can determine the relations mentioned above between two graphs are usually computationally expensive.
Therefore we decided to parallelize these processes on GPU and hopefully achieve some performance gains. 

### Graph vs image pattern matching
Besides graphs, the application can also accept images as inputs. 
Then machine learning algorithms produce graph drawings corresponding to shapes visible in given pictures.
That way we may investigate how pattern matching in graphs relates to pattern matching in images.

### Interface
The graphs are visualized in real time using force-based simulations that provide a clear overview of their structure.
At the same time, a user-friendly interface simplifies making small modifications to these graphs to see how little adjustments influence their relationship.

### Proposed division of work

| Author           | Area                        |
|------------------|-----------------------------|
| Piotr Kucharczyk | Graph visualisation         |
| Borys Kurdek     | Pattern matching algorithms |
| Bartosz Maj      | Image to graph conversion   |
