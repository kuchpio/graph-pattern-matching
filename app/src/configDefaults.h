#pragma once

#include "wx/wx.h"

class ConfigDefaults {
  public:
    const wxString TRIANGULATE_ID = "Triangulate";
    const wxString CONTRACTION_TIME_ID = "ContractionTime";
    const wxString ALIGNMENT_TIME_ID = "AlignmentTime";
    const wxString SPRING_STRENGTH_ID = "SpringStrength";
    const wxString SPRING_LENGTH_ID = "SpringLength";
    const wxString NODE_REPULSION_ID = "NodeRepulsion";
    const wxString NODE_DRAG_ID = "NodeDrag";
    const std::string SELECTED_SUBGRAPH_ALGORITHM_ID = "Subgraph";
    const std::string SELECTED_INDUCED_SUBGRAPH_ALGORITHM_ID = "InducedSubgraph";
    const std::string SELECTED_MINOR_ALGORITHM_ID = "Minor";
    const std::string SELECTED_INDUCED_MINOR_ALGORITHM_ID = "InducedMinor";
    const std::string SELECTED_TOPOLOGICAL_MINOR_ALGORITHM_ID = "TopologicalMinor";
    const std::string SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM_ID = "InducedTopologicalMinor";
    const wxString ENABLE_EXTERNAL_ALGORITHM_ID = "EnableExternalAlgorithm";
    const wxString EXTERNAL_ALGORITHM_NAME_ID = "ExternalAlgorithmName";
    const wxString EXTERNAL_ALGORITHM_PATH_ID = "ExternalAlgorithmPath";

    const std::unordered_map<std::string, std::vector<wxString>> CHOICES = {
        {SELECTED_SUBGRAPH_ALGORITHM_ID,
         {
#ifdef CUDA_ENABLED
             "Exact (GPU)",
#endif // CUDA_ENABLED
             "Heuristic", "Exact (CPU)"}},
        {SELECTED_INDUCED_SUBGRAPH_ALGORITHM_ID, {"Heuristic", "Exact"}},
        {SELECTED_MINOR_ALGORITHM_ID, {"Heuristic", "Exact"}},
        {SELECTED_INDUCED_MINOR_ALGORITHM_ID, {"Heuristic", "Exact"}},
        {SELECTED_TOPOLOGICAL_MINOR_ALGORITHM_ID, {"Heuristic", "Exact"}},
        {SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM_ID, {"Heuristic", "Exact"}}};

    const bool TRIANGULATE = false;
    const float CONTRACTION_TIME = 1.0f;
    const float ALIGNMENT_TIME = 2.0f;
    const float SPRING_STRENGTH = 5.0f;
    const float SPRING_LENGTH = 0.5f;
    const float NODE_REPULSION = 10.0f;
    const float NODE_DRAG = 4.0f;
    const int SELECTED_SUBGRAPH_ALGORITHM = 0;
    const int SELECTED_INDUCED_SUBGRAPH_ALGORITHM = 0;
    const int SELECTED_MINOR_ALGORITHM = 0;
    const int SELECTED_INDUCED_MINOR_ALGORITHM = 0;
    const int SELECTED_TOPOLOGICAL_MINOR_ALGORITHM = 0;
    const int SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM = 0;
    const bool ENABLE_EXTERNAL_ALGORITHM = false;
    const std::string EXTERNAL_ALGORITHM_NAME = "";
    const std::string EXTERNAL_ALGORITHM_PATH = "";
};
