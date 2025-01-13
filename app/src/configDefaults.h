#pragma once

#include "wx/wx.h"

class ConfigDefaults {
  public:
    const wxString CONTRACTION_TIME_ID = "ContractionTime";
    const wxString ALIGNMENT_TIME_ID = "AlignmentTime";
    const wxString SPRING_STRENGTH_ID = "SpringStrength";
    const wxString SPRING_LENGTH_ID = "SpringLength";
    const wxString NODE_REPULSION_ID = "NodeRepulsion";
    const wxString NODE_DRAG_ID = "NodeDrag";
    const wxString ENABLE_EXTERNAL_ALGORITHM_ID = "EnableExternalAlgorithm";
    const wxString EXTERNAL_ALGORITHM_NAME_ID = "ExternalAlgorithmName";
    const wxString EXTERNAL_ALGORITHM_PATH_ID = "ExternalAlgorithmPath";

    const float CONTRACTION_TIME = 1.0f;
    const float ALIGNMENT_TIME = 2.0f;
    const float SPRING_STRENGTH = 5.0f;
    const float SPRING_LENGTH = 0.5f;
    const float NODE_REPULSION = 10.0f;
    const float NODE_DRAG = 4.0f;
    const bool ENABLE_EXTERNAL_ALGORITHM = false;
    const std::string EXTERNAL_ALGORITHM_NAME = "";
    const std::string EXTERNAL_ALGORITHM_PATH = "";
};