#pragma once

#include "wx/wx.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
  public:
    GraphPanel(wxWindow* parent);

  private:
    GraphCanvas* canvas{nullptr};
};
