#include <iostream>

#include "wx/wx.h"

#include "app.h"
#include "frame.h"

bool App::OnInit() {
    auto frame = new Frame("Graph pattern matching");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP_NO_MAIN(App);

int main() {
    return wxEntry();
}
