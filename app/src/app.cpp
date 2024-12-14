#include "app.h"
#include "frame.h"

bool App::OnInit() {
    srand(100);
    MSWEnableDarkMode();
    auto frame = new Frame("Graph pattern matching");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP_NO_MAIN(App);

int main(int argc, char* argv[]) {
    return wxEntry(argc, argv);
}
