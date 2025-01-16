#include "app.h"
#include "frame.h"
#include "graphPatternMatchingCLI.h"
#include "generateSamples.h"
#include <ctime>

bool App::OnInit() {
    srand(time(NULL));
#ifdef __WXMSW__
    MSWEnableDarkMode();
#endif
    auto frame = new Frame("Graph pattern matching");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP_NO_MAIN(App);

int main(int argc, char* argv[]) {

    GraphPatternMatchingCLI cli;
    if (argc == 1) return wxEntry(argc, argv);

    try {
        cli.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return cli.exit(error);
    }
    cli.run();
    return 0;
}
