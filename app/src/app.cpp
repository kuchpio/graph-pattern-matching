﻿#include "app.h"
#include "frame.h"
#include "graphPatternMatchingCLI.h"
#include "generateSamples.h"

bool App::OnInit() {
    srand(100);
#ifdef __WXMSW__
    MSWEnableDarkMode();
#endif
    auto frame = new Frame("Graph pattern matching");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP_NO_MAIN(App);

int main(int argc, char* argv[]) {

    // utils::generateSamples("/home/borys/studia/Beng/graph-pattern-matching/samples/benchmarks", 10);
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
