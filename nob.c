#define NOB_IMPLEMENTATION
#include "include/nob.h"

int main(int argc, char* argv[]) {
    GO_REBUILD_URSELF(argc, argv);

    Cmd cmd = {0};
    cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-Iinclude", "-Linclude", "-lm");
    cmd_append(&cmd, "-o", "main");
    cmd_append(&cmd, "src/main.c");

    if (!cmd_run(&cmd)) { return 1; }

    return 0;
}
