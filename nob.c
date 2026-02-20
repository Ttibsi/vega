#define NOB_IMPLEMENTATION
#include "include/nob.h"

const char* exec_name = "vega";

int main(int argc, char* argv[]) {
    GO_REBUILD_URSELF(argc, argv);

    Cmd cmd = {0};
    cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-std=c23", "-g");
    cmd_append(&cmd, "-Iinclude", "-Linclude", "-lm");
    cmd_append(&cmd, "-o", exec_name);
    cmd_append(&cmd, "src/main.c", "src/network.c", "src/matrix.c");

    if (!cmd_run(&cmd)) { return 1; }

    return 0;
}
