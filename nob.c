#define NOB_IMPLEMENTATION
#include "include/nob.h"

const char* exec_name = "vega";

int buildFile(Cmd* cmd, char* file, char* exec) {
    nob_log(NOB_INFO, "Bulding: %s\n", exec);
    cmd_append(cmd, "gcc", "-std=c23", "-g");
    cmd_append(cmd, "-Wall", "-Wextra");
    cmd_append(cmd, "-lm");
    cmd_append(cmd, file);
    cmd_append(cmd, "-o", exec);
    if (!cmd_run(cmd)) { return 1; }
}


int main(int argc, char* argv[]) {
    GO_REBUILD_URSELF(argc, argv);

    char* files[] = {
        "src/main2.c",
         "src/xor.c",
    };

    char* executables[] = {
        "vega", "xor"
    };

    const int file_count = sizeof(files) / sizeof(files[0]);

    Cmd cmd = {0};

    for (int i = 0; i < file_count; i++) {
        if (buildFile(&cmd, files[i], executables[i])) { return 1; }
    }

    return 0;
}
