#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "ERROR: No file provided\n");
        return 1;
    }

    int dim_x, dim_y, channels; // out params
    unsigned char* data = stbi_load(argv[1], &dim_x, &dim_y, &channels, 1);
    const int data_len = dim_x * dim_y;

    printf("X: %d, Y: %d, chann: %d\n", dim_x, dim_y, channels);
}
