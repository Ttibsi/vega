#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "ERROR: No file provided\n");
        return 1;
    }

    // Load image
    int dim_x, dim_y, channels; // out params
    unsigned char* data = stbi_load(argv[1], &dim_x, &dim_y, &channels, 1);
    const int data_len = dim_x * dim_y;

    // Draw image to terminal
    printf("X: %d, Y: %d, chann: %d\n", dim_x, dim_y, channels);
    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {
            printf("%3u ", data[y*dim_x + x]);
        }
        printf("\n");
    }
}
