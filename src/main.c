#include <stdio.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "matrix.h"
#include "network.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "ERROR: No file provided\n");
        return 1;
    }

    srand(time(NULL));

    // Load image
    int dim_x, dim_y, channels; // out params
    unsigned char* data = stbi_load(argv[1], &dim_x, &dim_y, &channels, 1);
    const int data_len = dim_x * dim_y;

    // Draw image to terminal
    printf("X: %d, Y: %d, chann: %d\n", dim_x, dim_y, channels);
    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {
            if (data[y*dim_x + x]) {
                printf("%3u ", data[y*dim_x + x]);
            } else {
                printf("    ");
            }
        }
        printf("\n");
    }

    // Turn image data into a 1-col matrix
    Matrix a0 = matrixAlloc(dim_x*dim_y, 1);

    // Create a neural network with random weights and biases to start
    size_t arch[] = {dim_x*dim_y, 16, 16, 10};
    Network nn = networkAlloc(&a0, arch, sizeof(arch)/sizeof(arch[0]));
    networkRandomize(nn, 0);

    networkPrint(&nn);

    // learn
    const int epoch = 100;

    for (int i = 0; i < epoch; i++) {
        networkFeedForward(nn);
        networkBackPropagate(nn);
    }
}
