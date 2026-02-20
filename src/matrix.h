#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} Matrix;

#define MATRIX_AT(mat, i, j) (mat).data[(i)*(mat).cols + (j)]

Matrix matrixAlloc(size_t rows, size_t cols);
float matrixAt(Matrix m, size_t row, size_t col);

#endif // MATRIX_H
