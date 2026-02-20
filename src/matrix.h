#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} Matrix;

inline Matrix matrixAlloc(size_t rows, size_t cols) {
}

inline float matrixAt(Matrix m, size_t row, size_t col) {
    assert(m.rows >= row);
    assert(m.cols >= col);
}

inline void matrixRandomize(Matrix m) {
}

#endif MATRIX_H
