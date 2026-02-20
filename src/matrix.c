#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

Matrix matrixAlloc(size_t rows, size_t cols) {
    assert(rows > 0);
    assert(cols > 0);

    return (Matrix){
        .data = calloc(rows*cols, sizeof(float)),
        .rows = rows,
        .cols = cols
    };
}

float matrixAt(Matrix m, size_t row, size_t col) {
    assert(m.rows >= row);
    assert(m.cols >= col);
    return MATRIX_AT(m, row, col);
}
