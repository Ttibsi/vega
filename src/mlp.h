// vim: set ft=c
// -*- mode: c; -*-
#ifndef MLP_H
#define MLP_H

#include <stddef.h>

// TODO: Defined by user? Not sure where this should be placed
constexpr float margin = 0.05f;

static float sigmoidf(float);
static float sigmoidfDerivative(float);
static float randWeight(void);

typedef struct {
    // pointer arithmetic on void* isn't standard C
    unsigned char* memory;
    size_t capacity;
    size_t size;
} Arena;

static int   arenaInit(Arena* a, size_t capacity);
static void* arenaAlloc(Arena* a, size_t allocSize);
static void  arenaReset(Arena* a);
static void  arenaDestroy(Arena* a);

typedef enum {
    OP_NONE,
    OP_PLUS,
    OP_MUL,
    OP_ACT
} Op;

typedef struct _Value {
    float x;
    float grad;
    struct _Value* prev[2];
    Op op;
} Value;

static Value* NULLPREV[2] = { NULL, NULL };

static Value newValue(float x, Value* prev[static 2], Op op);
static Value plusValue(Value* lhs, Value* rhs);
static Value mulValue(Value* lhs, Value* rhs);
static Value activateValue(Value* val, float(*activation)(float));
static void backprop(Value* this);

typedef struct {
    Value* weights;
    size_t weightCount;
    Value bias;
} Neuron;

// note: inCount is the number of connections into this neuron
// wCount == inCount
static Neuron newNeuron(size_t wCount, Value* inputs[], Arena* a);
static Value activateNeuron(Neuron*, Value inputs[], size_t inCount);

// functionally an array of neurons, but doesn't need to be a dynamic array
typedef struct {
    Neuron* neurons;
    size_t neuronCount;
    size_t out_conns;
} Layer;


static Layer newLayer(Layer* prevLayer, size_t outputs, Arena* a);
static Value* activateLayer(Layer* l, Value* vs, size_t inCount, Arena* a);

typedef struct {
    Layer* layers;
    size_t layerCount;
} MLP;

static MLP newPerceptron(const size_t* arch, const size_t arch_sz, Arena* a);
static Value* activatePerceptron(MLP* mlp, float* inputs, size_t inCount, Arena* a);
static void zeroGrad(MLP* mlp);
static void gradientDescent(MLP* mlp, float learnRate);
static bool trainPerceptron(MLP* mlp, Arena* a, float* inputs, size_t inCount, float* expected, float learnRate);
static void displayPerceptron(MLP* mlp);

#ifdef MLP_IMPLEMENTATION

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float sigmoidf(float x) {
    return 1.f / (1.f + expf(x));
}

static float sigmoidfDerivative(float x) {
    return x * (1 - x);
}

// https://stackoverflow.com/a/13409005
static float randWeight(void) {
    const float min = -1.0f;
    const float max = 1.0f;

    return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

static int arenaInit(Arena* a, size_t capacity) {
    if (!a || capacity == 0) { return 0; }

    a->memory = (unsigned char*)malloc(capacity);
    if (!a->memory) { return 0; }

    a->capacity = capacity;
    a->size = 0;

    return 1;
}

static void* arenaAlloc(Arena* a, size_t allocSize) {
    if (!a || allocSize == 0) { return NULL; }
    if (a->size + allocSize > a->capacity) { return NULL; }

    void* ptr = a->memory + a->size;
    a->size += allocSize;

    return ptr;
}

static void arenaReset(Arena* a) {
    if (!a) { return; }
    a->size = 0;
}

static void arenaDestroy(Arena* a) {
    if (!a) { return; }

    free(a->memory);
    a->memory = NULL;
    a->capacity = 0;
    a->size = 0;
}

static Value newValue(float x, Value* prev[static 2], Op op) {
    return (Value){
        .x = x,
        .grad = 0.0f,
        .prev[0] = prev[0],
        .prev[1] = prev[1],
        .op = op
    };
}

static Value plusValue(Value* lhs, Value* rhs) {
    Value* prev[2] = {lhs, rhs};
    return newValue(lhs->x + rhs->x, prev, OP_PLUS);
}

static Value mulValue(Value* lhs, Value* rhs) {
    Value* prev[2] = {lhs, rhs};
    return newValue(lhs->x * rhs->x, prev, OP_MUL);
}

static Value activateValue(Value* val, float(*activation)(float)) {
    Value* prev[2] = {val};
    return newValue(activation(val->x), prev, OP_ACT);
}

static void backprop(Value* this) {
    // base case - the final element should have a gradient of 1.0
    if (!this->grad) { this->grad = 1.0; }

    if (this->op == OP_PLUS) {
        this->prev[0]->grad += this->grad;
        this->prev[1]->grad += this->grad;

        backprop(this->prev[0]);
        backprop(this->prev[1]);

    } else if (this->op == OP_MUL) {
        this->prev[0]->grad += this->grad * this->prev[1]->grad;
        this->prev[1]->grad += this->grad * this->prev[0]->grad;

        backprop(this->prev[0]);
        backprop(this->prev[1]);

    } else if (this->op == OP_ACT) {
        // TODO: Adjust for other activation functions
        this->prev[0]->grad += sigmoidfDerivative(this->grad);
        backprop(this->prev[0]);
    }
}

static Neuron newNeuron(size_t wCount, Value* inputs[], Arena* a) {
    Neuron n = {0};
    n.weights = arenaAlloc(a, sizeof(Value) * wCount);

    while (n.weightCount < wCount) {
        Value v = newValue(randWeight(), inputs, OP_NONE);
        n.weights[n.weightCount] = v;
        n.weightCount++;
    }

    n.bias = newValue(0.0f, NULLPREV, OP_NONE);
    return n;
}

static Value activateNeuron(Neuron* n, Value inputs[], size_t inCount) {
    assert(inCount >= n->weightCount);

    // TODO: Is this the right place to set ops and prevs?
    float val = n->bias.x;
    for (size_t i = 0; i < n->weightCount; i++) {
        val += (n->weights[i].x * inputs[i].x);
    }

    // TODO: Should this have no previous?
    // TODO: we might also want to genericise this if we use different activation functions
    Value v = newValue(sigmoidf(val), NULLPREV, OP_ACT);
    return v;
}

static Layer newLayer(Layer* prevLayer, size_t outputs, Arena* a) {
    Layer l = {0};
    l.neurons = arenaAlloc(a, sizeof(Neuron) * outputs);

    while (l.neuronCount < outputs) {
        Value* vs;
        if (prevLayer) {
            vs = arenaAlloc(a, sizeof(Value)*prevLayer->neuronCount);
            for (size_t i = 0; i < prevLayer->neuronCount; i++) {
                vs[i] = prevLayer->neurons[i].weights[i];
            }
        } else {
            vs = *NULLPREV;
        }

        Neuron n = newNeuron(outputs, &vs, a);
        l.neurons[l.neuronCount] = n;
        l.neuronCount++;
    }

    return l;
}

static Value* activateLayer(Layer* l, Value* vs, size_t inCount, Arena* a) {
    if (inCount < l->neuronCount) {
        Value* newVs = arenaAlloc(a, sizeof(Value) * l->neuronCount);
        vs = memmove(newVs, vs, inCount);
    }

    for (size_t i = 0; i < l->neuronCount; i++) {
        vs[i] = activateNeuron(&l->neurons[i], vs, inCount);
    }

    return vs;
}

static MLP newPerceptron(const size_t* arch, const size_t arch_sz, Arena* a) {
    MLP mlp = {0};
    mlp.layers = arenaAlloc(a, sizeof(Layer) * arch_sz);
    mlp.layerCount = arch_sz;

    Layer* prev = NULL;
    for (size_t i = 0; i < arch_sz; i++) {
        Layer l = newLayer(prev, arch[i], a);
        mlp.layers[i] = l;
        prev = &mlp.layers[i];
    }

    return mlp;
}

static Value* activatePerceptron(MLP* mlp, float* inputs, size_t inCount, Arena* a) {
    Value* vs = arenaAlloc(a, sizeof(Value) * inCount);

    for (size_t i = 0; i < inCount; i++) {
        vs[i] = newValue(inputs[i], NULLPREV, OP_NONE);
    }

    for (size_t i = 0; i < mlp->layerCount; i++) {
        vs = activateLayer(&mlp->layers[i], vs, inCount, a);
    }

    return vs;
}

static void zeroGrad(MLP* mlp) {
    for (size_t i = mlp->layerCount - 1; i > 0; i--) {
        for (size_t j = 0; j < mlp->layers[i].neuronCount; j++) {
            for (size_t k = 0; k < mlp->layers[i].neurons[j].weightCount; k++) {
                Value* v = &mlp->layers[i].neurons[j].weights[k];
                v->grad = 0.0f;
            }
        }
    }
}

static void gradientDescent(MLP* mlp, float learnRate) {
    for (size_t i = mlp->layerCount - 1; i > 0; i--) {
        for (size_t j = 0; j < mlp->layers[i].neuronCount; j++) {
            for (size_t k = 0; k < mlp->layers[i].neurons[j].weightCount; k++) {
                // do something to each value
                Value* v = &mlp->layers[i].neurons[j].weights[k];
                v->x += -learnRate * v->grad;

            }
        }
    }
}

static bool trainPerceptron(MLP* mlp, Arena* a, float* inputs, size_t inCount, float* expected, float learnRate) {
    // Feed forward
    Value* vs = activatePerceptron(mlp, inputs, inCount, a);

    // Calculate mean-squared error loss
    float loss = 0.0;
    for (size_t i = 0; i < inCount; i++) {
        loss += powf(vs[i].x - expected[i], 2);
    }

    // printf("Loss: %.5f ", loss);
    if (fabs(loss - expected[0]) <= margin) { return false; }

    // backpropagate
    zeroGrad(mlp);
    for (size_t i = mlp->layerCount - 1; i > 0; i--) {
        for (size_t j = 0; j < mlp->layers[i].neuronCount; j++) {
            for (size_t k = 0; k < mlp->layers[i].neurons[j].weightCount; k++) {
                Value* v = &mlp->layers[i].neurons[j].weights[k];
                backprop(v);
            }
        }
    }

    gradientDescent(mlp, learnRate);
    return true;
}

static void displayPerceptron(MLP* mlp) {
    for (size_t i = 0; i < mlp->layerCount; i++) {
        printf("Layer: %zu\n", i);
        for (size_t j = 0; j < mlp->layers[i].neuronCount; j++) {
            printf("\tNeuron: %zu\n", j);
            for (size_t k = 0; k < mlp->layers[i].neurons[j].weightCount; k++) {
                // do something to each value
                Value* v = &mlp->layers[i].neurons[j].weights[k];
                printf("\t\tValue: %.4f Gradient: %.4f, Op: %i\n", v->x, v->grad, v->op);
            }
        }
    }
}

#endif // MLP_IMPLEMENTATION
#endif // MLP_H
