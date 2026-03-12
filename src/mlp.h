#ifndef MLP_H
#define MLP_H

#include <stddef.h>

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
    size_t weight_count;
    Value bias;
} Neuron;

// note: inCount is the number of connections into this neuron
// wCount == inCount
static Neuron newNeuron(size_t wCount, Value* inputs[], Arena* a);
static Value activateNeuron(Neuron*, float inputs[], size_t inCount);

// functionally an array of neurons, but doesn't need to be a dynamic array
typedef struct {
    Neuron* neurons;
    size_t count;
    size_t out_conns;
} Layer;

static Layer newLayer(Layer* prevLayer, size_t outputs, Arena* a);
// returns an array of values of length inCount
static Value* activateLayer(Layer* l, float inputs[], int inCount, Arena* a);

typedef struct {
    Layer* layers;
    size_t layerCount;
} MLP;

static MLP newPerceptron(size_t* arch, size_t arch_sz);
static Value* activatePerceptron(MLP* mlp, float* inputs, float inCount);
// note: will need to set all grads to 0 on each iteration
static void gradientDescent(MLP* mlp, size_t learnRate);
// for iterations, step forward, calculate loss, backprop, update backwar
static void trainPerceptron(MLP* mlp, size_t iterations, size_t learnRate);

#ifdef MLP_IMPLEMENTATION

#include <assert.h>
#include <math.h>
#include <stdlib.h>

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

    while (n.weight_count < wCount) {
        Value v = newValue(randWeight(), inputs, OP_NONE);
        n.weights[n.weight_count] = v;
        n.weight_count++;
    }

    n.bias = newValue(0.0f, NULLPREV, OP_NONE);
    return n;
}

static Value activateNeuron(Neuron* n, float inputs[], size_t inCount) {
    assert(inCount >= n->weight_count);

    float val = n->bias.x;
    for (size_t i = 0; i < n->weight_count; i++) {
        val += (n->weights[i].x * inputs[i]);
    }

    // TODO: Should this have no previous?
    // TODO: we might also want to genericise this if we use different activation functions
    return newValue(sigmoidf(val), NULLPREV, OP_ACT);
}

static Layer newLayer(Layer* prevLayer, size_t outputs, Arena* a) {
    Layer l = {0};
    l.neurons = arenaAlloc(a, sizeof(Neuron) * outputs);

    while (l.count < outputs) {
        Value* vs;
        if (prevLayer) {
            vs = arenaAlloc(a, sizeof(Value)*prevLayer->count);
            for (size_t i = 0; i < prevLayer->count; i++) {
                vs[i] = prevLayer->neurons[i].weights[i];
            }
        } else {
            vs = *NULLPREV;
        }

        Neuron n = newNeuron(outputs, &vs, a);
        l.neurons[l.count] = n;
        l.count++;
    }

    return l;
}

static Value* activateLayer(Layer* l, float inputs[], int inCount, Arena* a) {
    Value* vs = arenaAlloc(a, sizeof(Value) * inCount);
    
    for (size_t i = 0; i < l->count; i++) {
        vs[i] = activateNeuron(&l->neurons[i], inputs, inCount);
    }

    return vs;
}

#endif // MLP_IMPLEMENTATION
#endif // MLP_H
