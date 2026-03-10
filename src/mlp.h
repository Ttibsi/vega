#ifndef MLP_H
#define MLP_H

#include <stddef.h>

static float sigmoidf(float);
static float sigmoidfDerivative(float);

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

typedef struct _Value {
    float x;
    float grad;
    struct _Value* prev[2];
    char op;
} Value;

static Value newValue(float, Value*[static 2], char);
static Value plusValue(Value*, Value*);
static Value mulValue(Value*, Value*);
// switch on op, backprop as expected
// could use funcptrs here like in micrograd?
// This should be recursive? 
static void backprop(Value*); 
static Value activateValue(Value*, float(*)(float));

typedef struct {
    Value* weights;
    size_t weight_count;
    Value bias;
} Neuron;

// note: inCount is the number of connections into this neuron
// wCount == inCount
static Neuron newNeuron(size_t wCount);
static Value activateNeuron(Neuron*, float* inputs[], int inCount);

// functionally an array of neurons, but doesn't need to be a dynamic array
typedef struct {
    Neuron* neurons;
    size_t count;
} Layer;

static Layer newLayer(size_t inCount, size_t outCount);
// returns an array of values of length inCount
static Value* activateLayer(Layer* l, float* inputs[], int inCount);

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

#endif // MLP_IMPLEMENTATION
#endif // MLP_H
