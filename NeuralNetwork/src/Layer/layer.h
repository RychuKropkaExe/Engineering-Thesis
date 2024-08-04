#ifndef LAYER_H
#define LAYER_H

#include "FastMatrix.h"
#include <utility>
#include <string>

using std::pair;

#define GET_ROWS_FROM_PAIR(PAIR) (PAIR).first
#define GET_COLS_FROM_PAIR(PAIR) (PAIR).second
#define SET_ROWS_IN_PAIR(PAIR, VAL) (PAIR).first = (VAL)
#define SET_COLS_IN_PAIR(PAIR, VAL)  (PAIR).second = (VAL)

enum ActivationFunctionE{
    SIGMOID,
    RELU,
    SOFTMAX,
    NO_ACTIVATION
};

enum LayerTypeE{
    INPUT_LAYER,
    INTERMEDIATE_LAYER,
    OUTPUT_LAYER,
    EMPTY_LAYER // <-- Used for default constructor
};

class Layer{

    public:

        FastMatrix weights;
        FastMatrix biases;
        FastMatrix output;

        LayerTypeE layerType;

        ActivationFunctionE functionType;

        Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions,
              pair<size_t, size_t> biasesDimensions, ActivationFunctionE f, bool randomize,
              LayerTypeE type);
        Layer(pair<size_t, size_t> inputDimensions);
        Layer();

        void xavierInitialization(size_t prevLayerSize);

        FastMatrix forward(FastMatrix input);
        void activate();

};

#endif