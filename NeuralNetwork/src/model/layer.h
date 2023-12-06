#ifndef LAYER_H
#define LAYER_H

#include "../fastMatrix/FastMatrix.h"
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
    NO_ACTIVATION
};

class Layer{

    public:

        FastMatrix weights;
        FastMatrix biases;
        FastMatrix output;
        ActivationFunctionE functionType;

        Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions,
              pair<size_t, size_t> biasesDimensions, ActivationFunctionE f, bool randomize);
        Layer();

        void xavierInitialization(size_t prevLayerSize);

        FastMatrix forward(FastMatrix input);
        void activate();

};

#endif