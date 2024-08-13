#include "layer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
//======================= CONSTRUCTORS ==========================================

Layer::Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions,
             pair<size_t, size_t> biasesDimensions, ActivationFunctionE f, bool randomize,
             LayerTypeE type)
{
    this->output = FastMatrix(GET_ROWS_FROM_PAIR(outputDimensions), GET_COLS_FROM_PAIR(outputDimensions));
    this->weights = FastMatrix(GET_ROWS_FROM_PAIR(weightsDimensions), GET_COLS_FROM_PAIR(weightsDimensions));
    this->biases = FastMatrix(GET_ROWS_FROM_PAIR(biasesDimensions), GET_COLS_FROM_PAIR(biasesDimensions));

    if (randomize)
    {
        this->weights.randomize(-1.0, 1.0);
        this->biases.randomize(-1.0, 1.0);
    }

    this->functionType = f;
    this->layerType = type;
}

// Default constructor
// When any object that uses layers is initialized
// The layers are empty.
Layer::Layer()
{
    this->layerType = EMPTY_LAYER;
}

// Constructor for input layer
Layer::Layer(pair<size_t, size_t> inputDimensions)
{
    this->output = FastMatrix(GET_ROWS_FROM_PAIR(inputDimensions), GET_COLS_FROM_PAIR(inputDimensions));
    this->layerType = INPUT_LAYER;
}

//======================= UTILITIES ============================================

void Layer::xavierInitialization(size_t prevLayerSize)
{
    biases.set(0.0);
    weights.randomize(-(sqrt(6) / sqrt(prevLayerSize + weights.cols)), (sqrt(6) / sqrt(prevLayerSize + weights.cols)));
}

//======================= ACTIVATION FUNCTIONS ============================================

inline double sigmoidf(double x)
{
    return (double)1.0 / ((double)1.0 + std::exp(-x));
}

void Layer::activate()
{
    switch (functionType)
    {
    case SIGMOID:
    {
        for (size_t i = 0; i < output.rows; ++i)
        {
            for (size_t j = 0; j < output.cols; ++j)
            {
                MAT_ACCESS(output, i, j) = sigmoidf(MAT_ACCESS(output, i, j));
            }
        }
        break;
    }
    case RELU:
    {
        for (size_t i = 0; i < output.rows; ++i)
        {
            for (size_t j = 0; j < output.cols; ++j)
            {
                MAT_ACCESS(output, i, j) = std::max(0.0, MAT_ACCESS(output, i, j));
            }
        }
        break;
    }
    case SOFTMAX:
    {
        double maxValue = *max_element(std::begin(output.mat), std::end(output.mat));
        double sum = 0.0;
        for (size_t i = 0; i < output.cols; ++i)
        {
            sum += exp(MAT_ACCESS(output, 0, i) - maxValue);
        }

        // double constant = maxValue + log(sum);
        for (size_t i = 0; i < output.cols; ++i)
        {
            MAT_ACCESS(output, 0, i) = exp(MAT_ACCESS(output, 0, i) - maxValue) / sum;
        }
    }
    case NO_ACTIVATION:
    {
        break;
    }
    }
}

FastMatrix Layer::forward(FastMatrix input)
{
    output = (input * (weights)) + (biases);
    activate();
    return output;
}
