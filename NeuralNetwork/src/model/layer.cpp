#include "layer.h"
#include <cmath>
#include <iostream>
//======================= CONSTRUCTORS ==========================================

Layer::Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions,
             pair<size_t, size_t> biasesDimensions, ActivationFunctionE f, bool randomize)
{
    this->output = FastMatrix(GET_ROWS_FROM_PAIR(outputDimensions), GET_COLS_FROM_PAIR(outputDimensions));
    this->weights = FastMatrix(GET_ROWS_FROM_PAIR(weightsDimensions), GET_COLS_FROM_PAIR(weightsDimensions));
    this->biases = FastMatrix(GET_ROWS_FROM_PAIR(biasesDimensions), GET_COLS_FROM_PAIR(biasesDimensions));
    if(randomize){
        this->weights.randomize(-1.0, 1.0);
        this->biases.randomize(-1.0, 1.0);
    }
    this->functionType = f;

}

Layer::Layer(){
    this->functionType = SIGMOID;
}

//======================= UTILITIES ============================================

void Layer::xavierInitialization(size_t prevLayerSize){
    biases.set(0.0);
    weights.randomize(-(sqrt(6)/sqrt(prevLayerSize + weights.cols)), (sqrt(6)/sqrt(prevLayerSize + weights.cols)));
}

inline double sigmoidf(double x){
    return (double)1.0/((double)1.0 + std::exp(-x));
}

void Layer::activate(){
    switch(functionType){
        case SIGMOID:
            {
                for(size_t i = 0; i < output.rows; ++i){
                   for(size_t j = 0; j < output.cols; ++j){
                        MAT_ACCESS(output, i, j) = sigmoidf(MAT_ACCESS(output, i, j));
                    } 
                }
                break;
            } 
        case RELU:
            {
                for(size_t i = 0; i < output.rows; ++i){
                   for(size_t j = 0; j < output.cols; ++j){
                        MAT_ACCESS(output, i, j) = std::max(0.0, MAT_ACCESS(output, i, j));
                    } 
                }
                break;
            }
        case NO_ACTIVATION:
            {
                break;
            }
    }
}

FastMatrix Layer::forward(FastMatrix input){
    output = (input*(weights)) + (biases);
    activate();
    return output;
}