#include "layer.h"
#include <cmath>

//======================= CONSTRUCTORS ==========================================

Layer::Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions,
             pair<size_t, size_t> biasesDimensions, ActivationFunctionE f)
{
    this->output = FastMatrix(GET_ROWS_FROM_PAIR(outputDimensions), GET_COLS_FROM_PAIR(outputDimensions));
    this->weights = FastMatrix(GET_ROWS_FROM_PAIR(weightsDimensions), GET_COLS_FROM_PAIR(weightsDimensions));
    this->biases = FastMatrix(GET_ROWS_FROM_PAIR(biasesDimensions), GET_COLS_FROM_PAIR(biasesDimensions));
    this->functionType = f;
}

Layer::Layer(){
    this->functionType = SIGMOID;
}

//======================= UTILITIES ============================================

inline float sigmoidf(float x){
    return 1.f/(1.f + std::exp(-x));
}

void Layer::activate(){
    switch(this->functionType){
        case SIGMOID:
            {
                for(size_t i = 0; i < (this->output).rows; ++i){
                   for(size_t j = 0; i < (this->output).cols; ++j){
                        MAT_ACCESS(this->output, i, j) = sigmoidf(MAT_ACCESS(this->output, i, j));
                    } 
                }
            } 
        case RELU:
            {
                return;
            }
    }
}

FastMatrix Layer::forward(FastMatrix input){
    this->output = input*(this->weights);
    this->output = (this->output) + (this->biases);
    activate();
    return this->output;
}