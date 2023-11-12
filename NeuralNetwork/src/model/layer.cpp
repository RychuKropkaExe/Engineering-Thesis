#include "layer.h"
#include <cmath>

//======================= CONSTRUCTORS ==========================================

Layer::Layer(pair<size_t, size_t> outputDimensions, pair<size_t, size_t> weightsDimensions, pair<size_t, size_t> biasesDimensions){
    this->output = FastMatrix(GET_ROWS_FROM_PAIR(outputDimensions), GET_COLS_FROM_PAIR(outputDimensions));
    this->weights = FastMatrix(GET_ROWS_FROM_PAIR(weightsDimensions), GET_COLS_FROM_PAIR(weightsDimensions));
    this->biases = FastMatrix(GET_ROWS_FROM_PAIR(biasesDimensions), GET_COLS_FROM_PAIR(biasesDimensions));
}

//======================= UTILITIES ============================================

inline float sigmoidf(float x){
    return 1.f/(1.f + std::exp(-x));
}

void activate(FastMatrix& mat, ActivationFunctionE functionTypeE){
    switch(functionTypeE){
        case SIGMOID:
            {
                for(size_t i = 0; i < mat.rows; ++i){
                   for(size_t j = 0; i < mat.cols; ++j){
                        MAT_ACCESS(mat, i, j) = sigmoidf(MAT_ACCESS(mat, i, j));
                    } 
                }
            } 
        case RELU:
            {
                return;
            }
    }
}

void Layer::forward(FastMatrix input){
    this->output = input*(this->weights);
    this->output = (this->output) + (this->biases);
    activate(this->output, this->functionType);
}