#include "model.h"
#include <cassert>

Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize){

    assert(archSize == actFunctionsSize);
    this->activationFunctions.reserve(actFunctionsSize);
    for(size_t i = 0; i < actFunctionsSize; ++i){
        this->activationFunctions[i] = actFunctions[i];
    }

    this->layers.reserve(archSize);
    this->numberOfLayers = archSize;
    for(size_t i = 0; i < archSize - 1; ++i){
        pair<size_t, size_t> weightsDimensions;
        pair<size_t, size_t> biasesDimensions;
        pair<size_t, size_t> outputDimensions;
        SET_ROWS_IN_PAIR(weightsDimensions, arch[i]);
        SET_COLS_IN_PAIR(weightsDimensions, arch[i+1]);
        SET_ROWS_IN_PAIR(biasesDimensions, arch[i]);
        SET_COLS_IN_PAIR(biasesDimensions, 1);
        SET_ROWS_IN_PAIR(outputDimensions, arch[i]);
        SET_COLS_IN_PAIR(outputDimensions, 1);

        this->layers[i] = Layer(outputDimensions, weightsDimensions , biasesDimensions, actFunctions[i]);
    }


}

void Model::setLearningRate(float val){
    this->learningRate = val;
}

void Model::setEps(float val){
    this->eps = val;
}

float Model::cost(){

    float totalCost = 0;

    for(size_t i = 0; i < this->trainingData.numOfSamples; ++i){

        // Result of each layer. So the first result is the input
        FastMatrix result = this->trainingData.inputs[i];

        for(size_t j = 0; j < this->numberOfLayers; j++){

            result = this->layers[j].forward(result);

        }

        for(size_t j = 0; j < result.cols; ++j){

            float d = MAT_ACCESS(result, j, 0) - MAT_ACCESS(this->trainingData.outputs[i], j, 0);
            totalCost += d*d;

        }

    }

    return totalCost/(this->trainingData.numOfSamples);

}

void Model::learn(TrainingData trainingData, size_t iterations){
    return;
}

FastMatrix Model::run(FastMatrix input){
    return FastMatrix();
}