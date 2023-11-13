#include "model.h"
#include <cassert>
#include <iostream>


Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize){
    std::cout << "CREATING MODEL" << "\n";
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

        this->layers[i] = Layer(outputDimensions, weightsDimensions , biasesDimensions, actFunctions[i], randomize);
    }

    this->arch = arch;
    this->archSize = archSize;
    std::cout << "CREATED MODEL" << "\n";
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

        FastMatrix result = run(this->trainingData.inputs[i]);

        for(size_t j = 0; j < result.cols; ++j){

            float d = MAT_ACCESS(result, j, 0) - MAT_ACCESS(this->trainingData.outputs[i], j, 0);
            totalCost += d*d;

        }

    }

    return totalCost/(this->trainingData.numOfSamples);

}

void Model::learn(TrainingData& trainingData, size_t iterations){

    std::cout << "STARTED LEARNING" << "\n";
    this->trainingData = trainingData;

    float percentage = 0.1f;
    for(size_t i = 0; i < iterations; ++i){
        if( (i / iterations) >= percentage )
            std::cout << "LEARNING COMPLETION: " << 100*percentage << "\n";

        finiteDifference();

    }


}

FastMatrix Model::run(FastMatrix input){

    FastMatrix result = input;

    for(size_t i = 0; i < this->numberOfLayers; i++){

        result = this->layers[i].forward(result);

    }

    return result;

}

void Model::finiteDifference(){
    Model fakeGradient(this->arch, this->archSize, this->activationFunctions, this->archSize, false);

    float saved;
    float curCost = cost();

    for(size_t i = 0; i < this->numberOfLayers; ++i){

        for(size_t j = 0; j < this->layers[i].weights.rows; ++j){
            for(size_t k = 0; k < this->layers[i].weights.cols; k++){
                saved = MAT_ACCESS(this->layers[i].weights, j, k);
                MAT_ACCESS(this->layers[i].weights, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].weights, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(this->layers[i].weights, j, k) = saved;
            }
        }

        for(size_t j = 0; j < this->layers[i].biases.rows; ++j){
            for(size_t k = 0; k < this->layers[i].biases.cols; k++){
                saved = MAT_ACCESS(this->layers[i].biases, j, k);
                MAT_ACCESS(this->layers[i].biases, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].biases, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(this->layers[i].biases, j, k) = saved;
            }
        }

    }

    for(size_t i = 0; i < this->numberOfLayers; ++i){

        for(size_t j = 0; j < this->layers[i].weights.rows; ++j){
            for(size_t k = 0; k < this->layers[i].weights.cols; k++){
                MAT_ACCESS(this->layers[i].weights, j, k) -= (this->learningRate)*MAT_ACCESS(fakeGradient.layers[i].weights, j, k);
            }
        }

        for(size_t j = 0; j < this->layers[i].biases.rows; ++j){
            for(size_t k = 0; k < this->layers[i].biases.cols; k++){
                MAT_ACCESS(this->layers[i].biases, j, k) -= (this->learningRate)*MAT_ACCESS(fakeGradient.layers[i].biases, j, k);
            }
        }

    }

}