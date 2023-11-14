#include "model.h"
#include <cassert>
#include <iostream>


Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize){
    assert(archSize == actFunctionsSize);
    this->activationFunctions.resize(actFunctionsSize);
    for(size_t i = 0; i < actFunctionsSize; ++i){
        this->activationFunctions[i] = actFunctions[i];
    }

    this->layers.resize(archSize);
    this->numberOfLayers = archSize;
    for(size_t i = 0; i < archSize - 1; ++i){
        pair<size_t, size_t> weightsDimensions;
        pair<size_t, size_t> biasesDimensions;
        pair<size_t, size_t> outputDimensions;
        SET_ROWS_IN_PAIR(weightsDimensions, arch[i]);
        SET_COLS_IN_PAIR(weightsDimensions, arch[i+1]);
        SET_ROWS_IN_PAIR(biasesDimensions, 1);
        SET_COLS_IN_PAIR(biasesDimensions, arch[i+1]);
        SET_ROWS_IN_PAIR(outputDimensions, 1);
        SET_COLS_IN_PAIR(outputDimensions, arch[i+1]);
        this->layers[i] = Layer(outputDimensions, weightsDimensions , biasesDimensions, actFunctions[i], randomize);
    }

    this->arch = arch;
    this->archSize = archSize;
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

            float d = MAT_ACCESS(result, 0, j) - MAT_ACCESS(this->trainingData.outputs[i], 0, j);
            totalCost += d*d;

        }

    }

    return totalCost/(this->trainingData.numOfSamples);

}

void Model::learn(TrainingData& trainingData, size_t iterations){

    std::cout << "STARTED LEARNING" << "\n";
    this->trainingData = trainingData;
    std::cout << "COST FUNCTION VALUE: " << cost() << "\n";
    float percentage = 0.1f;
    for(size_t i = 0; i < iterations; ++i){
        if( ((float)i / (float)iterations) >= percentage ){
            std::cout << "LEARNING COMPLETION: " << 100*percentage << "\n";
            std::cout << "COST FUNCTION VALUE: " << cost() << "\n";
            percentage += 0.1f;
        }

        finiteDifference();
        //std::cout << "COST FUNCTION VALUE: " << cost() << "\n";

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
    vector<ActivationFunctionE> cp = this->activationFunctions;
    Model fakeGradient(this->arch, this->archSize, {SIGMOID, SIGMOID}, this->archSize, false);

    float saved;
    float curCost = cost();

    for(size_t i = 0; i < numberOfLayers; ++i){

        for(size_t j = 0; j < layers[i].weights.rows; ++j){
            for(size_t k = 0; k < this->layers[i].weights.cols; k++){
                saved = MAT_ACCESS(this->layers[i].weights, j, k);
                MAT_ACCESS(layers[i].weights, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].weights, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].weights, j, k) = saved;
            }
        }

        for(size_t j = 0; j < layers[i].biases.rows; ++j){
            for(size_t k = 0; k < layers[i].biases.cols; k++){
                saved = MAT_ACCESS(layers[i].biases, j, k);
                MAT_ACCESS(layers[i].biases, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].biases, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].biases, j, k) = saved;
            }
        }

    }
    for(size_t i = 0; i < this->numberOfLayers; ++i){

        for(size_t j = 0; j < this->layers[i].weights.rows; ++j){
            for(size_t k = 0; k < this->layers[i].weights.cols; ++k){
                MAT_ACCESS(layers[i].weights, j, k) -= learningRate*MAT_ACCESS(fakeGradient.layers[i].weights, j, k);
            }
        }

        for(size_t j = 0; j < this->layers[i].biases.rows; ++j){
            for(size_t k = 0; k < this->layers[i].biases.cols; ++k){
                MAT_ACCESS(layers[i].biases, j, k) -= learningRate*MAT_ACCESS(fakeGradient.layers[i].biases, j, k);
            }
        }

    }

}

void Model::printModel(){
    
    std::cout << "MODEL PARAMETERS: " << "\n";
    std::cout << "LEARNING RATE: " << learningRate << "\n";
    std::cout << "EPS: " << eps << "\n";
    std::cout << "NUMBER OF LAYERS: " << numberOfLayers << "\n";
    std::cout << "ACTIVATION FUNCTIONS: " << activationFunctions[0] << "\n";
    std::cout << "LAYERS: " << "\n";

    for(size_t i = 0; i < numberOfLayers; ++i){
        std::cout << "LAYER NUMBER: " << i << "\n";
        std::cout << "WEIGHTS: " << "\n";
        printFastMatrix(layers[i].weights);
        std::cout << "BIASES: " << "\n";
        printFastMatrix(layers[i].biases);
        std::cout << "ACTIVATION FUNCTION: " << layers[i].functionType << "\n";
    }

}