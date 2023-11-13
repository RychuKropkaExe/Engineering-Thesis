#include "trainingData.h"
#include <cassert>
#include <iostream>

TrainingData::TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    std::cout << "CREATING TRAINING DATA" << "\n";
    assert(inputCount == outputCount);
    this->inputs.reserve(inputCount);
    this->outputs.reserve(outputCount);
    this->numOfSamples = inputCount;
    std::cout << "ELO1" << "\n";
    for(size_t i = 0; i < inputCount; ++i){
        std::cout << "ELO3" << "\n";
        this->inputs[i] = FastMatrix(trainingInputs[i], inputSize);
        std::cout << "ELO6" << "\n";
    }
    std::cout << "ELO 2" << "\n";
    for(size_t i = 0; i < inputCount; ++i){
        this->outputs[i] = FastMatrix(trainingOutputs[i], outputSize);
    }
    std::cout << "CREATED TRAINING DATA" << "\n";
}

TrainingData::TrainingData(){
    this->numOfSamples = 0;
}