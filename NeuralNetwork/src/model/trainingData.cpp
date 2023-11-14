#include "trainingData.h"
#include <cassert>
#include <iostream>

TrainingData::TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    std::cout << "CREATING TRAINING DATA" << "\n";
    assert(inputCount == outputCount);
    inputs.resize(inputCount);
    outputs.resize(outputCount);
    numOfSamples = inputCount;
    for(size_t i = 0; i < inputCount; ++i){
        inputs[i] = FastMatrix(trainingInputs[i], inputSize, ROW_VECTOR);
    }
    for(size_t i = 0; i < inputCount; ++i){
        FastMatrix t(trainingOutputs[i], outputSize, ROW_VECTOR);
        outputs[i] = t;
    }
}

TrainingData::TrainingData(){
    this->numOfSamples = 0;
    this->inputs.resize(1);
    this->outputs.resize(1);
}

void TrainingData::printTrainingData(){

    for(size_t i = 0; i < numOfSamples; ++i){
        std::cout << "SAMPLE: " << i << "\n";
        printFastMatrix(inputs[i]);
        std::cout << "EXPECTED RESULT: " << "\n";
        printFastMatrix(outputs[i]);
    }

}