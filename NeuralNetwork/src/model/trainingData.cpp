#include "trainingData.h"

TrainingData::TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    this->inputs.reserve(inputCount);
    this->outputs.reserve(outputCount);
    for(size_t i = 0; i < inputCount; ++i){
        this->inputs[i] = FastMatrix(trainingInputs[i], inputSize);
    }
    for(size_t i = 0; i < inputCount; ++i){
        this->outputs[i] = FastMatrix(trainingOutputs[i], outputSize);
    }
}