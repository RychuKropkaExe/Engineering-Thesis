#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "FastMatrix.h"
#include <vector>
using std::vector;

class TrainingData{

    public:
        vector<FastMatrix> inputs;
        vector<FastMatrix> outputs;
        size_t numOfSamples;
        TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                     vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount);
        TrainingData();

};

#endif