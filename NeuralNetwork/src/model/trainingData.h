#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "FastMatrix.h"
#include <vector>
using std::vector;

class TrainingData{

    vector<FastMatrix> inputs;
    vector<FastMatrix> outputs;

    public:
        TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                     vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount);

};

#endif