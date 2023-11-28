#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "../fastMatrix/FastMatrix.h"
#include <vector>
#include <string>
using std::vector;

class TrainingData{

    public:
        vector<FastMatrix> inputs;
        size_t inputSize = 0;
        vector<FastMatrix> outputs;
        size_t outputSize = 0;
        size_t numOfSamples;
        TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                     vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount);
        TrainingData(std::string filename);
        TrainingData();
        void printTrainingData();

};

#endif