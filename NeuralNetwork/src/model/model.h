#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "trainingData.h"
#include <vector>
using std::vector;

class Model{

    vector<Layer> layers;
    vector<ActivationFunctionE> activationFunctions;


    float learningRate;
    float eps;

    public:
        Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize);

        FastMatrix run(FastMatrix input);

        void setLearningRate(float val);
        void setEps(float val);
        void learn(TrainingData trainingData, size_t iterations);


};

#endif