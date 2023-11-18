#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "trainingData.h"
#include <vector>
using std::vector;

class Model{

    public:

        vector<Layer> layers;
        size_t numberOfLayers;
        vector<ActivationFunctionE> activationFunctions;

        // For easy copy of model dimensions for finite difference purposes
        vector<size_t> arch;
        size_t archSize;

        TrainingData trainingData {};
        float learningRate = 1.f;
        float eps = 1e-1;

        Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize);
        Model();

        FastMatrix run(FastMatrix input);

        float cost();
        void finiteDifference();
        void backPropagation();

        void setLearningRate(float val);
        void setEps(float val);
        void learn(TrainingData& trainingData, size_t iterations);

        void printModel();

        void printModelToFile(std::string filename);


};

Model parseModelFromFile(std::string filename);

#endif