#ifndef MODEL_TEST_H
#define MODEL_TEST_H
#include <vector>
#include "trainingData.h"
#include "model.h"
using std::vector;

void xorModelTest(){
    vector<vector<float>> trainingInputs {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    vector<vector<float>> trainingOutputs {
        {0},
        {1},
        {1},
        {0}
    };

    TrainingData td = TrainingData(trainingInputs, 2, 4, trainingOutputs, 1, 4);

    vector<size_t> arch = {2, 2, 1};
    size_t archSize = 3;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    float eps = 1e-1;
    float learningRate = 1e-1;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    // model.printModel();

    model.learn(td, 10000);

    std::cout << "COST FUNCTION VALUE: " << model.cost() << "\n";

    for(size_t i = 0; i < 4; ++i){
        FastMatrix inp(trainingInputs[i], 2, ROW_VECTOR);
        std::cout << "RESULT FOR: " << "\n";
        printFastMatrix(inp);
        FastMatrix res = model.run(inp);
        printFastMatrix(res);
    }
    // model.printModel();
}

void modelTests(){
    xorModelTest();
}

#endif