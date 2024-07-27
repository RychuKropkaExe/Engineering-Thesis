#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "FastMatrix.h"
#include "trainingData.h"
#include "model.h"
#include "pythonApi.h"

TrackmaniaAgent agent;

namespace py = pybind11;

void createMainModel(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize){
    agent.mainModel = Model(arch, archSize, actFunctions, actFunctionsSize, randomize);
    agent.mainModel.modelXavierInitialize();
}

void loadMainModel(std::string filename){
    agent.mainModel = parseModelFromFile(filename);
}

void createTargetModel(){
    agent.targetModel = Model(agent.mainModel.arch, agent.mainModel.archSize, agent.mainModel.activationFunctions, agent.mainModel.activationFunctions.size(), false);
    for(size_t i = 0; i < agent.mainModel.numberOfLayers; ++i){
        for(size_t j = 0; j < agent.mainModel.layers[i].weights.rows; ++j){
            for(size_t k = 0; k < agent.mainModel.layers[i].weights.cols; ++k){
                MAT_ACCESS(agent.targetModel.layers[i].weights, j, k) = MAT_ACCESS(agent.mainModel.layers[i].weights, j, k);
            }
        }

        for(size_t j = 0; j < agent.mainModel.layers[i].biases.rows; ++j){
            for(size_t k = 0; k < agent.mainModel.layers[i].biases.cols; ++k){
                MAT_ACCESS(agent.targetModel.layers[i].biases, j, k) = MAT_ACCESS(agent.mainModel.layers[i].biases, j, k);
            }
        }
    }
}

void setTrainingRate(double val){
    agent.mainModel.learningRate = val;
}

void setMinThreshold(double val){
    agent.mainModel.minThreshold = val;
}

void setMaxThreshold(double val){
    agent.mainModel.maxThreshold = val;
}

vector<double> runModel(vector<double> input, size_t inputSize){
    return agent.mainModel.run(FastMatrix(input, inputSize, ROW_VECTOR)).mat;
}

void do_something() {
    std::cout << "Hello" << "\n";
}

void initializeBuffers(size_t size){
    agent.stateBuffer.resize(size);
    agent.actionBuffer.resize(size);
    agent.rewardBuffer.resize(size);
    agent.nextStateBuffer.resize(size);
    agent.doneBuffer.resize(size);
    agent.maxBufferSize = size;
}

// 100000/10 = 10000 -> 10000 sekund -> 10000/60 minut -> ~150 minut -> 2,5 h

void remember(vector<double> state, ActionsE action, vector<double> nextState, double reward, bool done){
    agent.stateBuffer[agent.bufferSize] = state;
    agent.actionBuffer[agent.bufferSize] = action;
    agent.nextStateBuffer[agent.bufferSize] = nextState;
    agent.rewardBuffer[agent.bufferSize] = reward;
    agent.doneBuffer[agent.bufferSize] = done;
    agent.bufferSize++;
    //std::cout << "CUR_BUFFER_SIZE: " << agent.bufferSize << "\n";
}

void setTrainingData(size_t sampleCount, size_t inputSize, size_t outputSize, double discountRate, size_t batchSize){

    vector<vector<double>> inputs;
    vector<vector<double>> outputs;
    inputs.resize(batchSize);
    outputs.resize(batchSize);

    for(size_t i = 0; i < batchSize; ++i){

        int index = (int)randomdouble(0.f, (double)(agent.bufferSize - 1));
        //std::cout << "RANDOM STATE: " << index << "\n";
        vector<double> state = agent.stateBuffer[index];
        vector<double> nextState = agent.nextStateBuffer[index];
        double reward = agent.rewardBuffer[index];
        ActionsE action = agent.actionBuffer[index];
        bool done = agent.doneBuffer[index];

        //std::cout << "ACTION TAKEN: " << action << "\n";

        vector<double> currQs = agent.mainModel.run(FastMatrix(state, inputSize, ROW_VECTOR)).mat;
        vector<double> nextQs = agent.targetModel.run(FastMatrix(nextState, inputSize, ROW_VECTOR)).mat;
        //printFastMatrix(currsQs);
        //printFastMatrix(nextsQs);

        //FastMatrix nextsQs(nextQs, outputSize, ROW_VECTOR);
        //printFastMatrix(nextsQs);
        double maxNextQ = *max_element(std::begin(nextQs), std::end(nextQs));

        //std::cout << "MAXNEXTQ: " << maxNextQ << "\n";
        // for(size_t k = 0; k < outputSize; ++k){
        //     currQs[k] = 0;
        // }
        currQs[action] = reward+(1-(int)done)*maxNextQ*discountRate;
        //std::cout << "REWARD FOR STATE: " << reward << "\n";
        //FastMatrix currsQs(currQs, outputSize, ROW_VECTOR);
        //printFastMatrix(currsQs);
        inputs[i] = state;
        outputs[i] = currQs;
    }

    agent.trainingData = TrainingData(inputs, inputSize, batchSize, outputs, outputSize, batchSize);

}

void agentLearn(size_t numberOfIterations, size_t sampleCount, size_t inputSize, size_t outputSize, double discountRate, size_t batchSize, bool clipGradient){
    //std::cout << "SETTING TRAINING DATA" << "\n";
    setTrainingData(sampleCount, inputSize, outputSize, discountRate, batchSize);
    //std::cout << "TRAINING DATA SET" << "\n";
    agent.mainModel.learn(agent.trainingData, numberOfIterations, clipGradient);
    //std::cout << "LEARNING COMPLETED" << "\n";
}

void printTrainingData(){
    agent.trainingData.printTrainingData();
}

void printModels(){
    agent.mainModel.printModel();
    agent.targetModel.printModel();
}

void dumpModel(std::string filename){
    agent.mainModel.printModelToFile(filename);
}

void xorModelTest(){

    TrainingData td = TrainingData(std::string("C:\\Users\\Admin\\Desktop\\Umieralnia\\PracaDyplomowa\\Engineering-Thesis\\NeuralNetwork\\test\\modelTest\\xorData.txt"));
    td.printTrainingData();
    vector<size_t> arch = {2, 2, 4, 1};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    // model.printModel();

    float eps = 1e-3;
    float learningRate = 1e-1f;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 100000, true);

    //model.printModelToFile("/home/rychu/Engineering-Thesis/NeuralNetwork/printedModel.log");

    assert(model.cost() < 0.05f);
}

PYBIND11_MODULE(agent, m) {
    m.def("do_something", &do_something, "A function that does something");
    m.def("initializeBuffers", &initializeBuffers, "A function that does something");
    m.def("remember", &remember, "A function that does something");

    m.def("setTrainingData", &setTrainingData, "A function that does something");
    m.def("printTrainingData", &printTrainingData, "A function that does something");

    m.def("createMainModel", &createMainModel, "A function that does something");
    m.def("createTargetModel", &createTargetModel, "A function that does something");
    m.def("runModel", &runModel, "A function that does something");
    m.def("agentLearn", &agentLearn, "A function that does something");
    m.def("dumpModel", &dumpModel, "A function that does something");
    m.def("loadMainModel", &loadMainModel, "A function that does something");
    m.def("printModels", &printModels, "A function that does something");
    m.def("setTrainingRate", &setTrainingRate, "A function that does something");
    m.def("xorModelTest", &xorModelTest, "A function that does something");
    m.def("setMinThreshold", &setMinThreshold, "A function that does something");
    m.def("setMaxThreshold", &setMaxThreshold, "A function that does something");

    py::enum_<ActivationFunctionE>(m, "ActivationFunctionE")
    .value("SIGMOID", ActivationFunctionE::SIGMOID)
    .value("RELU", ActivationFunctionE::RELU)
    .value("SOFTMAX", ActivationFunctionE::SOFTMAX)
    .value("NO_ACTIVATION", ActivationFunctionE::NO_ACTIVATION);

    py::enum_<ActionsE>(m, "ActionsE")
    .value("NO_ACTION", ActionsE::NO_ACTION)
    .value("FORWARD", ActionsE::FORWARD)
    // .value("LEFT", ActionsE::LEFT)
    // .value("RIGHT", ActionsE::RIGHT)
    .value("FORWARD_RIGHT", ActionsE::FORWARD_RIGHT)
    .value("FORWARD_LEFT", ActionsE::FORWARD_LEFT)
    .value("ACTIONS_COUNT", ActionsE::ACTIONS_COUNT);
}