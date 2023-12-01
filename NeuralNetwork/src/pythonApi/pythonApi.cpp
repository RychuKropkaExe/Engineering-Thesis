#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "../fastMatrix/FastMatrix.h"
#include "../model/trainingData.h"
#include "../model/model.h"
#include "pythonApi.h"

TrackmaniaAgent agent;

namespace py = pybind11;

void createMainModel(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize){
    agent.mainModel = Model(arch, archSize, actFunctions, actFunctionsSize, randomize);
}

void loadMainModel(std::string filename){
    agent.mainModel = parseModelFromFile(filename);
}

void createTargetModel(){
    agent.targetModel = agent.mainModel;
}

vector<float> runModel(vector<float> input, size_t inputSize){
    return agent.mainModel.run(FastMatrix(input, inputSize, ROW_VECTOR)).mat;
}

void do_something() {
    std::cout << "SIEMA ENIU" << "\n";
}

void initializeBuffers(size_t size){
    agent.stateBuffer.resize(size);
    agent.actionBuffer.resize(size);
    agent.rewardBuffer.resize(size);
    agent.nextStateBuffer.resize(size);
    agent.doneBuffer.resize(size);
    std::cout << "STATE BUFFER SIZE: " << agent.stateBuffer.size() << "\n";
    std::cout << "STATE BUFFER CAPACITY: " << agent.stateBuffer.capacity()<< "\n";
    std::cout << "STATE actionBuffer SIZE: " << agent.actionBuffer.size()<< "\n";
    std::cout << "STATE actionBuffer CAPACITY: " << agent.actionBuffer.capacity()<< "\n";
    std::cout << "STATE rewardBuffer SIZE: " << agent.rewardBuffer.size()<< "\n";
    std::cout << "STATE rewardBuffer CAPACITY: " << agent.rewardBuffer.capacity()<< "\n";
    std::cout << "STATE nextStateBuffer SIZE: " << agent.nextStateBuffer.size()<< "\n";
    std::cout << "STATE nextStateBuffer CAPACITY: " << agent.nextStateBuffer.capacity()<< "\n";
    std::cout << "STATE doneBuffer SIZE: " << agent.doneBuffer.size()<< "\n";
    std::cout << "STATE doneBuffer CAPACITY: " << agent.doneBuffer.capacity()<< "\n";
    std::cout << "MAX_BUFFER_SIZE: " << size<< "\n";
    agent.maxBufferSize = size;
}

// 100000/10 = 10000 -> 10000 sekund -> 10000/60 minut -> ~150 minut -> 2,5 h

void remember(vector<float> state, ActionsE action, vector<float> nextState, float reward, bool done){
    agent.stateBuffer[agent.bufferSize] = state;
    agent.actionBuffer[agent.bufferSize] = action;
    agent.nextStateBuffer[agent.bufferSize] = nextState;
    agent.rewardBuffer[agent.bufferSize] = reward;
    agent.doneBuffer[agent.bufferSize] = done;
    agent.bufferSize++;
    std::cout << "CUR_BUFFER_SIZE: " << agent.bufferSize << "\n";
}

void setTrainingData(size_t sampleCount, size_t inputSize, size_t outputSize, float discountRate){

    vector<vector<float>> inputs;
    vector<vector<float>> outputs;
    inputs.resize(sampleCount);
    outputs.resize(sampleCount);

    for(size_t i = 0; i < sampleCount; ++i){

        int index = (int)randomFloat(0.f, (float)(agent.bufferSize - 1));
        
        vector<float> state = agent.stateBuffer[index];
        vector<float> nextState = agent.nextStateBuffer[index];
        float reward = agent.rewardBuffer[index];
        ActionsE action = agent.actionBuffer[index];

        std::cout << "ACTION TAKEN: " << action << "\n";

        vector<float> currQs = agent.mainModel.run(FastMatrix(state, inputSize, ROW_VECTOR)).mat;
        FastMatrix currsQs(currQs, outputSize, ROW_VECTOR);
        vector<float> nextQs = agent.targetModel.run(FastMatrix(nextState, inputSize, ROW_VECTOR)).mat;

        FastMatrix nextsQs(nextQs, outputSize, ROW_VECTOR);
        printFastMatrix(currsQs);
        printFastMatrix(nextsQs);

        float maxNextQ = *max_element(std::begin(nextQs), std::end(nextQs));

        std::cout << "MAXNEXTQ: " << maxNextQ << "\n";

        currQs[action] = reward+maxNextQ*discountRate;

        inputs[i] = state;
        outputs[i] = currQs;
    }

    agent.trainingData = TrainingData(inputs, inputSize, sampleCount, outputs, outputSize, sampleCount);

}

void agentLearn(size_t numberOfIterations, size_t sampleCount, size_t inputSize, size_t outputSize, float discountRate){
    std::cout << "SETTING TRAINING DATA" << "\n";
    setTrainingData(sampleCount, inputSize, outputSize, discountRate);
    std::cout << "TRAINING DATA SET" << "\n";
    agent.mainModel.learn(agent.trainingData, numberOfIterations);
    std::cout << "LEARNING COMPLETED" << "\n";
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

    py::enum_<ActivationFunctionE>(m, "ActivationFunctionE")
    .value("SIGMOID", ActivationFunctionE::SIGMOID)
    .value("RELU", ActivationFunctionE::RELU)
    .value("NO_ACTIVATION", ActivationFunctionE::NO_ACTIVATION);

    py::enum_<ActionsE>(m, "ActionsE")
    .value("NO_ACTION", ActionsE::NO_ACTION)
    .value("FORWARD", ActionsE::FORWARD)
    .value("LEFT", ActionsE::LEFT)
    .value("RIGHT", ActionsE::RIGHT)
    .value("FORWARD_RIGHT", ActionsE::FORWARD_RIGHT)
    .value("FORWARD_LEFT", ActionsE::FORWARD_LEFT)
    .value("ACTIONS_COUNT", ActionsE::ACTIONS_COUNT);
}