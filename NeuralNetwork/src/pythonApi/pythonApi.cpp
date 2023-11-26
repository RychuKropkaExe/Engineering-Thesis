#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "FastMatrix.h"
#include "trainingData.h"
#include "model.h"
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

vector<float> runModel(vector<float> input){
    return agent.mainModel.run(FastMatrix(input, agent.stateSize, ROW_VECTOR)).mat;
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
}

// 100000/10 = 10000 -> 10000 sekund -> 10000/60 minut -> ~150 minut -> 2,5 h

void remember(vector<float> state, ActionsE action, vector<float> nextState, float reward, bool done){
    if(agent.bufferSize >= agent.maxBufferSize){
        agent.stateBuffer.resize(agent.bufferSize+agent.maxBufferSize);
        agent.actionBuffer.resize(agent.bufferSize+agent.maxBufferSize);
        agent.rewardBuffer.resize(agent.bufferSize+agent.maxBufferSize);
        agent.nextStateBuffer.resize(agent.bufferSize+agent.maxBufferSize);
        agent.doneBuffer.resize(agent.bufferSize+agent.maxBufferSize);
    }
    agent.stateBuffer[agent.bufferSize] = state;
    agent.actionBuffer[agent.bufferSize] = action;
    agent.nextStateBuffer[agent.bufferSize] = nextState;
    agent.rewardBuffer[agent.bufferSize] = reward;
    agent.doneBuffer[agent.bufferSize] = done;
    agent.bufferSize++;
}

void setTrainingData(size_t sampleCount, size_t inputSize, size_t outputSize, float discountRate){
    //todo
    vector<vector<float>> inputs;
    vector<vector<float>> outputs;
    inputs.resize(sampleCount);
    outputs.resize(sampleCount);

    for(size_t i = 0; i < sampleCount; ++i){

        int index = (int)randomFloat(0.f, (float)agent.bufferSize);
        
        vector<float> state = agent.stateBuffer[index];
        vector<float> nextState = agent.nextStateBuffer[index];
        float reward = agent.rewardBuffer[index];
        ActionsE action = agent.actionBuffer[index];

        vector<float> currQs = agent.mainModel.run(FastMatrix(state, inputSize, ROW_VECTOR)).mat;
        vector<float> nextQs = agent.targetModel.run(FastMatrix(nextState, inputSize, ROW_VECTOR)).mat;

        float maxNextQ = *max_element(std::begin(nextQs), std::end(nextQs));

        currQs[action] = reward+maxNextQ*discountRate;

        inputs[index] = state;
        outputs[index] = currQs;


    }

    agent.trainingData = TrainingData(inputs, inputSize, sampleCount, outputs, outputSize, sampleCount);

}

void agentLearn(size_t numberOfIterations, size_t sampleCount, size_t inputSize, size_t outputSize, float discountRate){
    setTrainingData(sampleCount, inputSize, outputSize, discountRate);
    agent.mainModel.learn(agent.trainingData, sampleCount);
}

void printTrainingData(){
    agent.trainingData.printTrainingData();
}

void dumpModel(std::string filename){
    agent.mainModel.printModelToFile(filename);
}

PYBIND11_MODULE(example, m) {
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
    .value("FORWARD_LEFT", ActionsE::FORWARD_LEFT);
}