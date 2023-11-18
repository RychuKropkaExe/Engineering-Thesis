#include "trainingData.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>      

TrainingData::TrainingData(vector<vector<float>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<float>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    std::cout << "CREATING TRAINING DATA" << "\n";
    assert(inputCount == outputCount);

    this->inputSize = inputSize;
    this->outputSize = outputSize;

    inputs.resize(inputCount);
    outputs.resize(outputCount);
    numOfSamples = inputCount;
    for(size_t i = 0; i < inputCount; ++i){
        inputs[i] = FastMatrix(trainingInputs[i], inputSize, ROW_VECTOR);
    }
    for(size_t i = 0; i < inputCount; ++i){
        FastMatrix t(trainingOutputs[i], outputSize, ROW_VECTOR);
        outputs[i] = t;
    }
}

TrainingData::TrainingData(){
    this->numOfSamples = 0;
    this->inputs.resize(1);
    this->outputs.resize(1);
}

TrainingData::TrainingData(std::string filename){

    std::ifstream f(filename);
    std::string buffer;

    getline (f, buffer);
    numOfSamples = (size_t)stoi(buffer);
  
    getline (f, buffer);
    inputSize = (size_t)stoi(buffer);

    getline (f, buffer);
    outputSize = (size_t)stoi(buffer);

    inputs.resize(numOfSamples);
    outputs.resize(numOfSamples);
    

    for(size_t i = 0; i < numOfSamples; ++i){

        getline (f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        vector<float> sampleInput;

        sampleInput.resize(inputSize);

        while(getline(check, buffer, ' '))
        {   
            sampleInput[counter] = stof(buffer);
            counter++;
        }

        inputs[i] = FastMatrix(sampleInput, inputSize, ROW_VECTOR);


    }

    for(size_t i = 0; i < numOfSamples; ++i){

        getline (f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        vector<float> sampleOutput;

        sampleOutput.resize(outputSize);

        while(getline(check, buffer, ' '))
        {   
            sampleOutput[counter] = stof(buffer);
            counter++;
        }

        outputs[i] = FastMatrix(sampleOutput, outputSize, ROW_VECTOR);


    }

}


void TrainingData::printTrainingData(){

    for(size_t i = 0; i < numOfSamples; ++i){
        std::cout << "SAMPLE: " << i << "\n";
        printFastMatrix(inputs[i]);
        std::cout << "EXPECTED RESULT: " << "\n";
        printFastMatrix(outputs[i]);
    }

}