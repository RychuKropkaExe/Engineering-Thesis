#include "trainingData.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

TrainingData::TrainingData(vector<vector<double>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<double>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    assert(inputCount == outputCount);

    this->inputSize = inputSize;
    this->outputSize = outputSize;

    inputs.resize(inputCount);
    outputs.resize(outputCount);
    numOfSamples = inputCount;
    for (size_t i = 0; i < inputCount; ++i)
    {
        inputs[i] = FastMatrix(trainingInputs[i], inputSize, ROW_VECTOR);
    }
    for (size_t i = 0; i < inputCount; ++i)
    {
        FastMatrix t(trainingOutputs[i], outputSize, ROW_VECTOR);
        outputs[i] = t;
    }
}

TrainingData::TrainingData()
{
    this->numOfSamples = 0;
    this->inputs.resize(1);
    this->outputs.resize(1);
}

TrainingData::TrainingData(std::string filename)
{

    std::ifstream f(filename);
    std::string buffer;

    getline(f, buffer);
    numOfSamples = (size_t)stoi(buffer);

    getline(f, buffer);
    inputSize = (size_t)stoi(buffer);

    getline(f, buffer);
    outputSize = (size_t)stoi(buffer);

    inputs.resize(numOfSamples);
    outputs.resize(numOfSamples);

    for (size_t i = 0; i < numOfSamples; ++i)
    {

        getline(f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        vector<double> sampleInput;

        sampleInput.resize(inputSize);

        while (getline(check, buffer, ' '))
        {
            sampleInput[counter] = stof(buffer);
            counter++;
        }

        inputs[i] = FastMatrix(sampleInput, inputSize, ROW_VECTOR);
    }

    for (size_t i = 0; i < numOfSamples; ++i)
    {

        getline(f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        vector<double> sampleOutput;

        sampleOutput.resize(outputSize);

        while (getline(check, buffer, ' '))
        {
            sampleOutput[counter] = stof(buffer);
            counter++;
        }

        outputs[i] = FastMatrix(sampleOutput, outputSize, ROW_VECTOR);
    }
}

//====================================== DATA NORMALIZATION ============================================

double TrainingData::findMinInput()
{
    double curMin = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMin = *std::min_element(std::begin(inputs[i].mat), std::end(inputs[i].mat));
        if (curSampleMin < curMin)
            curMin = curSampleMin;
    }
    return curMin;
}

double TrainingData::findMaxInput()
{
    double curMax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMax = *std::max_element(std::begin(inputs[i].mat), std::end(inputs[i].mat));
        if (curSampleMax > curMax)
            curMax = curSampleMax;
    }
    return curMax;
}

double TrainingData::findMinOutput()
{
    double curMin = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMin = *std::min_element(std::begin(outputs[i].mat), std::end(outputs[i].mat));
        if (curSampleMin < curMin)
            curMin = curSampleMin;
    }
    return curMin;
}

double TrainingData::findMaxOutput()
{
    double curMax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMax = *std::max_element(std::begin(outputs[i].mat), std::end(outputs[i].mat));
        if (curSampleMax > curMax)
            curMax = curSampleMax;
    }
    return curMax;
}

double TrainingData::findMeanOutput()
{
    double meanValue = 0.f;
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMean = std::accumulate(std::begin(outputs[i].mat), std::end(outputs[i].mat), 0.0);
        meanValue += curSampleMean;
    }

    return meanValue / numOfSamples;
}

double TrainingData::findMeanInput()
{
    double meanValue = 0.f;
    for (size_t i = 0; i < numOfSamples; i++)
    {
        double curSampleMean = std::accumulate(std::begin(inputs[i].mat), std::end(inputs[i].mat), 0.0);
        meanValue += curSampleMean;
    }

    return meanValue / (numOfSamples * inputSize);
}

void TrainingData::normalizeData(NormalizationTypeE normType)
{
    switch (normType)
    {
    case (NORMALIZATION):
    {
        double minInput = findMinInput();
        double maxInput = findMaxInput();
        double minOutput = findMinOutput();
        double maxOutput = findMaxOutput();

        double meanInput = findMeanInput();
        double meanOutput = findMeanOutput();

        for (size_t i = 0; i < numOfSamples; i++)
        {
            for (size_t j = 0; j < inputSize; j++)
            {
                MAT_ACCESS(inputs[i], 0, j) = (MAT_ACCESS(inputs[i], 0, j) - meanInput) / (maxInput - minInput);
            }
            for (size_t j = 0; j < outputSize; j++)
            {
                MAT_ACCESS(outputs[i], 0, j) = (MAT_ACCESS(outputs[i], 0, j) - meanOutput) / (maxOutput - minOutput);
            }
        }
        normalizationData.minInputValue = minInput;
        normalizationData.maxInputValue = maxInput;

        normalizationData.minOutputValue = minOutput;
        normalizationData.maxOutputValue = maxOutput;

        normalizationData.meanInput = meanInput;
        normalizationData.meanOutput = meanOutput;
        return;
    }
    case (MIN_MAX_NORMALIZATION):
    {
        double minInput = findMinInput();
        double maxInput = findMaxInput();
        double minOutput = findMinOutput();
        double maxOutput = findMaxOutput();

        for (size_t i = 0; i < numOfSamples; i++)
        {
            for (size_t j = 0; j < inputSize; j++)
            {
                MAT_ACCESS(inputs[i], 0, j) = (MAT_ACCESS(inputs[i], 0, j) - minInput) / (maxInput - minInput);
            }
            for (size_t j = 0; j < outputSize; j++)
            {
                MAT_ACCESS(outputs[i], 0, j) = (MAT_ACCESS(outputs[i], 0, j) - minOutput) / (maxOutput - minOutput);
            }
        }
        minMaxNormalizationData.minInputValue = minInput;
        minMaxNormalizationData.maxInputValue = maxInput;
        minMaxNormalizationData.minOutputValue = minOutput;
        minMaxNormalizationData.maxOutputValue = maxOutput;
        return;
    }
    case (STANDARIZATION):
    {
        return;
    }
    default:
    {
        return;
    }
    }
}

void TrainingData::denomralizeOutput(NormalizationTypeE normType, FastMatrix &output)
{
    switch (normType)
    {
    case (NORMALIZATION):
    {
        double minOutput = normalizationData.minOutputValue;
        double maxOutput = normalizationData.maxOutputValue;
        double meanOutput = normalizationData.meanOutput;
        for (size_t j = 0; j < outputSize; j++)
        {
            MAT_ACCESS(output, 0, j) = MAT_ACCESS(output, 0, j) * (maxOutput - minOutput) + meanOutput;
        }
        return;
    }
    case (MIN_MAX_NORMALIZATION):
    {
        double minOutput = minMaxNormalizationData.minOutputValue;
        double maxOutput = minMaxNormalizationData.maxOutputValue;
        for (size_t j = 0; j < outputSize; j++)
        {
            MAT_ACCESS(output, 0, j) = MAT_ACCESS(output, 0, j) * (maxOutput - minOutput) + minOutput;
        }
        return;
    }
    case (STANDARIZATION):
    {
        return;
    }
    default:
    {
        return;
    }
    }
}

void TrainingData::printTrainingData()
{

    for (size_t i = 0; i < numOfSamples; ++i)
    {
        std::cout << "SAMPLE: " << i << "\n";
        printFastMatrix(inputs[i]);
        std::cout << "EXPECTED RESULT: " << "\n";
        printFastMatrix(outputs[i]);
    }
}
