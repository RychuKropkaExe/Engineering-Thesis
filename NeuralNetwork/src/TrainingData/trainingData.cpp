#include "trainingData.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

TrainingData::TrainingData(vector<vector<double>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<double>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    LOG(INFO_LEVEL, "TRAINING DATA INITIALIZED FROM VECTORS");
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

    LOG(INFO_LEVEL, "INITIALIZATION FINISHED, RESULTING TRAINING DATA: " << *this);
}

TrainingData::TrainingData()
{
    this->numOfSamples = 0;
    this->inputs.resize(1);
    this->outputs.resize(1);
}

TrainingData::TrainingData(std::string filename)
{
    LOG(INFO_LEVEL, "TRAINING DATA INITIALIZING FROM FILE");
    std::ifstream f(filename);
    std::string buffer;
    LOG(INFO_LEVEL, "FILE OPENED SUCCESSFULLY");

    getline(f, buffer);
    LOG(INFO_LEVEL, "GIVEN NUMBER OF SAMPLES: " << buffer);
    numOfSamples = (size_t)stoi(buffer);
    LOG(INFO_LEVEL, "PRASED NUMBER OF SAMPLES");

    getline(f, buffer);
    inputSize = (size_t)stoi(buffer);
    LOG(INFO_LEVEL, "PRASED INPUT SIZE");

    getline(f, buffer);
    outputSize = (size_t)stoi(buffer);

    inputs.resize(numOfSamples);
    outputs.resize(numOfSamples);
    LOG(INFO_LEVEL, "PARSED OUTPUT SIZE");

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

    LOG(INFO_LEVEL, "PARSED INPUT SAMPLES");

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
    LOG(INFO_LEVEL, "PARSED OUTPUT SAMPLES");
    LOG(INFO_LEVEL, "INITIALIZATION FINISHED, RESULTING TRAINING DATA: " << *this);
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

double TrainingData::findInputStandardDeviation()
{
    double meanInput = findMeanInput();

    double meanSquareSum = 0.f;

    for (size_t i = 0; i < numOfSamples; i++)
    {
        for (size_t j = 0; j < inputSize; j++)
        {
            double tmp = MAT_ACCESS(inputs[i], 0, j) - meanInput;
            meanSquareSum += tmp * tmp;
        }
    }

    double result = std::sqrt(meanSquareSum / (numOfSamples * inputSize));

    return result;
}

double TrainingData::findOutputStandardDeviation()
{
    double meanOutput = findMeanOutput();

    double meanSquareSum = 0.f;

    for (size_t i = 0; i < numOfSamples; i++)
    {
        for (size_t j = 0; j < outputSize; j++)
        {
            double tmp = MAT_ACCESS(outputs[i], 0, j) - meanOutput;
            meanSquareSum += tmp * tmp;
        }
    }

    double result = std::sqrt(meanSquareSum / (numOfSamples * outputSize));

    return result;
}

void TrainingData::normalizeData(NormalizationTypeE normType)
{

    LOG(INFO_LEVEL, "NORMALIZTING DATA USING NORM TYPE: " << normType);
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
        LOG(INFO_LEVEL, "DATA AFTER NORMALIZATION: " << *this);
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
        LOG(INFO_LEVEL, "DATA AFTER NORMALIZATION: " << *this);
        return;
    }
    case (STANDARIZATION):
    {
        double meanInput = findMeanInput();
        double meanOutput = findMeanOutput();

        double inputStandardDeviation = findInputStandardDeviation();
        double outputStandardDeviation = findOutputStandardDeviation();

        for (size_t i = 0; i < numOfSamples; i++)
        {
            for (size_t j = 0; j < inputSize; j++)
            {
                MAT_ACCESS(inputs[i], 0, j) = (MAT_ACCESS(inputs[i], 0, j) - meanInput) / (inputStandardDeviation);
            }
            for (size_t j = 0; j < outputSize; j++)
            {
                MAT_ACCESS(outputs[i], 0, j) = (MAT_ACCESS(outputs[i], 0, j) - meanOutput) / (outputStandardDeviation);
            }
        }

        standarizationData.meanInput = meanInput;
        standarizationData.meanOutput = meanOutput;

        standarizationData.inputStandardDeviation = inputStandardDeviation;
        standarizationData.outputStandardDeviation = outputStandardDeviation;
        LOG(INFO_LEVEL, "DATA AFTER NORMALIZATION: " << *this);
        return;
    }
    default:
    {
        LOG(ERROR_LEVEL, "UNRECOGNIZED NORM TYPE DURING NORMALIZATION" << normType);
        return;
    }
    }
}

void TrainingData::denomralizeOutput(NormalizationTypeE normType, FastMatrix &output)
{
    LOG(INFO_LEVEL, "DENORMALIZING USING NORM TYPE: " << normType << " VALUES: " << output);
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
        LOG(INFO_LEVEL, "VALUES AFTER DENORMALIZATION: " << output);
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
        LOG(INFO_LEVEL, "VALUES AFTER DENORMALIZATION: " << output);
        return;
    }
    case (STANDARIZATION):
    {
        double meanOutput = standarizationData.meanOutput;
        double outputStandardDeviation = standarizationData.outputStandardDeviation;
        for (size_t j = 0; j < outputSize; j++)
        {
            MAT_ACCESS(output, 0, j) = MAT_ACCESS(output, 0, j) * (outputStandardDeviation) + meanOutput;
        }
        LOG(INFO_LEVEL, "VALUES AFTER DENORMALIZATION: " << output);
        return;
    }
    default:
    {
        LOG(ERROR_LEVEL, "UNRECOGNIZED NORM TYPE");
        return;
    }
    }
}

std::ostream &operator<<(std::ostream &os, const TrainingData &td)
{

    for (size_t i = 0; i < td.numOfSamples; ++i)
    {
        os << "SAMPLE: " << i;
        os << td.inputs[i];
        os << "EXPECTED RESULT: ";
        os << td.outputs[i];
    }

    return os;
}
