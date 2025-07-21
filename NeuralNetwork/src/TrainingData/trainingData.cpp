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

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

/******************************************************************************
 * @brief Creates training data from vecotrs of inputs and outputs. Used mainly
 *        For tests
 *
 * @param trainingInputs    Vector of training input vectors
 * @param inputSize         Size of single input vector
 * @param inputCount        Number of input vectors
 * @param trainingOutputs   Vector of training output vectors
 * @param outputSize        Size of single output vector
 * @param outputCount       Number of output vectors
 *
 * @return TrainingData
 ******************************************************************************/
TrainingData::TrainingData(vector<vector<double>> trainingInputs, size_t inputSize, size_t inputCount,
                           vector<vector<double>> trainingOutputs, size_t outputSize, size_t outputCount)
{
    LOG(NORMAL_LOGS, INFO_TYPE, "TRAINING DATA INITIALIZED FROM VECTORS");
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

    LOG(HEAVY_LOGS, INFO_TYPE, "INITIALIZATION FINISHED, RESULTING TRAINING DATA: " << *this);
}

/******************************************************************************
 * @brief Default constructor. Used only when TrainingData is a member of another
 *        class.
 *
 * @return TrainingData
 ******************************************************************************/
TrainingData::TrainingData()
{
    this->numOfSamples = 0;
    this->inputs.resize(1);
    this->outputs.resize(1);
}

/******************************************************************************
 * @brief Parses TrainingData from file. The file format does not matter.
 *        Structure of file must be as follows:
 *
 *        <number of samples>
 *        <input size>
 *        <output size>
 *        <input vector nr 1>
 *        <input vector nr 2>
 *        <input vector nr n>
 *        <output vector nr 1>
 *        <output vector nr 2>
 *        <output vector nr n>
 *
 * @param filename  Path to file from which the training data is parsed
 *
 * @return TrainingData
 ******************************************************************************/
TrainingData::TrainingData(std::string filename)
{
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "TRAINING DATA INITIALIZING FROM FILE");
    std::ifstream f(filename);
    std::string buffer;
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "FILE OPENED SUCCESSFULLY");

    getline(f, buffer);
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "GIVEN NUMBER OF SAMPLES: " << buffer);
    numOfSamples = (size_t)stoi(buffer);
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "PARSED NUMBER OF SAMPLES");

    getline(f, buffer);
    inputSize = (size_t)stoi(buffer);
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "PRASED INPUT SIZE");

    getline(f, buffer);
    outputSize = (size_t)stoi(buffer);

    inputs.resize(numOfSamples);
    outputs.resize(numOfSamples);
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "PARSED OUTPUT SIZE");

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

    LOG(ESSENTIAL_LOGS, INFO_TYPE, "PARSED INPUT SAMPLES");

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
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "PARSED OUTPUT SAMPLES");
    LOG(HEAVY_LOGS, INFO_TYPE, "INITIALIZATION FINISHED, RESULTING TRAINING DATA: " << *this);
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

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

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

/******************************************************************************
 * @brief Finds minimal value from training input samples
 *
 * @return min value from input samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds maximal value from training input samples
 *
 * @return max value from input samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds minimal value from training otput samples
 *
 * @return min value from output samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds maximal value from training output samples
 *
 * @return max value from output samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds mean value of training output samples
 *
 * @return mean value of output samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds mean value of training input samples
 *
 * @return mean value of input samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds standard deviation of training input samples
 *
 * @return standard deviation of input samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Finds standard deviation of training output samples
 *
 * @return standard deviation of output samples
 ******************************************************************************/
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

/******************************************************************************
 * @brief Normalizes training input and output sample using chosen method
 *
 * @param normType Which normalization method to use
 *
 * @return Nothing
 ******************************************************************************/
void TrainingData::normalizeData(NormalizationTypeE normType)
{

    LOG(HEAVY_LOGS, INFO_TYPE, "NORMALIZTING DATA USING NORM TYPE: " << normType);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "DATA AFTER NORMALIZATION: " << *this);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "DATA AFTER NORMALIZATION: " << *this);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "DATA AFTER NORMALIZATION: " << *this);
        return;
    }
    default:
    {
        LOG(ESSENTIAL_LOGS, ERROR_TYPE, "UNRECOGNIZED NORM TYPE DURING NORMALIZATION" << normType);
        return;
    }
    }
}

/******************************************************************************
 * @brief Denormalizes given output of the model.
 *
 * @param normType      Which normalization method to use
 * @param output[out]   Model output to denormalize. The result
 *                      is stored in this FastMatrix
 *
 * @return Nothing
 ******************************************************************************/
void TrainingData::denormalizeOutput(NormalizationTypeE normType, FastMatrix &output)
{
    LOG(HEAVY_LOGS, INFO_TYPE, "DENORMALIZING USING NORM TYPE: " << normType);
    LOG(HEAVY_LOGS, INFO_TYPE, "VALUES TO DENORMALIZE: " << output);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "VALUES AFTER DENORMALIZATION: " << output);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "VALUES AFTER DENORMALIZATION: " << output);
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
        LOG(HEAVY_LOGS, INFO_TYPE, "VALUES AFTER DENORMALIZATION: " << output);
        return;
    }
    default:
    {
        LOG(ESSENTIAL_LOGS, ERROR_TYPE, "UNRECOGNIZED NORM TYPE");
        return;
    }
    }
}
