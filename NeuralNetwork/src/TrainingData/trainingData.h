#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "FastMatrix.h"
#include <string>
#include <vector>
using std::vector;

/******************************************************************************
 * @enum NormalizationTypeE
 *
 * @brief Types of normalization that can be used on training data
 ******************************************************************************/
enum NormalizationTypeE
{
    NORMALIZATION,
    MIN_MAX_NORMALIZATION,
    STANDARIZATION
};

/******************************************************************************
 * @class MinMaxNormalizationData
 *
 * @brief Wrapper for parameters used in min max normalization and
 *        denormalization
 *
 * @public @param minInputValue  Minmal value in training inputs
 * @public @param maxInputValue  Maximal value in training inputs
 * @public @param minOutputValue Minimal value in training outputs
 * @public @param maxOutputValue Maximal value in training outputs
 ******************************************************************************/
class MinMaxNormalizationData
{
public:
    double minInputValue{0.f};
    double maxInputValue{0.f};

    double minOutputValue{0.f};
    double maxOutputValue{0.f};
};

/******************************************************************************
 * @class NormalizationData
 *
 * @brief Wrapper for parameters used in standard normalization and
 *        denormalization
 *
 * @public @param minInputValue  Minmal value in training inputs
 * @public @param maxInputValue  Maximal value in training inputs
 * @public @param meanInput      Mean value of training inputs
 * @public @param minOutputValue Minimal value in training outputs
 * @public @param maxOutputValue Maximal value in training outputs
 * @public @param meanOutput     Mean value of training outputs
 ******************************************************************************/
class NormalizationData
{
public:
    double minInputValue{0.f};
    double maxInputValue{0.f};
    double meanInput{0.f};

    double minOutputValue{0.f};
    double maxOutputValue{0.f};
    double meanOutput{0.f};
};

/******************************************************************************
 * @class StandarizationData
 *
 * @brief Wrapper for parameters used in standarization and
 *        destandarization
 *
 * @public @param meanInput                 Mean value of training inputs
 * @public @param inputStandardDeviation    Standard deviation of input
 * @public @param meanOutput                Mean value of training outputs
 * @public @param outputStandardDeviation   Standard deviation of output
 ******************************************************************************/
class StandarizationData
{
public:
    double meanInput{0.f};
    double inputStandardDeviation{0.f};

    double meanOutput{0.f};
    double outputStandardDeviation{0.f};
};

/******************************************************************************
 * @class TrainingData
 *
 * @brief Implementation of standardized way of handling training data for model
 *        with all of its transformations and utilities
 *
 * @public @param inputs                    Vector of training input samples
 * @public @param inputSize                 Size of single input sample
 * @public @param outputs                   Vector of training otput samples
 * @public @param outputSize                Size of single output sample
 * @public @param numOfSamples              Number of training samples
 * @public @param minMaxNormalizationData   Data for min max normalization
 * @public @param standarizationData        Data for standarization
 * @public @param normalizationData         Data for standard normalization
 ******************************************************************************/
class TrainingData
{

public:
    /******************************************************************************
     * CLASS MEMBERS
     ******************************************************************************/

    vector<FastMatrix>
        inputs;
    size_t inputSize = 0;
    vector<FastMatrix>
        outputs;
    size_t outputSize = 0;
    size_t numOfSamples;

    MinMaxNormalizationData minMaxNormalizationData;
    StandarizationData standarizationData;
    NormalizationData normalizationData;

    /******************************************************************************
     * CONSTRUCTORS
     ******************************************************************************/

    TrainingData(vector<vector<double>> trainingInputs, size_t inputSize, size_t inputCount,
                 vector<vector<double>> trainingOutputs, size_t outputSize, size_t outputCount);
    TrainingData(std::string filename);
    TrainingData();

    /******************************************************************************
     * OPERATORS
     ******************************************************************************/

    friend std::ostream &operator<<(std::ostream &os, const TrainingData &td);

    /******************************************************************************
     * UTILITIES
     ******************************************************************************/

    double findMinInput();
    double findMaxInput();
    double findMinOutput();
    double findMaxOutput();

    double findMeanInput();
    double findMeanOutput();

    double findInputStandardDeviation();
    double findOutputStandardDeviation();

    void normalizeData(NormalizationTypeE normType);
    void denomralizeOutput(NormalizationTypeE normType, FastMatrix &output);
};

#endif
