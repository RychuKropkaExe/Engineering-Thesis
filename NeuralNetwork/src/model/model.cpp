#include "model.h"
#include "logger.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

/******************************************************************************
 * @brief Creates model of given architecture, and with given properties
 *
 * @param arch              Vector describing architecture of model. Contains sizes
 *                          of each layer
 * @param archSize          Size of @arch vector
 * @param actFunctions      Which activation function each layer should use
 * @param actFunctionsSize  Size of @actFunctions vector
 * @param randomize         Tells if each layer should randomize its weights
 *                          and biases
 *
 * @return Fully functional Model
 ******************************************************************************/
Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize)
{
    assert(archSize == actFunctionsSize);
    this->activationFunctions.resize(actFunctionsSize);
    for (size_t i = 1; i < archSize; ++i)
    {
        this->activationFunctions[i] = actFunctions[i - 1];
    }

    this->layers.resize(archSize);
    this->numberOfLayers = archSize;

    pair<size_t, size_t> inputDimensions;
    SET_ROWS_IN_PAIR(inputDimensions, 1);
    SET_COLS_IN_PAIR(inputDimensions, arch[0]);
    this->layers[0] = Layer(inputDimensions);

    for (size_t i = 1; i < archSize; ++i)
    {

        pair<size_t, size_t> weightsDimensions;
        pair<size_t, size_t> biasesDimensions;
        pair<size_t, size_t> outputDimensions;
        SET_ROWS_IN_PAIR(weightsDimensions, arch[i - 1]);
        SET_COLS_IN_PAIR(weightsDimensions, arch[i]);
        SET_ROWS_IN_PAIR(biasesDimensions, 1);
        SET_COLS_IN_PAIR(biasesDimensions, arch[i]);
        SET_ROWS_IN_PAIR(outputDimensions, 1);
        SET_COLS_IN_PAIR(outputDimensions, arch[i]);

        LayerTypeE type = INTERMEDIATE_LAYER;

        if (i == 0)
        {
            type = INPUT_LAYER;
        }
        if (i == archSize)
        {
            type = OUTPUT_LAYER;
        }

        this->layers[i] = Layer(outputDimensions, weightsDimensions, biasesDimensions, actFunctions[i - 1], randomize, type);
    }

    this->arch = arch;
    this->archSize = archSize;
}

/******************************************************************************
 * @brief Default constructor for Model. Used only when Model is part of another
 *        object and is not yet fully initialized.
 *
 * @return Empty Model
 ******************************************************************************/
Model::Model()
{
    this->layers.resize(1);
    this->activationFunctions.resize(1);
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const Model &model)
{
    os << "MODEL PARAMETERS: " << "\n";
    os << "LEARNING RATE: " << model.learningRate << "\n";
    os << "EPS: " << model.eps << "\n";
    os << "NUMBER OF LAYERS: " << model.numberOfLayers << "\n";
    os << "ACTIVATION FUNCTIONS: " << model.activationFunctions[0] << "\n";
    os << "LAYERS: " << "\n";

    for (size_t i = 0; i < model.numberOfLayers; ++i)
    {
        os << "LAYER NUMBER: " << i;
        os << "WEIGHTS: ";
        os << model.layers[i].weights;
        os << "BIASES: ";
        os << model.layers[i].biases;
        os << "INTERMIEDIATE: ";
        os << model.layers[i].output;
        os << "ACTIVATION FUNCTION: " << model.layers[i].functionType << "\n";
    }
    os << std::flush;
    return os;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

/******************************************************************************
 * @brief Initilizes model layers using Xavier method
 *
 * @return Nothing
 ******************************************************************************/
void Model::modelXavierInitialize()
{

    layers[0].xavierInitialization(arch[0]);

    for (size_t i = 1; i < archSize; ++i)
    {
        layers[i].xavierInitialization(arch[i]);
    }
}

/******************************************************************************
 * @brief Setter for learning rate
 *
 * @param val Value of learining rate
 *
 * @return Nothing
 ******************************************************************************/
void Model::setLearningRate(double val)
{
    this->learningRate = val;
}

/******************************************************************************
 * @brief Setter for epsilon
 *
 * @param val Value of epsilon
 *
 * @return Nothing
 ******************************************************************************/
void Model::setEps(double val)
{
    this->eps = val;
}

/******************************************************************************
 * @brief Calculates current cost using cross entropy method
 *
 * @return Model cost
 ******************************************************************************/
double Model::costCrossEntropy()
{
    double totalCost = 0;
    for (size_t i = 0; i < trainingData.numOfSamples; ++i)
    {

        FastMatrix result = run(trainingData.inputs[i]);

        for (size_t j = 0; j < result.cols; ++j)
        {

            double d = std::log(MAT_ACCESS(result, 0, j)) * (MAT_ACCESS(trainingData.outputs[i], 0, j));
            totalCost -= d;
        }
    }

    COND_LOG(trainingData.numOfSamples == 0, ERROR_TYPE, "NUMBER OF SAMPLES == 0");

    return totalCost / (trainingData.numOfSamples);
}

/******************************************************************************
 * @brief Calculates current cost using mean squared error method
 *
 * @return Model cost
 ******************************************************************************/
double Model::costMeanSquare()
{
    LOG(NORMAL_LOGS, INFO_TYPE, "CALCULATING MEAN SQUARE COST");
    double totalCost = 0;

    for (size_t i = 0; i < trainingData.numOfSamples; ++i)
    {
        FastMatrix result = run(trainingData.inputs[i]);
        for (size_t j = 0; j < result.cols; ++j)
        {
            double d = MAT_ACCESS(result, 0, j) - MAT_ACCESS(trainingData.outputs[i], 0, j);
            totalCost += d * d;
        }
    }

    COND_LOG(trainingData.numOfSamples == 0, ERROR_TYPE, "NUMBER OF SAMPLES == 0");

    return totalCost / (trainingData.numOfSamples);
}

/******************************************************************************
 * @brief Learning process for a model. Runs back propagation
 *        algorithm @iterations number of times
 *
 * @param trainingDataIn    Data on which the training should be performed
 * @param iterations        How many iterations of training
 * @param clipGradient      Tells if gradient clipping should be used
 * @param batchSize         Size of single training batch
 *
 * @return Nohing
 ******************************************************************************/
void Model::learn(TrainingData &trainingDataIn, size_t iterations, bool clipGradient, uint32_t batchSize)
{
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "STARTING LEARNING");
    this->trainingData = trainingDataIn;
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "NUMBER OF SAMPLES: " << trainingData.numOfSamples);
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "INITIAL MODEL: " << *this);
    double costBeforeLearning = costMeanSquare();
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "COST BEFORE LEARNING: " << costBeforeLearning);
    for (size_t i = 0; i < iterations; ++i)
    {
        backPropagation(clipGradient, batchSize);
    }
    double costAfterLearing = costMeanSquare();
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "COST AFTER LEARNING: " << costAfterLearing);
}

/******************************************************************************
 * @brief Forward feed model with given input
 *
 * @param input Input for the model
 *
 * @return Model output
 ******************************************************************************/
FastMatrix Model::run(FastMatrix input)
{

    this->layers[0].output = input;
    for (size_t i = 1; i < numberOfLayers; i++)
    {
        layers[i].forward(layers[i - 1].output);
    }
    return this->layers[this->numberOfLayers - 1].output;
}

/******************************************************************************
 * @brief Learning method using finite differnce algorithm. Calculates rough
 *        derivative of each weight/bias using formula:
 *        derivative = (f(x + eps) - f(x)) / eps
 *
 * @return Model output
 ******************************************************************************/
void Model::finiteDifference()
{

    Model fakeGradient(arch, archSize, activationFunctions, archSize, false);

    double saved;
    double curCost = costCrossEntropy();

    for (size_t i = 0; i < numberOfLayers; ++i)
    {

        for (size_t j = 0; j < layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].weights.cols; k++)
            {
                saved = MAT_ACCESS(layers[i].weights, j, k);
                MAT_ACCESS(layers[i].weights, j, k) += this->eps;
                double newCost = costCrossEntropy();
                MAT_ACCESS(fakeGradient.layers[i].weights, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].weights, j, k) = saved;
            }
        }

        for (size_t j = 0; j < layers[i].biases.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].biases.cols; k++)
            {
                saved = MAT_ACCESS(layers[i].biases, j, k);
                MAT_ACCESS(layers[i].biases, j, k) += this->eps;
                double newCost = costCrossEntropy();
                MAT_ACCESS(fakeGradient.layers[i].biases, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].biases, j, k) = saved;
            }
        }
    }
    for (size_t i = 0; i < this->numberOfLayers; ++i)
    {

        for (size_t j = 0; j < this->layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < this->layers[i].weights.cols; ++k)
            {
                MAT_ACCESS(layers[i].weights, j, k) -= learningRate * MAT_ACCESS(fakeGradient.layers[i].weights, j, k);
            }
        }

        for (size_t j = 0; j < this->layers[i].biases.rows; ++j)
        {
            for (size_t k = 0; k < this->layers[i].biases.cols; ++k)
            {
                MAT_ACCESS(layers[i].biases, j, k) -= learningRate * MAT_ACCESS(fakeGradient.layers[i].biases, j, k);
            }
        }
    }
}
/******************************************************************************
 * @brief Gradient clipping. If gradient for given weight/bias is greater than
 *        @maxThreshold or lesser than @minThreshold then this gradient becomes
 *        one of those values accordingly. Used to fight exploding gradient
 *
 * @return Nothing
 ******************************************************************************/
void Model::clipValues()
{
    for (size_t i = 0; i < numberOfLayers; ++i)
    {
        for (size_t j = 0; j < layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].weights.cols; ++k)
            {
                if (MAT_ACCESS(layers[i].weights, j, k) > maxThreshold)
                    MAT_ACCESS(layers[i].weights, j, k) = maxThreshold;
                if (MAT_ACCESS(layers[i].weights, j, k) < minThreshold)
                    MAT_ACCESS(layers[i].weights, j, k) = minThreshold;
            }
        }
        for (size_t j = 0; j < layers[i].biases.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].biases.cols; ++k)
            {
                if (MAT_ACCESS(layers[i].biases, j, k) > maxThreshold)
                    MAT_ACCESS(layers[i].biases, j, k) = maxThreshold;
                if (MAT_ACCESS(layers[i].biases, j, k) < minThreshold)
                    MAT_ACCESS(layers[i].biases, j, k) = minThreshold;
            }
        }
    }
}

/******************************************************************************
 * @brief Implementation of back propagation algorithm for mean squared error
 *        cost function.
 *
 * @param clipGradient Tells if gradient clipping should be used
 * @param batchSize    Size of one batch. If 0 then all samples are taken
 *
 * @return Nothing
 ******************************************************************************/
void Model::backPropagation(bool clipGradient, uint32_t batchSize)
{
    size_t n = batchSize == 0 ? trainingData.numOfSamples : batchSize;
    Model gradient(arch, archSize, activationFunctions, archSize, false);

    for (size_t i = 1; i < gradient.numberOfLayers; ++i)
    {
        gradient.layers[i].weights.set(0.0);
        gradient.layers[i].biases.set(0.0);
        gradient.layers[i].output.set(0.0);
    }

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i)
    {
        size_t currentIndex = (size_t)rand() % trainingData.numOfSamples;
        FastMatrix tmp{run(trainingData.inputs[currentIndex])};

        for (size_t j = 0; j < numberOfLayers; ++j)
        {
            gradient.layers[j].output.set(0.0);
        }

        // std::cout << "OUTPUT SIZE: " << trainingData.outputSize << "\n";

        for (size_t j = 0; j < trainingData.outputSize; ++j)
        {
            // std::cout << "OUTPUT: " << MAT_ACCESS(trainingData.outputs[currentIndex], 0, j) << "\n";
            MAT_ACCESS(gradient.layers[numberOfLayers - 1].output, 0, j) =
                (double)((double)MAT_ACCESS(layers[numberOfLayers - 1].output, 0, j) - (double)MAT_ACCESS(trainingData.outputs[currentIndex], 0, j));
            // std::cout << "MODEL OUTPUT: " << MAT_ACCESS(layers[numberOfLayers - 1].output, 0, j) << "\n";
        }

        for (size_t l = numberOfLayers - 1; l > 0; --l)
        {
            // std::cout << "LAYER: " << l << " OUTPUT COLS: " << layers[l].output.cols << "\n";
            for (size_t j = 0; j < layers[l].output.cols; ++j)
            {
                double a = MAT_ACCESS(layers[l].output, 0, j);
                double da = MAT_ACCESS(gradient.layers[l].output, 0, j);
                double actFuncDerivative = Layer::activationFunctionDerivative(a, layers[l].functionType);
                MAT_ACCESS(gradient.layers[l].biases, 0, j) += da * actFuncDerivative;
                // std::cout << "LAYER: " << l-1 << " OUTPUT COLS: " << layers[l-1].output.cols << "\n";
                for (size_t k = 0; k < layers[l - 1].output.cols; ++k)
                {
                    // j - weight matrix col
                    // k - weight matrix row
                    double pa = MAT_ACCESS(layers[l - 1].output, 0, k);
                    double w = MAT_ACCESS(layers[l].weights, k, j);

                    if (activationFunctions[l] == SOFTMAX)
                    {
                        actFuncDerivative = a * ((j == k) - pa);
                    }
                    MAT_ACCESS(gradient.layers[l].weights, k, j) += da * actFuncDerivative * pa;
                    MAT_ACCESS(gradient.layers[l - 1].output, 0, k) += da * actFuncDerivative * w;
                }
            }
        }
    }

    for (size_t i = 0; i < numberOfLayers; ++i)
    {
        for (size_t j = 0; j < gradient.layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < gradient.layers[i].weights.cols; ++k)
            {
                MAT_ACCESS(gradient.layers[i].weights, j, k) /= n;
            }
        }
        for (size_t j = 0; j < gradient.layers[i].biases.rows; ++j)
        {
            for (size_t k = 0; k < gradient.layers[i].biases.cols; ++k)
            {
                MAT_ACCESS(gradient.layers[i].biases, j, k) /= n;
            }
        }
    }

    if (clipGradient)
        gradient.clipValues();

    for (size_t i = 0; i < numberOfLayers; ++i)
    {

        for (size_t j = 0; j < layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].weights.cols; ++k)
            {
                // std::cout << "CURRENT GRADIENT: " << MAT_ACCESS(gradient.layers[i].weights, j, k) << "\n";
                MAT_ACCESS(layers[i].weights, j, k) -= learningRate * MAT_ACCESS(gradient.layers[i].weights, j, k);
            }
        }

        for (size_t j = 0; j < this->layers[i].biases.rows; ++j)
        {
            for (size_t k = 0; k < this->layers[i].biases.cols; ++k)
            {
                MAT_ACCESS(layers[i].biases, j, k) -= learningRate * MAT_ACCESS(gradient.layers[i].biases, j, k);
            }
        }
    }
}
