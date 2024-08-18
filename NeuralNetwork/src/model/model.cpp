#include "model.h"
#include "logger.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

//======================= CONSTRUCTORS ==========================================

Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize)
{
    assert(archSize == actFunctionsSize);
    this->activationFunctions.resize(actFunctionsSize);
    for (size_t i = 0; i < actFunctionsSize; ++i)
    {
        this->activationFunctions[i] = actFunctions[i];
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

Model::Model()
{
    this->layers.resize(1);
    this->activationFunctions.resize(1);
}

//======================= INITIALIZATION ==========================================

void Model::modelXavierInitialize()
{

    layers[0].xavierInitialization(arch[0]);

    for (size_t i = 1; i < archSize; ++i)
    {
        layers[i].xavierInitialization(arch[i]);
    }
}

//======================= HYPERPARAMETERS SETTINGS ==========================================

void Model::setLearningRate(double val)
{
    this->learningRate = val;
}

void Model::setEps(double val)
{
    this->eps = val;
}

//======================= COST FUNCTIONS ==========================================

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

    COND_LOG(trainingData.numOfSamples == 0, ERROR_LEVEL, "NUMBER OF SAMPLES == 0");

    return totalCost / (trainingData.numOfSamples);
}

double Model::costMeanSquare()
{
    LOG(INFO_LEVEL, "CALCULATING MEAN SQUARE COST");
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

    COND_LOG(trainingData.numOfSamples == 0, ERROR_LEVEL, "NUMBER OF SAMPLES == 0");

    return totalCost / (trainingData.numOfSamples);
}

void Model::learn(TrainingData &trainingDataIn, size_t iterations, bool clipGradient, uint32_t batchSize)
{
    LOG(INFO_LEVEL, "STARTING LEARNING");
    this->trainingData = trainingDataIn;
    LOG(INFO_LEVEL, "NUMBER OF SAMPLES: " << trainingData.numOfSamples);
    LOG(INFO_LEVEL, "INITIAL MODEL: " << *this);
    double costBeforeLearning = costMeanSquare();
    LOG(INFO_LEVEL, "COST BEFORE LEARNING: " << costBeforeLearning);
    for (size_t i = 0; i < iterations; ++i)
    {
        backPropagation(clipGradient, batchSize);
    }
    double costAfterLearing = costMeanSquare();
    LOG(INFO_LEVEL, "COST AFTER LEARNING: " << costAfterLearing);
}

FastMatrix Model::run(FastMatrix input)
{

    this->layers[0].output = input;
    for (size_t i = 1; i < numberOfLayers; i++)
    {
        layers[i].forward(layers[i - 1].output);
    }
    return this->layers[this->numberOfLayers - 1].output;
}

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

inline uint32_t Model::random_u32(uint32_t prev)
{
    return prev * 1664525U + 1013904223U; // assuming complement-2 integers and non-signaling overflow
}

void Model::printModelToFile(std::string filename)
{
    std::ofstream f(filename);

    // Write to the file
    f << numberOfLayers << "\n";
    f << learningRate << "\n";
    f << eps << "\n";
    for (size_t i = 0; i < numberOfLayers; ++i)
    {

        f << layers[i].functionType << "\n";
        f << layers[i].weights.rows << "\n";
        f << layers[i].weights.cols << "\n";
        for (size_t j = 0; j < layers[i].weights.rows * layers[i].weights.cols; ++j)
        {
            f << layers[i].weights.mat[j] << " ";
        }
        f << "\n";
        f << layers[i].biases.rows << "\n";
        f << layers[i].biases.cols << "\n";
        for (size_t j = 0; j < layers[i].biases.rows * layers[i].biases.cols; ++j)
        {
            f << layers[i].biases.mat[j] << " ";
        }
        f << "\n";
    }

    // Close the file
    f.close();
}

Model parseModelFromFile(std::string filename)
{
    std::ifstream f(filename);
    std::string buffer;

    getline(f, buffer);
    size_t numberOfLayers = (size_t)stoi(buffer);

    getline(f, buffer);
    double learningRate = (double)stof(buffer);

    getline(f, buffer);
    double eps = (double)stof(buffer);

    vector<Layer> layers;
    layers.resize(numberOfLayers);
    for (size_t i = 0; i < numberOfLayers; ++i)
    {

        getline(f, buffer);
        ActivationFunctionE funcType = (ActivationFunctionE)stoi(buffer);
        getline(f, buffer);
        size_t weightsRows = (size_t)stoi(buffer);
        getline(f, buffer);
        size_t weightsCols = (size_t)stoi(buffer);
        vector<double> wieghtsMat;
        wieghtsMat.resize(weightsRows * weightsCols);
        getline(f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        while (getline(check, buffer, ' '))
        {
            wieghtsMat[counter] = stof(buffer);
            counter++;
        }

        getline(f, buffer);
        size_t biasesRows = (size_t)stoi(buffer);
        getline(f, buffer);
        size_t biasesCols = (size_t)stoi(buffer);
        vector<double> biasesMat;
        biasesMat.resize(biasesCols * biasesRows);

        counter = 0;
        getline(f, buffer);
        std::cout << buffer << "\n";

        std::stringstream check2(buffer);
        while (getline(check2, buffer, ' '))
        {
            std::cout << buffer << "\n";
            biasesMat[counter] = stof(buffer);
            counter++;
        }

        FastMatrix weights;
        FastMatrix biases;
        FastMatrix output;
        weights.cols = weightsCols;
        weights.rows = weightsRows;
        weights.mat = wieghtsMat;
        biases.cols = biasesCols;
        biases.rows = biasesRows;
        biases.mat = biasesMat;
        output.cols = biasesCols;
        output.rows = biasesRows;
        output.mat = biasesMat;

        Layer l;

        l.weights = weights;
        l.biases = biases;
        l.output = output;
        l.functionType = funcType;

        layers[i] = l;
    }

    Model model;

    model.layers = layers;
    model.numberOfLayers = numberOfLayers;
    model.archSize = numberOfLayers;
    model.learningRate = learningRate;
    model.eps = eps;

    return model;
}
