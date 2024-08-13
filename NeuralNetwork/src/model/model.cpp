#include "model.h"
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
    // std::cout << "COST FUNCTION!!! " << "\n";
    // std::cout << "NUMBER OF SAMPLES: " << "\n";
    // std::cout << trainingData.numOfSamples << "\n";
    // trainingData.printTrainingData();
    for (size_t i = 0; i < trainingData.numOfSamples; ++i)
    {

        FastMatrix result = run(trainingData.inputs[i]);

        for (size_t j = 0; j < result.cols; ++j)
        {

            double d = std::log(MAT_ACCESS(result, 0, j)) * (MAT_ACCESS(trainingData.outputs[i], 0, j));
            totalCost -= d;
        }
    }

    if (trainingData.numOfSamples == 0)
    {
        std::cout << "ERROR, NUMBER OF SAMPLES == 0" << "\n";
    }

    return totalCost / (trainingData.numOfSamples);
}

double Model::costMeanSquare()
{

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

    if (trainingData.numOfSamples == 0)
    {
        std::cout << "ERROR, NUMBER OF SAMPLES == 0" << "\n";
        return -1;
    }

    return totalCost / (trainingData.numOfSamples);
}

void Model::learn(TrainingData &trainingDataIn, size_t iterations, bool clipGradient)
{

    // std::cout << "STARTED LEARNING" << "\n";
    this->trainingData = trainingDataIn;
    std::cout << "NUMBER OF SAMPLES: " << this->trainingData.numOfSamples << "\n";
    std::cout << "COST BEFORE LEARNING: " << costMeanSquare() << "\n";
    // printModel();
    for (size_t i = 0; i < iterations; ++i)
    {

        backPropagation(clipGradient);
    }

    std::cout << "COST AFTER LEARNING: " << costMeanSquare() << "\n";
    // printModel();
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

void Model::backPropagation(bool clipGradient)
{
    size_t n = trainingData.numOfSamples;
    // std::cout << "NUMBER OF SAMPLES IN BACK PROPAGATION: " << n << "\n";
    Model gradient(arch, archSize, activationFunctions, archSize, false);

    // std::cout << "ARCH SIZE: " << archSize << "\n";
    // for(size_t i = 0; i < archSize; ++i){
    //     std::cout << "ARCH: " << i << " " << arch[i] << "\n";
    // }

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

        run(trainingData.inputs[i]);

        // std::cout << "NUMBER OF LAYERS: " << numberOfLayers << "\n";
        for (size_t j = 0; j < numberOfLayers; ++j)
        {
            gradient.layers[j].output.set(0.0);
        }

        // std::cout << "OUTPUT SIZE: " << trainingData.outputSize << "\n";

        for (size_t j = 0; j < trainingData.outputSize; ++j)
        {
            MAT_ACCESS(gradient.layers[numberOfLayers - 1].output, 0, j) = MAT_ACCESS(layers[numberOfLayers - 1].output, 0, j) - MAT_ACCESS(trainingData.outputs[i], 0, j);
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
                        actFuncDerivative = a * ((j == k) - w);
                    }
                    MAT_ACCESS(gradient.layers[l].weights, k, j) += da * actFuncDerivative * pa;
                    MAT_ACCESS(gradient.layers[l - 1].output, 0, k) += da * actFuncDerivative * w;
                }
            }
        }
    }

    // gradient.printModel();

    // std::cout << "NUMBER WHICH DIVIDES: " << n << "\n";

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

    // std::cout << "APPLYING GRADIENT: " << numberOfLayers << "\n";
    if (clipGradient)
        gradient.clipValues();
    // gradient.printModel();
    for (size_t i = 0; i < numberOfLayers; ++i)
    {

        for (size_t j = 0; j < layers[i].weights.rows; ++j)
        {
            for (size_t k = 0; k < layers[i].weights.cols; ++k)
            {
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

    // std::cout << "BACK PROPAGATION APPLIED " << "\n";
}

void Model::printModel()
{

    std::cout << "MODEL PARAMETERS: " << "\n";
    std::cout << "LEARNING RATE: " << learningRate << "\n";
    std::cout << "EPS: " << eps << "\n";
    std::cout << "NUMBER OF LAYERS: " << numberOfLayers << "\n";
    std::cout << "ACTIVATION FUNCTIONS: " << activationFunctions[0] << "\n";
    std::cout << "LAYERS: " << "\n";

    for (size_t i = 0; i < numberOfLayers; ++i)
    {
        std::cout << "LAYER NUMBER: " << i << "\n";
        std::cout << "WEIGHTS: " << "\n";
        printFastMatrix(layers[i].weights);
        std::cout << "BIASES: " << "\n";
        printFastMatrix(layers[i].biases);
        std::cout << "INTERMIEDIATE: " << "\n";
        printFastMatrix(layers[i].output);
        std::cout << "ACTIVATION FUNCTION: " << layers[i].functionType << "\n";
    }
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
