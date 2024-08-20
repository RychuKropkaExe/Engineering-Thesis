#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "trainingData.h"
#include <vector>
using std::vector;

/******************************************************************************
 * @class Model
 *
 * @brief Implementation of Neural Network with all its functionalities
 *
 * @public @param layers                List of all layers in the model
 * @public @param numberOfLayers        Number of Layers in the model
 * @public @param activationFunctions   List of activation functions of the layers
 * @public @param arch                  A vector containing size of each layer. It
 *                                      is used for copying purposes.
 * @public @param archSize              Size of @arch vector
 * @public @param trainingData          Data used for learning process
 * @public @param learningRate          One of hyperparameters
 * @public @param eps                   One of hyperparameters
 * @public @param maxThreshold          Maximal value of gradient if clipping is used
 * @public @param minThreshold          Minimal value(negative) of gradient if clipping
 *                                      is used
 ******************************************************************************/
class Model
{

public:
    /******************************************************************************
     * CLASS MEMBERS
     ******************************************************************************/

    vector<Layer> layers;
    size_t numberOfLayers;
    vector<ActivationFunctionE> activationFunctions;

    vector<size_t> arch;
    size_t archSize;

    TrainingData trainingData{};
    double learningRate = 1e-8;
    double eps = 1e-3;

    double maxThreshold{0.1};
    double minThreshold{-0.1};

    /******************************************************************************
     * CONSTRUCTORS
     ******************************************************************************/

    Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize);
    Model();

    /******************************************************************************
     * OPERATORS
     ******************************************************************************/

    friend std::ostream &operator<<(std::ostream &os, const Model &dt);

    /******************************************************************************
     * UTILITIES
     ******************************************************************************/

    void modelXavierInitialize();

    FastMatrix run(FastMatrix input);

    double costMeanSquare();
    double costCrossEntropy();
    void finiteDifference();
    void backPropagation(bool clipGradientmz, uint32_t batchSize);

    void setLearningRate(double val);
    void setEps(double val);
    void learn(TrainingData &trainingDataIn, size_t iterations, bool clipGradient, uint32_t batchSize);

    void clipValues();
};

#endif
