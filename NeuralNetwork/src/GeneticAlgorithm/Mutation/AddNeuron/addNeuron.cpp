#include "addNeuron.h"
#include <cassert>

/******************************************************************************
 * Local functions
 ******************************************************************************/

static void expandLayerRows(Layer &layer)
{

  size_t newWeightsRows = layer.weights.rows + 1;
  size_t newWeightsCols = layer.weights.cols;

  FastMatrix newWeights = FastMatrix(newWeightsRows, newWeightsCols);

  // While adding rows we can use std::copy since we use flat vector
  // to represent matrix. When adding a row to it we add it at the
  // end of the vector:
  // [1.1, 1.2, 1.3, 2.1, 2.2, 2.3] -> [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]
  std::copy(layer.weights.mat.begin(), layer.weights.mat.end(), newWeights.mat.begin());

  for (size_t col = 0; col < newWeights.cols; col++)
  {
    newWeights.setElement(newWeights.rows - 1, col, randomdouble());
  }

  layer.weights = newWeights;
}

static void expandLayerCols(Layer &layer)
{
  size_t newWeightsRows = layer.weights.rows;
  size_t newWeightsCols = layer.weights.cols + 1;

  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " max row: " << newWeightsRows);
  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " max col: " << newWeightsCols);
  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " existing row: " << layer.weights.rows);
  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " existing col: " << layer.weights.cols);

  FastMatrix newWeights = FastMatrix(newWeightsRows, newWeightsCols);

  // While adding columns we cant use std::copy since we use flat vector
  // to represent matrix. When adding a column to it we dont add it at the
  // end of the vector like in rows, we add one value at the end of each row
  // [1.1, 1.2, 1.3, 2.1, 2.2, 2.3] -> [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4]
  for (size_t row = 0; row < newWeights.rows; row++)
  {
    for (size_t col = 0; col < newWeights.cols - 1; col++)
    {
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, " row: " << row);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, " col: " << col);
      double value = layer.weights.getElement(row, col);
      newWeights.setElement(row, col, value);
    }
    newWeights.setElement(row, newWeights.cols - 1, randomdouble());
  }

  layer.weights = newWeights;

  // For layer before we need to also expand number of biases to include the additional neuron
  // in the next layer
  FastMatrix newBiases = FastMatrix(1, newWeightsCols);

  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " TUTAJ??: ");

  std::copy(layer.biases.mat.begin(), layer.biases.mat.end(), newBiases.mat.begin());

  // LOG(ESSENTIAL_LOGS, INFO_TYPE, " A MOŻE TUTAJ??: ");

  newBiases.setElement(0, newWeightsCols - 1, randomdouble());

  layer.biases = newBiases;

  vector<ActivationFunctionE> newActivations(newWeightsCols, NO_ACTIVATION);

  for (size_t index = 0; index < newWeightsCols - 1; index++)
  {
    newActivations[index] = layer.functionTypes[index];
  }

  newActivations[newWeightsCols - 1] = (ActivationFunctionE)(rand() % ACTIVATION_COUNT);

  layer.functionTypes = newActivations;

}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/
void AddNeuron::mutateIndividual(Individual &individual)
{
  Model &genotype = individual.genotype;

  //LOG(ESSENTIAL_LOGS, INFO_TYPE, "MODEL BEFORE: " << genotype);

  assert(genotype.archSize > 3);

  // We cant expand input and output layer
  size_t numberOfHiddenLayers = genotype.archSize - 2;

  size_t expandedLayerIndex = 2 + (rand() % numberOfHiddenLayers);

  Layer &expandedLayer = genotype.layers[expandedLayerIndex];

  expandLayerRows(expandedLayer);

  // We know its not input layer so its safe to subtract 1
  Layer &previousLayer = genotype.layers[expandedLayerIndex - 1];

  // We also need to expand number of weights in previous layer to account for the new neuron
  expandLayerCols(previousLayer);

  //LOG(ESSENTIAL_LOGS, INFO_TYPE, "MODEL AFTER: " << genotype);
}
