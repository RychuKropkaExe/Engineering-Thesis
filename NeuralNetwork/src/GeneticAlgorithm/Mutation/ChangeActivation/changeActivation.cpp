#include "changeActivation.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

ChangeActivation::ChangeActivation(size_t numberOfChanges)
{
  this->numberOfChanges = numberOfChanges;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void ChangeActivation::mutateIndividual(Individual &individual)
{

  Model &genotype = individual.genotype;

  // archSize - 1 since we don't want to change activation functions of
  // output layers
  size_t numberOfLayers = genotype.archSize - 2;
  size_t minLayer = 1;

  for (size_t unused = 0; unused < numberOfChanges; unused++)
  {
    size_t layerIndex = minLayer % numberOfLayers;

    Layer &layer = genotype.layers[layerIndex];

    size_t numberOfActivations = layer.weights.cols;

    size_t changeIndex = rand() % numberOfActivations;

    ActivationFunctionE newActivation = (ActivationFunctionE)(rand() % ACTIVATION_COUNT);

    layer.functionTypes[changeIndex] = newActivation;
  }
}
