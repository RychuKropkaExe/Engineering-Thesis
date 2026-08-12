#include "flipBias.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

FlipBias::FlipBias(size_t numberOfFlips)
{
  this->numberOfFlips = numberOfFlips;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void FlipBias::mutateIndividual(Individual &individual)
{
  size_t numberOfBiases = 0;

  Model &genotype = individual.genotype;

  for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
  {

    size_t rowsInLayer = genotype.layers[layerIndex].biases.rows;
    size_t colsInLayer = genotype.layers[layerIndex].biases.cols;

    numberOfBiases += rowsInLayer * colsInLayer;
  }
  for (size_t flipIndex = 0; flipIndex < numberOfFlips; flipIndex++)
  {
    size_t flipBiasIndex = rand() % numberOfBiases;

    for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
    {

      size_t rowsInLayer = genotype.layers[layerIndex].biases.rows;
      size_t colsInLayer = genotype.layers[layerIndex].biases.cols;

      if (flipBiasIndex - (rowsInLayer * colsInLayer) <= 0)
      {
        size_t weightIndex = flipBiasIndex - 1;

        double elementValue = genotype.layers[layerIndex].biases.mat[weightIndex];

        if (elementValue == 0)
        {
          double newValue = randomdouble();
          genotype.layers[layerIndex].biases.mat[weightIndex] = newValue;
        }
        else
        {
          genotype.layers[layerIndex].biases.mat[weightIndex] = 0;
        }
      }
    }
  }
}
