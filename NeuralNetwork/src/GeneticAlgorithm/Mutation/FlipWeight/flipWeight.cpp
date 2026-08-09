#include "flipWeight.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

FlipWeight::FlipWeight(size_t numberOfFlips)
{
  this->numberOfFlips = numberOfFlips;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void FlipWeight::mutateIndividual(Individual &individual)
{
  size_t numberOfWeights = 0;

  Model &genotype = individual.genotype;

  for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
  {

    size_t rowsInLayer = genotype.layers[layerIndex].weights.rows;
    size_t colsInLayer = genotype.layers[layerIndex].weights.cols;

    numberOfWeights += rowsInLayer * colsInLayer;
  }
  for (size_t flipIndex = 0; flipIndex < numberOfFlips; flipIndex++)
  {
    size_t flipWeightIndex = rand() % numberOfWeights;

    for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
    {

      size_t rowsInLayer = genotype.layers[layerIndex].weights.rows;
      size_t colsInLayer = genotype.layers[layerIndex].weights.cols;

      if (flipWeightIndex - (rowsInLayer * colsInLayer) <= 0)
      {
        size_t weightIndex = flipWeightIndex - 1;

        double elementValue = genotype.layers[layerIndex].weights.mat[weightIndex];

        if (elementValue == 0)
        {
          double newValue = randomdouble();
          genotype.layers[layerIndex].weights.mat[weightIndex] = newValue;
        }
        else
        {
          genotype.layers[layerIndex].weights.mat[weightIndex] = 0;
        }
      }
    }
  }
}
