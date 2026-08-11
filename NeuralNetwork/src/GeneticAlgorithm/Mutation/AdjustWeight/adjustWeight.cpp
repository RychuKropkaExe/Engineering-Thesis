#include "adjustWeight.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

AdjustWeight::AdjustWeight(size_t numberOfAdjustments, double adjustmentRate)
{
  this->numberOfAdjustments = numberOfAdjustments;
  this->adjustmentRate = adjustmentRate;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void AdjustWeight::mutateIndividual(Individual &individual)
{
  size_t numberOfWeights = 0;

  Model &genotype = individual.genotype;

  for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
  {

    size_t rowsInLayer = genotype.layers[layerIndex].weights.rows;
    size_t colsInLayer = genotype.layers[layerIndex].weights.cols;

    numberOfWeights += rowsInLayer * colsInLayer;
  }
  for (size_t flipIndex = 0; flipIndex < numberOfAdjustments; flipIndex++)
  {
    size_t adjustmentWeightIndex = rand() % numberOfWeights;

    for (size_t layerIndex = 0; layerIndex < genotype.archSize; layerIndex++)
    {

      size_t rowsInLayer = genotype.layers[layerIndex].weights.rows;
      size_t colsInLayer = genotype.layers[layerIndex].weights.cols;

      if (adjustmentWeightIndex - (rowsInLayer * colsInLayer) <= 0)
      {
        size_t weightIndex = adjustmentWeightIndex - 1;

        int sign = (rand() % 2) == 0 ? -1 : 1;

        double elementValue = genotype.layers[layerIndex].weights.mat[weightIndex];
        genotype.layers[layerIndex].weights.mat[weightIndex] = elementValue * adjustmentRate * sign;
      }
    }
  }
}
