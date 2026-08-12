#include "weightedCrossover.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

WeightedCrossover::WeightedCrossover(double scalingFactor)
{
  this->scalingFactor = scalingFactor;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/
std::vector<Individual> WeightedCrossover::produceOffspring(std::vector<Parents> &parents)
{
  size_t numberOfOffspring = parents.size();

  std::vector<Individual> offspringList{};

  offspringList.resize(numberOfOffspring);

  for (size_t index = 0; index < numberOfOffspring; index++)
  {

    Model &firstParentGenotype = parents[index].firstParent.genotype;
    Model &secondParentGenotype = parents[index].secondParent.genotype;

    vector<size_t> offspringArch{};
    vector<ActivationFunctionE> actFunctions{};

    offspringArch.resize(firstParentGenotype.archSize);
    actFunctions.resize(firstParentGenotype.archSize);

    for (size_t layerIndex = 0; layerIndex < firstParentGenotype.archSize;
         layerIndex++)
    {
      size_t firstParentNeuronsNum = firstParentGenotype.arch[layerIndex];
      size_t secondParentNeuronsNum = secondParentGenotype.arch[layerIndex];

      if (firstParentNeuronsNum > secondParentNeuronsNum)
      {
        offspringArch[layerIndex] = firstParentNeuronsNum;
      }
      else
      {
        offspringArch[layerIndex] = secondParentNeuronsNum;
      }
      // Activation functions will be set alongside weights later.
      actFunctions[layerIndex] = NO_ACTIVATION;
    }

    Model offspringModel(offspringArch, actFunctions, false);

    for (size_t layerIndex = 0; layerIndex < firstParentGenotype.archSize;
         layerIndex++)
    {

      Layer &firstParLayer = firstParentGenotype.layers[layerIndex];

      Layer &secondParLayer = secondParentGenotype.layers[layerIndex];

      FastMatrix &offspringWeights = offspringModel.layers[layerIndex].weights;
      FastMatrix &offspringBiases = offspringModel.layers[layerIndex].biases;

      for (size_t row = 0; row < offspringWeights.rows; row++)
      {
        for (size_t col = 0; col < offspringWeights.cols; col++)
        {

          double firstValue = 0;
          double secondValue = 0;

          double firstParFitness = parents[index].firstParent.fitness;
          double secondParFitness = parents[index].secondParent.fitness;

          if (firstParLayer.weights.cols > col && firstParLayer.weights.rows > row)
          {
            firstValue = firstParLayer.weights.getElement(row, col);
          }
          else
          {
            // If weight is not present in first parent, dont include it in weighted calculations
            firstParFitness = 0;
          }

          if (secondParLayer.weights.cols > col && secondParLayer.weights.rows > row)
          {
            secondValue = secondParLayer.weights.getElement(row, col);
          }
          else
          {
            // If weight is not present in second parent, dont include it in weighted calculations
            secondParFitness = 0;
          }

          double newValue = ((firstParFitness * firstValue) + (secondParFitness * secondValue)) / (firstParFitness + secondParFitness);

          offspringWeights.setElement(row, col, newValue);

          // TODO: verify that is a corrent way to index activation functions
          if (firstParFitness >= secondParFitness && firstParFitness != 0)
          {
            actFunctions[layerIndex] = firstParLayer.functionTypes[col];
          }
          else if (firstParFitness < secondParFitness && secondParFitness != 0)
          {
            actFunctions[layerIndex] = secondParLayer.functionTypes[col];
          }
        }

        for (size_t col = 0; col < offspringBiases.cols; col++)
        {

          double firstValue = 0;
          double secondValue = 0;

          double firstParFitness = parents[index].firstParent.fitness;
          double secondParFitness = parents[index].secondParent.fitness;

          if (firstParLayer.biases.cols > col && firstParLayer.biases.rows > row)
          {
            firstValue = firstParLayer.biases.getElement(row, col);
          }
          else
          {
            // If bias is not present in first parent, dont include it in weighted calculations
            firstParFitness = 0;
          }

          if (secondParLayer.biases.cols > col && secondParLayer.biases.rows > row)
          {
            secondValue = secondParLayer.biases.getElement(row, col);
          }
          else
          {
            // If bias is not present in second parent, dont include it in weighted calculations
            secondParFitness = 0;
          }

          double newValue = ((firstParFitness * firstValue) + (secondParFitness * secondValue)) / (firstParFitness + secondParFitness);

          offspringBiases.setElement(row, col, newValue);
        }
      }
    }

    size_t gracePeriodLength = parents[index].firstParent.gracePeriodLength;

    if (parents[index].secondParent.gracePeriodLength > gracePeriodLength)
    {
      gracePeriodLength = parents[index].secondParent.gracePeriodLength;
    }

    if (gracePeriodLength > 0)
    {
      gracePeriodLength--;
    }

    offspringList[index] = Individual(offspringModel, gracePeriodLength);
  }

  return offspringList;
}
