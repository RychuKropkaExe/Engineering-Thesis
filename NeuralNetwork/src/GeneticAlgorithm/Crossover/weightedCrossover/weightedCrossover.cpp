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
void WeightedCrossover::produceOffspring(std::vector<Parents> &parents, vector<Individual> &population)
{
  size_t numberOfOffspring = parents.size();

  // LOG(ESSENTIAL_LOGS, INFO_TYPE, "POPULATION SIZE: " << population.size() << " NUMBER OF OFFSPRING: " << numberOfOffspring);

  for (size_t index = 0; index < numberOfOffspring; index++)
  {
    // LOG(ESSENTIAL_LOGS, INFO_TYPE, "OFFSPRING INDEX: " << index);
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

      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " secondParentNeuronsNum: " << firstParentNeuronsNum);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " secondParentNeuronsNum: " << secondParentNeuronsNum);

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

    for (size_t layerIndex = 1; layerIndex < firstParentGenotype.archSize;
         layerIndex++)
    {

      Layer &firstParLayer = firstParentGenotype.layers[layerIndex];

      Layer &secondParLayer = secondParentGenotype.layers[layerIndex];

      FastMatrix &offspringWeights = offspringModel.layers[layerIndex].weights;
      FastMatrix &offspringBiases = offspringModel.layers[layerIndex].biases;
      vector<ActivationFunctionE> &offspringAct = offspringModel.layers[layerIndex].functionTypes;

      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " firstParLayerRows: " << firstParLayer.weights.rows);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " firstParLayerCols: " << firstParLayer.weights.cols);

      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " secondParLayerRows: " << secondParLayer.weights.rows);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " secondParLayerCols: " << secondParLayer.weights.cols);

      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " offspringWeightsRows: " << offspringWeights.rows);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " offspringWeightsCols: " << offspringWeights.cols);
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " offspringActSize: " << offspringAct.size());

      for (size_t row = 0; row < offspringWeights.rows; row++)
      {
        for (size_t col = 0; col < offspringWeights.cols; col++)
        {

          // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " row: " << row);
          // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Layer: " << layerIndex << " col: " << col);

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

          // double newValue = ((firstParFitness * firstValue) + (secondParFitness * secondValue)) / (firstParFitness + secondParFitness);

          double newValue = 0;

          if (firstParFitness > secondParFitness)
          {
            newValue = firstValue;
          }
          else
          {
            newValue = secondValue;
          }

          offspringWeights.setElement(row, col, newValue);
        }
      }

      for (size_t col = 0; col < offspringBiases.cols; col++)
      {

        double firstValue = 0;
        double secondValue = 0;

        double firstParFitness = parents[index].firstParent.fitness;
        double secondParFitness = parents[index].secondParent.fitness;

        if (firstParLayer.biases.cols > col)
        {
          firstValue = firstParLayer.biases.getElement(0, col);
        }
        else
        {
          // If bias is not present in first parent, dont include it in weighted calculations
          firstParFitness = 0;
        }

        if (secondParLayer.biases.cols > col)
        {
          secondValue = secondParLayer.biases.getElement(0, col);
        }
        else
        {
          // If bias is not present in second parent, dont include it in weighted calculations
          secondParFitness = 0;
        }

        double newValue = ((firstParFitness * firstValue) + (secondParFitness * secondValue)) / (firstParFitness + secondParFitness);

        offspringBiases.setElement(0, col, newValue);

        // TODO: verify that is a corrent way to index activation functions
        if (firstParFitness >= secondParFitness && firstParFitness != 0)
        {
          offspringAct[col] = firstParLayer.functionTypes[col];
        }
        else if (firstParFitness < secondParFitness && secondParFitness != 0)
        {
          offspringAct[col] = secondParLayer.functionTypes[col];
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

    // LOG(ESSENTIAL_LOGS, INFO_TYPE, offspringModel);

    population[index] = Individual(offspringModel, gracePeriodLength);
  }
}
