#include "rouletteSelection.h"
#include <algorithm>
#include <random>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

RouletteSelection::RouletteSelection(double scalingFactor)
{
  isScalingUsed = true;
  this->scalingFactor = scalingFactor;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

/******************************************************************************
 * @brief Roulette selection algorithm
 *
 * @param population            Current population
 * @param numberOfParentsPairs  How many parents are to be chosen for crossover
 *
 * @return Vector of paris of parents
 ******************************************************************************/
vector<Parents> RouletteSelection::selectParents(vector<Individual> &population, size_t numberOfParentsPairs)
{
  size_t populationSize = population.size();

  vector<Parents> parentsList{};

  constexpr size_t numberOfParents = 2;

  parentsList.resize(numberOfParentsPairs);

  // Sort in descending order, so the most fit individuals are at the beggining
  std::sort(population.begin(), population.end(), std::greater<Individual>());

  size_t initialIndex = fasttrackProtectedIndividuals(population, parentsList, 0);

  double fitnessSum = 0;

  for (auto individual : population)
  {
    fitnessSum += individual.fitness;
  }

  double fitnessMean = fitnessSum / population.size();

  double scaledFitnessSum = 0;
  double scaledMaxFitness = fitnessMean * scalingFactor;

  if (isScalingUsed)
  {
    for (auto individual : population)
    {
      scaledFitnessSum += (individual.fitness > scaledMaxFitness) ? scaledMaxFitness : individual.fitness;
    }
  }

  double maxRandValue = isScalingUsed ? scaledFitnessSum : fitnessSum;

  std::uniform_real_distribution<double> unif(0,
                                              maxRandValue);

  std::default_random_engine randomEngine;

  for (size_t index = initialIndex; index < numberOfParentsPairs; index++)
  {
    Parents parents;

    for (size_t parentIndex = 0; parentIndex < numberOfParents; parentIndex++)
    {
      double randomValue = unif(randomEngine);

      size_t chosenIndividualIndex = 0;

      for (size_t individualIndex = 0; individualIndex < populationSize; individualIndex++)
      {
        double individualFitness = population[individualIndex].fitness;

        if (isScalingUsed)
        {
          randomValue -= individualFitness > scaledMaxFitness ? scaledMaxFitness : individualFitness;
        }
        else
        {
          randomValue -= individualFitness;
        }

        if (randomValue < 0)
        {
          chosenIndividualIndex = individualIndex;
          break;
        }
      }

      parents.addParent(population[chosenIndividualIndex], chosenIndividualIndex);
    }
    parentsList[index] = parents;
  }

  return parentsList;
}
