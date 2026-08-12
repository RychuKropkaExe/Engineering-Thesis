#include "selectionAlgorithm.h"

/******************************************************************************
 * @brief Adds protected individuals to parents list regardless of their fitness
 *
 * @param population Current population
 * @param parents    Reference to parents list
 * @param startIndex Start index for where pair should be inserted into parents list
 *
 * @return Current index of last non filled entry in parents list
 ******************************************************************************/
size_t SelectionAlgorithm::fasttrackProtectedIndividuals(vector<Individual> &population, vector<Parents> &parents, size_t startIndex)
{
  size_t populationSize = population.size();
  size_t parentsSize = parents.size();

  size_t currentParentsPairIndex = startIndex;

  for (size_t index = 0; index < populationSize; index++)
  {
    if (population[index].gracePeriodLength > 0)
    {
      size_t secondParentIndex = rand() % populationSize;

      Individual firstParent = population[index];
      Individual secondParent = population[secondParentIndex];

      Parents parentsPair = Parents(firstParent, index, secondParent, secondParentIndex);

      parents[currentParentsPairIndex] = parentsPair;

      currentParentsPairIndex++;
    }

    if (currentParentsPairIndex == parentsSize)
    {
      LOG(ESSENTIAL_LOGS, ERROR_TYPE, "TOO MANY FASTTRACKED PROTECTED INDIVIDUALS");
    }
  }

  return currentParentsPairIndex;
}
