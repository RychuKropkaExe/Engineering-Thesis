#include "dtmgeneticAlgorithm.h"

/******************************************************************************
 * @brief Groups population individuals by the maximum depth of their models
 *
 * @return Lists of population indexes grouped by model depth
 ******************************************************************************/
vector<vector<size_t>> DTMGeneticAlgorithm::divideIntoSpecies()
{

  size_t maxDepth = 0;

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    if (population[index].model.maxDepth > maxDepth)
    {
      maxDepth = population[index].model.maxDepth;
    }
  }

  const size_t bufferInterval = hyperparameters.populationSize / maxDepth;

  vector<vector<size_t>> speciesIndexes;

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    speciesIndexes[index].reserve(bufferInterval);
  }

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    size_t individualDepth = population[index].model.maxDepth;
    if (speciesIndexes[individualDepth].size() == speciesIndexes[individualDepth].capacity())
    {
      speciesIndexes[individualDepth].reserve(speciesIndexes[individualDepth].size() + bufferInterval);
    }

    speciesIndexes[individualDepth].push_back(index);
  }

  return speciesIndexes;
}

