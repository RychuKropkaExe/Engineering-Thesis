#include "dtmgeneticAlgorithm.h"
#include <cassert>
#include <cstdlib>

/******************************************************************************
 * @brief Selects one offspring's parent pair per individual within each species
 *
 * Protected individuals first receive self-parent pairs. Remaining pairs use
 * tournaments sampled with replacement within their species. Grace periods
 * are decremented by crossover only, once per generation.
 *
 * @param species Lists of population indexes grouped into species
 *
 * @return Parent pairs grouped by species, preserving each species' size
 ******************************************************************************/
vector<vector<GAParents>> DTMGeneticAlgorithm::tournamentSelection(
    const vector<vector<size_t>> &species)
{
  assert(hyperparameters.tournamentSize > 0);
  vector<vector<GAParents>> parentsLists(species.size());

  for (size_t speciesIndex = 0; speciesIndex < species.size(); speciesIndex++)
  {
    const auto &members = species[speciesIndex];
    auto &parents = parentsLists[speciesIndex];
    parents.reserve(members.size());

    for (size_t individualIndex : members)
    {
      if (population.at(individualIndex).gracePeriodLength > 0)
      {
        GAParents protectedPair;
        protectedPair.addParent(individualIndex);
        protectedPair.addParent(individualIndex);
        parents.push_back(protectedPair);
      }
    }

    while (parents.size() < members.size())
    {
      GAParents pair;
      for (size_t parent = 0; parent < 2; parent++)
      {
        // A random offset belongs to the species list. Resolve it to the
        // population index before comparing fitness, including for the winner.
        size_t bestIndex = members[rand() % members.size()];
        for (size_t sample = 1; sample < hyperparameters.tournamentSize; sample++)
        {
          const size_t candidate = members[rand() % members.size()];
          if (population[candidate].fitness > population[bestIndex].fitness)
          {
            bestIndex = candidate;
          }
        }
        pair.addParent(bestIndex);
      }
      parents.push_back(pair);
    }
  }
  return parentsLists;
}
