#include "dtmgeneticAlgorithm.h"
#include <algorithm>

/******************************************************************************
 * @brief Groups population individuals by the maximum depth of their models
 *
 * Refreshes unsorted models before grouping. Empty depths remain empty species;
 * empty populations return no species. Population indexes, not individual IDs,
 * are stored so selection can access the owning individuals directly.
 *
 * @return Lists of population indexes grouped by model depth
 ******************************************************************************/
vector<vector<size_t>> DTMGeneticAlgorithm::divideIntoSpecies()
{
  if (population.empty())
  {
    return {};
  }

  size_t maximumDepth = 0;
  for (DTIndividual &individual : population)
  {
    if (!individual.model.isSorted)
    {
      individual.model.sortTopologically();
    }
    maximumDepth = std::max(maximumDepth, individual.model.maxDepth);
  }

  vector<vector<size_t>> species(maximumDepth + 1);
  vector<size_t> speciesSizes(species.size(), 0);
  for (const DTIndividual &individual : population)
  {
    speciesSizes[individual.model.maxDepth]++;
  }
  for (size_t depth = 0; depth < species.size(); depth++)
  {
    species[depth].reserve(speciesSizes[depth]);
  }
  for (size_t index = 0; index < population.size(); index++)
  {
    species[population[index].model.maxDepth].push_back(index);
  }
  return species;
}
