#include "dtmgeneticAlgorithm.h"
#include <cstdlib>

/******************************************************************************
 * @brief Selects parent pairs using tournament selection within each species
 *
 * Individuals still in their grace period are copied by selecting them as
 * both parents. The remaining parent pairs are selected by comparing the
 * fitness of individuals sampled from the corresponding species.
 *
 * @param species Lists of population indexes grouped into species
 *
 * @return Parent pairs grouped according to their species
 ******************************************************************************/
vector<vector<GAParents>> DTMGeneticAlgorithm::tournamentSelection(vector<vector<size_t>> species)
{

  vector<vector<GAParents>> parentsLists;

  for (size_t index = 0; index < species.size(); index++)
  {
    parentsLists.reserve(species[index].size());
  }

  // Fast forward protected individuals
  for (size_t index = 0; index < species.size(); index++)
  {
    for (auto individualIndex : species[index])
    {
      if (population[individualIndex].gracePeriodLength > 0)
      {
        // Pairs of parents with the same individual
        // are processed by just copying the individual to the next generation
        GAParents parents;
        parents.addParent(individualIndex);
        parents.addParent(individualIndex);
        parentsLists[index].push_back(parents);
        population[individualIndex].gracePeriodLength--;
      }
    }
  }


  for (size_t index = 0; index < parentsLists.size(); index++)
  {
    while(parentsLists[index].size() != parentsLists[index].capacity())
    {

      constexpr size_t numberOfParents = 2;

      GAParents parents;

      for (size_t _ = 0; _ < numberOfParents; _++)
      {

        vector<size_t> tournament;

        tournament.reserve(hyperparameters.tournamentSize);

        while (tournament.size() != tournament.capacity())
        {
          size_t individualIndex = rand() % species[index].size();
          tournament.push_back(individualIndex);
        }

        size_t bestIndividualIndex = 0;

        double bestFitness = 0;

        for (auto individualIndex : tournament)
        {
          if (population[individualIndex].fitness > bestFitness)
          {
            bestFitness = population[individualIndex].fitness;
            bestIndividualIndex = individualIndex;
          }
        }

        parents.addParent(bestIndividualIndex);

      }

      parentsLists[index].push_back(parents);


    }
  }

  return parentsLists;

}

