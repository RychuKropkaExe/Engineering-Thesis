#include "tournamentSelection.h"

/******************************************************************************
 * @brief Tournament selection algorithms. Parents are chosen via competition
 *        between individuals, two randomly chosen are compared and one with
 *        better fitness is chosen as parent.
 *
 * @param population            Current population
 * @param numberOfParentsPairs  How many parents are to be chosen for crossover
 *
 * @return Vector of paris of parents
 ******************************************************************************/
vector<Parents> TournamentSelection::selectParents(vector<Individual> &population, size_t numberOfParentsPairs)
{

  vector<Parents> parentsList{};

  constexpr size_t numberOfParents = 2;

  parentsList.resize(numberOfParentsPairs);

  size_t populationSize = population.size();

  size_t initialIndex = fasttrackProtectedIndividuals(population, parentsList, 0);

  for (size_t index = initialIndex; index < numberOfParentsPairs; index++)
  {
    Parents parents;

    for (size_t parentIndex = 0; parentIndex < numberOfParents; parentIndex++)
    {
      size_t fighterOneIndex = rand() % populationSize;
      size_t fighterTwoIndex = rand() % populationSize;

      double fighterOneFitness = population[fighterOneIndex].fitness;
      double fighterTwoFitness = population[fighterTwoIndex].fitness;

      size_t winnerIndex = fighterOneFitness < fighterTwoFitness ? fighterTwoFitness : fighterOneFitness;

      parents.addParent(population[winnerIndex], winnerIndex);
    }

    parentsList[index] = parents;
  }

  return parentsList;
}
