#pragma once

#include "crossoverAlgorithm.h"
#include "fitnessEvaluation.h"
#include "individual.h"
#include "model.h"
#include "mutationAlgorithm.h"
#include "selectionAlgorithm.h"

/******************************************************************************
 * @class GeneticAlgorithm
 *
 * @brief Handling of all data and processing regarding genetic algortihm
 *
 * @public @param population          Container for population
 * @public @param selectionAlgorithm  Algorithm for selecting parents for next population
 * @public @param crossoverAlgorithm  Algorithm for creating new individuals
 * @public @param fitnessEvaluation   Algorithm for evaluating individual fitness
 * @public @param mutationAlgorithm   Algorithm for mutating new individuals
 ******************************************************************************/
class GeneticAlgorithm
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  std::vector<Individual> population;

  SelectionAlgorithm *selectionAlgorithm;
  CrossoverAlgorithm *crossoverAlgorithm;
  FitnessEvaluation *fitnessEvaluation;
  vector<MutationAlgorithm *> mutationAlgorithms;

  vector<size_t> mutationRates;
  size_t gracePeriodLength;
  size_t populationSize;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  GeneticAlgorithm(SelectionAlgorithm *selectionAlgorithm,
                   CrossoverAlgorithm *crossoverAlgorithm,
                   FitnessEvaluation *fitnessEvaluation,
                   vector<MutationAlgorithm *> mutationAlgorithms,
                   vector<size_t> mutationRates,
                   size_t gracePeriodLength);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void initializePopulation(size_t populationSize, Model limit);
  Individual runGeneticAlgorithm(size_t numberOfIterations);

  /******************************************************************************
   * OPERATORS
   ******************************************************************************/
  friend std::ostream &
  operator<<(std::ostream &os, const GeneticAlgorithm &dt);
};
