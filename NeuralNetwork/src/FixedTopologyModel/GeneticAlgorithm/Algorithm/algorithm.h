#pragma once

#include "individual.h"
#include "model.h"

/******************************************************************************
 * @class GeneticAlgorithm
 *
 * @brief Handling of all data and processing regarding genetic algortihm
 *
 * @public @param population  Container for population
 ******************************************************************************/
class GeneticAlgorithm
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  std::vector<Individual> population;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  GeneticAlgorithm(size_t populationSize, Model limit);

  /******************************************************************************
   * OPERATORS
   ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const GeneticAlgorithm &dt);
};
