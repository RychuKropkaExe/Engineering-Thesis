#pragma once

#include "crossoverAlgorithm.h"
#include "individual.h"
#include "model.h"

/******************************************************************************
 * @class NoCrossover
 *
 * @brief Class used when no crossover is to be used
 *
 * @public @param population Current population to be returned
 *
 ******************************************************************************/
class NoCrossover : public CrossoverAlgorithm
{
public:
  vector<Individual> &population;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  NoCrossover(vector<Individual> &population);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  std::vector<Individual> produceOffspring(std::vector<Parents> &parents);
};
