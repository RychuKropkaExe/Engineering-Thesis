#pragma once

#include "crossoverAlgorithm.h"
#include "individual.h"
#include "model.h"

/******************************************************************************
 * @class WeightedCrossover
 *
 * @brief Implementation of one point crossover
 *
 * @public @param scalingFactor Scaling factor for selecting cut point
 *
 ******************************************************************************/
class WeightedCrossover : public CrossoverAlgorithm
{
public:
  double scalingFactor;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  WeightedCrossover(double scalingFactor);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  std::vector<Individual> produceOffspring(std::vector<Parents> &parents);
};
