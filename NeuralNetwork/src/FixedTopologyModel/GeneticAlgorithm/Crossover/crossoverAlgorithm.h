#pragma once

#include "individual.h"
#include "model.h"
#include "parents.h"

/******************************************************************************
 * @class CrossoverAlgorithm
 *
 * @brief Base class for all crossover algorithms
 *
 *
 ******************************************************************************/
class CrossoverAlgorithm
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  virtual std::vector<Individual> produceOffspring(std::vector<Parents> &parents) = 0;
};
