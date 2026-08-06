#include "noCrossover.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

NoCrossover::NoCrossover(vector<Individual> &population) : population(population)
{
  this->population = population;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/
std::vector<Individual> NoCrossover::produceOffspring(std::vector<Parents> &parents)
{
  (void)parents;
  return vector{population};
}
