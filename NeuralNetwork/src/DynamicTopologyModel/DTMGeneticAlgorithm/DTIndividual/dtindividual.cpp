#include "dtindividual.h"
#include <cassert>
#include <cmath>
/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

DTIndividual::DTIndividual(size_t id, size_t generation, DTModel model)
{
  this->id = id;
  this->model = model;
  this->generation = generation;
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const DTIndividual &dtindividual)
{
  os << "DT INDIVIDUAL ID: " << dtindividual.id << std::endl;
  os << "DT INDIVIDUAL GENERATION: " << dtindividual.generation << std::endl;
  os << "DT INDIVIDUAL FITNESS: " << dtindividual.fitness << std::endl;
  os << "DT INDIVIDUAL MODEL: " << dtindividual.model << std::endl;
  return os;
}

/******************************************************************************
* UTILITIES
******************************************************************************/
