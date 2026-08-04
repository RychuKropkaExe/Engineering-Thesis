#include "individual.h"
#include <iostream>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

Individual::Individual(Model model, size_t gracePeriodLength)
{
  genotype = model;
  this->gracePeriodLength = gracePeriodLength;
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const Individual &dt)
{
  os << "INDIVIDUAL WITH ID: " << dt.id << std::endl;
  os << dt.genotype << std::endl;
  os << std::flush;

  return os;
}

bool Individual::operator<(const Individual &obj)
{
  return fitness < obj.fitness;
}

bool Individual::operator>(const Individual &obj) const
{
  return fitness > obj.fitness;
}
