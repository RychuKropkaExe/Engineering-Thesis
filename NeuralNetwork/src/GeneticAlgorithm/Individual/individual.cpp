#include "individual.h"
#include <iostream>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

Individual::Individual(Model model)
{
  genotype = model;
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
