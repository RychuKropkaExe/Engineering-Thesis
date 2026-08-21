#include "hyperparameters.h"

/******************************************************************************
* CONSTRUCTORS
******************************************************************************/

/******************************************************************************
* OPERATORS
******************************************************************************/
std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparameters)
{
  os << "HYPERPARAMETERS: " << std::endl;
  os << "POPULATION SIZE: " << hyperparameters.populationSize << std::endl;
  os << "MAX NUMBER OF NEURONS: " << hyperparameters.maxNumberOfNeurons << std::endl;

  return os;
}
