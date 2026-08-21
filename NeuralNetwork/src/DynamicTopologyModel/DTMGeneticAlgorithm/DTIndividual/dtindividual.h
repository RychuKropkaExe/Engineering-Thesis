
#include "dtmodel.h"

/******************************************************************************
 * @class DTIndividual
 *
 * @brief Represents an indivudal and its dtindividual
 *
 * @public @param id          Synapse id
 ******************************************************************************/
class DTIndividual
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  size_t id;

  size_t generation;

  DTModel model;

  double fitness;

  size_t gracePeriodLength{0};

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  DTIndividual() = default;
  DTIndividual(size_t id, size_t generation, DTModel model);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const DTIndividual &DTIndividual);
};
