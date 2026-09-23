#pragma once

#include "dtmodel.h"

/******************************************************************************
 * @class DTIndividual
 *
 * @brief Represents an individual in the dynamic topology genetic algorithm
 *
 * @public @param id                  Individual unique id
 * @public @param generation          Generation in which the individual was created
 * @public @param model               Dynamic topology model represented by the individual
 * @public @param fitness             Fitness value assigned to the individual
 * @public @param gracePeriodLength   Number of generations for which the individual is
 *                                    protected from removal
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

  double fitness{0.0};

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
