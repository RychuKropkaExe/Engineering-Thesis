#pragma once

#include "model.h"

/******************************************************************************
 * @class Individual
 *
 * @brief Represents individual in genetic algorithm
 *
 * @public @param genotype          Individual genotype represented as a neural network
 * @public @param id                Individual id
 * @public @param generation        Expresses from which generation in genethic algorithm
 *                                  individual comes from
 * @public @param fitness           Fitness of given individual
 * @public @param gracePeriodLength Indicates how many generations a mutated individual
 *                                  should be allowed to be a parent regardless of their fitness
 ******************************************************************************/
class Individual
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  Model genotype;
  size_t id{0};
  size_t generation{0};
  double fitness{0.0};
  size_t gracePeriodLength{0};

  // Find better name for those
  static size_t maxId;
  static size_t maxGeneration;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  Individual() = default;
  Individual(Model model, size_t gracePeriodLength = 0);

  /******************************************************************************
   * OPERATORS
   ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Individual &dt);

  bool operator<(const Individual &obj);
  bool operator>(const Individual &obj) const;
};
