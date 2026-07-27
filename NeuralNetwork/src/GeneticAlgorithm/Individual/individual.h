#pragma once

#include "model.h"

/******************************************************************************
 * @class Individual
 *
 * @brief Represents individual in genetic algorithm
 *
 * @public @param genotype    Individual genotype represented as a neural network
 * @public @param id          Individual id
 * @public @param generation  Expresses from which generation in genethic algorithm
 *                            individual comes from
 * @public @param fitness     Fitness of given individual
 ******************************************************************************/
class Individual
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  Model genotype;
  int id{};
  int generation{};
  double fitness{};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  Individual() = delete;
  Individual(Model model);

  /******************************************************************************
   * OPERATORS
   ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Individual &dt);
};
