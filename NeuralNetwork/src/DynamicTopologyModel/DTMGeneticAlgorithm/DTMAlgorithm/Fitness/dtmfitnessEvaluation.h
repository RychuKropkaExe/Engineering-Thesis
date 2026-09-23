#pragma once

#include <stdexcept>

class DTIndividual;

/******************************************************************************
 * @class DTMFitnessEvaluation
 *
 * @brief Base strategy for assigning fitness to a dynamic-topology individual
 *
 * Higher fitness is better. DTMGeneticAlgorithm borrows this strategy and may
 * evaluate distinct individuals concurrently. Derived implementations must
 * keep shared state read-only or synchronize access to it.
 ******************************************************************************/
class DTMFitnessEvaluation
{
public:
  // Allow derived evaluators to be destroyed through a base-class pointer.
  virtual ~DTMFitnessEvaluation() = default;

  /******************************************************************************
   * @brief Calculate fitness of individual
   *
   * @param individual Individual whose model is evaluated and fitness assigned
   * @throws std::logic_error If the base implementation is used directly
   ******************************************************************************/
  virtual void evaluateIndividual([[maybe_unused]] DTIndividual &individual)
  {
    throw std::logic_error("DTMFitnessEvaluation::evaluateIndividual must be overridden");
  }
};
