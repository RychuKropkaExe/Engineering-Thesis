#pragma once

#include "dtmfitnessEvaluation.h"
#include "trainingData.h"

/******************************************************************************
 * @class MeanSquaredEval
 *
 * @brief Assigns reciprocal mean-squared-error fitness using fixed training data
 *
 * Squared errors are summed across outputs and averaged over samples, matching
 * Model::costMeanSquare. Data is stored as an immutable snapshot and is not
 * normalized automatically. Evaluation updates only the supplied individual.
 *
 * @private @param trainingData Validated samples and target outputs
 ******************************************************************************/
class MeanSquaredEval : public DTMFitnessEvaluation
{
public:
  /******************************************************************************
   * @brief Validates and stores the dataset used for subsequent evaluations
   *
   * @param trainingData Nonempty samples, normalized by the caller if needed;
   *                     copied from an lvalue or moved from an rvalue
   * @throws std::invalid_argument If dimensions, counts or sample sizes disagree
   ******************************************************************************/
  explicit MeanSquaredEval(TrainingData trainingData);

  /******************************************************************************
   * @brief Evaluates the individual and calculate its fitness with 1/MSE
   *
   * Forward propagation updates neuron values and sorts the model if needed.
   *
   * @param individual Individual with input/output dimensions matching the data
   * @throws std::invalid_argument If the model dimensions do not match the data
   ******************************************************************************/
  void evaluateIndividual(DTIndividual &individual) override;

private:
  const TrainingData trainingData;
};
