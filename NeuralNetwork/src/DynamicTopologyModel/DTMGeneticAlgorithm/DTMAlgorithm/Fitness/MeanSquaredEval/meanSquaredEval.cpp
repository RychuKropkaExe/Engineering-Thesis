#include "meanSquaredEval.h"
#include "dtindividual.h"
#include <stdexcept>
#include <utility>

/******************************************************************************
 * @brief Stores a dataset and checks all data-only invariants once
 *
 * @param trainingData Dataset of input and matching expected outputs.
 * @throws std::invalid_argument If the dataset is empty or inconsistent
 ******************************************************************************/
MeanSquaredEval::MeanSquaredEval(TrainingData trainingData)
    : trainingData(std::move(trainingData))
{
  const TrainingData &data = this->trainingData;
  if (data.inputSize == 0 || data.outputSize == 0 || data.numOfSamples == 0 ||
      data.inputs.size() != data.numOfSamples ||
      data.outputs.size() != data.numOfSamples)
  {
    throw std::invalid_argument("Training data must be nonempty with consistent dimensions and sample counts");
  }
  for (size_t index = 0; index < data.numOfSamples; index++)
  {
    if (data.inputs[index].mat.size() != data.inputSize ||
        data.outputs[index].mat.size() != data.outputSize)
    {
      throw std::invalid_argument("Training sample dimensions do not match the dataset dimensions");
    }
  }
}

/******************************************************************************
 * @brief Assigns reciprocal error averaged over samples, not output values
 *
 * @param individual Individual whose model is evaluated and fitness replaced
 * @throws std::invalid_argument If the model dimensions do not match the data
 ******************************************************************************/
void MeanSquaredEval::evaluateIndividual(DTIndividual &individual)
{
  DTModel &model = individual.model;
  // Compatibility depends on the individual; dataset integrity is checked
  // only at construction and protected by the evaluator's immutable snapshot.
  if (model.inputSize != trainingData.inputSize ||
      model.outputSize != trainingData.outputSize)
  {
    throw std::invalid_argument("Training data dimensions do not match the individual model");
  }

  double totalSquaredError = 0.0;
  for (size_t sampleIndex = 0; sampleIndex < trainingData.numOfSamples; sampleIndex++)
  {
    // Reuse the flat sample vectors without copying or normalizing them.
    const vector<double> &input = trainingData.inputs[sampleIndex].mat;
    const vector<double> &expected = trainingData.outputs[sampleIndex].mat;
    const vector<double> actual = model.feedForward(input);
    for (size_t outputIndex = 0; outputIndex < model.outputSize; outputIndex++)
    {
      const double difference = actual[outputIndex] - expected[outputIndex];
      totalSquaredError += difference * difference;
    }
  }
  const double mmse = totalSquaredError / static_cast<double>(trainingData.numOfSamples);
  individual.fitness = 1.0 / mmse;
}
