#include "dtmgeneticAlgorithm.h"
#include "trainingData.h"
#include <cassert>

/******************************************************************************
 * @brief Calculates the model's mean squared error over supplied training data
 *
 * Matches Model::costMeanSquare: sums squared errors across all outputs and
 * divides by the number of samples, without an additional output-count divisor.
 * Data must be nonempty and contain input/output vectors matching the model's
 * dimensions. Values are used as supplied; any normalization should be performed
 * once by the caller before evaluating individuals.
 *
 * @param model        Model to evaluate; feedForward updates neuron values and
 *                     sorts the model when necessary
 * @param trainingData Training samples and expected outputs, read by reference
 *
 * @return Sum of squared output errors divided by the number of samples
 ******************************************************************************/
double DTMGeneticAlgorithm::calculateMMSE(DTModel &model, const TrainingData &trainingData)
{
  assert(trainingData.numOfSamples > 0);
  assert(trainingData.inputs.size() == trainingData.numOfSamples);
  assert(trainingData.outputs.size() == trainingData.numOfSamples);
  assert(trainingData.inputSize == model.inputSize);
  assert(trainingData.outputSize == model.outputSize);

  double totalSquaredError = 0.0;
  for (size_t sampleIndex = 0; sampleIndex < trainingData.numOfSamples; sampleIndex++)
  {
    // FastMatrix stores each sample in its flat mat vector. Use those vectors
    // directly, without copying the dataset or converting samples to matrices.
    const vector<double> &input = trainingData.inputs[sampleIndex].mat;
    const vector<double> &expected = trainingData.outputs[sampleIndex].mat;
    assert(input.size() == model.inputSize);
    assert(expected.size() == model.outputSize);

    const vector<double> actual = model.feedForward(input);
    for (size_t outputIndex = 0; outputIndex < model.outputSize; outputIndex++)
    {
      const double difference = actual[outputIndex] - expected[outputIndex];
      totalSquaredError += difference * difference;
    }
  }

  return totalSquaredError / static_cast<double>(trainingData.numOfSamples);
}

/******************************************************************************
 * @brief Assigns reciprocal-MMSE fitness to an individual
 *
 * Uses exactly 1.0 / MMSE, so lower model error produces higher fitness.
 * Assumes a nonzero MMSE. Training data has the same requirements as
 * calculateMMSE and is neither copied nor normalized here.
 *
 * @param individual   Individual whose model is evaluated and fitness replaced
 * @param trainingData Training samples and expected outputs, read by reference
 ******************************************************************************/
void DTMGeneticAlgorithm::evaluateIndividual(DTIndividual &individual,
                                           const TrainingData &trainingData)
{
  const double mmse = calculateMMSE(individual.model, trainingData);
  individual.fitness = 1.0 / mmse;
}
