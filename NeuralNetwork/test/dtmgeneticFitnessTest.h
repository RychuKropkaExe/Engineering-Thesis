#pragma once

#include "dtmgeneticAlgorithm.h"
#include "trainingData.h"
#include <gtest/gtest.h>
#include <utility>

/******************************************************************************
 * @brief Tests MMSE averaging over samples with multiple inputs and outputs
 ******************************************************************************/
TEST(DTMGeneticFitnessTest, calculateMMSETest)
{
  constexpr double delta = 1e-9;
  DTModel model(2, 2, ActivationE::NO_ACTIVATION);
  model.setBias(2, 1.0);
  model.setBias(3, 2.0);
  model.addOutSynapse(Synapse(100, 0, 2, 2.0), false);
  model.addOutSynapse(Synapse(101, 1, 2, 1.0), false);
  model.addOutSynapse(Synapse(102, 0, 3, -1.0), false);
  model.addOutSynapse(Synapse(103, 1, 3, 3.0), false);

  const TrainingData data({{1.0, 2.0}, {2.0, 1.0}, {-1.0, 3.0}}, 2, 3,
                          {{1.0, 1.0}, {4.0, 3.0}, {-2.0, 4.0}}, 2, 3);
  const auto inputsBefore = data.inputs;
  const auto outputsBefore = data.outputs;

  // The model computes [2*x0 + x1 - 1, -x0 + 3*x1 - 2].
  // Predictions are [3, 3], [4, -1], [0, 8]; squared error sums are 8, 16, 20.
  // Match the fixed model's divisor of 3 samples, not 6 output values.
  EXPECT_NEAR(DTMGeneticAlgorithm::calculateMMSE(model, data), 44.0 / 3.0, delta);
  EXPECT_TRUE(model.isSorted);
  EXPECT_NEAR(DTMGeneticAlgorithm::calculateMMSE(model, data), 44.0 / 3.0, delta);
  EXPECT_EQ(data.inputs, inputsBefore);
  EXPECT_EQ(data.outputs, outputsBefore);
}

/******************************************************************************
 * @brief Tests reciprocal fitness through a hidden neuron and repeated evaluation
 ******************************************************************************/
TEST(DTMGeneticFitnessTest, evaluateIndividualTest)
{
  constexpr double delta = 1e-9;
  DTModel model(1, 1, ActivationE::RELU);
  model.setBias(1, 2.0);
  Neuron hidden(10, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);
  hidden.bias = 1.0;
  model.addNeuron(hidden, Synapse(100, 0, 10, 2.0), Synapse(101, 10, 1, 3.0), false);

  DTIndividual individual(7, 3, std::move(model));
  individual.fitness = 123.0;
  individual.gracePeriodLength = 8;
  const TrainingData data({{-1.0}, {2.0}}, 1, 2, {{1.0}, {5.0}}, 1, 2);

  // y = RELU(3 * RELU(2*x - 1) - 2) gives [0, 7].
  // MMSE = ((0 - 1)^2 + (7 - 5)^2) / 2 = 2.5, so fitness = 0.4.
  ASSERT_FALSE(individual.model.isSorted);
  DTMGeneticAlgorithm::evaluateIndividual(individual, data);
  EXPECT_NEAR(individual.fitness, 0.4, delta);
  EXPECT_TRUE(individual.model.isSorted);
  EXPECT_EQ(individual.model.maxDepth, 2u);
  EXPECT_EQ(individual.id, 7u);
  EXPECT_EQ(individual.generation, 3u);
  EXPECT_EQ(individual.gracePeriodLength, 8u);

  DTMGeneticAlgorithm::evaluateIndividual(individual, data);
  EXPECT_NEAR(individual.fitness, 0.4, delta);

  // New targets give MMSE = 0.25. Fitness must be the raw reciprocal (4.0),
  // using the newly supplied data and replacing the previous fitness value.
  const TrainingData closerTargets({{-1.0}, {2.0}}, 1, 2, {{0.5}, {7.5}}, 1, 2);
  DTMGeneticAlgorithm::evaluateIndividual(individual, closerTargets);
  EXPECT_NEAR(individual.fitness, 4.0, delta);
}

/******************************************************************************
 * @brief Tests MMSE and reciprocal fitness with fractional values and tolerance
 ******************************************************************************/
TEST(DTMGeneticFitnessTest, fractionalValuesTest)
{
  // Allow small rounding differences when decimal values are stored as doubles.
  constexpr double delta = 1e-9;
  DTModel model(1, 1, ActivationE::NO_ACTIVATION);
  model.setBias(1, 0.1);
  model.addOutSynapse(Synapse(100, 0, 1, 0.3), false);
  DTIndividual individual(7, 3, std::move(model));
  const TrainingData data({{0.1}, {0.2}, {0.7}}, 1, 3,
                          {{0.2}, {-0.1}, {0.3}}, 1, 3);

  // y = 0.3*x - 0.1 gives [-0.07, -0.04, 0.11].
  // Errors [-0.27, 0.06, -0.19] give squared errors [0.0729, 0.0036, 0.0361].
  // Both expectations come from this calculation, independently of the result.
  constexpr double expectedMMSE = 0.1126 / 3.0;
  constexpr double expectedFitness = 30000.0 / 1126.0;

  EXPECT_NEAR(DTMGeneticAlgorithm::calculateMMSE(individual.model, data),
              expectedMMSE, delta);
  DTMGeneticAlgorithm::evaluateIndividual(individual, data);
  EXPECT_NEAR(individual.fitness, expectedFitness, delta);
}
