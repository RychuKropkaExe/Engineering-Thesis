#pragma once

#include "dtmgeneticAlgorithm.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include <utility>

/******************************************************************************
 * @brief Tests signed weight adjustments, incoming mirrors and impossible attempts
 ******************************************************************************/
TEST(DTMGeneticParameterMutationTest, adjustWeightTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 2;
  parameters.weightMutationStrength = 2.0;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::SIGMOID);
  const Synapse connection(100, 0, 1, 0.0);
  model.addOutSynapse(connection, true);
  model.neurons[model.indexMap.at(1)].addInSynapse(connection);
  bool sawNegative = false;
  bool sawPositive = false;
  for (unsigned seed = 0; seed < 32; seed++)
  {
    DTIndividual individual(7, 3, model);
    srand(seed);
    ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADJUST_WEIGHT));
    const Synapse &changed = individual.model.neurons[0].outSynapses[0];
    EXPECT_NE(changed.weight, 0.0);
    EXPECT_GE(changed.weight, -2.0);
    EXPECT_LE(changed.weight, 2.0);
    EXPECT_NEAR(individual.model.neurons[1].inSynapses[0].weight, changed.weight, 1e-12);
    EXPECT_EQ(changed.id, connection.id);
    EXPECT_EQ(changed.inNeuronId, connection.inNeuronId);
    EXPECT_EQ(changed.outNeuronId, connection.outNeuronId);
    EXPECT_EQ(changed.isActive, connection.isActive);
    EXPECT_EQ(individual.model.indexMap, model.indexMap);
    EXPECT_TRUE(individual.model.isSorted);
    EXPECT_EQ(individual.gracePeriodLength, 2u);
    EXPECT_EQ(individual.id, 7u);
    EXPECT_EQ(individual.generation, 3u);
    sawNegative |= changed.weight < 0.0;
    sawPositive |= changed.weight > 0.0;
  }
  EXPECT_TRUE(sawNegative);
  EXPECT_TRUE(sawPositive);

  // Canonical outgoing connections are sufficient even without incoming mirrors.
  model.neurons[1].inSynapses.clear();
  DTIndividual outgoingOnly(7, 3, model);
  EXPECT_TRUE(algorithm.adjustWeightMutation(outgoingOnly));
  EXPECT_TRUE(outgoingOnly.model.neurons[1].inSynapses.empty());

  DTIndividual empty(8, 3, DTModel(1, 1, ActivationE::SIGMOID));
  empty.gracePeriodLength = 7;
  EXPECT_FALSE(algorithm.adjustWeightMutation(empty));
  EXPECT_EQ(empty.gracePeriodLength, 7u);
  algorithm.hyperparameters.weightMutationStrength = 0.0;
  DTIndividual disabled(9, 3, model);
  disabled.gracePeriodLength = 7;
  EXPECT_FALSE(algorithm.adjustWeightMutation(disabled));
  EXPECT_DOUBLE_EQ(disabled.model.neurons[0].outSynapses[0].weight, 0.0);
  EXPECT_EQ(disabled.gracePeriodLength, 7u);

  for (double strength : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()})
  {
    algorithm.hyperparameters.weightMutationStrength = strength;
    EXPECT_FALSE(algorithm.adjustWeightMutation(disabled));
    EXPECT_DOUBLE_EQ(disabled.model.neurons[0].outSynapses[0].weight, 0.0);
    EXPECT_EQ(disabled.gracePeriodLength, 7u);
  }
}

/******************************************************************************
 * @brief Tests signed bias adjustments on hidden/output neurons, excluding inputs
 ******************************************************************************/
TEST(DTMGeneticParameterMutationTest, adjustBiasTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 2;
  parameters.biasMutationStrength = 2.0;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::SIGMOID);
  Neuron hidden(10, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);
  hidden.bias = 0.0;
  model.addNeuron(hidden, Synapse(100, 0, 10, 0.5), Synapse(101, 10, 1, 0.5), true);
  model.setBias(0, 42.0);
  model.setBias(1, 0.0);
  bool sawHidden = false;
  bool sawOutput = false;
  bool sawNegative = false;
  for (unsigned seed = 0; seed < 32; seed++)
  {
    DTIndividual individual(7, 3, model);
    srand(seed);
    ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADJUST_BIAS));
    size_t changedCount = 0;
    for (const Neuron &neuron : individual.model.neurons)
    {
      const Neuron &before = model.neurons[model.indexMap.at(neuron.id)];
      EXPECT_EQ(neuron.activation, before.activation);
      if (neuron.bias != before.bias)
      {
        changedCount++;
        EXPECT_NE(neuron.type, NeuronTypeE::INPUT_NEURON);
        EXPECT_GE(neuron.bias, -2.0);
        EXPECT_LE(neuron.bias, 2.0);
        sawHidden |= neuron.type == NeuronTypeE::HIDDEN_NEURON;
        sawOutput |= neuron.type == NeuronTypeE::OUTPUT_NEURON;
        sawNegative |= neuron.bias < 0.0;
      }
    }
    EXPECT_EQ(changedCount, 1u);
    EXPECT_EQ(individual.gracePeriodLength, 2u);
    EXPECT_EQ(individual.model.indexMap, model.indexMap);
    EXPECT_TRUE(individual.model.isSorted);
  }
  EXPECT_TRUE(sawHidden);
  EXPECT_TRUE(sawOutput);
  EXPECT_TRUE(sawNegative);
  algorithm.hyperparameters.biasMutationStrength = 0.0;
  DTIndividual disabled(7, 3, model);
  disabled.gracePeriodLength = 7;
  EXPECT_FALSE(algorithm.adjustBiasMutation(disabled));
  EXPECT_EQ(disabled.gracePeriodLength, 7u);
  for (const Neuron &neuron : disabled.model.neurons)
  {
    EXPECT_DOUBLE_EQ(neuron.bias, model.neurons[model.indexMap.at(neuron.id)].bias);
  }
  for (double strength : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()})
  {
    algorithm.hyperparameters.biasMutationStrength = strength;
    EXPECT_FALSE(algorithm.adjustBiasMutation(disabled));
    EXPECT_EQ(disabled.gracePeriodLength, 7u);
    for (const Neuron &neuron : disabled.model.neurons)
    {
      EXPECT_DOUBLE_EQ(neuron.bias, model.neurons[model.indexMap.at(neuron.id)].bias);
    }
  }
}

/******************************************************************************
 * @brief Tests that activation mutations always change a hidden neuron only
 ******************************************************************************/
TEST(DTMGeneticParameterMutationTest, changeActivationTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 2;
  DTMGeneticAlgorithm algorithm(parameters);
  for (ActivationE activation : {ActivationE::SIGMOID, ActivationE::RELU, ActivationE::NO_ACTIVATION})
  {
    DTModel model(1, 1, ActivationE::SIGMOID);
    model.addNeuron(Neuron(10, NeuronTypeE::HIDDEN_NEURON, activation),
                    Synapse(100, 0, 10, 0.5), Synapse(101, 10, 1, 0.5), true);
    DTIndividual individual(7, 3, model);
    ASSERT_TRUE(algorithm.mutate(individual, MutationE::CHANGE_ACTIVATION));
    for (const Neuron &neuron : individual.model.neurons)
    {
      const Neuron &before = model.neurons[model.indexMap.at(neuron.id)];
      if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
      {
        EXPECT_NE(neuron.activation, activation);
      }
      else
      {
        EXPECT_EQ(neuron.activation, before.activation);
      }
      EXPECT_DOUBLE_EQ(neuron.bias, before.bias);
    }
    EXPECT_EQ(individual.gracePeriodLength, 2u);
    EXPECT_EQ(individual.model.indexMap, model.indexMap);
    EXPECT_TRUE(individual.model.isSorted);
  }
  DTIndividual noHidden(7, 3, DTModel(1, 1, ActivationE::SIGMOID));
  noHidden.gracePeriodLength = 7;
  EXPECT_FALSE(algorithm.changeActivationMutation(noHidden));
  EXPECT_EQ(noHidden.gracePeriodLength, 7u);
  EXPECT_EQ(noHidden.model.neurons[0].activation, ActivationE::NO_ACTIVATION);
  EXPECT_EQ(noHidden.model.neurons[1].activation, ActivationE::SIGMOID);
}
