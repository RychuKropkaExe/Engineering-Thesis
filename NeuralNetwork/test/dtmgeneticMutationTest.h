#pragma once

#include "dtmgeneticAlgorithm.h"
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

using std::vector;
using std::pair;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

using DTMUtils::MutationE;

namespace DTMutationTestUtils
{

// Compare every stored field so a rejected mutation cannot silently reorder
// neurons, alter metadata, or leave an incoming-synapse mirror changed.
void expectSameSynapses(const vector<Synapse> &actual, const vector<Synapse> &expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); index++)
  {
    EXPECT_EQ(actual[index].id, expected[index].id);
    EXPECT_EQ(actual[index].inNeuronId, expected[index].inNeuronId);
    EXPECT_EQ(actual[index].outNeuronId, expected[index].outNeuronId);
    EXPECT_DOUBLE_EQ(actual[index].weight, expected[index].weight);
    EXPECT_EQ(actual[index].isActive, expected[index].isActive);
  }
}

void expectSameIndividual(const DTIndividual &actual, const DTIndividual &expected)
{
  EXPECT_EQ(actual.id, expected.id);
  EXPECT_EQ(actual.generation, expected.generation);
  EXPECT_DOUBLE_EQ(actual.fitness, expected.fitness);
  EXPECT_EQ(actual.gracePeriodLength, expected.gracePeriodLength);
  EXPECT_EQ(actual.model.inputSize, expected.model.inputSize);
  EXPECT_EQ(actual.model.outputSize, expected.model.outputSize);
  EXPECT_EQ(actual.model.maxDepth, expected.model.maxDepth);
  EXPECT_EQ(actual.model.indexMap, expected.model.indexMap);
  EXPECT_EQ(actual.model.isSorted, expected.model.isSorted);
  ASSERT_EQ(actual.model.neurons.size(), expected.model.neurons.size());
  for (size_t index = 0; index < actual.model.neurons.size(); index++)
  {
    const Neuron &neuron = actual.model.neurons[index];
    const Neuron &previous = expected.model.neurons[index];
    EXPECT_EQ(neuron.id, previous.id);
    EXPECT_EQ(neuron.depthId, previous.depthId);
    EXPECT_EQ(neuron.depth, previous.depth);
    EXPECT_EQ(neuron.type, previous.type);
    EXPECT_EQ(neuron.activation, previous.activation);
    EXPECT_DOUBLE_EQ(neuron.value, previous.value);
    EXPECT_DOUBLE_EQ(neuron.bias, previous.bias);
    expectSameSynapses(neuron.inSynapses, previous.inSynapses);
    expectSameSynapses(neuron.outSynapses, previous.outSynapses);
  }
}

size_t synapseCount(const DTModel &model)
{
  size_t count = 0;
  for (const Neuron &neuron : model.neurons)
  {
    count += neuron.outSynapses.size();
  }
  return count;
}

void expectValidModel(DTModel &model)
{
  EXPECT_TRUE(model.isSorted);
  EXPECT_TRUE(model.validateModel());
  ASSERT_EQ(model.indexMap.size(), model.neurons.size());
  std::set<size_t> synapseIds;
  std::set<pair<size_t, size_t>> endpoints;
  size_t maximumDepth = 0;
  for (size_t index = 0; index < model.neurons.size(); index++)
  {
    const Neuron &neuron = model.neurons[index];
    EXPECT_EQ(model.indexMap.at(neuron.id), index);
    maximumDepth = std::max(maximumDepth, neuron.depth);
    for (const Synapse &synapse : neuron.outSynapses)
    {
      EXPECT_TRUE(synapseIds.insert(synapse.id).second);
      EXPECT_TRUE(endpoints.emplace(synapse.inNeuronId, synapse.outNeuronId).second);
      EXPECT_EQ(synapse.inNeuronId, neuron.id);
      ASSERT_TRUE(model.indexMap.contains(synapse.outNeuronId));
      const size_t targetIndex = model.indexMap.at(synapse.outNeuronId);
      EXPECT_LT(index, targetIndex);
      EXPECT_LT(neuron.depth, model.neurons[targetIndex].depth);
      EXPECT_NE(neuron.type, NeuronTypeE::OUTPUT_NEURON);
      EXPECT_NE(model.neurons[targetIndex].type, NeuronTypeE::INPUT_NEURON);
    }
    // Existing models may omit incoming mirrors, but any mirror that is
    // present must still correspond to an actual outgoing connection.
    for (const Synapse &mirror : neuron.inSynapses)
    {
      EXPECT_EQ(mirror.outNeuronId, neuron.id);
      ASSERT_TRUE(model.indexMap.contains(mirror.inNeuronId));
      const auto &outgoing = model.neurons[model.indexMap.at(mirror.inNeuronId)].outSynapses;
      const auto match = std::find_if(outgoing.begin(), outgoing.end(),
          [&](const Synapse &synapse) { return synapse.id == mirror.id; });
      ASSERT_NE(match, outgoing.end());
      EXPECT_EQ(match->outNeuronId, mirror.outNeuronId);
      EXPECT_DOUBLE_EQ(match->weight, mirror.weight);
      EXPECT_EQ(match->isActive, mirror.isActive);
    }
  }
  EXPECT_EQ(model.maxDepth, maximumDepth);
}

}

/******************************************************************************
 * @brief Tests neuron insertion, new gene attributes, ID uniqueness and limits
 ******************************************************************************/
TEST(DTModelTest, addNeuronMutationTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.maxNumberOfNeurons = 4;
  parameters.gracePeriodLength = 9;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(100, 0, 40, 2.5), Synapse(101, 40, 1, 3.5), true);
  DTIndividual individual(7, 3, std::move(model));
  individual.fitness = 4.5;
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;

  srand(12345);
  ASSERT_TRUE(algorithm.addNeuronMutation(individual));
  EXPECT_EQ(individual.id, before.id);
  EXPECT_EQ(individual.generation, before.generation);
  EXPECT_DOUBLE_EQ(individual.fitness, before.fitness);
  EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
  ASSERT_EQ(individual.model.neurons.size(), 4u);
  EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 4u);
  DTMutationTestUtils::expectValidModel(individual.model);

  size_t addedNeuronCount = 0;
  for (const Neuron &neuron : individual.model.neurons)
  {
    if (before.model.indexMap.contains(neuron.id))
    {
      const Neuron &previous = before.model.neurons[before.model.indexMap.at(neuron.id)];
      EXPECT_EQ(neuron.type, previous.type);
      EXPECT_EQ(neuron.activation, previous.activation);
      EXPECT_DOUBLE_EQ(neuron.bias, previous.bias);
      continue;
    }
    addedNeuronCount++;
    EXPECT_GT(neuron.id, 40u);
    EXPECT_EQ(neuron.type, NeuronTypeE::HIDDEN_NEURON);
    EXPECT_EQ(neuron.activation, ActivationE::SIGMOID);
    EXPECT_GE(neuron.bias, 0.0);
    EXPECT_LE(neuron.bias, 1.0);
    ASSERT_EQ(neuron.inSynapses.size(), 1u);
    ASSERT_EQ(neuron.outSynapses.size(), 1u);
    const Synapse &incoming = neuron.inSynapses[0];
    const Synapse &outgoing = neuron.outSynapses[0];
    EXPECT_EQ(incoming.outNeuronId, neuron.id);
    EXPECT_EQ(outgoing.inNeuronId, neuron.id);
    EXPECT_GT(incoming.id, 101u);
    EXPECT_GT(outgoing.id, 101u);
    EXPECT_NE(incoming.id, outgoing.id);
    EXPECT_TRUE(incoming.isActive);
    EXPECT_TRUE(outgoing.isActive);
    EXPECT_GE(incoming.weight, 0.0);
    EXPECT_LE(incoming.weight, 1.0);
    EXPECT_GE(outgoing.weight, 0.0);
    EXPECT_LE(outgoing.weight, 1.0);
    EXPECT_LT(before.model.indexMap.at(incoming.inNeuronId),
              before.model.indexMap.at(outgoing.outNeuronId));
  }
  EXPECT_EQ(addedNeuronCount, 1u);
  EXPECT_TRUE(individual.model.hasSynapse(0, 40));
  EXPECT_TRUE(individual.model.hasSynapse(40, 1));

  individual.gracePeriodLength = 2;
  const DTIndividual atLimit = individual;
  EXPECT_FALSE(algorithm.addNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, atLimit);
  algorithm.hyperparameters.maxNumberOfNeurons = 3;
  EXPECT_FALSE(algorithm.addNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, atLimit);
}

/******************************************************************************
 * @brief Tests removing only hidden neurons and cleaning both adjacency lists
 ******************************************************************************/
TEST(DTModelTest, removeNeuronMutationTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 8;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 1, 70, 0.4), Synapse(103, 70, 2, 0.5), true);
  DTIndividual individual(7, 3, std::move(model));

  for (size_t remainingHidden = 2; remainingHidden != 0; remainingHidden--)
  {
    const DTIndividual before = individual;
    ASSERT_TRUE(algorithm.removeNeuronMutation(individual));
    EXPECT_EQ(individual.model.neurons.size(), before.model.neurons.size() - 1);
    EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
    EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 2 * (remainingHidden - 1));
    DTMutationTestUtils::expectValidModel(individual.model);
    for (const Neuron &neuron : before.model.neurons)
    {
      if (!individual.model.indexMap.contains(neuron.id))
      {
        EXPECT_EQ(neuron.type, NeuronTypeE::HIDDEN_NEURON);
      }
      else
      {
        const Neuron &remaining = individual.model.neurons[individual.model.indexMap.at(neuron.id)];
        EXPECT_DOUBLE_EQ(remaining.bias, neuron.bias);
        EXPECT_EQ(remaining.activation, neuron.activation);
      }
    }
  }

  individual.gracePeriodLength = 2;
  const DTIndividual beforeFailure = individual;
  EXPECT_FALSE(algorithm.removeNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeFailure);
}

/******************************************************************************
 * @brief Tests rejecting either removal from a chain of dependent hidden neurons
 ******************************************************************************/
TEST(DTModelTest, removeNeuronMutationRejectsDanglingTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 8;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 40, 70, 0.4), Synapse(103, 70, 1, 0.5), false);
  model.removeSynapse(40, 1, true);
  DTIndividual individual(7, 3, std::move(model));
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;

  for (unsigned seed = 0; seed < 16; seed++)
  {
    srand(seed);
    EXPECT_FALSE(algorithm.removeNeuronMutation(individual));
    DTMutationTestUtils::expectSameIndividual(individual, before);
  }
}

/******************************************************************************
 * @brief Tests adding a connection and rejecting duplicates, including inactive ones
 ******************************************************************************/
TEST(DTModelTest, addSynapseMutationTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.gracePeriodLength = 6;
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual individual(7, 3, DTModel(1, 1, ActivationE::RELU));

  ASSERT_TRUE(algorithm.addSynapseMutation(individual));
  EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
  ASSERT_EQ(individual.model.neurons.size(), 2u);
  ASSERT_EQ(individual.model.neurons[0].outSynapses.size(), 1u);
  const Synapse &synapse = individual.model.neurons[0].outSynapses[0];
  EXPECT_EQ(synapse.inNeuronId, 0u);
  EXPECT_EQ(synapse.outNeuronId, 1u);
  EXPECT_TRUE(synapse.isActive);
  EXPECT_GE(synapse.weight, 0.0);
  EXPECT_LE(synapse.weight, 1.0);
  DTMutationTestUtils::expectSameSynapses(individual.model.neurons[1].inSynapses,
                                       individual.model.neurons[0].outSynapses);
  DTMutationTestUtils::expectValidModel(individual.model);

  individual.gracePeriodLength = 2;
  const DTIndividual beforeDuplicate = individual;
  EXPECT_FALSE(algorithm.addSynapseMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeDuplicate);

  individual.model.neurons[0].outSynapses[0].isActive = false;
  individual.model.neurons[1].inSynapses[0].isActive = false;
  const DTIndividual beforeInactiveDuplicate = individual;
  EXPECT_FALSE(algorithm.addSynapseMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeInactiveDuplicate);
}

/******************************************************************************
 * @brief Tests forward endpoint selection with hidden neurons and one-attempt failures
 ******************************************************************************/
TEST(DTModelTest, addSynapseMutationTopologyTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 6;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 2, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 1, 70, 0.4), Synapse(103, 70, 3, 0.5), true);
  const DTIndividual original(7, 3, std::move(model));
  size_t successCount = 0;
  size_t failureCount = 0;

  for (unsigned seed = 0; seed < 64; seed++)
  {
    DTIndividual individual = original;
    srand(seed);
    if (!algorithm.addSynapseMutation(individual))
    {
      failureCount++;
      DTMutationTestUtils::expectSameIndividual(individual, original);
      continue;
    }
    successCount++;
    EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
    EXPECT_EQ(individual.model.neurons.size(), original.model.neurons.size());
    EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 5u);
    DTMutationTestUtils::expectValidModel(individual.model);
    for (const Neuron &neuron : individual.model.neurons)
    {
      for (const Synapse &synapse : neuron.outSynapses)
      {
        if (!original.model.hasSynapse(synapse.inNeuronId, synapse.outNeuronId))
        {
          EXPECT_GT(synapse.id, 103u);
          EXPECT_TRUE(synapse.isActive);
          EXPECT_GE(synapse.weight, 0.0);
          EXPECT_LE(synapse.weight, 1.0);
          EXPECT_LT(original.model.indexMap.at(synapse.inNeuronId),
                    original.model.indexMap.at(synapse.outNeuronId));
        }
      }
    }
  }
  // A graph with both missing and existing pairs must produce rejected
  // attempts too: duplicate selection must not trigger another draw.
  EXPECT_GT(successCount, 0u);
  EXPECT_GT(failureCount, 0u);
}

/******************************************************************************
 * @brief Tests valid edge removal with both complete and absent incoming mirrors
 ******************************************************************************/
TEST(DTModelTest, removeSynapseMutationTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 5;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 2, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  const Synapse extraIn(102, 1, 40, 0.4);
  const Synapse extraOut(103, 40, 3, 0.5);
  model.addOutSynapse(extraIn, false);
  model.addOutSynapse(extraOut, false);
  model.neurons[model.indexMap.at(40)].addInSynapse(extraIn);
  model.neurons[model.indexMap.at(3)].addInSynapse(extraOut);
  model.sortTopologically();

  for (bool omitIncoming : {false, true})
  {
    for (unsigned seed = 0; seed < 16; seed++)
    {
      DTIndividual individual(7, 3, model);
      if (omitIncoming)
      {
        for (Neuron &neuron : individual.model.neurons)
        {
          neuron.inSynapses.clear();
        }
      }
      srand(seed);
      // Two inputs and two outputs at the hidden neuron make every single
      // edge removable, regardless of whether incoming mirrors are available.
      ASSERT_TRUE(algorithm.removeSynapseMutation(individual));
      EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
      EXPECT_EQ(individual.model.neurons.size(), model.neurons.size());
      EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 3u);
      DTMutationTestUtils::expectValidModel(individual.model);
      for (const Neuron &neuron : individual.model.neurons)
      {
        for (const Synapse &synapse : neuron.outSynapses)
        {
          EXPECT_TRUE(model.hasSynapse(synapse.inNeuronId, synapse.outNeuronId));
        }
      }
    }
  }
}

/******************************************************************************
 * @brief Tests rejected dangling-edge removals without retrying an optional edge
 ******************************************************************************/
TEST(DTModelTest, removeSynapseMutationRejectsDanglingTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 5;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), true);
  DTIndividual individual(7, 3, model);
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;
  for (unsigned seed = 0; seed < 16; seed++)
  {
    srand(seed);
    EXPECT_FALSE(algorithm.removeSynapseMutation(individual));
    DTMutationTestUtils::expectSameIndividual(individual, before);
  }

  // Only the direct input/output shortcut can be removed. Selecting either
  // edge on the hidden-neuron path must fail even though the shortcut exists.
  model.addOutSynapse(Synapse(102, 0, 1, 0.4), true);
  size_t successCount = 0;
  size_t failureCount = 0;
  for (unsigned seed = 0; seed < 32; seed++)
  {
    DTIndividual attempt(7, 3, model);
    attempt.gracePeriodLength = 2;
    const DTIndividual original = attempt;
    srand(seed);
    if (algorithm.removeSynapseMutation(attempt))
    {
      successCount++;
      EXPECT_FALSE(attempt.model.hasSynapse(0, 1));
      EXPECT_TRUE(attempt.model.hasSynapse(0, 40));
      EXPECT_TRUE(attempt.model.hasSynapse(40, 1));
      DTMutationTestUtils::expectValidModel(attempt.model);
    }
    else
    {
      failureCount++;
      DTMutationTestUtils::expectSameIndividual(attempt, original);
    }
  }
  EXPECT_GT(successCount, 0u);
  EXPECT_GT(failureCount, 0u);

  DTIndividual empty(8, 3, DTModel(1, 1, ActivationE::RELU));
  empty.gracePeriodLength = 2;
  const DTIndividual beforeEmpty = empty;
  EXPECT_FALSE(algorithm.removeSynapseMutation(empty));
  DTMutationTestUtils::expectSameIndividual(empty, beforeEmpty);
}

/******************************************************************************
 * @brief Tests all dispatcher values and unchanged individuals on invalid requests
 ******************************************************************************/
TEST(DTModelTest, mutationDispatchTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.maxNumberOfNeurons = 3;
  // A successful mutation must reset the grace period even when it is zero.
  parameters.gracePeriodLength = 0;
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual individual(7, 3, DTModel(1, 1, ActivationE::RELU));

  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADD_SYNAPSE));
  EXPECT_TRUE(individual.model.hasSynapse(0, 1));
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::REMOVE_SYNAPSE));
  EXPECT_FALSE(individual.model.hasSynapse(0, 1));
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADD_NEURON));
  EXPECT_EQ(individual.model.neurons.size(), 3u);
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::REMOVE_NEURON));
  EXPECT_EQ(individual.model.neurons.size(), 2u);
  EXPECT_EQ(individual.gracePeriodLength, 0u);

  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;
  EXPECT_FALSE(algorithm.mutate(individual, static_cast<MutationE>(999)));
  DTMutationTestUtils::expectSameIndividual(individual, before);
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::REMOVE_NEURON));
  DTMutationTestUtils::expectSameIndividual(individual, before);

  // Addition requires sorted input and must not sort as a side effect of a
  // rejected call. The caller can sort explicitly before its next attempt.
  individual.model.isSorted = false;
  const DTIndividual unsorted = individual;
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::ADD_NEURON));
  DTMutationTestUtils::expectSameIndividual(individual, unsorted);
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::ADD_SYNAPSE));
  DTMutationTestUtils::expectSameIndividual(individual, unsorted);
}
