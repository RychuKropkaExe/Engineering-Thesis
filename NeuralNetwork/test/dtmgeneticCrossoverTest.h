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

/******************************************************************************
 * @brief Tests finding similar neurons and synapses in two DTModels
 ******************************************************************************/
TEST(DTModelTest, findSimilarNeuronsAndSynapsesTest)
{
  auto createNeuron = [](size_t id, size_t depthId)
  {
    Neuron neuron(id, NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);
    neuron.depthId = depthId;
    return neuron;
  };

  DTModel firstModel;
  firstModel.neurons = {
      createNeuron(10, 0),
      createNeuron(11, 1000),
      createNeuron(12, 2000),
      createNeuron(13, 4000)};
  firstModel.indexMap = {{10, 0}, {11, 1}, {12, 2}, {13, 3}};

  firstModel.neurons[0].addOutSynapse(Synapse(1, 10, 11, 0.1));
  firstModel.neurons[1].addOutSynapse(Synapse(2, 11, 12, 0.2));
  firstModel.neurons[0].addOutSynapse(Synapse(3, 10, 12, 0.3));
  firstModel.neurons[0].addOutSynapse(Synapse(4, 10, 13, 0.4));

  DTModel secondModel;
  // Different global IDs and vector order ensure that a match cannot rely on
  // neuron IDs or vector positions. Endpoints must be resolved with indexMap.
  secondModel.neurons = {
      createNeuron(102, 2000),
      createNeuron(100, 0),
      createNeuron(103, 3000),
      createNeuron(101, 1000)};
  secondModel.indexMap = {{100, 1}, {101, 3}, {102, 0}, {103, 2}};

  secondModel.neurons[1].addOutSynapse(Synapse(20, 100, 101, 9.1));
  secondModel.neurons[3].addOutSynapse(Synapse(21, 101, 102, 9.2));
  secondModel.neurons[1].addOutSynapse(Synapse(22, 100, 103, 9.3));

  // Synapse activity, weight and global ID are intentionally different and
  // must not affect similarity when both endpoint depthIds are equal.
  secondModel.neurons[1].outSynapses[0].isActive = false;

  const SimilarDTModelElements result =
      DTMGeneticAlgorithm::findSimilarNeuronsAndSynapses(firstModel, secondModel);

  const vector<size_t> expectedFirstModelNeuronIds{10, 11, 12};
  const vector<size_t> expectedSecondModelNeuronIds{100, 101, 102};

  EXPECT_EQ(result.firstModelNeuronIds, expectedFirstModelNeuronIds);
  EXPECT_EQ(result.secondModelNeuronIds, expectedSecondModelNeuronIds);

  ASSERT_EQ(result.firstModelSynapses.size(), 2);
  ASSERT_EQ(result.secondModelSynapses.size(), 2);

  // Entries at equal positions describe the same depth-based connection.
  EXPECT_EQ(result.firstModelSynapses[0].id, 1);
  EXPECT_EQ(result.secondModelSynapses[0].id, 20);
  EXPECT_EQ(result.firstModelSynapses[1].id, 2);
  EXPECT_EQ(result.secondModelSynapses[1].id, 21);

  EXPECT_NE(result.firstModelSynapses[0].weight,
            result.secondModelSynapses[0].weight);
  EXPECT_NE(result.firstModelSynapses[0].isActive,
            result.secondModelSynapses[0].isActive);
}

/******************************************************************************
 * @brief Tests finding model elements not included in its similar elements
 ******************************************************************************/
TEST(DTModelTest, findNonSimilarNeuronsAndSynapsesTest)
{
  auto createNeuron = [](size_t id, size_t depthId)
  {
    Neuron neuron(id, NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);
    neuron.depthId = depthId;
    return neuron;
  };

  DTModel model;
  model.neurons = {
      createNeuron(10, 0),
      createNeuron(11, 1000),
      createNeuron(12, 2000)};
  model.indexMap = {{10, 0}, {11, 1}, {12, 2}};

  model.neurons[0].addOutSynapse(Synapse(1, 10, 11, 0.1));
  model.neurons[1].addOutSynapse(Synapse(2, 11, 12, 0.2));
  model.neurons[0].addOutSynapse(Synapse(3, 10, 12, 0.3));

  const vector<size_t> similarNeuronIds{10, 12};
  const vector<Synapse> similarSynapses{
      model.neurons[0].outSynapses[0],
      model.neurons[1].outSynapses[0]};

  const NonSimilarDTModelElements result =
      DTMGeneticAlgorithm::findNonSimilarNeuronsAndSynapses(
          model, similarNeuronIds, similarSynapses);

  const vector<size_t> expectedNeuronIds{11};

  EXPECT_EQ(result.neuronIds, expectedNeuronIds);
  ASSERT_EQ(result.synapses.size(), 1);
  EXPECT_EQ(result.synapses[0].id, 3);
  EXPECT_EQ(result.synapses[0].inNeuronId, 10);
  EXPECT_EQ(result.synapses[0].outNeuronId, 12);
}

/******************************************************************************
 * @brief Tests unique hidden-neuron IDs without consuming fixed input/output IDs
 ******************************************************************************/
TEST(DTModelTest, getNewUniqueNeuronIdTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 2;
  parameters.outputSize = 1;
  DTMFitnessEvaluation fitnessEvaluation;
  DTMGeneticAlgorithm algorithm(parameters, &fitnessEvaluation);

  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 3u);
  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 4u);
  EXPECT_EQ(algorithm.getNewUniqueSynapseId(), 0u);
  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 5u);
}

/******************************************************************************
 * @brief Tests crossover inheritance, endpoint remapping and generation replacement
 ******************************************************************************/
TEST(DTModelTest, crossoverTest)
{
  auto createModel = [](bool first)
  {
    DTModel model(2, 1, ActivationE::RELU);
    const size_t hiddenId = first ? 10 : 20;
    const double weight = first ? 5.0 : 9.0;
    size_t synapseId = first ? 100 : 200;
    Neuron hidden(hiddenId, NeuronTypeE::HIDDEN_NEURON,
                  first ? ActivationE::RELU : ActivationE::SIGMOID);
    hidden.bias = first ? 1.0 : 2.0;
    const Synapse hiddenIn(synapseId++, 0, hiddenId, weight);
    const Synapse hiddenOut(synapseId++, hiddenId, 2, weight);
    model.addNeuron(hidden, hiddenIn, hiddenOut, false);
    model.setBias(2, first ? 3.0 : 4.0);

    // Each parent has an exclusive shortcut. Only the first has another
    // hidden neuron, at the same depth as the shared hidden neuron.
    if (first)
    {
      Neuron extra(11, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID);
      extra.bias = 7.0;
      const Synapse extraIn(synapseId++, 1, 11, weight);
      const Synapse extraOut(synapseId++, 11, 2, weight);
      model.addNeuron(extra, extraIn, extraOut, false);
    }
    model.addOutSynapse(Synapse(synapseId, first ? 0 : 1, 2, weight), false);

    for (Neuron &neuron : model.neurons)
    {
      for (Synapse &synapse : neuron.outSynapses)
      {
        synapse.isActive = first;
      }
    }
    model.sortTopologically();
    return model;
  };

  const DTModel firstModel = createModel(true);
  const DTModel secondModel = createModel(false);
  const vector<pair<double, double>> fitnessCases{{10.0, 1.0}, {1.0, 10.0}, {5.0, 5.0}};

  // A fixed seed makes the coverage repeatable without depending on an exact
  // random sequence. Check inheritance choices, not a statistical 50% quota.
  srand(271828);
  for (const auto &fitness : fitnessCases)
  {
    SCOPED_TRACE(testing::Message() << "fitness: " << fitness.first << ", " << fitness.second);
    Hyperparameters parameters{};
    parameters.inputSize = 2;
    parameters.outputSize = 1;
    parameters.populationSize = 2;
    DTMFitnessEvaluation fitnessEvaluation;
    DTMGeneticAlgorithm algorithm(parameters, &fitnessEvaluation);
    algorithm.currentGeneration = 7;
    algorithm.population.emplace_back(40, 1, firstModel);
    algorithm.population.emplace_back(50, 1, secondModel);
    algorithm.population[0].fitness = fitness.first;
    algorithm.population[1].fitness = fitness.second;

    // Force the crossover to prepare parents whose vector indexes differ
    // from their ordinary IDs and whose sorted flags are stale.
    for (DTIndividual &parent : algorithm.population)
    {
      std::reverse(parent.model.neurons.begin(), parent.model.neurons.end());
      for (size_t index = 0; index < parent.model.neurons.size(); index++)
      {
        parent.model.indexMap[parent.model.neurons[index].id] = index;
      }
      parent.model.isSorted = false;
    }

    GAParents forward;
    forward.addParent(0);
    forward.addParent(1);
    GAParents reverse;
    reverse.addParent(1);
    reverse.addParent(0);
    const size_t pairsPerSpecies = 32;
    const vector<vector<GAParents>> parentsLists{
        vector<GAParents>(pairsPerSpecies, forward), {}, vector<GAParents>(pairsPerSpecies, reverse)};

    algorithm.crossover(parentsLists);

    ASSERT_EQ(algorithm.population.size(), 2 * pairsPerSpecies);
    EXPECT_EQ(algorithm.hyperparameters.populationSize, algorithm.population.size());
    EXPECT_EQ(algorithm.currentGeneration, 7u);
    std::set<size_t> individualIds;
    std::set<size_t> hiddenIds;
    std::set<size_t> synapseIds;
    bool sawMixedNeuronAttributes = false;
    bool sawMixedSynapseAttributes = false;
    bool sawFirstOutputBias = false;
    bool sawSecondOutputBias = false;

    for (size_t childIndex = 0; childIndex < algorithm.population.size(); childIndex++)
    {
      DTIndividual &child = algorithm.population[childIndex];
      DTModel &model = child.model;
      const bool firstDominates = fitness.first > fitness.second ||
          (fitness.first == fitness.second && childIndex < pairsPerSpecies);
      const DTModel &dominant = firstDominates ? firstModel : secondModel;

      EXPECT_GT(child.id, 50u);
      EXPECT_TRUE(individualIds.insert(child.id).second);
      EXPECT_EQ(child.generation, 7u);
      EXPECT_DOUBLE_EQ(child.fitness, 0.0);
      EXPECT_EQ(model.inputSize, 2u);
      EXPECT_EQ(model.outputSize, 1u);
      EXPECT_TRUE(model.isSorted);
      EXPECT_EQ(model.maxDepth, 2u);
      ASSERT_EQ(model.neurons.size(), dominant.neurons.size());
      ASSERT_EQ(model.indexMap.size(), model.neurons.size());
      EXPECT_TRUE(model.validateModel());

      // A topology keyed by endpoint depthIds checks exact inherited edges
      // while allowing every hidden-neuron and synapse ID to be regenerated.
      std::set<pair<size_t, size_t>> expectedConnections;
      for (const Neuron &neuron : dominant.neurons)
      {
        for (const Synapse &synapse : neuron.outSynapses)
        {
          expectedConnections.emplace(
              dominant.neurons[dominant.indexMap.at(synapse.inNeuronId)].depthId,
              dominant.neurons[dominant.indexMap.at(synapse.outNeuronId)].depthId);
        }
      }
      std::set<pair<size_t, size_t>> actualConnections;
      size_t connectionCount = 0;
      size_t incomingCount = 0;

      for (size_t index = 0; index < model.neurons.size(); index++)
      {
        const Neuron &neuron = model.neurons[index];
        EXPECT_EQ(model.indexMap.at(neuron.id), index);
        EXPECT_DOUBLE_EQ(neuron.value, 0.0);
        incomingCount += neuron.inSynapses.size();

        if (neuron.type == NeuronTypeE::INPUT_NEURON)
        {
          EXPECT_LT(neuron.id, 2u);
          EXPECT_EQ(neuron.activation, ActivationE::NO_ACTIVATION);
        }
        else if (neuron.type == NeuronTypeE::OUTPUT_NEURON)
        {
          EXPECT_EQ(neuron.id, 2u);
          EXPECT_EQ(neuron.activation, ActivationE::RELU);
          EXPECT_TRUE(neuron.bias == 3.0 || neuron.bias == 4.0);
          sawFirstOutputBias |= neuron.bias == 3.0;
          sawSecondOutputBias |= neuron.bias == 4.0;
        }
        else
        {
          EXPECT_GT(neuron.id, 20u);
          EXPECT_TRUE(hiddenIds.insert(neuron.id).second);
          if (neuron.depthId == 1000)
          {
            EXPECT_TRUE(neuron.bias == 1.0 || neuron.bias == 2.0);
            EXPECT_TRUE(neuron.activation == ActivationE::RELU ||
                        neuron.activation == ActivationE::SIGMOID);
            sawMixedNeuronAttributes |= (neuron.bias == 1.0) !=
                                        (neuron.activation == ActivationE::RELU);
          }
          else
          {
            EXPECT_TRUE(firstDominates);
            EXPECT_EQ(neuron.depthId, 1001u);
            EXPECT_DOUBLE_EQ(neuron.bias, 7.0);
            EXPECT_EQ(neuron.activation, ActivationE::SIGMOID);
          }
        }

        for (const Synapse &synapse : neuron.outSynapses)
        {
          connectionCount++;
          EXPECT_GT(synapse.id, 202u);
          EXPECT_TRUE(synapseIds.insert(synapse.id).second);
          EXPECT_EQ(synapse.inNeuronId, neuron.id);
          const size_t outIndex = model.indexMap.at(synapse.outNeuronId);
          EXPECT_LT(index, outIndex);
          const Neuron &outNeuron = model.neurons[outIndex];
          actualConnections.emplace(neuron.depthId, outNeuron.depthId);
          const bool shared = (neuron.depthId == 0 && outNeuron.depthId == 1000) ||
                              (neuron.depthId == 1000 && outNeuron.depthId == 2000);
          if (shared)
          {
            EXPECT_TRUE(synapse.weight == 5.0 || synapse.weight == 9.0);
            sawMixedSynapseAttributes |= (synapse.weight == 5.0) != synapse.isActive;
          }
          else
          {
            EXPECT_DOUBLE_EQ(synapse.weight, firstDominates ? 5.0 : 9.0);
            EXPECT_EQ(synapse.isActive, firstDominates);
          }

          const auto incoming = std::find_if(outNeuron.inSynapses.begin(), outNeuron.inSynapses.end(),
              [&](const Synapse &mirror) { return mirror.id == synapse.id; });
          ASSERT_NE(incoming, outNeuron.inSynapses.end());
          EXPECT_EQ(incoming->inNeuronId, synapse.inNeuronId);
          EXPECT_EQ(incoming->outNeuronId, synapse.outNeuronId);
          EXPECT_DOUBLE_EQ(incoming->weight, synapse.weight);
          EXPECT_EQ(incoming->isActive, synapse.isActive);
        }
      }
      EXPECT_EQ(actualConnections, expectedConnections);
      EXPECT_EQ(connectionCount, expectedConnections.size());
      EXPECT_EQ(incomingCount, connectionCount);
    }

    EXPECT_TRUE(sawMixedNeuronAttributes);
    EXPECT_TRUE(sawMixedSynapseAttributes);
    EXPECT_TRUE(sawFirstOutputBias);
    EXPECT_TRUE(sawSecondOutputBias);
    EXPECT_GT(algorithm.getNewUniqueNeuronId(), *hiddenIds.rbegin());
    EXPECT_GT(algorithm.getNewUniqueSynapseId(), *synapseIds.rbegin());
    EXPECT_GT(algorithm.getNewUniqueIndividualCounter(), *individualIds.rbegin());
  }
}

/******************************************************************************
 * @brief Tests self-crossover, repeated generations and empty parent lists
 ******************************************************************************/
TEST(DTModelTest, crossoverSameParentAndEmptyListsTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  DTMFitnessEvaluation fitnessEvaluation;
  DTMGeneticAlgorithm algorithm(parameters, &fitnessEvaluation);
  DTModel model(1, 1, ActivationE::NO_ACTIVATION);
  Neuron hidden(10, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);
  hidden.bias = 1.0;
  model.addNeuron(hidden, Synapse(100, 0, 10, 2.0), Synapse(101, 10, 1, 3.0), false);
  model.setBias(1, 4.0);
  algorithm.population.emplace_back(30, 0, std::move(model));
  algorithm.population[0].fitness = 10.0;
  GAParents parents;
  parents.addParent(0);
  parents.addParent(0);

  size_t previousIndividualId = 30;
  size_t previousHiddenId = 10;
  size_t previousSynapseId = 101;
  for (size_t generation = 1; generation <= 2; generation++)
  {
    algorithm.currentGeneration = generation;
    algorithm.crossover({{}, {parents}});
    ASSERT_EQ(algorithm.population.size(), 1u);
    DTIndividual &child = algorithm.population[0];
    EXPECT_GT(child.id, previousIndividualId);
    EXPECT_EQ(child.generation, generation);
    EXPECT_TRUE(child.model.isSorted);
    EXPECT_EQ(child.model.maxDepth, 2u);
    ASSERT_EQ(child.model.neurons.size(), 3u);
    EXPECT_EQ(child.model.neurons[1].type, NeuronTypeE::HIDDEN_NEURON);
    EXPECT_GT(child.model.neurons[1].id, previousHiddenId);
    EXPECT_DOUBLE_EQ(child.model.neurons[1].bias, 1.0);
    EXPECT_EQ(child.model.neurons[1].activation, ActivationE::RELU);
    ASSERT_EQ(child.model.neurons[0].outSynapses.size(), 1u);
    ASSERT_EQ(child.model.neurons[1].outSynapses.size(), 1u);
    EXPECT_GT(child.model.neurons[0].outSynapses[0].id, previousSynapseId);
    EXPECT_GT(child.model.neurons[1].outSynapses[0].id, previousSynapseId);
    EXPECT_DOUBLE_EQ(child.model.neurons[0].outSynapses[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(child.model.neurons[1].outSynapses[0].weight, 3.0);
    const vector<double> output = child.model.feedForward({2.0});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_DOUBLE_EQ(output[0], 5.0);

    previousIndividualId = child.id;
    previousHiddenId = child.model.neurons[1].id;
    previousSynapseId = std::max(child.model.neurons[0].outSynapses[0].id,
                                 child.model.neurons[1].outSynapses[0].id);
  }

  algorithm.crossover({});
  EXPECT_TRUE(algorithm.population.empty());
  EXPECT_EQ(algorithm.hyperparameters.populationSize, 0u);
}
