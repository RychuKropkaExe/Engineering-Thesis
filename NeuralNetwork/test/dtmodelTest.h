#pragma once

#include "dtmodel.h"
#include <cassert>
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @brief Tests adding a synapse to Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, addOutSynapseTest)
{

  size_t inputSize = 3;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1, 2};
  vector<size_t> outputNeuronsIds{3, 4};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  // Verify that every input neuron connects to every output neuron
  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::OUTPUT_NEURON)
    {
      continue;
    }
    EXPECT_EQ(neuron.outSynapses.size(), outputSize) << "FOR NEURON WITH ID: " << neuron.id;

    vector<bool> isOutputIdPresent{false, false};

    for (auto synapse : neuron.outSynapses)
    {
      assert(synapse.outNeuronId >= inputSize);
      isOutputIdPresent[synapse.outNeuronId - inputSize] = true;
    }

    EXPECT_TRUE(isOutputIdPresent[0]);
    EXPECT_TRUE(isOutputIdPresent[1]);

  }

}

/******************************************************************************
 * @brief Tests adding a neuron to Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, addNeuronTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  size_t eachLayerSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  vector<size_t> neuronsIdToAdd{4, 5};

  for (size_t index = 0; index < eachLayerSize; index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, false);
  }

  vector<bool> newNeuronsFound{false, false};
  vector<bool> newInConnectionsFound{false, false};
  vector<bool> newOutConnectionsFound{false, false};

  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
    {
      size_t boolTableIndex = neuron.id - (inputSize + outputSize);
      newNeuronsFound[boolTableIndex] = true;
    }
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.outNeuronId >= (inputSize + outputSize))
      {
        // New connection can only come from input neurons
        EXPECT_LT(synapse.inNeuronId, inputSize) << "NEURON ID COMING INTO NEW NEURON IS TOO HIGH";
        newOutConnectionsFound[synapse.outNeuronId - (inputSize + outputSize)] = true;
      }

      if (synapse.inNeuronId >= (inputSize + outputSize))
      {
        // New connection can only feed into output neuron
        EXPECT_GE(synapse.outNeuronId, inputSize) << "NEURON ID COMING OUT OF NEW NEURON IS TOO LOW";
        // Ensure it does not go into one of the new neurons
        EXPECT_LT(synapse.outNeuronId, inputSize + outputSize) << "NEURON ID COMING OUT OF NEW NEURON IS TOO HIGH";
        newInConnectionsFound[synapse.inNeuronId - (inputSize + outputSize)] = true;
      }
    }
  }

  EXPECT_TRUE(newNeuronsFound[0]) << "DID NOT FOUND FIRST NEW NEURON";
  EXPECT_TRUE(newNeuronsFound[1]) << "DID NOT FOUND SECOND NEW NEURON";

  EXPECT_TRUE(newInConnectionsFound[0]) << "DID NOT FOUND CONNECTION FEEDING INTO FIRST NEW NEURON";
  EXPECT_TRUE(newInConnectionsFound[1]) << "DID NOT FOUND CONNECTION FEEDING INTO SECOND NEW NEURON";

  EXPECT_TRUE(newOutConnectionsFound[0]) << "DID NOT FOUND CONNECTION GOING OUT FROM FIRST NEW NEURON";
  EXPECT_TRUE(newOutConnectionsFound[0]) << "DID NOT FOUND CONNECTION GOING OUT FROM SECOND NEW NEURON";

}

/******************************************************************************
 * @brief Tests removing a neuron from Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, removeNeuronTest)
{
  TIME_MEASURE_BEGIN(REMOVE_NEURON_TEST);
  size_t inputSize = 2;
  size_t outputSize = 2;

  size_t eachLayerSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  vector<size_t> neuronsIdToAdd{4, 5};

  for (size_t index = 0; index < eachLayerSize; index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, false);
  }

  testModel.removeNeuron(neuronsIdToAdd[0], false);
  testModel.removeNeuron(neuronsIdToAdd[1], false);

  vector<bool> newNeuronsFound{false, false};
  vector<bool> newInConnectionsFound{false, false};

  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
    {
      size_t boolTableIndex = neuron.id - (inputSize + outputSize);
      newNeuronsFound[boolTableIndex] = true;
    }
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.inNeuronId >= (inputSize + outputSize))
      {
        newInConnectionsFound[synapse.inNeuronId - (inputSize + outputSize)] = true;
      }
    }
  }

  EXPECT_FALSE(newNeuronsFound[0]) << "DID FOUND FIRST REMOVED NEURON";
  EXPECT_FALSE(newNeuronsFound[1]) << "DID FOUND SECOND REMOVED NEURON";

  EXPECT_FALSE(newInConnectionsFound[0]) << "DID FOUND CONNECTION FEEDING INTO FIRST REMOVED NEURON";
  EXPECT_FALSE(newInConnectionsFound[1]) << "DID FOUND CONNECTION FEEDING INTO SECOND REMOVED NEURON";

  TIME_MEASURE_END(REMOVE_NEURON_TEST);
}

/******************************************************************************
 * @brief Tests removing a synapse from Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, removeSynapseTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  size_t removedSynapseInNeuronId = inputNeuronsIds[0];
  size_t removedSynapseOutNeuronId = outputNeuronsIds[1];

  testModel.removeSynapse(removedSynapseInNeuronId, removedSynapseOutNeuronId, false);

  bool isRemovedOutSynapsePresent = false;
  bool isRemovedInSynapsePresent = false;

  for (auto neuron : testModel.neurons)
  {
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.inNeuronId == removedSynapseInNeuronId && synapse.outNeuronId == removedSynapseOutNeuronId)
      {
        isRemovedOutSynapsePresent = true;
      }
    }

    for (auto synapse : neuron.inSynapses)
    {
      if (synapse.inNeuronId == removedSynapseInNeuronId && synapse.outNeuronId == removedSynapseOutNeuronId)
      {
        isRemovedInSynapsePresent = true;
      }
    }
  }

  EXPECT_FALSE(isRemovedOutSynapsePresent) << "DID FOUND REMOVED OUT CONNECTION";
  EXPECT_FALSE(isRemovedInSynapsePresent) << "DID FOUND REMOVED IN CONNECTION";

}

/******************************************************************************
 * @brief Tests directed synapse lookup, including disabled and missing edges
 ******************************************************************************/
TEST(DTModelTest, hasSynapseTest)
{
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), true);
  model.neurons[model.indexMap.at(0)].outSynapses[0].isActive = false;
  const auto originalIndexMap = model.indexMap;
  const DTModel &readOnlyModel = model;

  EXPECT_TRUE(readOnlyModel.hasSynapse(0, 40));
  EXPECT_TRUE(readOnlyModel.hasSynapse(40, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(40, 0));
  EXPECT_FALSE(readOnlyModel.hasSynapse(0, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(40, 40));
  EXPECT_FALSE(readOnlyModel.hasSynapse(999, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(0, 999));
  EXPECT_EQ(model.indexMap, originalIndexMap);
}
