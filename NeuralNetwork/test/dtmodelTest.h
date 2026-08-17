#pragma once
#include "dtmodel.h"
#include "utils.h"
#include "testUtils.h"
#include "trainingData.h"
#include <gtest/gtest.h>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @brief Tests adding a synapse to Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, addSynapseTest)
{

  size_t inputSize = 3;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1, 2};
  vector<size_t> outputNeuronsIds{3, 4};

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addSynapse(newSynapse, false);
    }
  }

  // Verify that every input neuron connects to every output neuron
  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::OUTPUT_NEURON)
    {
      continue;
    }
    EXPECT_EQ(neuron.synapses.size(), outputSize) << "FOR NEURON WITH ID: " << neuron.id;

    vector<bool> isOutputIdPresent{false, false};

    for (auto synapse : neuron.synapses)
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

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addSynapse(newSynapse, false);
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
    for (auto synapse : neuron.synapses)
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

  size_t inputSize = 2;
  size_t outputSize = 2;

  size_t eachLayerSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addSynapse(newSynapse, false);
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
    for (auto synapse : neuron.synapses)
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

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addSynapse(newSynapse, false);
    }
  }

  size_t removedSynapseInNeuronId = inputNeuronsIds[0];
  size_t removedSynapseOutNeuronId = outputNeuronsIds[1];

  testModel.removeSynapse(removedSynapseInNeuronId, removedSynapseOutNeuronId, false);

  bool isRemovedSynapsePresent = false;

  for (auto neuron : testModel.neurons)
  {
    for (auto synapse : neuron.synapses)
    {
      if (synapse.inNeuronId == removedSynapseInNeuronId && synapse.outNeuronId == removedSynapseOutNeuronId)
      {
        isRemovedSynapsePresent = true;
      }
    }
  }

  EXPECT_FALSE(isRemovedSynapsePresent) << "DID FOUND REMOVED CONNECTION";

}

/******************************************************************************
 * @brief Tests topologicall sorting of Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, topologicallSortingTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addSynapse(newSynapse, false);
    }
  }

  // Since sorting may not preserve order of neurons that
  // are equall in placement we define boundries for them.
  vector<size_t> neuronMinimalIndexAfterSort{0, 0, 2, 2};
  vector<size_t> neuronMaximalIndexAfterSort{1, 1, 3, 3};

  testModel.sortTopologically();

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronId = testModel.neurons[index].id;

    size_t neuronIndex = testModel.indexMap[neuronId];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);

  }

  vector<size_t> neuronsIdToAdd{4, 5};

  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  // First two are input neurons, next two are outputs, last two are hidden
  neuronMinimalIndexAfterSort = vector<size_t>{0, 0, 4, 4, 2, 2};
  neuronMaximalIndexAfterSort = vector<size_t>{1, 1, 5, 5, 3, 3};

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronIndex = testModel.indexMap[index];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);

  }

  // Add "third layer" to the network
  vector<size_t> previousNeuronsIds{4, 5};
  neuronsIdToAdd = vector<size_t>{6, 7};

  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, previousNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  // First two are input neurons, next two are outputs, next two are hidden, last two are hidden
  // going from the previous hidden ones.
  neuronMinimalIndexAfterSort = vector<size_t>{0, 0, 6, 6, 2, 2, 4, 4};
  neuronMaximalIndexAfterSort = vector<size_t>{1, 1, 7, 7, 3, 3, 5, 5};

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronIndex = testModel.indexMap[index];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);

  }

}

/******************************************************************************
 * @brief Tests feed forward of Dynamic Topology Model on following scenarios:
 *
 * All inputs set to 1.0 in all scenarios
 *
 *  First scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * expected output:
 * O1 = RELU((1.0*2.0 + 4.0*2.0) - 1.0) = 5
 * O2 = RELU((1.0*4.0 + 1.0*8.0) - 1.0) = 13
 *
 *
 *  Second scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  - 1 hidden neuron, bias set to 0.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * - I3 to H1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * expected output:
 * O1 = RELU((1.0*2.0 + 4.0*2.0 + RELU(1.0*2.0 - 0.0)*2.0) - 1.0) = 9.0
 * O2 = RELU((1.0*4.0 + 1.0*8.0) + + RELU(1.0*2.0 - 0.0)*2.0 - 1.0) = 17.0
 *
 *
 *  Second scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  - 2 hidden neurons, bias set to 0.0 and 1.0
 *  - 1 deeper hidden neuron, bias set to 1.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I1 to H2 with weight 8.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * - I3 to H1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to DH1 with weight 2.0
 * - H2 to DH1 with weight 1.0
 * - DH1 to O1 with weight 3.0
 * - DH1 to O1 with weight 3.0
 * expected output:
 * H1 = RELU(2.0 - 0.0) = 2
 * H2 = RELU(8.0 - 1.0) = 7
 * DH1 = RELU(H1*2.0 + H2*1.0 - 1.0) = 10
 * O1 = RELU(2.0 + H1*2.0 + DH1*3.0 + 4.0 - 1.0) = 39.0
 * O2 = RELU(8.0 + H1*2.0 + DH1*3.0 + 6.0) = 47.0
 *
 ******************************************************************************/
TEST(DTModelTest, feedForwardTest)
{

  size_t inputSize = 3;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1, 2};
  vector<size_t> outputNeuronsIds{3, 4};

  size_t testModelId = 0;

  DTModel testModel(testModelId, inputSize, outputSize, ActivationE::RELU);

  testModel.setBias(outputNeuronsIds[0], 1.0);
  testModel.setBias(outputNeuronsIds[1], 1.0);

  size_t synapseId = 0;

  // Connect every input to every output

  vector<size_t> synapseInNeuronIds{0, 0, 1, 2};
  vector<size_t> synapseOutNeuronIds{3, 4, 3, 4};

  vector<double> synapsesWeights{2.0, 6.0, 4.0, 8.0};

  for (size_t index = 0; index < synapseInNeuronIds.size(); index++)
  {
    // Connect every input to every output
    Synapse newSynapse(synapseId, synapseInNeuronIds[index], synapseOutNeuronIds[index], synapsesWeights[index]);
    synapseId++;
    testModel.addSynapse(newSynapse, false);
  }

  vector<double> input{1.0, 1.0, 1.0};
  vector<double> expectedOutput{5.0, 13.0};

  vector<double> result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

  // Add another neuron to network, a hidden neuron connected to third input neuron
  // and to both output neurons.
  size_t neuronToAddId1 = 5;

  Synapse inSynapse1(synapseId, inputNeuronsIds[2], neuronToAddId1, 2.0);
  synapseId++;
  Synapse outSynapse1(synapseId, neuronToAddId1, outputNeuronsIds[0], 2.0);
  synapseId++;

  Neuron neuronToAdd1(neuronToAddId1, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd1, inSynapse1, outSynapse1, true);

  testModel.setBias(neuronToAddId1, 0.0);

  Synapse additionalSynapse1(synapseId, neuronToAddId1, outputNeuronsIds[1], 2.0);
  synapseId++;

  testModel.addSynapse(additionalSynapse1, true);

  input = vector<double>{1.0, 1.0, 1.0};
  expectedOutput = vector<double>{9.0, 17.0};

  result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

  // Add another two neurons to network, a hidden neuron connected to first input neuron
  // and to deeper hidden neuron, and a deeper hidden neuron connected to both outputs which
  // feed from both hidden neurons
  size_t neuronToAddId2 = 6;

  // Create and add first neuron, deeper neuron
  Synapse inSynapse2(synapseId, neuronToAddId1, neuronToAddId2, 2.0);
  synapseId++;
  Synapse outSynapse2(synapseId, neuronToAddId2, outputNeuronsIds[0], 3.0);
  synapseId++;

  Neuron neuronToAdd2(neuronToAddId2, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd2, inSynapse2, outSynapse2, true);

  testModel.setBias(neuronToAddId2, 1.0);

  Synapse additionalSynapse2(synapseId, neuronToAddId2, outputNeuronsIds[1], 3.0);
  synapseId++;

  testModel.addSynapse(additionalSynapse2, true);

  // Create and add second neuron, connected to deeper hidden neuron
  size_t neuronToAddId3 = 7;

  Synapse inSynapse3(synapseId, inputNeuronsIds[0], neuronToAddId3, 8.0);
  synapseId++;
  Synapse outSynapse3(synapseId, neuronToAddId3, neuronToAddId2, 1.0);
  synapseId++;

  Neuron neuronToAdd3(neuronToAddId3, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd3, inSynapse3, outSynapse3, true);

  testModel.setBias(neuronToAddId3, 1.0);

  input = vector<double>{1.0, 1.0, 1.0};
  expectedOutput = vector<double>{39.0, 47.0};

  result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

}
