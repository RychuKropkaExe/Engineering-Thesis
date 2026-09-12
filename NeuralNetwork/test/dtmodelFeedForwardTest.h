#pragma once

#include "dtmodel.h"
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

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

  DTModel testModel(inputSize, outputSize, ActivationE::RELU);

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
    testModel.addOutSynapse(newSynapse, false);
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

  testModel.addOutSynapse(additionalSynapse1, true);

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

  testModel.addOutSynapse(additionalSynapse2, true);

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

