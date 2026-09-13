#pragma once
#include <string>
#include <iostream>

#ifndef NDEBUG
#define Assert(Expr, Msg) \
    myAssert(Expr, __FILE__, __LINE__, Msg)
#else
#define Assert(Expr, Msg) ;
#endif

void myAssert(bool expr, const char *file, int line, std::string msg);

namespace DTMUtils{

/******************************************************************************
 * @enum NeuronTypeE
 *
 * @brief Describes types of neurons
 *
 ******************************************************************************/
enum class NeuronTypeE
{
  INPUT_NEURON,
  HIDDEN_NEURON,
  OUTPUT_NEURON
};

/******************************************************************************
 * @enum ActivationE
 *
 * @brief Describes types of supported activation functions
 *
 ******************************************************************************/
enum class ActivationE
{
    SIGMOID,
    RELU,
    NO_ACTIVATION
};

/******************************************************************************
 * @enum MutationE
 *
 * @brief Selects a topology or parameter mutation for a dynamic-topology individual
 ******************************************************************************/
enum class MutationE
{
    ADD_NEURON,     // Insert a hidden neuron with an incoming and outgoing synapse.
    REMOVE_NEURON,  // Remove a hidden neuron and its incident synapses.
    ADD_SYNAPSE,    // Add a forward connection between two existing neurons.
    REMOVE_SYNAPSE, // Remove a connection without leaving hidden neurons dangling.
    ADJUST_WEIGHT,  // Perturb a synapse weight, allowing either sign.
    ADJUST_BIAS,    // Perturb a hidden or output neuron's bias.
    CHANGE_ACTIVATION // Switch a hidden neuron's activation to another supported function.
};

std::string neuronTypeToString(NeuronTypeE type);
std::string activationFunctionToString(ActivationE activation);

/******************************************************************************
 * @brief Returns random double
 *
 * @return random double
 ******************************************************************************/
double randomdouble();

}
