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

std::string neuronTypeToString(NeuronTypeE type);
std::string activationFunctionToString(ActivationE activation);

/******************************************************************************
 * @brief Returns random double
 *
 * @return random double
 ******************************************************************************/
double randomdouble();

}
