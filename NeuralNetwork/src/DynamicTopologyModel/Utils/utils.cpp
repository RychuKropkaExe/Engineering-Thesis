#include "utils.h"

/******************************************************************************
 * @brief Returns random double
 *
 * @return random double
 ******************************************************************************/
double DTMUtils::randomdouble()
{
    return (double)(rand()) / (double)(RAND_MAX);
}

/******************************************************************************
 * @brief Returns neuron type in form of string
 *
 * @param type Neuron type
 *
 * @return Neuron type enum value in string form
 ******************************************************************************/
std::string DTMUtils::neuronTypeToString(NeuronTypeE type)
{

  switch (type)
  {
  case INPUT_NEURON:
  {
    return std::string("INPUT_NEURON");
  }
  case HIDDEN_NEURON:
  {
    return std::string("HIDDEN_NEURON");
  }
  case OUTPUT_NEURON:
  {
    return std::string("OUTPUT_NEURON");
  }
  }
}

/******************************************************************************
 * @brief Returns activation function type in form of string
 *
 * @param activation Activation function enum value
 *
 * @return Activation function type enum value in string form
 ******************************************************************************/
std::string DTMUtils::activationFunctionToString(ActivationE activation)
{
  switch (type)
  {
  case RELU:
  {
    return std::string("RELU");
  }
  case SIGMOID:
  {
    return std::string("SIGMOID");
  }
  case NO_ACTIVATION:
  {
    return std::string("NO_ACTIVATION");
  }
  }
}
