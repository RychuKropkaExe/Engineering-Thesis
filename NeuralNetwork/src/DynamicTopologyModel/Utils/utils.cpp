#include "utils.h"


/******************************************************************************
 * @brief Matrix assert used for debugging
 *
 * @param expr Tested expresion
 * @param file File name from which the assert was called
 * @param line Line number from which the assert was called
 * @param msg Message to display if the assert is false
 *
 * @return Nothing.
 ******************************************************************************/
void myAssert(bool expr, const char *file, int line, std::string msg)
{
    if (!expr)
    {
        std::cerr << file << " " << line << ": " << " ASSERT FAILED: " << msg << "\n";
        abort();
    }
}

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
  case NeuronTypeE::INPUT_NEURON:
  {
    return std::string("INPUT_NEURON");
  }
  case NeuronTypeE::HIDDEN_NEURON:
  {
    return std::string("HIDDEN_NEURON");
  }
  case NeuronTypeE::OUTPUT_NEURON:
  {
    return std::string("OUTPUT_NEURON");
  }
  }

  return std::string("NO VALID TYPE MATCHED");
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
  switch (activation)
  {
  case ActivationE::RELU:
  {
    return std::string("RELU");
  }
  case ActivationE::SIGMOID:
  {
    return std::string("SIGMOID");
  }
  case ActivationE::NO_ACTIVATION:
  {
    return std::string("NO_ACTIVATION");
  }
  }

  return std::string("NO MATCHING FUNCTION MATCHED");
}
