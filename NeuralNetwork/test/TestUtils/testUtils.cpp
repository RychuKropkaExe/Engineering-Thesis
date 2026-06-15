#include "testUtils.h"
#include <iostream>
/******************************************************************************
 * HELPER FUNCTIONS
 ******************************************************************************/

/******************************************************************************
 * @brief Returns absolute path to given test data file using environment
 *        variable
 *
 * @param fileName name of the test data file
 *
 * @return path string
 ******************************************************************************/
std::string getTestDataPath(const std::string &fileName)
{
  // Get the path to the test data directory
  char *env_p = std::getenv("GIT_REPOSITORY_NEURAL_NETWORK_PATH");
  if (env_p == nullptr)
  {
    throw std::runtime_error("Environment variable GIT_REPOSITORY_NEURAL_NETWORK_PATH is not set.");
  }
  std::string testDataDir = std::string(env_p) + std::string("/test/TestData/");
  return testDataDir + fileName;
}
