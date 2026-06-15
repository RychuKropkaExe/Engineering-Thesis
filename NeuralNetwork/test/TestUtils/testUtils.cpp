#include "testUtils.h"

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
  std::string testDataDir = std::getenv("GIT_REPOSITORY_NEURAL_NETWORK_PATH") + std::string("/test/TestData/");
  return testDataDir + fileName;
}
