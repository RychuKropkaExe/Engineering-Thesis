#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include "logger.h"
#include <iomanip>
#include <iostream>
#include <string>

inline int passed = 0;
inline int failed = 0;
inline int assertPassed = 0;
inline int assertFailed = 0;
inline bool testResult = true;

inline const std::string c_reset = "\033[0m";
inline const std::string c_red = "\033[31m";
inline const std::string c_green = "\033[32m";
inline const std::string c_blue = "\033[34m";

#define TEST_PASS                                                 \
  (LOG(INFO_LEVEL, "Test: " << __FUNCTION__ << "\t\t  [PASSED]"), \
   passed++, testResult = true,                                   \
   std::cout << c_blue << "Test: " << c_reset << __FUNCTION__ << c_green << "\t\t  [PASSED]" << c_reset << "\n")

#define TEST_FAIL                                                  \
  (LOG(ERROR_LEVEL, "Test: " << __FUNCTION__ << "\t\t  [FAILED]"), \
   failed++, testResult = true,                                    \
   std::cout << c_blue << "Test: " << c_reset << __FUNCTION__ << c_red << "\t\t  [FAILED]" << c_reset << "\n")

#define TEST_SET                                                      \
  (LOG(INFO_LEVEL, " ===== Test set: " << __FUNCTION__ << " ===== "), \
   testResult = true)

#define ASSERT_PASS() (assertPassed++)

#define ASSERT_FAILED(expr, line, file, value)                                                              \
  (LOG(ERROR_LEVEL, file << ":" << line << " Assert: " << expr << " [FAILED] " << "Real Value: " << value), \
   assertFailed++, testResult = false)

#define MY_TEST_ASSERT(expr, value) (expr ? ASSERT_PASS() : ASSERT_FAILED(#expr, __LINE__, __FILE__, value))

#define TEST_START                                                 \
  (LOG(INFO_LEVEL, "Test: " << __FUNCTION__ << "\t\t  [STARTED]"), \
   std::cout << c_blue << "Test: " << c_reset << __FUNCTION__ << c_blue << "\t\t  [STARTED]" << c_reset << "\n")

#define TEST_RESULT() (testResult ? TEST_PASS : TEST_FAIL)

#endif
