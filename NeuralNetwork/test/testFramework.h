#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <iomanip>
#include <iostream>
#include <string>

inline const std::string c_reset = "\033[0m";
inline const std::string c_red = "\033[31m";
inline const std::string c_green = "\033[32m";
inline const std::string c_blue = "\033[34m";

inline int passed = 0;
inline int failed = 0;
inline int assertPassed = 0;
inline int assertFailed = 0;
inline bool testResult = true;

#define TEST_PASS                                                                                                \
  (std::cout << c_blue << "Test: " << c_reset << __FUNCTION__ << c_green << "\t\t  [PASSED]" << c_reset << "\n", \
   passed++, testResult = true)

#define TEST_FAIL                                                                                              \
  (std::cout << c_blue << "Test: " << c_reset << __FUNCTION__ << c_red << "\t\t  [FAILED]" << c_reset << "\n", \
   failed++, testResult = true)

#define TEST_SET                                                                                \
  (std::cout << c_blue << "\n===== Test set: " << __FUNCTION__ << " ===== " << c_reset << "\n", \
   testResult = true)

#define ASSERT_PASS() (assertPassed++)

#define ASSERT_FAILED(expr, line, file, value)                                                                                                          \
  (std::cout << c_blue << file << ":" << line << " Assert: " << c_reset << expr << c_red << " [FAILED] " << c_reset << "Real Value: " << value << "\n", \
   assertFailed++, testResult = false)

#define MY_TEST_ASSERT(expr, value) (expr ? ASSERT_PASS() : ASSERT_FAILED(#expr, __LINE__, __FILE__, value))

#define TEST_RESULT() (testResult ? TEST_PASS : TEST_FAIL)

#endif
