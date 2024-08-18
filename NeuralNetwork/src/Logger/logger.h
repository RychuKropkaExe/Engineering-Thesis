#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <iostream>
#include <string>

#define INFO_LEVEL "[INFO]"
#define ERROR_LEVEL "[ERROR]"

class Logger
{
public:
  static std::ofstream logFile;
};

#ifdef LOGGING_ACTIVATED

#define LOG(DEBUG_LEVEL, EXPR)                            \
  (Logger::logFile << DEBUG_LEVEL << ": " << EXPR << "\n" \
                   << std::flush)

#define FLUSH_LOG() \
  (Logger::logFile << std::flush)

#define COND_LOG(COND, DEBUG_LEVEL, EXPR) \
  (COND ? LOG(DEBUG_LEVEL, EXPR) : FLUSH_LOG())

#else
#define LOG(...)
#define COND_LOG(...)
#endif

#endif
