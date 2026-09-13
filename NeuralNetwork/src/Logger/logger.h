#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>

using namespace std::chrono;

constexpr std::string INFO_TYPE{"[INFO]"};
constexpr std::string ERROR_TYPE{"[ERROR]"};
constexpr std::string TIME_STAMP{"[TIME_STAMP]"};

constexpr int ESSENTIAL_LOGS{1};
constexpr int NORMAL_LOGS{2};
constexpr int HEAVY_LOGS{3};

#ifndef DEBUG_PRIO
constexpr int MAX_DEBUG_PRIO{ESSENTIAL_LOGS};
#else
constexpr int MAX_DEBUG_PRIO{DEBUG_PRIO};
#endif

/******************************************************************************
 * @class Logger
 *
 * @brief Wrapper for log file used for easier managing of access to it
 *
 * @public @param logFile  File to which logs are saved
 *
 ******************************************************************************/
class Logger
{
public:
  static std::ofstream logFile;
  static time_point<steady_clock> programClock;

  static void startTimeStamp(std::string eventName);

  static void endTimeStamp(std::string eventName);
};

#ifdef LOGGING_ACTIVATED

#define LOG(PRIO_TYPE, DEBUG_TYPE, EXPR)                                                \
  (PRIO_TYPE <= MAX_DEBUG_PRIO ? (Logger::logFile << DEBUG_TYPE << ": " << EXPR << "\n" \
                                                  << std::flush)                        \
                               : FLUSH_LOG())

#define TIME_MEASURE_BEGIN(EVENT_NAME) Logger::startTimeStamp(std::string(#EVENT_NAME));
#define TIME_MEASURE_END(EVENT_NAME) Logger::endTimeStamp(std::string(#EVENT_NAME));

#define FLUSH_LOG() \
  (Logger::logFile << std::flush)

// Conditional logs are always essential, by their nature they are only used
// if some condition is satisfied so its important to log it
#define COND_LOG(COND, DEBUG_TYPE, EXPR) \
  (COND ? LOG(ESSENTIAL_LOGS, DEBUG_TYPE, EXPR) : FLUSH_LOG())

#else
#define LOG(...)
#define COND_LOG(...)
#define TIME_MEASURE_BEGIN(EVENT_NAME)
#define TIME_MEASURE_END(EVENT_NAME)
#endif
