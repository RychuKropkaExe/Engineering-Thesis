#include "logger.h"

std::ofstream Logger::logFile = std::ofstream("logs.log");

time_point<steady_clock> Logger::programClock{steady_clock::now()};

void Logger::startTimeStamp([[maybe_unused]] std::string_view eventName)
{
#ifdef LOGGING_ACTIVATED
  if constexpr (MAX_DEBUG_PRIO >= ESSENTIAL_LOGS)
  {
    const auto elapsed = duration_cast<microseconds>(steady_clock::now() - programClock).count();
    TimeStampLogger::getInstance().record(eventName, TimeStampEventE::EVENT_BEGIN, elapsed);
  }
#endif
}

void Logger::endTimeStamp([[maybe_unused]] std::string_view eventName)
{
#ifdef LOGGING_ACTIVATED
  if constexpr (MAX_DEBUG_PRIO >= ESSENTIAL_LOGS)
  {
    const auto elapsed = duration_cast<microseconds>(steady_clock::now() - programClock).count();
    TimeStampLogger::getInstance().record(eventName, TimeStampEventE::EVENT_END, elapsed);
  }
#endif
}
