#include "logger.h"

std::ofstream Logger::logFile = std::ofstream("logs.log");

time_point<steady_clock> Logger::programClock{steady_clock::now()};

void Logger::startTimeStamp(std::string eventName)
{
  auto elapsed = duration_cast<microseconds>(steady_clock::now() - programClock).count();
  LOG(ESSENTIAL_LOGS, TIME_STAMP, "EVENT_START " << eventName << " " << elapsed);
}

void Logger::endTimeStamp(std::string eventName)
{
  auto elapsed = duration_cast<microseconds>(steady_clock::now() - programClock);
  LOG(ESSENTIAL_LOGS, TIME_STAMP, "EVENT_END " << eventName << " " << elapsed.count());
}
