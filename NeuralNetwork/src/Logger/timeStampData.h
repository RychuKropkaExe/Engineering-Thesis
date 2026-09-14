#pragma once

#include <cstdint>
#include <string>

enum class TimeStampEventE
{
  EVENT_BEGIN,
  EVENT_END
};

/******************************************************************************
 * @class TimeStampData
 *
 * @brief A timestamp captured by the producer and formatted by the logger thread
 *
 * @public @param eventName Event name supplied through the timing macro
 * @public @param eventType Beginning or end of the measured operation
 * @public @param timeStamp Elapsed microseconds since Logger::programClock
 ******************************************************************************/
class TimeStampData
{
public:
  std::string eventName;
  TimeStampEventE eventType{TimeStampEventE::EVENT_BEGIN};
  std::int64_t timeStamp{};
};
