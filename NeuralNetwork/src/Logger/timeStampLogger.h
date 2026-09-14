#pragma once

#include "timeStampData.h"
#include <array>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

/******************************************************************************
 * @class TimeStampLogger
 *
 * @brief Single-producer, double-buffered asynchronous timestamp logger
 *
 * The main thread must call start(), record() and stop(). Only the worker
 * writes timestamp records to timeStampLog.log. Calls from multiple producers
 * are not supported: the per-record path deliberately avoids mutex locking.
 *
 * @private @param logFile        Static output stream, opened by start()
 * @private @param buffers        Two permanently sized vectors, reused by index
 * @private @param ready          True while a buffer is published or being written
 * @private @param sizes          Number of published entries in each buffer
 * @private @param producerBuffer Buffer exclusively owned by the main thread
 * @private @param producerSize   Number of entries filled in the current buffer
 * @private @param bufferMutex    Protects buffer handoffs and the stop signal
 * @private @param bufferChanged  Signals published buffers or released ownership
 * @private @param worker         Thread responsible for formatting and file I/O
 * @private @param stopping       Requests worker exit after all buffers are drained
 * @private @param running        Main-thread lifecycle flag
 * @private @param writeFailed    Worker I/O status, inspected after joining
 ******************************************************************************/
class TimeStampLogger
{
public:
  static constexpr std::size_t BUFFER_CAPACITY = 4096;

  static TimeStampLogger &getInstance();

  // Opens the output file in overwrite mode and starts the waiting worker.
  void start();

  /******************************************************************************
   * @brief Copies an event into the current buffer without formatting or I/O
   *
   * Publishes a full buffer and waits only if the other buffer is still owned
   * by the worker. start() must have been called on this same producer thread.
   *
   * @param eventName Name copied into reusable string storage
   * @param eventType Beginning or end of an operation
   * @param timeStamp Timestamp already captured on the producer thread
   ******************************************************************************/
  void record(std::string_view eventName, TimeStampEventE eventType,
              std::int64_t timeStamp);

  /******************************************************************************
   * @brief Publishes partial data, waits for draining, then stops and joins
   *
   * Safe to call again after stopping. Reports file errors after joining so
   * failed writes cannot leave the worker running. No records may be submitted
   * concurrently with shutdown.
   ******************************************************************************/
  void stop();

  ~TimeStampLogger();
  TimeStampLogger(const TimeStampLogger &) = delete;
  TimeStampLogger &operator=(const TimeStampLogger &) = delete;

private:
  TimeStampLogger();
  void processBuffers();

  static std::ofstream logFile;
  std::array<std::vector<TimeStampData>, 2> buffers;
  std::array<bool, 2> ready{};
  std::array<std::size_t, 2> sizes{};
  std::size_t producerBuffer{};
  std::size_t producerSize{};
  std::mutex bufferMutex;
  std::condition_variable bufferChanged;
  std::thread worker;
  bool stopping{};
  bool running{};
  bool writeFailed{};
};
