#include "timeStampLogger.h"
#include <stdexcept>

std::ofstream TimeStampLogger::logFile;

TimeStampLogger::TimeStampLogger()
  : buffers{std::vector<TimeStampData>(BUFFER_CAPACITY),
            std::vector<TimeStampData>(BUFFER_CAPACITY)}
{
}

TimeStampLogger &TimeStampLogger::getInstance()
{
  static TimeStampLogger instance;
  return instance;
}

TimeStampLogger::~TimeStampLogger()
{
  // Fallback for stack unwinding or an early return from the test runner.
  // Normal shutdown calls stop() explicitly, where I/O errors can be reported.
  try
  {
    stop();
  }
  catch (...)
  {
  }
}

void TimeStampLogger::start()
{
  if (running)
  {
    throw std::logic_error("Timestamp logger is already running");
  }

  logFile.clear();
  logFile.open("timeStampLog.log", std::ios::out | std::ios::trunc);
  if (!logFile)
  {
    throw std::runtime_error("Cannot open timeStampLog.log");
  }

  producerBuffer = 0;
  producerSize = 0;
  ready = {};
  sizes = {};
  stopping = false;
  writeFailed = false;
  try
  {
    worker = std::thread(&TimeStampLogger::processBuffers, this);
    running = true;
  }
  catch (...)
  {
    logFile.close();
    throw;
  }
}

void TimeStampLogger::record(std::string_view eventName,
                             TimeStampEventE eventType, std::int64_t timeStamp)
{
  if (!running)
  {
    throw std::logic_error("Timestamp logger must be started before recording");
  }

  // The producer has exclusive ownership until publication. assign() can
  // reuse the string's capacity when this slot is filled on a later cycle.
  TimeStampData &entry = buffers[producerBuffer][producerSize];
  entry.eventName.assign(eventName);
  entry.eventType = eventType;
  entry.timeStamp = timeStamp;
  ++producerSize;

  if (producerSize == BUFFER_CAPACITY)
  {
    std::unique_lock<std::mutex> lock(bufferMutex);
    sizes[producerBuffer] = producerSize;
    ready[producerBuffer] = true;
    producerSize = 0;
    producerBuffer ^= 1;
    bufferChanged.notify_all();

    // ready stays true throughout writing, not just until the worker wakes.
    // This prevents overwriting entries while the worker still reads them.
    bufferChanged.wait(lock, [this] { return !ready[producerBuffer]; });
  }
}

void TimeStampLogger::processBuffers()
{
  std::size_t consumerBuffer = 0;
  std::unique_lock<std::mutex> lock(bufferMutex);
  for (;;)
  {
    bufferChanged.wait(lock, [this, consumerBuffer] {
      return ready[consumerBuffer] || stopping;
    });
    if (!ready[consumerBuffer])
    {
      return;
    }

    const std::size_t count = sizes[consumerBuffer];
    lock.unlock();

    // Keep ownership but release the handoff mutex during I/O so the producer
    // can fill and publish the other buffer. Strict alternation preserves order.
    for (std::size_t index = 0; index < count; ++index)
    {
      const TimeStampData &entry = buffers[consumerBuffer][index];
      logFile << "[TIME_STAMP]: "
              << (entry.eventType == TimeStampEventE::EVENT_BEGIN
                    ? "EVENT_BEGIN" : "EVENT_END")
              << ' ' << entry.eventName << ' ' << entry.timeStamp << '\n';
    }
    // Flush once per buffer, rather than once per timestamp.
    logFile.flush();
    writeFailed = writeFailed || !logFile;

    lock.lock();
    ready[consumerBuffer] = false;
    consumerBuffer ^= 1;
    bufferChanged.notify_all();
  }
}

void TimeStampLogger::stop()
{
  if (!running)
  {
    return;
  }

  {
    std::unique_lock<std::mutex> lock(bufferMutex);
    if (producerSize != 0)
    {
      sizes[producerBuffer] = producerSize;
      ready[producerBuffer] = true;
      producerSize = 0;
      bufferChanged.notify_all();
    }

    // Drain both published buffers before asking the worker to exit. This
    // also covers zero events, exact-capacity batches and a partial last batch.
    bufferChanged.wait(lock, [this] { return !ready[0] && !ready[1]; });
    stopping = true;
    bufferChanged.notify_all();
  }

  worker.join();
  running = false;
  logFile.close();
  if (writeFailed || logFile.fail())
  {
    throw std::runtime_error("Failed to write timeStampLog.log");
  }
}
