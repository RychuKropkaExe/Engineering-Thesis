#pragma once

#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/input_state.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

/******************************************************************************
 * @struct TrackmaniaSimulatorOptions
 *
 * @brief Configures the race time limit, random input seed and replay recording
 *
 * The horizon excludes the countdown and must be a positive multiple of 10 ms.
 * Replay recording retains the source map and each observed state in memory.
 ******************************************************************************/
struct TrackmaniaSimulatorOptions
{
  // Race-relative limit, in milliseconds; must be a positive multiple of 10.
  std::uint32_t simulationHorizonMs = 120000;
  // A local generator keeps runs reproducible without affecting std::rand().
  std::uint32_t randomSeed = 2026;
  // Retain the map and car states for exportReplay(). Disabled by default.
  bool recordReplay = false;
};

/******************************************************************************
 * @class TrackmaniaSimulator
 *
 * @brief Runs a fresh TrackMania race using ForeverValidator's physics sandbox
 *
 * Loads a standalone Challenge.Gbx or the embedded map of a Replay.Gbx.
 * Recorded controls/outcomes are ignored.
 * Requires the Packs directory from TrackMania United Forever.
 *
 * Uses the experimental PhysicsSandbox API; errors from the library are
 * reported as std::runtime_error with the operation and diagnostic.
 *
 * @private @param sandbox         Physics state and canonical input timeline
 * @private @param randomEngine    Local generator for reproducible controls
 * @private @param recordingReplay Whether map and state recording is enabled
 * @private @param replayMap       Original scenario bytes retained for export
 * @private @param replayStates    Recorded states starting at race time zero
 ******************************************************************************/
class TrackmaniaSimulator
{
public:
  using State = forevervalidator::experimental::PhysicsSandboxStateView;

  static constexpr std::uint32_t tickDurationMs = 10;

  /******************************************************************************
   * @struct Input
   *
   * @brief Complete driving controls applied to the next simulation tick
   *
   * Accelerate and brake may both be pressed. Steering uses native analog
   * units; race lifecycle and respawn events are not represented here.
   ******************************************************************************/
  struct Input
  {
    bool accelerate = false;
    bool brake = false;
    // Native analog units, in [kAnalogInputMinimum, kAnalogInputMaximum].
    forevervalidator::AnalogInputState steering = 0;
  };

  /******************************************************************************
   * @enum StopReason
   *
   * @brief Distinguishes completing the race from reaching the simulation limit
   ******************************************************************************/
  enum class StopReason
  {
    Finished,
    TimeLimitReached
  };

  /******************************************************************************
   * @struct RunResult
   *
   * @brief Captures why run() stopped and the final observed simulation state
   ******************************************************************************/
  struct RunResult
  {
    StopReason reason;
    State finalState;
  };

  /******************************************************************************
   * @brief Loads a scenario and advances its countdown to race time zero
   *
   * @param scenarioPath Path to a Challenge.Gbx or a replay containing a map
   * @param options      Race horizon, random seed and recording configuration
   *
   * @throws std::invalid_argument If the simulation horizon is invalid
   * @throws std::runtime_error If the repository environment variable is absent,
   *                            or assets, scenario or sandbox cannot be loaded
   ******************************************************************************/
  TrackmaniaSimulator(const std::string &scenarioPath,
                      const TrackmaniaSimulatorOptions &options = {});

  /******************************************************************************
   * @brief Reads the current observation without advancing the simulation
   *
   * @return Current car, controls, checkpoint and race state by value
   * @throws std::runtime_error If the sandbox cannot provide its state
   ******************************************************************************/
  State readState() const;

  /******************************************************************************
   * @brief Applies driving controls and advances the simulation by 10 ms
   *
   * Preserves earlier inputs and records the new state if recording is enabled.
   *
   * @param input Complete driving controls for the next tick
   *
   * @return State after advancing one tick
   * @throws std::invalid_argument If steering is outside [-65536, 65536]
   * @throws std::logic_error If the race has finished or reached its time limit
   * @throws std::runtime_error If reading, updating or advancing the sandbox fails
   ******************************************************************************/
  State step(const Input &input);

  /******************************************************************************
   * @brief Runs random driving inputs until the race finishes or times out
   *
   * Continues from the current state. Observations do not influence the random
   * policy; reaching the time limit does not mark the race as completed.
   *
   * @return Stop reason and final simulation state
   * @throws std::runtime_error If observing or stepping the sandbox fails
   ******************************************************************************/
  RunResult run();

  /******************************************************************************
   * @brief Exports the recorded run as an experimental TMF Stadium replay
   *
   * Requires recordReplay, a standalone Challenge.Gbx and at least one tick.
   * Writes a scripted, possibly unfinished ghost. Parent directories are
   * created and an existing destination file is overwritten.
   *
   * @param replayPath Destination Replay.Gbx file
   *
   * @throws std::logic_error If recording is disabled or has insufficient states
   * @throws std::invalid_argument If the recording or map format is unsupported
   * @throws std::length_error If an encoded size exceeds the GBX limit
   * @throws std::runtime_error If input retrieval, compression or file I/O fails
   ******************************************************************************/
  void exportReplay(const std::string &replayPath) const;

private:
  /******************************************************************************
   * @brief Samples independent accelerate/brake switches and uniform steering
   *
   * @return Complete driving input drawn from the simulator's local generator
   ******************************************************************************/
  Input selectRandomInput();

  forevervalidator::experimental::PhysicsSandbox sandbox;
  std::mt19937 randomEngine;
  bool recordingReplay;
  forevervalidator::AssetBytes replayMap;
  std::vector<State> replayStates;
};
