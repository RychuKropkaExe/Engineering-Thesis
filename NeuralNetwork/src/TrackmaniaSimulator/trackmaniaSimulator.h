#pragma once

#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/input_state.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct TrackmaniaSimulatorOptions
{
  // Race-relative limit, in milliseconds; must be a positive multiple of 10.
  std::uint32_t simulationHorizonMs = 120000;
  // A local generator keeps runs reproducible without affecting std::rand().
  std::uint32_t randomSeed = 2026;
  // Retain the map and car states for exportReplay(). Disabled by default.
  bool recordReplay = false;
};

/**
 * Owns a ForeverValidator sandbox running a fresh race on a Challenge.Gbx or
 * the embedded map of a Replay.Gbx. Recorded controls/outcomes are ignored.
 * Requires the Packs directory from TrackMania United Forever.
 *
 * Uses the experimental PhysicsSandbox API; errors from the library are
 * reported as std::runtime_error with the operation and diagnostic.
 */
class TrackmaniaSimulator
{
public:
  using State = forevervalidator::experimental::PhysicsSandboxStateView;

  static constexpr std::uint32_t tickDurationMs = 10;

  struct Input
  {
    bool accelerate = false;
    bool brake = false;
    // Native analog units, in [kAnalogInputMinimum, kAnalogInputMaximum].
    forevervalidator::AnalogInputState steering = 0;
  };

  enum class StopReason
  {
    Finished,
    TimeLimitReached
  };

  struct RunResult
  {
    StopReason reason;
    State finalState;
  };

  TrackmaniaSimulator(const std::string &scenarioPath,
                      const TrackmaniaSimulatorOptions &options = {});

  // Returns the current car, controls, checkpoint and race state by value.
  State readState() const;

  // Applies a complete driving input to the next 10 ms tick and returns its
  // resulting state. Throws std::invalid_argument for invalid steering and
  // std::logic_error if the race has finished or its time limit was reached.
  State step(const Input &input);

  // Reads state, selects random controls, and advances one tick until the race
  // finishes or the time limit is reached. Continues from the current state.
  // The observation is deliberately unused by the input policy for now.
  RunResult run();

  // Experimental TMF Stadium replay export; requires recordReplay and a
  // standalone Challenge.Gbx. Writes a scripted, possibly unfinished ghost.
  void exportReplay(const std::string &replayPath) const;

private:
  Input selectRandomInput();

  forevervalidator::experimental::PhysicsSandbox sandbox;
  std::mt19937 randomEngine;
  bool recordingReplay;
  forevervalidator::AssetBytes replayMap;
  std::vector<State> replayStates;
};
