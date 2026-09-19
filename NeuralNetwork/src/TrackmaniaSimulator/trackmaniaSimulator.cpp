#include "trackmaniaSimulator.h"
#include "replayExport.h"

#include <forevervalidator/native.h>

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using namespace forevervalidator;
using namespace forevervalidator::experimental;

const std::filesystem::path PACKS_DIRECTORY =
    "NeuralNetwork/src/TrackmaniaSimulator/Packs";

/******************************************************************************
 * @brief Resolves the fixed Packs directory from the repository environment
 *
 * Accepts either the repository root or its NeuralNetwork subdirectory in
 * GIT_REPOSITORY_NEURAL_NETWORK_PATH.
 *
 * @return Lexically normalized path to TrackmaniaSimulator/Packs
 * @throws std::runtime_error If the environment variable is not set
 ******************************************************************************/
std::string getPacksDirectory()
{
  const char *environmentPath =
      std::getenv("GIT_REPOSITORY_NEURAL_NETWORK_PATH");
  if (environmentPath == nullptr)
  {
    throw std::runtime_error(
        "Environment variable GIT_REPOSITORY_NEURAL_NETWORK_PATH is not set.");
  }

  std::filesystem::path repositoryPath(environmentPath);
  // gitenv.sh is normally sourced from NeuralNetwork, while callers may also
  // provide the repository root. Support both forms when constructing the
  // absolute path to the fixed packs directory.
  if (repositoryPath.filename() == "NeuralNetwork")
  {
    repositoryPath = repositoryPath.parent_path();
  }
  return (repositoryPath / PACKS_DIRECTORY).lexically_normal().string();
}

/******************************************************************************
 * @brief Extracts a successful library result or reports its diagnostic
 *
 * @tparam T     Value type returned by the native or sandbox API
 * @tparam Error Error type exposing a diagnostic string
 * @param result    Checked result whose value may be moved out
 * @param operation Description prepended to an error diagnostic
 *
 * @return Value moved from a successful result
 * @throws std::runtime_error If the library operation failed
 ******************************************************************************/
template<typename T, typename Error>
T takeValue(DiscriminatedResult<T, Error> result, const std::string &operation)
{
  if (!result)
  {
    throw std::runtime_error(operation + ": " + result.Error().diagnostic);
  }
  return std::move(result).Value();
}

/******************************************************************************
 * @brief Creates a reference-CPU sandbox with a fresh canonical race timeline
 *
 * Loads the installed TMUF packs and configures 10 ms ticks with a 2600 ms
 * countdown. The horizon is bounded to keep input timestamps in signed range.
 *
 * @param options Simulation configuration supplying the race-relative horizon
 *
 * @return Initialized sandbox ready to load a scenario
 * @throws std::invalid_argument If the horizon is not a supported multiple of 10
 * @throws std::runtime_error If resolving packs or creating the sandbox fails
 ******************************************************************************/
PhysicsSandbox createSandbox(const TrackmaniaSimulatorOptions &options)
{
  constexpr std::uint32_t prestartDurationMs = 2600;
  constexpr auto maximumHorizonMs =
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) -
      prestartDurationMs;
  if (options.simulationHorizonMs < TrackmaniaSimulator::tickDurationMs ||
      options.simulationHorizonMs % TrackmaniaSimulator::tickDurationMs != 0 ||
      options.simulationHorizonMs > maximumHorizonMs)
  {
    throw std::invalid_argument(
        "Simulation horizon must be a positive multiple of 10 ms, at most " +
        std::to_string(maximumHorizonMs) + " ms");
  }

  const std::string packsPath = getPacksDirectory();
  auto assets = takeValue(OpenInstalledPackDirectory(packsPath),
                          "Open TMUF Packs directory '" + packsPath + "'");

  PhysicsSandboxOptions sandboxOptions;
  sandboxOptions.backend = SimulationBackend::Reference;
  sandboxOptions.tickDurationMs = TrackmaniaSimulator::tickDurationMs;
  sandboxOptions.prestartDurationMs = prestartDurationMs;
  sandboxOptions.timelineMode = PhysicsSandboxTimelineMode::Canonical;
  sandboxOptions.simulationHorizonMs = options.simulationHorizonMs;
  return takeValue(CreatePhysicsSandbox(std::move(assets), sandboxOptions),
                   "Create physics sandbox");
}

/******************************************************************************
 * @brief Creates a pressed or released switch event on the canonical timeline
 *
 * @param timeMs  Race-relative timestamp in milliseconds
 * @param action  Switch action, such as Accelerate or Brake
 * @param pressed Whether the switch should be pressed
 *
 * @return Input event with a canonical switch value
 ******************************************************************************/
PhysicsSandboxInputEvent switchEvent(std::int32_t timeMs,
                                    PhysicsSandboxInputAction action,
                                    bool pressed)
{
  PhysicsSandboxInputEvent event{};
  event.timeMs = timeMs;
  event.action = action;
  event.value.kind = PhysicsSandboxInputValueKind::Switch;
  event.value.switchState = pressed ? PhysicsSandboxSwitchState::Pressed
                                    : PhysicsSandboxSwitchState::Released;
  return event;
}
} // namespace

/******************************************************************************
 * @brief Loads a fresh race and optionally records its initial state
 *
 * @param scenarioPath Challenge.Gbx or Replay.Gbx providing the scenario map
 * @param options      Sandbox horizon, input generator seed and recording flag
 *
 * @throws std::invalid_argument If the requested horizon is invalid
 * @throws std::runtime_error If asset loading or scenario initialization fails
 ******************************************************************************/
TrackmaniaSimulator::TrackmaniaSimulator(
    const std::string &scenarioPath,
    const TrackmaniaSimulatorOptions &options)
    : sandbox(createSandbox(options)),
      randomEngine(options.randomSeed),
      recordingReplay(options.recordReplay)
{
  const forevervalidator::ReplayIdentity identity{scenarioPath};
  // Despite its name, this native helper only reads bytes; it also accepts a
  // Challenge.Gbx. LoadScenario performs the actual decoding.
  auto bytes = takeValue(forevervalidator::ReadNativeReplayFile(scenarioPath, identity),
                         "Read scenario '" + scenarioPath + "'");
  takeValue(sandbox.LoadScenario({bytes.data(), bytes.size()}, identity),
            "Load scenario '" + scenarioPath + "'");
  // LoadScenario runs the prestart internally and returns at race time 0 ms.
  if (recordingReplay)
  {
    replayMap = std::move(bytes);
    replayStates.push_back(readState());
  }
}

/******************************************************************************
 * @brief Obtains the current sandbox observation without changing the race
 *
 * @return Current simulation state
 * @throws std::runtime_error If reading the state fails
 ******************************************************************************/
TrackmaniaSimulator::State TrackmaniaSimulator::readState() const
{
  return takeValue(sandbox.ReadState(), "Read simulation state");
}

/******************************************************************************
 * @brief Replaces next-tick controls, advances physics and records the result
 *
 * @param input Accelerate, brake and native analog steering for the next tick
 *
 * @return State after one 10 ms simulation step
 * @throws std::invalid_argument If steering is outside the native analog range
 * @throws std::logic_error If the simulation has finished or timed out
 * @throws std::runtime_error If a sandbox operation fails
 ******************************************************************************/
TrackmaniaSimulator::State TrackmaniaSimulator::step(const Input &input)
{
  using namespace forevervalidator;
  using namespace forevervalidator::experimental;

  if (!IsAnalogInputStateValid(input.steering))
  {
    throw std::invalid_argument("Steering must be in [-65536, 65536]");
  }

  const State currentState = readState();
  if (currentState.raceCompleted || currentState.timeMs >= currentState.durationMs)
  {
    throw std::logic_error("Cannot step a finished or timed-out simulation");
  }

  // The sandbox samples the next tick at current race time + 10 ms.
  // Its validated horizon keeps this timestamp within the signed input range.
  const auto inputTimeMs =
      static_cast<std::int32_t>(currentState.timeMs + tickDurationMs);

  PhysicsSandboxInputEvent steering{};
  steering.timeMs = inputTimeMs;
  steering.action = PhysicsSandboxInputAction::Steer;
  steering.value.kind = PhysicsSandboxInputValueKind::Analog;
  steering.value.analog = input.steering;

  std::vector<PhysicsSandboxInputEvent> events{
      switchEvent(inputTimeMs, PhysicsSandboxInputAction::Accelerate, input.accelerate),
      switchEvent(inputTimeMs, PhysicsSandboxInputAction::Brake, input.brake),
      steering};

  // This is an inclusive, one-timestamp window. Preserve previous inputs and
  // especially the canonical RaceRunning event at 0 ms. Sending both pressed
  // and released values each tick prevents controls from remaining stuck on.
  takeValue(sandbox.ReplaceInputWindow(inputTimeMs, inputTimeMs, std::move(events)),
            "Set next-tick controls");
  const State state = takeValue(sandbox.AdvanceTicks(1), "Advance simulation by one tick");
  if (recordingReplay)
  {
    replayStates.push_back(state);
  }
  return state;
}

/******************************************************************************
 * @brief Writes retained states and sandbox inputs to a scripted replay
 *
 * @param replayPath Destination file, overwritten if it already exists
 *
 * @throws std::logic_error If recording was not enabled
 * @throws std::runtime_error If reading the input timeline fails
 * @see trackmania::writeReplay for format requirements and export errors
 ******************************************************************************/
void TrackmaniaSimulator::exportReplay(const std::string &replayPath) const
{
  if (!recordingReplay)
  {
    throw std::logic_error("Replay recording was not enabled for this simulation");
  }
  trackmania::writeReplay(replayPath, replayMap, replayStates,
                         takeValue(sandbox.ReadInputs(), "Read replay inputs"));
}

/******************************************************************************
 * @brief Samples driving controls independently of the observed game state
 *
 * Switches each have a 50 percent chance of being pressed. Steering is sampled
 * uniformly over the complete native integer range, including both endpoints.
 *
 * @return Random accelerate, brake and steering values
 ******************************************************************************/
TrackmaniaSimulator::Input TrackmaniaSimulator::selectRandomInput()
{
  std::bernoulli_distribution pressed(0.5);
  std::uniform_int_distribution<forevervalidator::AnalogInputState> steering(
      forevervalidator::kAnalogInputMinimum,
      forevervalidator::kAnalogInputMaximum);

  // Accelerate and brake are independent (both pressed is a valid input).
  // Analog Steer covers the range from full left through neutral to full right.
  // Respawn and race lifecycle events are not part of this driving policy.
  return {pressed(randomEngine), pressed(randomEngine), steering(randomEngine)};
}

/******************************************************************************
 * @brief Repeats observation, random input selection and a single physics tick
 *
 * Resumes at the current state and stops on race completion or the time limit.
 *
 * @return Stop reason together with the final observed state
 * @throws std::runtime_error If a sandbox operation fails
 ******************************************************************************/
TrackmaniaSimulator::RunResult TrackmaniaSimulator::run()
{
  State currentState = readState();
  while (!currentState.raceCompleted && currentState.timeMs < currentState.durationMs)
  {
    // 1. Observe the current state. TODO: pass it to the neural network here.
    // For now, the input policy ignores the car and checkpoint information.

    // 2. Select a fresh random driving input using the library's valid ranges.
    const Input input = selectRandomInput();

    // 3. Run one physics tick and retain its state for the next iteration.
    currentState = step(input);
  }

  return {currentState.raceCompleted ? StopReason::Finished
                                     : StopReason::TimeLimitReached,
          currentState};
}
