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

// Both the native API and sandbox API return checked results. Keep all
// failures visible to the caller, including loading and input replacement.
template<typename T, typename Error>
T takeValue(DiscriminatedResult<T, Error> result, const std::string &operation)
{
  if (!result)
  {
    throw std::runtime_error(operation + ": " + result.Error().diagnostic);
  }
  return std::move(result).Value();
}

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

TrackmaniaSimulator::State TrackmaniaSimulator::readState() const
{
  return takeValue(sandbox.ReadState(), "Read simulation state");
}

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

void TrackmaniaSimulator::exportReplay(const std::string &replayPath) const
{
  if (!recordingReplay)
  {
    throw std::logic_error("Replay recording was not enabled for this simulation");
  }
  trackmania::writeReplay(replayPath, replayMap, replayStates,
                         takeValue(sandbox.ReadInputs(), "Read replay inputs"));
}

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
