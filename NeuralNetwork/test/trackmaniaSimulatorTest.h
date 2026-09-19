#pragma once

#include "trackmaniaSimulator.h"

#include <forevervalidator/native.h>
#include <format/replay/replay_file.h>

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
std::string getChallengeMapPath()
{
  const char *repositoryPath =
      std::getenv("GIT_REPOSITORY_NEURAL_NETWORK_PATH");
  if (repositoryPath == nullptr)
  {
    throw std::runtime_error(
        "Environment variable GIT_REPOSITORY_NEURAL_NETWORK_PATH is not set.");
  }
  return std::string(repositoryPath) +
         "/src/TrackmaniaSimulator/Maps/A03-Race.Challenge.Gbx";
}
}

TEST(TrackmaniaSimulatorTest, runCompletesWithoutCrashing)
{
  TrackmaniaSimulatorOptions options;
  // One tick is enough to exercise construction, map loading, random input,
  // input replacement, and PhysicsSandbox::AdvanceTicks without turning this
  // smoke test into a full random race.
  options.simulationHorizonMs = TrackmaniaSimulator::tickDurationMs;
  options.randomSeed = 2026;
  // An optional longer run produces something viewable without slowing down
  // the normal one-tick smoke test. The simulator validates tick alignment.
  if (const char *duration = std::getenv("TRACKMANIA_REPLAY_DURATION_MS"))
  {
    const std::string_view text(duration);
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                        options.simulationHorizonMs);
    ASSERT_EQ(parsed.ec, std::errc{});
    ASSERT_EQ(parsed.ptr, text.data() + text.size());
  }
  options.recordReplay = true;
  const std::string CHALLENGE_MAP = getChallengeMapPath();
  const auto SIMULATOR_DIRECTORY = std::filesystem::path(CHALLENGE_MAP).parent_path().parent_path();
  const auto REPLAY_PATH = SIMULATOR_DIRECTORY.parent_path().parent_path() / "build/replays" /
      ("A03-Race-" + std::to_string(options.simulationHorizonMs) + "ms.Replay.Gbx");

  EXPECT_NO_THROW({
    TrackmaniaSimulator simulator(CHALLENGE_MAP, options);
    EXPECT_THROW(simulator.exportReplay(REPLAY_PATH.string()), std::logic_error);
    const TrackmaniaSimulator::RunResult result = simulator.run();
    EXPECT_EQ(result.finalState.mapEnvironment, forevervalidator::MapEnvironment::Stadium);
    EXPECT_EQ(result.finalState.vehicleModel, forevervalidator::VehicleModel::StadiumCar);

    if (result.reason == TrackmaniaSimulator::StopReason::TimeLimitReached)
    {
      EXPECT_EQ(result.finalState.timeMs, options.simulationHorizonMs);
      EXPECT_FALSE(result.finalState.raceCompleted);
    }
    else
    {
      EXPECT_TRUE(result.finalState.raceCompleted);
    }

    simulator.exportReplay(REPLAY_PATH.string());
    EXPECT_GT(std::filesystem::file_size(REPLAY_PATH), 0u);
    RecordProperty("replay", REPLAY_PATH.string());

    // Decode and re-simulate the exported file, checking the recorded ghost
    // against the controls rather than merely checking the GBX magic bytes.
    using namespace forevervalidator;
    auto assets = OpenInstalledPackDirectory((SIMULATOR_DIRECTORY / "Packs").string());
    ASSERT_TRUE(assets) << assets.Error().diagnostic;
    auto context = CreateValidationContext(std::move(assets).Value());
    ASSERT_TRUE(context) << context.Error().diagnostic;
    auto bytes = ReadNativeReplayFile(REPLAY_PATH.string());
    ASSERT_TRUE(bytes) << bytes.Error().diagnostic;

    // Physics validation selects the model by name alone. The game resolves
    // the entire Ident; compare all three parts with the installed Stadium.pak
    // vehicle collector, whose collection is Vehicles (not the map's Stadium).
    ReplayFile decodedReplay;
    const auto decodeResult = ReadReplayBytes(
        reinterpret_cast<const std::uint8_t *>(bytes.Value().data()),
        bytes.Value().size(), &decodedReplay);
    ASSERT_EQ(decodeResult, ReplayFileReadError::Success)
        << ReplayFileReadErrorName(decodeResult);
    EXPECT_EQ(decodedReplay.VehicleIdentifier().id, "StadiumCar");
    EXPECT_EQ(decodedReplay.VehicleIdentifier().collection, "Vehicles");
    EXPECT_EQ(decodedReplay.VehicleIdentifier().author, "Nadeo");

    auto validation = ValidateReplay(context.Value(),
        {bytes.Value().data(), bytes.Value().size()}, {REPLAY_PATH.string()});
    ASSERT_TRUE(validation) << validation.Error().diagnostic;
    const auto &report = validation.Value();
    RecordProperty("replay_sample_count", static_cast<int>(report.metadata.sampleCount));
    RecordProperty("replay_max_deviation", std::to_string(report.maxDeviation));
    EXPECT_EQ(report.metadata.inputDurationMs, result.finalState.timeMs);
    EXPECT_EQ(report.metadata.sampleCount, result.finalState.timeMs / 10 + 1);
    EXPECT_EQ(report.metadata.replayProvenance, ReplayProvenance::Scripted);
    EXPECT_EQ(report.inputGhostMatch, InputGhostMatch::Match);
    EXPECT_EQ(report.simulation.raceCompleted, result.finalState.raceCompleted);
  });
}
