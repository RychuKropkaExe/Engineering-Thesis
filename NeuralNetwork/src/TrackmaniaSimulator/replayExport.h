#pragma once

#include <forevervalidator/experimental/physics_sandbox.h>

#include <string>
#include <vector>

namespace trackmania
{
/******************************************************************************
 * @brief Writes a scripted TMF Stadium replay with an embedded map and ghost
 *
 * This is a minimal GBX v6 writer, not a general-purpose GBX serializer. States
 * must start at race time zero and be spaced by 10 ms. At least two states are
 * required. Missing cosmetic vehicle data uses defaults, and an unfinished run
 * is not marked as finished. Creates parent directories and overwrites the file.
 *
 * @param path   Destination Replay.Gbx file
 * @param map    Complete standalone binary GBX v6 Challenge.Gbx bytes
 * @param states Recorded Stadium car states, including the initial observation
 * @param inputs Canonical RaceRunning, Accelerate, Brake and Steer events
 *
 * @throws std::logic_error If the recording is too short or does not start at zero
 * @throws std::invalid_argument If map, states or input events are unsupported
 * @throws std::length_error If an encoded size exceeds the GBX limit
 * @throws std::runtime_error If state encoding, compression or file writing fails
 * @throws std::filesystem::filesystem_error If creating parent directories fails
 ******************************************************************************/
void writeReplay(
    const std::string &path,
    const forevervalidator::AssetBytes &map,
    const std::vector<forevervalidator::experimental::PhysicsSandboxStateView> &states,
    const std::vector<forevervalidator::experimental::PhysicsSandboxInputEvent> &inputs);
}
