#include "replayExport.h"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>

namespace
{
using namespace forevervalidator;
using namespace forevervalidator::experimental;
constexpr std::uint32_t FACADE = 0xfacade01;
constexpr std::string_view NICKNAME = "TrackmaniaSimulator (scripted)";
constexpr float PI = std::numbers::pi_v<float>;

std::uint32_t size32(std::size_t size)
{
  if (size > std::numeric_limits<std::int32_t>::max())
    throw std::length_error("Replay exceeds the GBX size limit");
  return static_cast<std::uint32_t>(size);
}

struct Writer
{
  AssetBytes bytes;
  void u8(std::uint8_t value) { bytes.push_back(static_cast<std::byte>(value)); }
  void u16(std::uint16_t value) { u8(value & 255); u8(value >> 8); }
  void u32(std::uint32_t value) { u16(value & 65535); u16(value >> 16); }
  void f32(float value) { u32(std::bit_cast<std::uint32_t>(value)); }
  void append(std::span<const std::byte> value)
  { bytes.insert(bytes.end(), value.begin(), value.end()); }
  void string(std::string_view value)
  {
    u32(size32(value.size()));
    append(std::as_bytes(std::span(value.data(), value.size())));
  }
  void id(std::string_view value) { u32(0x40000000); string(value); }
  void wrapped(std::uint32_t chunk, const Writer &payload)
  {
    u32(chunk); u32(0x534b4950); u32(size32(payload.bytes.size())); append(payload.bytes);
  }
};

struct Reader
{
  std::span<const std::byte> bytes;
  std::size_t offset = 0;
  std::span<const std::byte> take(std::size_t count)
  {
    if (offset > bytes.size() || count > bytes.size() - offset)
      throw std::invalid_argument("Truncated Challenge.Gbx header");
    const auto result = bytes.subspan(offset, count);
    offset += count;
    return result;
  }
  std::uint32_t u32()
  {
    const auto value = take(4);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8) |
           (std::to_integer<std::uint32_t>(value[2]) << 16) |
           (std::to_integer<std::uint32_t>(value[3]) << 24);
  }
};

// The map's header Ident consists of UID, environment and author. Re-encode
// its IDs so references in the map's ID table never leak into the replay.
Writer mapIdent(const AssetBytes &map)
{
  Reader file{map};
  const auto prefix = file.take(9);
  if (prefix[0] != std::byte{'G'} || prefix[1] != std::byte{'B'} ||
      prefix[2] != std::byte{'X'} || prefix[3] != std::byte{6} ||
      prefix[4] != std::byte{0} || prefix[5] != std::byte{'B'} ||
      file.u32() != 0x03043000)
    throw std::invalid_argument("Replay export requires a binary GBX v6 Challenge.Gbx");
  const auto headerSize = file.u32();
  Reader header{file.take(headerSize)};
  const auto count = header.u32();
  if (count > (header.bytes.size() - 4) / 8)
    throw std::invalid_argument("Invalid Challenge.Gbx header table");
  std::size_t payloadOffset = 4 + count * 8;
  for (std::uint32_t i = 0; i < count; ++i)
  {
    const auto chunk = header.u32();
    const auto length = header.u32() & 0x7fffffff;
    Reader payload{header.bytes, payloadOffset};
    const auto data = payload.take(length);
    payloadOffset += length;
    if (chunk != 0x03043003) continue;

    Reader ident{data};
    ident.take(1); // Challenge header chunk version.
    if (ident.u32() != 3)
      throw std::invalid_argument("Unsupported map identifier format");
    Writer result;
    result.u32(3); // Start a new identifier table in the replay header.
    std::vector<std::string> names;
    for (int part = 0; part < 3; ++part)
    {
      const auto word = ident.u32();
      if (word == 0xffffffff || (word & 0xc0000000) == 0)
      {
        result.u32(word);
        continue;
      }
      const auto index = word & 0x0fffffff;
      if (index == 0)
      {
        const auto text = ident.take(ident.u32());
        names.emplace_back(reinterpret_cast<const char *>(text.data()), text.size());
        result.id(names.back());
      }
      else
      {
        if (index > names.size()) throw std::invalid_argument("Invalid map identifier reference");
        result.id(names[index - 1]);
      }
    }
    return result;
  }
  throw std::invalid_argument("Challenge.Gbx has no map identifier header");
}

std::uint16_t quantize(float value, float minimum, float maximum, unsigned scale)
{
  if (!std::isfinite(value)) throw std::runtime_error("Non-finite replay state");
  return static_cast<std::uint16_t>(std::lround(
      std::clamp((value - minimum) / (maximum - minimum), 0.0f, 1.0f) * scale));
}

void packedVelocity(Writer &out, Vector3 v)
{
  const float magnitude = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (magnitude < 0.00001f)
  {
    out.u32(0x8000);
    return;
  }
  out.u16(static_cast<std::int16_t>(std::clamp(std::lround(std::log(magnitude) * 1000),
                                            -32767l, 32767l)));
  out.u8(static_cast<std::int8_t>(std::lround(std::atan2(v.y, v.x) * 127 / PI)));
  out.u8(static_cast<std::int8_t>(std::lround(std::asin(std::clamp(v.z / magnitude,
                                                 -1.0f, 1.0f)) * 254 / PI)));
}

// TMF state version 9: 61 bytes per sample. Layout and quantization follow
// GBX.NET's CSceneVehicleCar.Sample and GbxWriter (see README references).
void sample(Writer &out, const PhysicsSandboxStateView &state)
{
  const auto &car = state.car;
  out.f32(car.position.x); out.f32(car.position.y); out.f32(car.position.z);
  const float angle = std::acos(std::clamp(car.rotationW, -1.0f, 1.0f));
  const float sine = std::sin(angle);
  const Vector3 axis = std::abs(sine) > 0.00001f
      ? Vector3{car.rotationX / sine, car.rotationY / sine, car.rotationZ / sine}
      : Vector3{1, 0, 0};
  out.u16(quantize(angle, 0, PI, 65535));
  out.u16(static_cast<std::int16_t>(std::atan2(axis.y, axis.x) * 32767 / PI));
  out.u16(static_cast<std::int16_t>(std::asin(std::clamp(axis.z, -1.0f, 1.0f)) * 65534 / PI));
  packedVelocity(out, car.linearSpeed);
  packedVelocity(out, car.angularSpeed);
  out.u16(quantize(car.signedSpeed, -1000, 10000, 65535));
  out.u16(quantize(car.localSpeed.x, -1000, 1000, 65535));
  out.u16(quantize(car.rpm, 0, 30000, 65535));
  for (int wheel = 0; wheel < 4; ++wheel) out.u16(0); // Wheel spin is not exposed.
  out.u8(quantize(state.steering, -1, 1, 255));
  out.u8(quantize(state.accelerate, 0, 1, 255));
  out.u8(quantize(state.brake, 0, 1, 255));
  out.u8(0); out.u8(0); out.u8(128); out.u8(128); // Unknown visual fields.
  out.u8(quantize(car.turbo, 0, 1, 255));
  out.u8(128); // Front-wheel angle unavailable; neutral presentation.
  for (int wheel = 0; wheel < 4; ++wheel)
  {
    out.u8(128); // Suspension length unavailable.
    out.u8(static_cast<std::uint8_t>(car.wheelSurface[wheel]));
  }
  out.u8(static_cast<std::uint8_t>(std::clamp(car.gear, 0, 7)));
  out.u8((car.wheelSliding[0] ? 64 : 0) | (car.wheelContact[0] ? 128 : 0));
  unsigned flags = 0;
  for (int wheel = 1; wheel < 4; ++wheel)
    flags |= ((car.wheelSliding[wheel] ? 1 : 0) | (car.wheelContact[wheel] ? 2 : 0))
             << ((wheel - 1) * 2);
  out.u8(flags);
  out.u8(0); // Dirt blend unavailable.
}

Writer ghostStates(const std::vector<PhysicsSandboxStateView> &states)
{
  Writer raw;
  raw.u32(0x0a02b000); raw.u32(1); raw.u32(0); raw.u32(10); raw.u32(9);
  raw.u32(size32(states.size() * 61));
  for (const auto &state : states) sample(raw, state);
  raw.u32(size32(states.size())); raw.u32(0); raw.u32(61);

  AssetBytes compressed(compressBound(raw.bytes.size()));
  uLongf length = compressed.size();
  if (compress2(reinterpret_cast<Bytef *>(compressed.data()), &length,
                reinterpret_cast<const Bytef *>(raw.bytes.data()), raw.bytes.size(),
                Z_BEST_COMPRESSION) != Z_OK)
    throw std::runtime_error("Cannot compress replay ghost states");
  compressed.resize(length);
  Writer chunk;
  chunk.u32(0x0303f005); chunk.u32(size32(raw.bytes.size()));
  chunk.u32(size32(compressed.size())); chunk.append(compressed);
  return chunk;
}

void writeInputs(Writer &out, const std::vector<PhysicsSandboxInputEvent> &inputs,
                 std::uint32_t duration)
{
  constexpr std::array ACTIONS{PhysicsSandboxInputAction::RaceRunning,
      PhysicsSandboxInputAction::Accelerate, PhysicsSandboxInputAction::Brake,
      PhysicsSandboxInputAction::Steer};
  constexpr std::array NAMES{"_FakeIsRaceRunning", "Accelerate", "Brake", "Steer"};
  out.u32(0x03092019); out.u32(duration); out.u32(0); out.u32(ACTIONS.size());
  for (const auto name : NAMES) out.id(name);
  out.u32(size32(inputs.size())); out.u32(size32(inputs.size()));
  for (const auto &input : inputs)
  {
    const auto action = std::find(ACTIONS.begin(), ACTIONS.end(), input.action);
    if (action == ACTIONS.end() || input.timeMs < 0 ||
        static_cast<std::uint32_t>(input.timeMs) > duration)
      throw std::invalid_argument("Unsupported replay input or timestamp");
    // TMF clock base + TMInterface's scripted-clock marker. ForeverValidator
    // recognizes and normalizes this offset, identifying the run as scripted.
    out.u32(100000u + 65535u + static_cast<std::uint32_t>(input.timeMs));
    out.u8(static_cast<std::uint8_t>(action - ACTIONS.begin()));
    out.u32(input.value.kind == PhysicsSandboxInputValueKind::Analog
                ? static_cast<std::uint32_t>(-input.value.analog) & 0x00ffffff
                : (input.value.switchState == PhysicsSandboxSwitchState::Pressed ? 1 : 0));
  }
  out.string("TrackmaniaSimulator/ForeverValidator");
  out.u32(0); out.u32(0); out.u32(0); out.string(""); out.u32(0);
}

// A standards-compatible literal-only LZO1X stream. The embedded map is
// already compressed; avoiding match compression needs no extra dependency.
Writer lzoLiteral(const AssetBytes &bytes)
{
  Writer out;
  if (bytes.size() <= 238) out.u8(static_cast<std::uint8_t>(17 + bytes.size()));
  else
  {
    out.u8(0);
    auto remaining = bytes.size() - 18;
    while (remaining > 255) { out.u8(0); remaining -= 255; }
    out.u8(static_cast<std::uint8_t>(remaining));
  }
  out.append(bytes); out.u8(0x11); out.u8(0); out.u8(0);
  return out;
}
} // namespace

void trackmania::writeReplay(
    const std::string &path, const AssetBytes &map,
    const std::vector<PhysicsSandboxStateView> &states,
    const std::vector<PhysicsSandboxInputEvent> &inputs)
{
  if (states.size() < 2 || states.front().timeMs != 0)
    throw std::logic_error("Replay export needs a recording starting at 0 ms and at least one tick");
  for (std::size_t i = 0; i < states.size(); ++i)
    if (states[i].timeMs != i * 10 || states[i].vehicleModel != VehicleModel::StadiumCar ||
        states[i].mapEnvironment != MapEnvironment::Stadium)
      throw std::invalid_argument("Replay export supports only contiguous 10 ms Stadium recordings");
  const auto &last = states.back();
  const auto duration = size32(last.timeMs);
  const auto finish = last.finishTimeMs.value_or(0xffffffff);

  Writer info;
  info.u32(7); info.append(mapIdent(map).bytes); info.u32(finish);
  info.string(NICKNAME); info.string("");
  Writer header;
  header.u32(1); header.u32(0x03093000); header.u32(size32(info.bytes.size()));
  header.append(info.bytes);

  Writer body;
  body.u32(0x03093002); body.u32(size32(map.size())); body.append(map);
  body.u32(0x03093014); body.u32(10); body.u32(1); // Deprecated array version, count.
  body.u32(1); body.u32(0x03092000); // Ghost node index and class.
  body.append(ghostStates(states).bytes);
  body.u32(0x03092018); body.u32(3);
  // Match the vehicle collector's complete Ident in Stadium.pak. Vehicles is
  // its asset collection; Stadium is the map environment, not this collection.
  // The game needs the author too, although FV selects physics by model name.
  body.id("StadiumCar"); body.id("Vehicles"); body.id("Nadeo");
  Writer name;
  name.u32(0); name.string(NICKNAME); name.string("");
  body.wrapped(0x03092017, name);
  Writer time;
  time.u32(finish); body.wrapped(0x03092005, time);
  Writer respawns;
  respawns.u32(last.respawnCount); body.wrapped(0x03092008, respawns);
  writeInputs(body, inputs, duration);
  body.u32(FACADE); body.u32(0); body.u32(0);
  body.u32(0x03093015); body.u32(0xffffffff); body.u32(FACADE);

  const auto compressed = lzoLiteral(body.bytes);
  Writer file;
  file.u8('G'); file.u8('B'); file.u8('X'); file.u16(6);
  file.u8('B'); file.u8('U'); file.u8('C'); file.u8('R'); file.u32(0x03093000);
  file.u32(size32(header.bytes.size())); file.append(header.bytes);
  file.u32(2); file.u32(0); // Root + ghost; no external node references.
  file.u32(size32(body.bytes.size())); file.u32(size32(compressed.bytes.size()));
  file.append(compressed.bytes);

  const std::filesystem::path outputPath(path);
  if (outputPath.has_parent_path()) std::filesystem::create_directories(outputPath.parent_path());
  std::ofstream stream(outputPath, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(file.bytes.data()), file.bytes.size());
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write replay '" + path + "'");
}
