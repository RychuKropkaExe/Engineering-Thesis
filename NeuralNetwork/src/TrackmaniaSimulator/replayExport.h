#pragma once

#include <forevervalidator/experimental/physics_sandbox.h>

#include <string>
#include <vector>

namespace trackmania
{
// Minimal GBX v6 / TMF Stadium writer, not a general-purpose GBX serializer.
void writeReplay(
    const std::string &path,
    const forevervalidator::AssetBytes &map,
    const std::vector<forevervalidator::experimental::PhysicsSandboxStateView> &states,
    const std::vector<forevervalidator::experimental::PhysicsSandboxInputEvent> &inputs);
}
