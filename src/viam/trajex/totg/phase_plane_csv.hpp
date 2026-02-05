#pragma once

#include <cstddef>
#include <string>

#include <viam/trajex/totg/trajectory.hpp>

namespace viam::trajex::totg {

/// Writes two CSV files for phase plane visualization:
///   {prefix}_trajectory.csv  — time, s, s_dot, s_ddot
///   {prefix}_limits.csv      — s, s_dot_max_acc, s_dot_max_vel
///
/// @param traj          Completed trajectory to export
/// @param prefix        File path prefix (e.g. "/tmp/my_traj" produces
///                      "/tmp/my_traj_trajectory.csv" and "/tmp/my_traj_limits.csv")
/// @param num_samples   Number of uniformly-spaced arc length samples for limit curves (default 1000)
void write_phase_plane_csv(const trajectory& traj,
                           const std::string& prefix,
                           std::size_t num_samples = 1000);

}  // namespace viam::trajex::totg
