#include <viam/trajex/totg/phase_plane_csv.hpp>

#include <fstream>
#include <stdexcept>

namespace viam::trajex::totg {

void write_phase_plane_csv(const trajectory& traj,
                           const std::string& prefix,
                           std::size_t num_samples) {
    // Step 1: Write trajectory CSV from integration points
    {
        std::ofstream out(prefix + "_trajectory.csv");
        if (!out) {
            throw std::runtime_error("Failed to open " + prefix + "_trajectory.csv for writing");
        }
        out << "time,s,s_dot,s_ddot\n";
        for (const auto& pt : traj.get_integration_points()) {
            out << pt.time.count() << ','
                << static_cast<double>(pt.s) << ','
                << static_cast<double>(pt.s_dot) << ','
                << static_cast<double>(pt.s_ddot) << '\n';
        }
    }

    // Step 2: Write limit curves CSV sampled at uniform arc lengths
    {
        std::ofstream out(prefix + "_limits.csv");
        if (!out) {
            throw std::runtime_error("Failed to open " + prefix + "_limits.csv for writing");
        }
        out << "s,s_dot_max_acc,s_dot_max_vel\n";

        const auto& p = traj.path();
        auto cursor = p.create_cursor();
        const double length = static_cast<double>(p.length());

        for (std::size_t i = 0; i <= num_samples; ++i) {
            const double s = length * static_cast<double>(i) / static_cast<double>(num_samples);
            cursor.seek(arc_length{s});
            auto limits = traj.get_velocity_limits(cursor);
            out << s << ','
                << static_cast<double>(limits.s_dot_max_acc) << ','
                << static_cast<double>(limits.s_dot_max_vel) << '\n';
        }
    }
}

}  // namespace viam::trajex::totg
