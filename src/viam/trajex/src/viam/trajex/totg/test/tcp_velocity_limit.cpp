// TCP velocity limit tests for TOTG trajectory generation

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>

#include <boost/test/unit_test.hpp>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#if __has_include(<xtensor/generators/xbuilder.hpp>)
#include <xtensor/generators/xbuilder.hpp>
#else
#include <xtensor/xbuilder.hpp>
#endif

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <jacobian.hpp>
#include <urdf_parser.hpp>

#include <viam/trajex/totg/json_serialization.hpp>
#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>

namespace {

using namespace viam::trajex::totg;

// Identity Jacobian for 3-DOF: maps joint velocities directly to TCP velocity.
// This means TCP velocity = joint velocity, simplifying analytical predictions.
auto identity_jacobian = [](const xt::xarray<double>&) -> xt::xarray<double> {
    return xt::eye<double>(3);
};

// Zero Jacobian: simulates a singularity where no joint motion produces TCP motion.
auto zero_jacobian = [](const xt::xarray<double>&) -> xt::xarray<double> {
    return xt::zeros<double>({3u, 3u});
};

trajectory::tcp_limit make_tcp(double max_vel,
                               std::function<xt::xarray<double>(const xt::xarray<double>&)> jac) {
    return {.max_velocity = max_vel, .jacobian = std::move(jac)};
}

void write_phase_plane_json(const std::string& filename,
                            const trajectory& traj,
                            const trajectory_integration_event_collector& collector) {
    std::ofstream out(filename);
    write_trajectory_json(out, traj, collector);
    out.close();
    BOOST_TEST_MESSAGE("Wrote phase plane JSON to " << filename);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(tcp_velocity_limit_tests)

// Validation: tcp_limit with no jacobian must be rejected
BOOST_AUTO_TEST_CASE(rejects_tcp_without_jacobian) {
    const xt::xarray<double> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const auto p = path::create(waypoints);

    trajectory::options opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
        .tcp = trajectory::tcp_limit{.max_velocity = 0.5, .jacobian = nullptr},
    };

    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opts)), std::invalid_argument);
}

// Validation: non-positive max_velocity must be rejected
BOOST_AUTO_TEST_CASE(rejects_non_positive_tcp_velocity) {
    const xt::xarray<double> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const auto p = path::create(waypoints);

    trajectory::options opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
        .tcp = make_tcp(0.0, identity_jacobian),
    };

    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opts)), std::invalid_argument);

    opts.tcp = make_tcp(-1.0, identity_jacobian);
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opts)), std::invalid_argument);
}

// Large TCP limit: when TCP limit is much larger than joint limits, the trajectory
// should be identical to the no-TCP case (joint limits dominate).
BOOST_AUTO_TEST_CASE(large_tcp_limit_matches_no_tcp) {
    const xt::xarray<double> waypoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
    const auto p = path::create(waypoints);

    // Baseline: no TCP limit
    trajectory_integration_event_collector baseline_collector;
    trajectory::options baseline_opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
    };
    baseline_opts.observer = &baseline_collector;
    const auto baseline = trajectory::create(p, baseline_opts);
    write_phase_plane_json("large_tcp_limit_baseline.json", baseline, baseline_collector);

    // With TCP limit much larger than any joint can produce
    trajectory_integration_event_collector tcp_collector;
    trajectory::options tcp_opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
        .tcp = make_tcp(1000.0, identity_jacobian),
    };
    tcp_opts.observer = &tcp_collector;
    const auto with_tcp = trajectory::create(p, tcp_opts);
    write_phase_plane_json("large_tcp_limit_with_tcp.json", with_tcp, tcp_collector);

    // Durations should be identical (within floating point tolerance)
    BOOST_CHECK_CLOSE(baseline.duration().count(), with_tcp.duration().count(), 0.1);
}

// Small TCP limit: when TCP limit is smaller than joint limits, the trajectory
// duration should increase because the TCP constraint is more restrictive.
BOOST_AUTO_TEST_CASE(small_tcp_limit_increases_duration) {
    const xt::xarray<double> waypoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
    const auto p = path::create(waypoints);

    // Baseline: no TCP limit
    trajectory_integration_event_collector baseline_collector;
    trajectory::options baseline_opts{
        .max_velocity = xt::xarray<double>{2.0, 2.0, 2.0},
        .max_acceleration = xt::xarray<double>{2.0, 2.0, 2.0},
    };
    baseline_opts.observer = &baseline_collector;
    const auto baseline = trajectory::create(p, baseline_opts);
    write_phase_plane_json("small_tcp_limit_baseline.json", baseline, baseline_collector);

    // With restrictive TCP limit (0.1 m/s is much slower than joint limits)
    trajectory_integration_event_collector tcp_collector;
    trajectory::options tcp_opts{
        .max_velocity = xt::xarray<double>{2.0, 2.0, 2.0},
        .max_acceleration = xt::xarray<double>{2.0, 2.0, 2.0},
        .tcp = make_tcp(0.1, identity_jacobian),
    };
    tcp_opts.observer = &tcp_collector;
    const auto with_tcp = trajectory::create(p, tcp_opts);
    write_phase_plane_json("small_tcp_limit_with_tcp.json", with_tcp, tcp_collector);

    // TCP-constrained trajectory should be slower
    BOOST_CHECK_GT(with_tcp.duration().count(), baseline.duration().count());
}

// Singularity: zero Jacobian means TCP is never moving, so the TCP constraint
// is non-constraining (returns infinity). Trajectory should match baseline.
BOOST_AUTO_TEST_CASE(zero_jacobian_is_non_constraining) {
    const xt::xarray<double> waypoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const auto p = path::create(waypoints);

    // Baseline: no TCP limit
    trajectory_integration_event_collector baseline_collector;
    trajectory::options baseline_opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
    };
    baseline_opts.observer = &baseline_collector;
    const auto baseline = trajectory::create(p, baseline_opts);
    write_phase_plane_json("zero_jacobian_baseline.json", baseline, baseline_collector);

    // With zero Jacobian: TCP velocity is always zero, so limit is infinity
    trajectory_integration_event_collector tcp_collector;
    trajectory::options tcp_opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
        .tcp = make_tcp(0.5, zero_jacobian),
    };
    tcp_opts.observer = &tcp_collector;
    const auto with_tcp = trajectory::create(p, tcp_opts);
    write_phase_plane_json("zero_jacobian_with_tcp.json", with_tcp, tcp_collector);

    // Duration should be identical since TCP constraint is non-constraining
    BOOST_CHECK_CLOSE(baseline.duration().count(), with_tcp.duration().count(), 0.1);
}

// Straight-line path with identity Jacobian: TCP velocity limit should produce
// a constant constraint along the path, giving predictable behavior.
BOOST_AUTO_TEST_CASE(straight_line_with_identity_jacobian) {
    // Single straight segment along first axis
    const xt::xarray<double> waypoints = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    const auto p = path::create(waypoints);

    // With identity Jacobian, the TCP velocity along a straight line in joint 0
    // equals the joint 0 velocity. Setting TCP limit = 0.5 should be equivalent
    // to setting joint 0 max velocity to 0.5 (since tangent is [1,0,0]).
    trajectory_integration_event_collector tcp_collector;
    trajectory::options tcp_opts{
        .max_velocity = xt::xarray<double>{2.0, 2.0, 2.0},
        .max_acceleration = xt::xarray<double>{2.0, 2.0, 2.0},
        .tcp = make_tcp(0.5, identity_jacobian),
    };
    tcp_opts.observer = &tcp_collector;
    const auto with_tcp = trajectory::create(p, tcp_opts);
    write_phase_plane_json("straight_line_identity_jacobian_tcp.json", with_tcp, tcp_collector);

    // Compare with equivalent joint limit
    trajectory_integration_event_collector joint_collector;
    trajectory::options joint_opts{
        .max_velocity = xt::xarray<double>{0.5, 2.0, 2.0},
        .max_acceleration = xt::xarray<double>{2.0, 2.0, 2.0},
    };
    joint_opts.observer = &joint_collector;
    const auto with_joint = trajectory::create(p, joint_opts);
    write_phase_plane_json("straight_line_identity_jacobian_joint.json", with_joint, joint_collector);

    // Durations should be very close since constraints are equivalent
    BOOST_CHECK_CLOSE(with_tcp.duration().count(), with_joint.duration().count(), 1.0);
}

// Both options unset: trajectory should work normally (opt-in behavior)
BOOST_AUTO_TEST_CASE(no_tcp_options_works_normally) {
    const xt::xarray<double> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const auto p = path::create(waypoints);

    trajectory::options opts{
        .max_velocity = xt::xarray<double>{1.0, 1.0, 1.0},
        .max_acceleration = xt::xarray<double>{1.5, 1.5, 1.5},
    };

    BOOST_CHECK_NO_THROW(static_cast<void>(trajectory::create(p, opts)));
}

// UR20 URDF spiral path: exercises the real Jacobian over a complex multi-waypoint
// trajectory with TCP velocity limits.
BOOST_AUTO_TEST_CASE(ur20_spiral_phase_plane) {
    // Resolve the UR20 URDF relative to this source file
    const auto urdf_path =
        std::filesystem::path(__FILE__).parent_path() / "../../../../kinematics/ur20.urdf";
    BOOST_REQUIRE(std::filesystem::exists(urdf_path));

    auto jac_model = std::make_shared<jacobian::Model>(
        jacobian::parseURDF(urdf_path.string()));

    // Build Jacobian callback matching the production code in ur_arm.cpp
    auto jac_fn = [jac_model](const xt::xarray<double>& q) -> xt::xarray<double> {
        std::array<double, 6> joint_angles{};
        for (std::size_t i = 0; i < 6; ++i) {
            joint_angles[i] = q(i);
        }
        jacobian::Data data;
        jacobian::computeJacobian(*jac_model, joint_angles, data);

        xt::xarray<double> result = xt::zeros<double>({3u, 6u});
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 6; ++c) {
                result(r, c) = data.Jv(
                    static_cast<Eigen::Index>(r),
                    static_cast<Eigen::Index>(c));
            }
        }
        return result;
    };

    // Degrees to radians
    constexpr auto deg = [](double d) { return d * std::numbers::pi / 180.0; };

    // Path through UR20 joint space exercising joints 1 and 2 through a
    // significant range. The Jacobian varies meaningfully over this range,
    // producing a non-trivial TCP velocity limit curve in the phase plane.
    // clang-format off
    const xt::xarray<double> waypoints = {
        {deg(  0), deg(  0), deg(0), deg(0), deg(0), deg(0)},
        {deg(-45), deg(-45), deg(0), deg(0), deg(0), deg(0)},
        {deg(-45), deg(-90), deg(0), deg(0), deg(0), deg(0)},
    };
    // clang-format on

    const auto p = path::create(waypoints, path::options{}.set_max_deviation(0.1));

    // Use velocity/acceleration limits matching the working integration tests
    const xt::xarray<double> max_vel = {deg(50), deg(50), deg(50), deg(50), deg(50), deg(50)};
    const xt::xarray<double> max_acc = {deg(150), deg(150), deg(150), deg(150), deg(150), deg(150)};

    // Baseline: no TCP limit
    trajectory_integration_event_collector baseline_collector;
    trajectory::options baseline_opts{
        .max_velocity = max_vel,
        .max_acceleration = max_acc,
    };
    baseline_opts.observer = &baseline_collector;
    const auto baseline = trajectory::create(p, baseline_opts);
    write_phase_plane_json("ur20_spiral_baseline.json", baseline, baseline_collector);
    BOOST_TEST_MESSAGE("Baseline (no TCP) duration: " << baseline.duration().count() << "s");
    BOOST_CHECK_GT(baseline.duration().count(), 0.0);

    // With TCP velocity limit from UR20 Jacobian
    trajectory_integration_event_collector tcp_collector;
    trajectory::options tcp_opts{
        .max_velocity = max_vel,
        .max_acceleration = max_acc,
        .tcp = trajectory::tcp_limit{
            .max_velocity = 1.0,
            .jacobian = jac_fn,
        },
    };
    tcp_opts.observer = &tcp_collector;
    const auto traj = trajectory::create(p, tcp_opts);
    write_phase_plane_json("ur20_spiral_with_tcp.json", traj, tcp_collector);

    BOOST_CHECK_GT(traj.duration().count(), 0.0);
    BOOST_TEST_MESSAGE("TCP-constrained trajectory duration: " << traj.duration().count() << "s");
    BOOST_TEST_MESSAGE("Integration points: " << traj.get_integration_points().size());

    // TCP constraint should make trajectory slower (or equal)
    BOOST_CHECK_GE(traj.duration().count(), baseline.duration().count() - 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
