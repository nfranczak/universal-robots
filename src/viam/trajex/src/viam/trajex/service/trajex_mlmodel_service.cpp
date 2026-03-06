#include <viam/trajex/service/trajex_mlmodel_service.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <boost/variant/get.hpp>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/sdk/log/logging.hpp>

#include "jacobian.hpp"
#include "model.hpp"
#include "urdf_parser.hpp"

#include <viam/trajex/service/sampling_utils.hpp>
#include <viam/trajex/service/trajectory_planner.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace viam::trajex {

namespace vsdk = ::viam::sdk;

namespace {

// Result type for the trajectory planner. Accumulates samples as flat vectors
// for zero-copy output packing.
struct service_result {
    std::size_t dof = 0;
    std::vector<double> times;
    std::vector<double> configurations;
    std::vector<double> velocities;
    std::optional<std::vector<double>> accelerations;
    std::vector<double> tcp_velocities;  // ||J(q) * q_dot|| per sample
    double total_duration = 0.0;
};

// Holds owned output data and the named_tensor_views that reference it.
// Returned via an aliasing shared_ptr so the views remain valid for the caller.
struct infer_result {
    service_result data;
    trajex_mlmodel_service::named_tensor_views views;
};

const auto& get_double_tensor(const trajex_mlmodel_service::named_tensor_views& inputs, std::string_view name) {
    const auto it = inputs.find(std::string(name));
    if (it == inputs.end()) {
        throw std::invalid_argument("missing required input tensor: " + std::string(name));
    }
    const auto* view = boost::get<trajex_mlmodel_service::tensor_view<double>>(&it->second);
    if (!view) {
        throw std::invalid_argument("input tensor '" + std::string(name) + "' has wrong type (expected float64)");
    }
    return *view;
}

const auto& get_int64_tensor(const trajex_mlmodel_service::named_tensor_views& inputs, std::string_view name) {
    const auto it = inputs.find(std::string(name));
    if (it == inputs.end()) {
        throw std::invalid_argument("missing required input tensor: " + std::string(name));
    }
    const auto* view = boost::get<trajex_mlmodel_service::tensor_view<std::int64_t>>(&it->second);
    if (!view) {
        throw std::invalid_argument("input tensor '" + std::string(name) + "' has wrong type (expected int64)");
    }
    return *view;
}

double get_scalar_double(const trajex_mlmodel_service::named_tensor_views& inputs, std::string_view name) {
    const auto& view = get_double_tensor(inputs, name);
    if (view.size() != 1) {
        throw std::invalid_argument("input tensor '" + std::string(name) + "' must be a scalar (shape [1])");
    }
    return view.flat(0);
}

}  // namespace

std::shared_ptr<jacobian::Model> trajex_mlmodel_service::get_or_parse_urdf(const std::string& urdf_xml) {
    const std::lock_guard lock{urdf_cache_mutex_};

    // Check cache (keyed on full content string — no hash collisions)
    auto it = urdf_cache_map_.find(urdf_xml);
    if (it != urdf_cache_map_.end()) {
        // Move to front (most recently used)
        urdf_cache_list_.splice(urdf_cache_list_.begin(), urdf_cache_list_, it->second);
        return it->second->model;
    }

    // Parse new model. For typical 6-DOF robot URDFs this is ~microseconds.
    auto model = std::make_shared<jacobian::Model>(jacobian::parseURDFFromString(urdf_xml));

    // Evict least recently used if at capacity
    if (urdf_cache_list_.size() >= k_urdf_cache_capacity) {
        auto& back = urdf_cache_list_.back();
        VIAM_SDK_LOG(debug) << "URDF cache evicting model '" << back.model->name
                            << "' (" << back.content.size() << " bytes, capacity=" << k_urdf_cache_capacity << ")";
        urdf_cache_map_.erase(back.content);
        urdf_cache_list_.pop_back();
    }

    urdf_cache_list_.push_front({urdf_xml, model});
    // Key is a string_view into the list node's content — pointer-stable because
    // std::list never moves its nodes.
    urdf_cache_map_.emplace(std::string_view{urdf_cache_list_.front().content}, urdf_cache_list_.begin());

    return model;
}

std::vector<std::string> trajex_mlmodel_service::validate(const vsdk::ResourceConfig& cfg) {
    std::vector<std::string> errors;

    auto seq_attr = cfg.attributes().find("generator_sequence");
    if (seq_attr != cfg.attributes().end()) {
        const auto* arr = seq_attr->second.get<std::vector<vsdk::ProtoValue>>();
        if (!arr || arr->empty()) {
            errors.emplace_back("generator_sequence must be a non-empty array of strings");
        } else {
            for (const auto& elem : *arr) {
                const auto* str = elem.get<std::string>();
                if (!str) {
                    errors.emplace_back("generator_sequence entries must be strings");
                    break;
                }
                if (*str != "totg" && *str != "legacy") {
                    errors.emplace_back("generator_sequence: unknown algorithm '" + *str + "' (expected 'totg' or 'legacy')");
                }
            }
        }
    }

    auto seg_attr = cfg.attributes().find("segment_for_totg");
    if (seg_attr != cfg.attributes().end()) {
        if (!seg_attr->second.get<bool>()) {
            errors.emplace_back("segment_for_totg must be a boolean");
        }
    }

    return errors;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param): Signature fixed by ModelRegistration factory.
trajex_mlmodel_service::trajex_mlmodel_service(vsdk::Dependencies deps, vsdk::ResourceConfig config) : MLModelService(config.name()) {
    reconfigure(deps, config);
}

void trajex_mlmodel_service::reconfigure(const vsdk::Dependencies&, const vsdk::ResourceConfig& cfg) {
    config new_config;

    // Parse generator_sequence
    auto seq_attr = cfg.attributes().find("generator_sequence");
    if (seq_attr != cfg.attributes().end()) {
        const auto* arr = seq_attr->second.get<std::vector<vsdk::ProtoValue>>();
        if (!arr || arr->empty()) {
            throw std::invalid_argument("generator_sequence must be a non-empty array of strings");
        }
        new_config.generator_sequence.clear();
        for (const auto& elem : *arr) {
            const auto* str = elem.get<std::string>();
            if (!str) {
                throw std::invalid_argument("generator_sequence entries must be strings");
            }
            if (*str != "totg" && *str != "legacy") {
                throw std::invalid_argument("generator_sequence: unknown algorithm '" + *str + "' (expected 'totg' or 'legacy')");
            }
            new_config.generator_sequence.push_back(*str);
        }
    }

    // Parse segment_for_totg
    auto seg_attr = cfg.attributes().find("segment_for_totg");
    if (seg_attr != cfg.attributes().end()) {
        const auto* val = seg_attr->second.get<bool>();
        if (!val) {
            throw std::invalid_argument("segment_for_totg must be a boolean");
        }
        new_config.segment_for_totg = *val;
    }

    const std::unique_lock lock{config_mutex_};
    config_ = std::move(new_config);
}

std::shared_ptr<trajex_mlmodel_service::named_tensor_views> trajex_mlmodel_service::infer(const named_tensor_views& inputs,
                                                                                          const vsdk::ProtoStruct& extra) {
    // Snapshot config under the read lock, then release
    config local_config;
    {
        const std::shared_lock lock{config_mutex_};
        local_config = config_;
    }

    // Extract TCP parameters from extra field (per-call)
    std::shared_ptr<jacobian::Model> local_jac_model;
    std::optional<double> tcp_max_velocity;

    auto urdf_it = extra.find("urdf_xml");
    if (urdf_it != extra.end()) {
        const auto* urdf_xml = urdf_it->second.get<std::string>();
        if (!urdf_xml || urdf_xml->empty()) {
            throw std::invalid_argument("extra.urdf_xml must be a non-empty string");
        }
        local_jac_model = get_or_parse_urdf(*urdf_xml);
    }

    auto tcp_vel_it = extra.find("tcp_max_velocity_m_per_s");
    if (tcp_vel_it != extra.end()) {
        const auto* val = tcp_vel_it->second.get<double>();
        if (!val || *val <= 0.0) {
            throw std::invalid_argument("extra.tcp_max_velocity_m_per_s must be a positive number");
        }
        if (!local_jac_model) {
            throw std::invalid_argument("extra.tcp_max_velocity_m_per_s requires extra.urdf_xml to be set");
        }
        tcp_max_velocity = *val;
    }

    // Parse inputs
    const auto& waypoints_view = get_double_tensor(inputs, "waypoints_rads");
    if (waypoints_view.dimension() != 2) {
        throw std::invalid_argument("waypoints_rads must be 2-dimensional [n_waypoints, n_dof]");
    }

    const auto& velocity_limits_view = get_double_tensor(inputs, "velocity_limits_rads_per_sec");
    const auto& acceleration_limits_view = get_double_tensor(inputs, "acceleration_limits_rads_per_sec2");

    // Derive DOF from velocity limits and validate consistency
    const auto dof = velocity_limits_view.size();
    if (acceleration_limits_view.size() != dof) {
        throw std::invalid_argument("acceleration_limits size (" + std::to_string(acceleration_limits_view.size()) +
                                    ") does not match velocity_limits size (" + std::to_string(dof) + ")");
    }
    if (waypoints_view.shape(1) != dof) {
        throw std::invalid_argument("waypoints DOF (" + std::to_string(waypoints_view.shape(1)) + ") does not match limits DOF (" +
                                    std::to_string(dof) + ")");
    }

    const double path_tolerance = get_scalar_double(inputs, "path_tolerance_delta_rads");
    const double colinearization_ratio_val = get_scalar_double(inputs, "path_colinearization_ratio");
    const std::optional<double> colinearization_ratio =
        colinearization_ratio_val > 0.0 ? std::optional{colinearization_ratio_val} : std::nullopt;
    const double dedup_tolerance = get_scalar_double(inputs, "waypoint_deduplication_tolerance_rads");

    const auto& sampling_freq_view = get_int64_tensor(inputs, "trajectory_sampling_freq_hz");
    if (sampling_freq_view.size() != 1) {
        throw std::invalid_argument("trajectory_sampling_freq_hz must be a scalar (shape [1])");
    }
    const auto sampling_freq = static_cast<double>(sampling_freq_view.flat(0));

    // Copy inputs into owned xtensor arrays for the planner
    xt::xarray<double> velocity_limits(velocity_limits_view);
    xt::xarray<double> acceleration_limits(acceleration_limits_view);

    // Build TCP limit if jacobian model and tcp velocity are provided
    std::optional<totg::trajectory::tcp_limit> tcp_limit;
    if (local_jac_model && tcp_max_velocity) {
        auto jac_data = std::make_shared<jacobian::Data>(*local_jac_model);
        tcp_limit = totg::trajectory::tcp_limit{
            .max_velocity = *tcp_max_velocity,
            .jacobian = [model = local_jac_model, data = std::move(jac_data)](
                             const xt::xarray<double>& q) -> xt::xarray<double> {
                const auto n = static_cast<Eigen::Index>(q.size());
                const Eigen::Map<const Eigen::VectorXd> q_eigen(q.data(), n);
                jacobian::computeJacobian(*model, q_eigen, *data);
                xt::xarray<double> result = xt::zeros<double>({3u, static_cast<unsigned>(n)});
                for (Eigen::Index r = 0; r < 3; ++r) {
                    for (Eigen::Index c = 0; c < n; ++c) {
                        result(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) = data->J(r, c);
                    }
                }
                return result;
            },
        };
    }

    // Build the planner
    auto planner = trajectory_planner<service_result>({
        .velocity_limits = std::move(velocity_limits),
        .acceleration_limits = std::move(acceleration_limits),
        .path_blend_tolerance = path_tolerance,
        .colinearization_ratio = colinearization_ratio,
        .segment_trajex = local_config.segment_for_totg,
        .tcp = std::move(tcp_limit),
    });

    planner
        .with_waypoint_provider([&](auto& p) {
            // TODO(zero-copy): waypoint_accumulator should accept tensor views
            // directly to avoid this copy for large waypoint sets.
            auto wp = p.stash(xt::xarray<double>(waypoints_view));
            return totg::waypoint_accumulator{*wp};
        })
        .with_waypoint_preprocessor([dedup_tolerance](auto&, totg::waypoint_accumulator& accumulator) {
            accumulator = totg::deduplicate_waypoints(accumulator, dedup_tolerance);
        })
        .with_segmenter([](auto&, totg::waypoint_accumulator accumulator) { return totg::segment_at_reversals(std::move(accumulator)); });

    // Register algorithms based on configured sequence
    for (const auto& algo : local_config.generator_sequence) {
        if (algo == "totg") {
            planner.with_trajex([&, dof](service_result& acc, const totg::waypoint_accumulator&, const totg::trajectory& traj, auto) {
                acc.dof = dof;
                const double time_offset = acc.total_duration;
                acc.total_duration += traj.duration().count();
                if (!acc.accelerations) {
                    acc.accelerations.emplace();
                }

                auto sampler = totg::uniform_sampler::quantized_for_trajectory(traj, types::hertz{sampling_freq});
                for (const auto& sample : traj.samples(sampler) | std::views::drop(1)) {
                    acc.times.push_back(time_offset + sample.time.count());
                    std::ranges::copy(sample.configuration, std::back_inserter(acc.configurations));
                    std::ranges::copy(sample.velocity, std::back_inserter(acc.velocities));
                    std::ranges::copy(sample.acceleration, std::back_inserter(*acc.accelerations));
                }
            });
        } else if (algo == "legacy") {
            planner.with_legacy(
                [&, dof](service_result& acc, const totg::waypoint_accumulator&, const Path&, const Trajectory& traj, auto) {
                    acc.dof = dof;
                    const double time_offset = acc.total_duration;
                    acc.total_duration += traj.getDuration();

                    for_each_sample(traj.getDuration(), sampling_freq, [&](double t, double) {
                        acc.times.push_back(time_offset + t);
                        auto p = traj.getPosition(t);
                        auto v = traj.getVelocity(t);
                        for (Eigen::Index j = 0; j < p.size(); ++j) {
                            acc.configurations.push_back(p[j]);
                            acc.velocities.push_back(v[j]);
                        }
                    });
                });
        }
    }

    // Execute the planner
    auto planner_result = planner.execute([](const auto& p, auto trajex, auto legacy) -> std::optional<service_result> {
        if (trajex.receiver) {
            return std::move(trajex.receiver);
        }
        if (legacy.receiver) {
            return std::move(legacy.receiver);
        }

        // Legacy errors are authoritative (stable reference algorithm)
        if (legacy.error) {
            std::rethrow_exception(legacy.error);
        }
        if (trajex.error) {
            std::rethrow_exception(trajex.error);
        }

        if (p.processed_waypoint_count() < 2) {
            return std::nullopt;
        }

        throw std::logic_error("trajectory generation returned neither results nor an error");
    });

    if (!planner_result) {
        // No waypoints to process -- return empty output
        auto result_holder = std::make_shared<infer_result>();
        auto* views = &result_holder->views;
        return {std::move(result_holder), views};
    }

    // Pack output using aliasing shared_ptr for zero-copy output
    auto result_holder = std::make_shared<infer_result>();
    result_holder->data = std::move(*planner_result);

    const auto n_samples = result_holder->data.times.size();
    const auto output_dof = result_holder->data.dof;

    // Validate output consistency: flat vector sizes must be n_samples * dof
    if (result_holder->data.configurations.size() != n_samples * output_dof ||
        result_holder->data.velocities.size() != n_samples * output_dof) {
        throw std::logic_error("output tensor size mismatch: expected " + std::to_string(n_samples) + " samples x " +
                               std::to_string(output_dof) + " dof");
    }
    if (result_holder->data.accelerations && result_holder->data.accelerations->size() != n_samples * output_dof) {
        throw std::logic_error("acceleration output tensor size mismatch");
    }

    result_holder->views.emplace("sample_times_sec", make_tensor_view(result_holder->data.times.data(), n_samples, {n_samples}));

    result_holder->views.emplace(
        "configurations_rads",
        make_tensor_view(result_holder->data.configurations.data(), n_samples * output_dof, {n_samples, output_dof}));

    result_holder->views.emplace("velocities_rads_per_sec",
                                 make_tensor_view(result_holder->data.velocities.data(), n_samples * output_dof, {n_samples, output_dof}));

    if (result_holder->data.accelerations) {
        result_holder->views.emplace(
            "accelerations_rads_per_sec2",
            make_tensor_view(result_holder->data.accelerations->data(), n_samples * output_dof, {n_samples, output_dof}));
    }

    // Compute TCP velocities if jacobian model was provided for this call
    if (local_jac_model) {
        auto jac_data = std::make_unique<jacobian::Data>(*local_jac_model);
        auto& tcp_vels = result_holder->data.tcp_velocities;
        tcp_vels.resize(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            const auto n = static_cast<Eigen::Index>(output_dof);
            const Eigen::Map<const Eigen::VectorXd> q(&result_holder->data.configurations[i * output_dof], n);
            const Eigen::Map<const Eigen::VectorXd> q_dot(&result_holder->data.velocities[i * output_dof], n);
            jacobian::computeJacobian(*local_jac_model, q, *jac_data);
            // Linear velocity = top 3 rows of J * q_dot
            Eigen::Vector3d linear_vel = jac_data->J.topRows(3) * q_dot;
            tcp_vels[i] = linear_vel.norm();
        }
        result_holder->views.emplace("tcp_velocities_m_per_sec",
                                     make_tensor_view(tcp_vels.data(), n_samples, {n_samples}));
    }

    auto* views = &result_holder->views;
    return {std::move(result_holder), views};
}

struct trajex_mlmodel_service::metadata trajex_mlmodel_service::metadata(const vsdk::ProtoStruct&) {
    // No lock needed — metadata is static and does not depend on config_.
    return {
        .name = "trajex",
        .type = "other",
        .description = "Time-optimal trajectory generation via TOTG",
        .inputs =
            {
                {.name = "waypoints_rads",
                 .description = "Joint configurations (in radians) [n_waypoints, n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1, -1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "waypoint_deduplication_tolerance_rads",
                 .description = "Waypoint deduplication tolerance (in radians) [scalar]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "path_tolerance_delta_rads",
                 .description = "Path tolerance delta (in radians) [scalar]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "path_colinearization_ratio",
                 .description = "Path colinearization ratio [scalar]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "velocity_limits_rads_per_sec",
                 .description = "Max joint velocities (in radians per second) [n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "acceleration_limits_rads_per_sec2",
                 .description = "Max joint accelerations (in radians per second squared) [n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "trajectory_sampling_freq_hz",
                 .description = "Trajectory sampling frequency in Hz [scalar]",
                 .data_type = tensor_info::data_types::k_int64,
                 .shape = {1},
                 .associated_files = {},
                 .extra = {}},
            },
        .outputs =
            {
                {.name = "sample_times_sec",
                 .description = "Time values for each sample (in seconds) [n_samples]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "configurations_rads",
                 .description = "Joint configurations over time (in radians) [n_samples, n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1, -1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "velocities_rads_per_sec",
                 .description = "Joint velocities over time (in radians per second) [n_samples, n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1, -1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "accelerations_rads_per_sec2",
                 .description = "Joint accelerations over time (in radians per second squared) [n_samples, n_dof]",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1, -1},
                 .associated_files = {},
                 .extra = {}},
                {.name = "tcp_velocities_m_per_sec",
                 .description = "TCP linear velocity magnitude per sample (m/s) [n_samples]. "
                                "Conditional: only present when extra.urdf_xml is provided.",
                 .data_type = tensor_info::data_types::k_float64,
                 .shape = {-1},
                 .associated_files = {},
                 .extra = {}},
            },
    };
}

}  // namespace viam::trajex
