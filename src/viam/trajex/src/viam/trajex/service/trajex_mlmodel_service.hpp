#pragma once

#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/resource/reconfigurable.hpp>
#include <viam/sdk/services/mlmodel.hpp>

namespace jacobian {
struct Model;
}  // namespace jacobian

namespace viam::trajex {

class trajex_mlmodel_service final : public ::viam::sdk::MLModelService, public ::viam::sdk::Reconfigurable {
   public:
    trajex_mlmodel_service(::viam::sdk::Dependencies deps, ::viam::sdk::ResourceConfig config);

    void reconfigure(const ::viam::sdk::Dependencies&, const ::viam::sdk::ResourceConfig&) override;

    std::shared_ptr<named_tensor_views> infer(const named_tensor_views& inputs, const ::viam::sdk::ProtoStruct& extra) override;

    struct metadata metadata(const ::viam::sdk::ProtoStruct& extra) override;

    static std::vector<std::string> validate(const ::viam::sdk::ResourceConfig& cfg);

   private:
    struct config {
        std::vector<std::string> generator_sequence = {"totg", "legacy"};
        bool segment_for_totg = true;
        std::optional<std::string> urdf_file_path;
        std::optional<double> tcp_max_velocity_m_per_s;
    };

    mutable std::shared_mutex config_mutex_;
    config config_;
    std::shared_ptr<jacobian::Model> jac_model_;
};

}  // namespace viam::trajex
