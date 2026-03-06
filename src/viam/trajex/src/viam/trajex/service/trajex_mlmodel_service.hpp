#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
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
    };

    // LRU cache for parsed URDF models. The list owns the URDF content strings;
    // the map keys are string_views into those list nodes, avoiding a redundant
    // copy of each (potentially large) URDF XML string.
    // Guarded by its own mutex, independent of config_mutex_.
    static constexpr std::size_t k_urdf_cache_capacity = 5;
    struct urdf_cache_entry {
        std::string content;
        std::shared_ptr<jacobian::Model> model;
    };
    std::mutex urdf_cache_mutex_;
    std::list<urdf_cache_entry> urdf_cache_list_;  // front = most recently used
    std::unordered_map<std::string_view, std::list<urdf_cache_entry>::iterator> urdf_cache_map_;

    std::shared_ptr<jacobian::Model> get_or_parse_urdf(const std::string& urdf_xml);

    mutable std::shared_mutex config_mutex_;
    config config_;
};

}  // namespace viam::trajex
