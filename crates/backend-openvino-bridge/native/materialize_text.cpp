#include "bridge.h"

#include <string>

loci_ov_text_materialize_response
loci_ov_materialize_text_artifact(
    const loci_ov_text_materialize_request* request
) {
    loci_ov_text_materialize_response response{};

    if (request == nullptr || request->source_root == nullptr || request->prepared_root == nullptr) {
        response.status = LOCI_OV_BRIDGE_INVALID_ARGUMENT;
        response.error_message =
            loci_ov_bridge_dup_cstr("invalid request: source_root and prepared_root are required");
        return response;
    }

    response.status = LOCI_OV_BRIDGE_UNSUPPORTED;
    response.artifact_root = loci_ov_bridge_dup_cstr(request->prepared_root);
    response.entrypoint_path = loci_ov_bridge_dup_cstr("openvino_model.xml");

    std::string metadata =
        std::string("{\"bridge\":\"openvino\",\"status\":\"stub\",\"note\":\"native text materialization is not implemented yet\"}");
    response.metadata_json = loci_ov_bridge_dup_cstr(metadata.c_str());
    response.error_message =
        loci_ov_bridge_dup_cstr("native OpenVINO text artifact materialization is not implemented yet");
    return response;
}

