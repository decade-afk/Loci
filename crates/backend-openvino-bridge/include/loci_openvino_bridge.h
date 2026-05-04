#ifndef LOCI_OPENVINO_BRIDGE_H
#define LOCI_OPENVINO_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loci_ov_bridge_status {
    LOCI_OV_BRIDGE_OK = 0,
    LOCI_OV_BRIDGE_UNSUPPORTED = 1,
    LOCI_OV_BRIDGE_INVALID_ARGUMENT = 2,
    LOCI_OV_BRIDGE_IO_ERROR = 3,
    LOCI_OV_BRIDGE_CONVERSION_FAILED = 4,
    LOCI_OV_BRIDGE_RUNTIME_ERROR = 5
} loci_ov_bridge_status;

typedef struct loci_ov_text_materialize_request {
    const char* source_root;
    const char* prepared_root;
    const char* model_name;
    const char* architecture;
    const char* config_json_path;
    const char* tokenizer_json_path;
    const char* safetensors_index_path;
    const char* options_json;
} loci_ov_text_materialize_request;

typedef struct loci_ov_text_materialize_response {
    loci_ov_bridge_status status;
    char* artifact_root;
    char* entrypoint_path;
    char* metadata_json;
    char* error_message;
} loci_ov_text_materialize_response;

const char* loci_ov_bridge_version(void);

loci_ov_text_materialize_response
loci_ov_materialize_text_artifact(
    const loci_ov_text_materialize_request* request
);

void loci_ov_bridge_free_string(char* ptr);

#ifdef __cplusplus
}
#endif

#endif

