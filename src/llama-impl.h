#pragma once

#include "ggml.h" // for ggml_log_level
#include "llama.h"

#include <string>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */ }
};

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to llama_format ?
LLAMA_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llama_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LLAMA_TENSOR_NAME_FATTN "__fattn__"





// Layer output capture
extern std::vector<float> g_layer17_vision_output;
extern std::vector<float> g_layer17_text_output;
extern std::vector<float> g_layer21_vision_output;
extern std::vector<float> g_layer21_text_output;
extern bool g_enable_layer_capture;
extern int g_layer17_vision_captured;  // bool → int로 변경!
extern bool g_layer17_text_captured;
extern int g_layer21_vision_captured;  // bool → int로 변경!
extern bool g_layer21_text_captured;

// 별도로 복사된 tensor들
extern ggml_tensor * g_layer17_output_copy;
extern ggml_tensor * g_layer21_output_copy;


// Skip pattern definition
struct llama_skip_pattern {
    const char* weight_file;
    int skip_start;       // First layer to skip
    int skip_end;         // Last layer to skip (inclusive)
    int replace_idx;      // Layer index where merged weight is used
    ggml_tensor* merged_weight;
};

// Global skip patterns
extern llama_skip_pattern g_skip_patterns[4];
extern bool g_skip_weights_loaded;

// Load/cleanup functions
void llama_load_skip_weights();
void llama_cleanup_skip_weights();