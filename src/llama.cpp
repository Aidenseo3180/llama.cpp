#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "llama-context.h"


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


llama_skip_pattern g_skip_patterns[3] = {
    // LLAMA_SKIP_NONE
    {
        /*.weight_file   =*/ nullptr,
        /*.skip_start    =*/ -1,
        /*.skip_end      =*/ -1,
        /*.replace_idx   =*/ -1,
        /*.merged_weight =*/ nullptr
    },
    // LLAMA_SKIP_25-28 (keep in mind, this layer index is when you start from 1. If from 0, 24-27. So, replace idx 23)
    {
        /*.weight_file   =*/ "transform_layer25_to_layer28_fp16.bin",
        /*.skip_start    =*/ 25,
        /*.skip_end      =*/ 28,
        /*.replace_idx   =*/ 24,
        /*.merged_weight =*/ nullptr
    },
    // LLAMA_SKIP_23_31
    {
        /*.weight_file   =*/ "merged_down_proj_layer23_to_layer30_q8_0.bin",
        /*.skip_start    =*/ 23,
        /*.skip_end      =*/ 30,
        /*.replace_idx   =*/ 22,
        /*.merged_weight =*/ nullptr
    },
};

bool g_skip_weights_loaded = false;


// Load skip weights from binary files
void llama_load_skip_weights() {
    if (g_skip_weights_loaded) {
        return;
    }
    
    LLAMA_LOG_INFO("Loading ReplaceMe skip weights...\n");
    
    for (int i = 1; i < 3; i++) {  // Skip NONE (index 0)
        auto & pattern = g_skip_patterns[i];
        
        FILE* f = fopen(pattern.weight_file, "rb");
        if (!f) {
            LLAMA_LOG_WARN("Failed to load %s, pattern %d disabled\n", 
                          pattern.weight_file, i);
            continue;
        }
        printf("✅ Opened: %s\n", pattern.weight_file);
        
        // Read dimensions [rows, cols] = [4096, 4096]
        int64_t dims[2];
        size_t read_size = fread(dims, sizeof(int64_t), 2, f);
        if (read_size != 2) {
            LLAMA_LOG_ERROR("Failed to read dimensions from %s\n", pattern.weight_file);
            fclose(f);
            continue;
        }
        
        // 🔴 FP16: Calculate data size (simple!)
        size_t n_elements = dims[0] * dims[1];
        size_t byte_size = n_elements * sizeof(ggml_fp16_t);  // 2 bytes per element
        
        // Allocate and read data
        ggml_fp16_t* data = (ggml_fp16_t*)aligned_alloc(GGML_MEM_ALIGN, byte_size);
        if (!data) {
            LLAMA_LOG_ERROR("Failed to allocate %zu bytes for %s\n", 
                           byte_size, pattern.weight_file);
            fclose(f);
            continue;
        }
        
        read_size = fread(data, 1, byte_size, f);
        fclose(f);
        
        if (read_size != byte_size) {
            LLAMA_LOG_ERROR("Failed to read data from %s (expected %zu, got %zu)\n",
                           pattern.weight_file, byte_size, read_size);
            free(data);
            continue;
        }
        
        // Create tensor
        pattern.merged_weight = (ggml_tensor*)calloc(1, sizeof(ggml_tensor));
        if (!pattern.merged_weight) {
            LLAMA_LOG_ERROR("Failed to allocate tensor for %s\n", pattern.weight_file);
            free(data);
            continue;
        }
        
        // 🔴 FP16 type
        pattern.merged_weight->type = GGML_TYPE_F16;
        pattern.merged_weight->data = data;
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(
            data, 
            byte_size
        );
        pattern.merged_weight->buffer = buf;
        
        // Set dimensions (dims = [4096, 4096])
        pattern.merged_weight->ne[0] = dims[0];  // 4096
        pattern.merged_weight->ne[1] = dims[1];  // 4096
        pattern.merged_weight->ne[2] = 1;
        pattern.merged_weight->ne[3] = 1;
        
        // 🔴 Calculate strides for FP16
        size_t type_size = sizeof(ggml_fp16_t);  // 2 bytes
        
        pattern.merged_weight->nb[0] = type_size;
        pattern.merged_weight->nb[1] = pattern.merged_weight->nb[0] * 
            pattern.merged_weight->ne[0];
        pattern.merged_weight->nb[2] = pattern.merged_weight->nb[1] * 
            pattern.merged_weight->ne[1];
        pattern.merged_weight->nb[3] = pattern.merged_weight->nb[2] * 
            pattern.merged_weight->ne[2];
        
        pattern.merged_weight->op = GGML_OP_NONE;
        pattern.merged_weight->flags = 0;
        pattern.merged_weight->src[0] = nullptr;
        pattern.merged_weight->src[1] = nullptr;
        
        LLAMA_LOG_INFO("  Loaded pattern %d: %s [%lld, %lld] (FP16) - skip layers %d-%d\n",
                      i, pattern.weight_file, dims[0], dims[1],
                      pattern.skip_start, pattern.skip_end);

        printf("  ✅ Loaded successfully\n");
        printf("  -> Loaded tensor dims: ne[0]=%lld, ne[1]=%lld\n",
               pattern.merged_weight->ne[0],
               pattern.merged_weight->ne[1]);
    }
    
    g_skip_weights_loaded = true;
}


// Cleanup skip weights
void llama_cleanup_skip_weights() {
    for (int i = 1; i < 3; i++) {
        if (g_skip_patterns[i].merged_weight) {
            if (g_skip_patterns[i].merged_weight->data) {
                free(g_skip_patterns[i].merged_weight->data);
            }
            free(g_skip_patterns[i].merged_weight);
            g_skip_patterns[i].merged_weight = nullptr;
        }
    }
    g_skip_weights_loaded = false;
    LLAMA_LOG_DEBUG("ReplaceMe skip weights cleaned up\n");
}


//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        std::vector<ggml_backend_dev_t> rpc_servers;

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU: {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        // check if there is already a GPU with the same device id
                        ggml_backend_dev_props props;
                        ggml_backend_dev_get_props(dev, &props);
                        auto it = std::find_if(gpus.begin(), gpus.end(), [&props](ggml_backend_dev_t d) {
                            ggml_backend_dev_props d_props;
                            ggml_backend_dev_get_props(d, &d_props);
                            if (props.device_id && d_props.device_id) {
                                return strcmp(props.device_id, d_props.device_id) == 0;
                            }
                            return false;
                        });

                        if (it != gpus.end()) {
                            LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                    __func__,
                                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                    props.device_id ? props.device_id : "unknown id",
                                    ggml_backend_dev_name(*it), ggml_backend_dev_description(*it));
                        } else {
                            gpus.push_back(dev);
                        }
                    }
                    break;
                }

                case GGML_BACKEND_DEVICE_TYPE_IGPU:
                    igpus.push_back(dev);
                    break;
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(*model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}


// llama.cpp - 파일 끝 부분에 추가

// ============================================================================
// ReplaceMe: Public API implementation
// ============================================================================

void llama_set_generation_skip_mode(
    struct llama_context * ctx, 
    enum llama_skip_mode mode) {
    
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(mode >= LLAMA_SKIP_NONE && mode <= LLAMA_SKIP_23_30);
    
    ctx->generation_skip_mode = mode;
    
    if (mode != LLAMA_SKIP_NONE) {
        const auto & pattern = g_skip_patterns[mode];
        LLAMA_LOG_INFO("Set generation skip mode to %d (will skip layers %d-%d)\n",
                      mode, pattern.skip_start, pattern.skip_end);
    } else {
        LLAMA_LOG_INFO("Set generation skip mode to NONE (original graph)\n");
    }
}

void llama_end_generation(struct llama_context * ctx) {
    GGML_ASSERT(ctx != nullptr);
    
    if (ctx->skip_mode != LLAMA_SKIP_NONE || ctx->is_generating) {
        LLAMA_LOG_DEBUG("Generation ended, reset to original graph\n");
    }
    
    ctx->skip_mode = LLAMA_SKIP_NONE;
    ctx->generation_skip_mode = LLAMA_SKIP_NONE;
    ctx->is_generating = false;
}

enum llama_skip_mode llama_get_skip_mode(struct llama_context * ctx) {
    GGML_ASSERT(ctx != nullptr);
    return ctx->skip_mode;
}