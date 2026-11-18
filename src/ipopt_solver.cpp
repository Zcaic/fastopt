#include "fastopt/ipopt_solver.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#endif

// If IpStdCInterface.h is available in the include path (e.g., from CasADi or system headers),
// include it directly; otherwise use the binary-compatible COIN-OR IpStdCInterface C ABI definitions.
#if defined(FASTOPT_HAS_IPOPT_HEADER) && __has_include(<coin-or/IpStdCInterface.h>)
#include <coin-or/IpStdCInterface.h>
#define FASTOPT_IPOPT_TYPES_DEFINED 1
#elif defined(FASTOPT_HAS_IPOPT_HEADER) && __has_include(<IpStdCInterface.h>)
#include <IpStdCInterface.h>
#define FASTOPT_IPOPT_TYPES_DEFINED 1
#endif

namespace fastopt {

namespace {

#ifndef FASTOPT_IPOPT_TYPES_DEFINED
// Official COIN-OR IpStdCInterface C ABI types (Ipopt 3.x 64-bit, matches CasADi's bundled libipopt)
using ipindex = int;
using ipnumber = double;

struct IpoptProblemInfo;
using IpoptProblem = IpoptProblemInfo*;
using UserDataPtr = void*;

enum ApplicationReturnStatus {
    Solve_Succeeded = 0,
    Solved_To_Acceptable_Level = 1,
    Infeasible_Problem_Detected = 2,
    Search_Direction_Becomes_Too_Small = 3,
    Diverging_Iterates = 4,
    User_Requested_Stop = 5,
    Feasible_Point_Found = 6,
    Maximum_Iterations_Exceeded = -1,
    Restoration_Failed = -2,
    Error_In_Step_Computation = -3,
    Maximum_CpuTime_Exceeded = -4,
    Maximum_WallTime_Exceeded = -5,
    Not_Enough_Degrees_Of_Freedom = -10,
    Invalid_Problem_Definition = -11,
    Invalid_Option = -12,
    Invalid_Number_Detected = -13,
    Unrecoverable_Exception = -100,
    NonIpopt_Exception_Thrown = -101,
    Insufficient_Memory = -102,
    Internal_Error = -199
};

using Eval_F_CB = bool (*)(ipindex n, ipnumber* x, bool new_x,
                           ipnumber* obj_value, UserDataPtr user_data);
using Eval_G_CB = bool (*)(ipindex n, ipnumber* x, bool new_x,
                           ipindex m, ipnumber* g, UserDataPtr user_data);
using Eval_Grad_F_CB = bool (*)(ipindex n, ipnumber* x, bool new_x,
                                ipnumber* grad_f, UserDataPtr user_data);
using Eval_Jac_G_CB = bool (*)(ipindex n, ipnumber* x, bool new_x,
                               ipindex m, ipindex nele_jac,
                               ipindex* iRow, ipindex* jCol,
                               ipnumber* values, UserDataPtr user_data);
using Eval_H_CB = bool (*)(ipindex n, ipnumber* x, bool new_x,
                           ipnumber obj_factor, ipindex m, ipnumber* lambda,
                           bool new_lambda, ipindex nele_hess,
                           ipindex* iRow, ipindex* jCol,
                           ipnumber* values, UserDataPtr user_data);
using Intermediate_CB = bool (*)(ipindex alg_mod, ipindex iter_count,
                                 ipnumber obj_value, ipnumber inf_pr,
                                 ipnumber inf_du, ipnumber mu,
                                 ipnumber d_norm, ipnumber regularization_size,
                                 ipnumber alpha_du, ipnumber alpha_pr,
                                 ipindex ls_trials, UserDataPtr user_data);
#endif

// Function pointer types for dynamic loading of the official Ipopt C library (bundled in CasADi)
using CreateIpoptProblemFn = IpoptProblem (*)(
    ipindex n, ipnumber* x_L, ipnumber* x_U,
    ipindex m, ipnumber* g_L, ipnumber* g_U,
    ipindex nele_jac, ipindex nele_hess, ipindex index_style,
    Eval_F_CB eval_f, Eval_G_CB eval_g, Eval_Grad_F_CB eval_grad_f,
    Eval_Jac_G_CB eval_jac_g, Eval_H_CB eval_h);

using FreeIpoptProblemFn = void (*)(IpoptProblem ipopt_problem);
using AddIpoptStrOptionFn = bool (*)(IpoptProblem ipopt_problem, char* keyword, char* val);
using AddIpoptNumOptionFn = bool (*)(IpoptProblem ipopt_problem, char* keyword, ipnumber val);
using AddIpoptIntOptionFn = bool (*)(IpoptProblem ipopt_problem, char* keyword, ipindex val);
using SetIntermediateCallbackFn = bool (*)(IpoptProblem ipopt_problem, Intermediate_CB intermediate_cb);
using IpoptSolveFn = ApplicationReturnStatus (*)(
    IpoptProblem ipopt_problem, ipnumber* x, ipnumber* g,
    ipnumber* obj_val, ipnumber* mult_g, ipnumber* mult_x_L,
    ipnumber* mult_x_U, UserDataPtr user_data);

struct IpoptCApi {
    void* handle{nullptr};
    CreateIpoptProblemFn CreateIpoptProblem{nullptr};
    FreeIpoptProblemFn FreeIpoptProblem{nullptr};
    AddIpoptStrOptionFn AddIpoptStrOption{nullptr};
    AddIpoptNumOptionFn AddIpoptNumOption{nullptr};
    AddIpoptIntOptionFn AddIpoptIntOption{nullptr};
    SetIntermediateCallbackFn SetIntermediateCallback{nullptr};
    IpoptSolveFn IpoptSolve{nullptr};
};

void* load_shared_lib(const std::string& path) {
#if defined(_WIN32)
    // Temporarily scope SetDllDirectoryA and LOAD_WITH_ALTERED_SEARCH_PATH to the library's
    // directory (e.g. site-packages/casadi) so CasADi's bundled MUMPS/Fortran/BLAS DLLs
    // resolve cleanly during LoadLibraryExA, then restore SetDllDirectoryA(nullptr).
    std::string win_path = path;
    for (char& ch : win_path) {
        if (ch == '/') ch = '\\';
    }
    std::size_t slash = win_path.find_last_of('\\');
    if (slash != std::string::npos) {
        std::string dir = win_path.substr(0, slash);
        SetDllDirectoryA(dir.c_str());
    }
    HMODULE h = LoadLibraryExA(win_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        h = LoadLibraryA(win_path.c_str());
    }
    if (slash != std::string::npos) {
        SetDllDirectoryA(nullptr);
    }
    return reinterpret_cast<void*>(h);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_shared_lib(void* handle) {
    if (!handle) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void* get_symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool path_is_regular_file(const std::string& path) {
#if defined(_WIN32)
    DWORD attr = GetFileAttributesA(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && ((attr & FILE_ATTRIBUTE_DIRECTORY) == 0);
#else
    struct stat st {};
    return (stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
#endif
}

bool path_is_directory(const std::string& path) {
#if defined(_WIN32)
    DWORD attr = GetFileAttributesA(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && ((attr & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    struct stat st {};
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
#endif
}

std::string join_path(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    char last = dir.back();
#if defined(_WIN32)
    if (last == '\\' || last == '/') return dir + leaf;
    return dir + "\\" + leaf;
#else
    if (last == '/') return dir + leaf;
    return dir + "/" + leaf;
#endif
}

void append_candidates_from_dir(const std::string& dir_str, std::vector<std::string>& candidates) {
    if (dir_str.empty()) return;
    if (path_is_regular_file(dir_str)) {
        candidates.push_back(dir_str);
        return;
    }
    if (!path_is_directory(dir_str)) {
        return;
    }

#if defined(_WIN32)
    const std::vector<std::string> exact_names = {
        "libipopt-3.dll", "libipopt.dll", "ipopt-3.dll", "ipopt.dll"
    };
#elif defined(__APPLE__)
    const std::vector<std::string> exact_names = {
        "libipopt.3.dylib", "libipopt.dylib", "libipopt.so", "libipopt.so.3"
    };
#else
    const std::vector<std::string> exact_names = {
        "libipopt.so.3", "libipopt.so", "libipopt.so.0"
    };
#endif

    for (const auto& nm : exact_names) {
        std::string cand = join_path(dir_str, nm);
        if (path_is_regular_file(cand)) {
            candidates.push_back(cand);
        }
        std::string cand_lib = join_path(join_path(dir_str, "lib"), nm);
        if (path_is_regular_file(cand_lib)) {
            candidates.push_back(cand_lib);
        }
        std::string cand_bin = join_path(join_path(dir_str, "bin"), nm);
        if (path_is_regular_file(cand_bin)) {
            candidates.push_back(cand_bin);
        }
    }

    // Scan directory using native OS APIs (avoiding std::filesystem so libstdc++ pthread locks are never pulled in)
#if defined(_WIN32)
    std::string pattern = join_path(dir_str, "*");
    WIN32_FIND_DATAA fdata{};
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fdata);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if ((fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) continue;
            std::string fname = to_lower_ascii(fdata.cFileName);
            if (fname.find("ipopt") == std::string::npos) continue;
            if (fname.find("casadi") != std::string::npos) continue;
            if (fname.find(".dll") != std::string::npos) {
                std::string full_str = join_path(dir_str, fdata.cFileName);
                if (std::find(candidates.begin(), candidates.end(), full_str) == candidates.end()) {
                    candidates.push_back(full_str);
                }
            }
        } while (FindNextFileA(hFind, &fdata) != 0);
        FindClose(hFind);
    }
#else
    DIR* dp = opendir(dir_str.c_str());
    if (dp != nullptr) {
        struct dirent* entry = nullptr;
        while ((entry = readdir(dp)) != nullptr) {
            std::string raw_name = entry->d_name;
            if (raw_name == "." || raw_name == "..") continue;
            std::string fname = to_lower_ascii(raw_name);
            if (fname.find("ipopt") == std::string::npos) continue;
            if (fname.find("casadi") != std::string::npos) continue;
            if (fname.find(".so") != std::string::npos || fname.find(".dylib") != std::string::npos) {
                std::string full_str = join_path(dir_str, raw_name);
                if (path_is_regular_file(full_str) &&
                    std::find(candidates.begin(), candidates.end(), full_str) == candidates.end()) {
                    candidates.push_back(full_str);
                }
            }
        }
        closedir(dp);
    }
#endif
}

const IpoptCApi& get_ipopt_api(const std::string& search_dir) {
    static IpoptCApi api;
    if (api.handle != nullptr) {
        return api;
    }

    std::vector<std::string> candidates;
    append_candidates_from_dir(search_dir, candidates);

#if defined(FASTOPT_DEFAULT_CASADI_DIR)
    if (std::string(FASTOPT_DEFAULT_CASADI_DIR) != search_dir) {
        append_candidates_from_dir(FASTOPT_DEFAULT_CASADI_DIR, candidates);
    }
#endif

#if defined(_WIN32)
    for (const char* nm : {"libipopt-3.dll", "libipopt.dll", "ipopt-3.dll", "ipopt.dll"}) {
        candidates.emplace_back(nm);
    }
#elif defined(__APPLE__)
    for (const char* nm : {"libipopt.3.dylib", "libipopt.dylib", "libipopt.so"}) {
        candidates.emplace_back(nm);
    }
#else
    for (const char* nm : {"libipopt.so.3", "libipopt.so", "libipopt.so.0"}) {
        candidates.emplace_back(nm);
    }
#endif

    for (const auto& cand : candidates) {
        void* h = load_shared_lib(cand);
        if (h == nullptr) continue;

        auto create_fn = reinterpret_cast<CreateIpoptProblemFn>(get_symbol(h, "CreateIpoptProblem"));
        auto free_fn = reinterpret_cast<FreeIpoptProblemFn>(get_symbol(h, "FreeIpoptProblem"));
        auto str_fn = reinterpret_cast<AddIpoptStrOptionFn>(get_symbol(h, "AddIpoptStrOption"));
        auto num_fn = reinterpret_cast<AddIpoptNumOptionFn>(get_symbol(h, "AddIpoptNumOption"));
        auto int_fn = reinterpret_cast<AddIpoptIntOptionFn>(get_symbol(h, "AddIpoptIntOption"));
        auto cb_fn = reinterpret_cast<SetIntermediateCallbackFn>(get_symbol(h, "SetIntermediateCallback"));
        auto solve_fn = reinterpret_cast<IpoptSolveFn>(get_symbol(h, "IpoptSolve"));

        if (create_fn && free_fn && str_fn && num_fn && int_fn && solve_fn) {
            api.handle = h;
            api.CreateIpoptProblem = create_fn;
            api.FreeIpoptProblem = free_fn;
            api.AddIpoptStrOption = str_fn;
            api.AddIpoptNumOption = num_fn;
            api.AddIpoptIntOption = int_fn;
            api.SetIntermediateCallback = cb_fn;
            api.IpoptSolve = solve_fn;
            break;
        }
        close_shared_lib(h);
    }

    if (api.handle == nullptr) {
        std::ostringstream oss;
        oss << "Failed to load COIN-OR IPOPT shared library (searched in '" << search_dir
            << "' for libipopt*.dll / libipopt*.so / libipopt*.dylib). "
            << "Please ensure CasADi is installed via 'pip install casadi' so that prebuilt "
            << "IPOPT and MUMPS shared libraries are available in site-packages/casadi.";
        throw std::runtime_error(oss.str());
    }

    return api;
}

std::string status_to_string(int status) {
    switch (status) {
        case Solve_Succeeded: return "Solve_Succeeded: Optimization terminated successfully.";
        case Solved_To_Acceptable_Level: return "Solved_To_Acceptable_Level: Solved to acceptable tolerance.";
        case Infeasible_Problem_Detected: return "Infeasible_Problem_Detected: Converged to a point of local infeasibility.";
        case Search_Direction_Becomes_Too_Small: return "Search_Direction_Becomes_Too_Small.";
        case Diverging_Iterates: return "Diverging_Iterates.";
        case User_Requested_Stop: return "User_Requested_Stop.";
        case Feasible_Point_Found: return "Feasible_Point_Found.";
        case Maximum_Iterations_Exceeded: return "Maximum_Iterations_Exceeded.";
        case Restoration_Failed: return "Restoration_Failed.";
        case Error_In_Step_Computation: return "Error_In_Step_Computation.";
        case Maximum_CpuTime_Exceeded: return "Maximum_CpuTime_Exceeded.";
        case Maximum_WallTime_Exceeded: return "Maximum_WallTime_Exceeded.";
        case Not_Enough_Degrees_Of_Freedom: return "Not_Enough_Degrees_Of_Freedom.";
        case Invalid_Problem_Definition: return "Invalid_Problem_Definition.";
        case Invalid_Option: return "Invalid_Option.";
        case Invalid_Number_Detected: return "Invalid_Number_Detected.";
        default: return "IPOPT terminated with status code " + std::to_string(status);
    }
}

// Context passed via UserDataPtr to IPOPT's 5 C callbacks (100% pure C++ execution)
struct IpoptCallbackContext {
    Evaluator* ev{nullptr};
    Eigen::Index n{0};
    Eigen::Index m{0};
    Eigen::Index m_eq{0};
    Eigen::Index m_ineq{0};
    bool has_eq{false};
    bool has_ineq{false};
    bool use_exact_hessian{true};

    // Symbolic sparsity patterns (0-based COO indices from Graph Coloring plans)
    Eigen::Index nnz_eq{0};
    Eigen::Index nnz_ineq{0};
    std::vector<ipindex> jac_rows;
    std::vector<ipindex> jac_cols;
    std::vector<ipindex> hess_rows;
    std::vector<ipindex> hess_cols;

    // Pre-allocated multiplier buffers to avoid per-iteration heap allocations in cb_eval_h
    std::vector<Eigen::VectorXd> mults_buf;

    // Iteration history & optional user callback
    int iter_count{0};
    std::vector<int> hist_iter;
    std::vector<double> hist_obj;
    std::vector<double> hist_inf_pr;
    std::vector<double> hist_inf_du;
    std::vector<double> hist_mu;
    std::vector<double> hist_alpha_pr;
    IpoptIterCallbackFn iter_callback{nullptr};
};

// Extract exact symbolic sparsity patterns from Evaluator Graph-Coloring plans
void build_sparsity_patterns(IpoptCallbackContext& ctx) {
    ctx.jac_rows.clear();
    ctx.jac_cols.clear();
    ctx.nnz_eq = 0;
    ctx.nnz_ineq = 0;

    if (ctx.has_eq) {
        const auto& plan_eq = ctx.ev->jacobian_sparsity_plan(1);
        ctx.nnz_eq = plan_eq.nnz();
        for (Eigen::Index k = 0; k < plan_eq.nnz(); ++k) {
            ctx.jac_rows.push_back(static_cast<ipindex>(plan_eq.rows(k)));
            ctx.jac_cols.push_back(static_cast<ipindex>(plan_eq.cols(k)));
        }
    }
    if (ctx.has_ineq) {
        const std::size_t ineq_idx = ctx.has_eq ? 2 : 1;
        const auto& plan_ineq = ctx.ev->jacobian_sparsity_plan(ineq_idx);
        ctx.nnz_ineq = plan_ineq.nnz();
        for (Eigen::Index k = 0; k < plan_ineq.nnz(); ++k) {
            ctx.jac_rows.push_back(static_cast<ipindex>(ctx.m_eq + plan_ineq.rows(k)));
            ctx.jac_cols.push_back(static_cast<ipindex>(plan_ineq.cols(k)));
        }
    }

    ctx.hess_rows.clear();
    ctx.hess_cols.clear();
    if (ctx.use_exact_hessian && ctx.n > 0) {
        const auto& h_plan = ctx.ev->hessian_sparsity_plan();
        ctx.hess_rows.reserve(static_cast<std::size_t>(h_plan.nnz_tril()));
        ctx.hess_cols.reserve(static_cast<std::size_t>(h_plan.nnz_tril()));
        for (Eigen::Index k = 0; k < h_plan.nnz_tril(); ++k) {
            ctx.hess_rows.push_back(static_cast<ipindex>(h_plan.tril_rows(k)));
            ctx.hess_cols.push_back(static_cast<ipindex>(h_plan.tril_cols(k)));
        }
    }
}

// ============================================================================
// Official C Callback Functions for IpStdCInterface
// ============================================================================
bool cb_eval_f(ipindex n, ipnumber* x, bool /*new_x*/,
               ipnumber* obj_value, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    try {
        Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
        const auto outs = ctx->ev->eval(x_vec);
        *obj_value = outs[0](0, 0);
        return std::isfinite(*obj_value);
    } catch (...) {
        return false;
    }
}

bool cb_eval_grad_f(ipindex n, ipnumber* x, bool /*new_x*/,
                    ipnumber* grad_f, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    try {
        Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
        Eigen::VectorXd g = ctx->ev->gradient(x_vec, 0);
        if (!g.allFinite()) return false;
        Eigen::Map<Eigen::VectorXd>(grad_f, n) = g;
        return true;
    } catch (...) {
        return false;
    }
}

bool cb_eval_g(ipindex n, ipnumber* x, bool /*new_x*/,
               ipindex m, ipnumber* g, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    if (m == 0) return true;
    try {
        Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
        const auto outs = ctx->ev->eval(x_vec);
        Eigen::Map<Eigen::VectorXd> g_out(g, m);
        Eigen::Index offset = 0;
        if (ctx->has_eq) {
            Eigen::Map<const Eigen::VectorXd> eq_v(outs[1].data(), ctx->m_eq);
            g_out.segment(offset, ctx->m_eq) = eq_v;
            offset += ctx->m_eq;
        }
        if (ctx->has_ineq) {
            const std::size_t ineq_idx = ctx->has_eq ? 2 : 1;
            Eigen::Map<const Eigen::VectorXd> ineq_v(outs[ineq_idx].data(), ctx->m_ineq);
            g_out.segment(offset, ctx->m_ineq) = ineq_v;
        }
        return g_out.allFinite();
    } catch (...) {
        return false;
    }
}

bool cb_eval_jac_g(ipindex n, ipnumber* x, bool /*new_x*/,
                   ipindex /*m*/, ipindex nele_jac,
                   ipindex* iRow, ipindex* jCol,
                   ipnumber* values, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    if (nele_jac == 0) return true;
    if (values == nullptr) {
        std::memcpy(iRow, ctx->jac_rows.data(), static_cast<std::size_t>(nele_jac) * sizeof(ipindex));
        std::memcpy(jCol, ctx->jac_cols.data(), static_cast<std::size_t>(nele_jac) * sizeof(ipindex));
        return true;
    }
    try {
        Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
        Eigen::Index offset = 0;
        if (ctx->has_eq && ctx->nnz_eq > 0) {
            Eigen::VectorXd v_eq = ctx->ev->sparse_jacobian_values(x_vec, 1);
            if (!v_eq.allFinite()) return false;
            Eigen::Map<Eigen::VectorXd>(values + offset, ctx->nnz_eq) = v_eq;
            offset += ctx->nnz_eq;
        }
        if (ctx->has_ineq && ctx->nnz_ineq > 0) {
            const std::size_t ineq_idx = ctx->has_eq ? 2 : 1;
            Eigen::VectorXd v_ineq = ctx->ev->sparse_jacobian_values(x_vec, ineq_idx);
            if (!v_ineq.allFinite()) return false;
            Eigen::Map<Eigen::VectorXd>(values + offset, ctx->nnz_ineq) = v_ineq;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool cb_eval_h(ipindex n, ipnumber* x, bool /*new_x*/,
               ipnumber obj_factor, ipindex /*m*/, ipnumber* lambda,
               bool /*new_lambda*/, ipindex nele_hess,
               ipindex* iRow, ipindex* jCol,
               ipnumber* values, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    if (!ctx->use_exact_hessian || nele_hess == 0) return true;
    if (values == nullptr) {
        std::memcpy(iRow, ctx->hess_rows.data(), static_cast<std::size_t>(nele_hess) * sizeof(ipindex));
        std::memcpy(jCol, ctx->hess_cols.data(), static_cast<std::size_t>(nele_hess) * sizeof(ipindex));
        return true;
    }
    try {
        Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
        Eigen::Index offset = 0;
        std::size_t buf_idx = 0;
        if (ctx->has_eq) {
            ctx->mults_buf[buf_idx] = Eigen::Map<const Eigen::VectorXd>(lambda + offset, ctx->m_eq);
            offset += ctx->m_eq;
            ++buf_idx;
        }
        if (ctx->has_ineq) {
            ctx->mults_buf[buf_idx] = Eigen::Map<const Eigen::VectorXd>(lambda + offset, ctx->m_ineq);
        }
        Eigen::VectorXd h_vals = ctx->ev->sparse_hessian_values(x_vec, obj_factor, ctx->mults_buf);
        if (!h_vals.allFinite()) return false;
        Eigen::Map<Eigen::VectorXd>(values, nele_hess) = h_vals;
        return true;
    } catch (...) {
        return false;
    }
}

bool cb_intermediate(ipindex /*alg_mod*/, ipindex iter_count,
                     ipnumber obj_value, ipnumber inf_pr,
                     ipnumber inf_du, ipnumber mu,
                     ipnumber /*d_norm*/, ipnumber /*regularization_size*/,
                     ipnumber /*alpha_du*/, ipnumber alpha_pr,
                     ipindex /*ls_trials*/, UserDataPtr user_data) {
    auto* ctx = static_cast<IpoptCallbackContext*>(user_data);
    const int it = static_cast<int>(iter_count);
    ctx->iter_count = it;
    ctx->hist_iter.push_back(it);
    ctx->hist_obj.push_back(static_cast<double>(obj_value));
    ctx->hist_inf_pr.push_back(static_cast<double>(inf_pr));
    ctx->hist_inf_du.push_back(static_cast<double>(inf_du));
    ctx->hist_mu.push_back(static_cast<double>(mu));
    ctx->hist_alpha_pr.push_back(static_cast<double>(alpha_pr));

    if (ctx->iter_callback) {
        try {
            return ctx->iter_callback(
                it,
                static_cast<double>(obj_value),
                static_cast<double>(inf_pr),
                static_cast<double>(inf_du),
                static_cast<double>(mu),
                static_cast<double>(alpha_pr));
        } catch (...) {
            return false;
        }
    }
    return true;
}

} // namespace

// ============================================================================
// Main C++ IPOPT Solver Entry Point
// ============================================================================
IpoptResult solve_ipopt_cpp(
    Evaluator& evaluator,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Ref<const Eigen::VectorXd>& x_L,
    const Eigen::Ref<const Eigen::VectorXd>& x_U,
    const Eigen::Ref<const Eigen::VectorXd>& g_L,
    const Eigen::Ref<const Eigen::VectorXd>& g_U,
    bool has_eq,
    bool has_ineq,
    bool use_exact_hessian,
    const std::map<std::string, std::string>& str_options,
    const std::map<std::string, double>& num_options,
    const std::map<std::string, int>& int_options,
    const std::string& lib_search_dir,
    IpoptIterCallbackFn iter_callback) {
    const auto t_start = std::chrono::steady_clock::now();
    const IpoptCApi& api = get_ipopt_api(lib_search_dir);

    IpoptCallbackContext ctx;
    ctx.ev = &evaluator;
    ctx.n = evaluator.num_vars();
    ctx.has_eq = has_eq;
    ctx.has_ineq = has_ineq;
    ctx.m_eq = has_eq ? evaluator.output_size(1) : 0;
    ctx.m_ineq = has_ineq ? evaluator.output_size(has_eq ? 2 : 1) : 0;
    ctx.m = ctx.m_eq + ctx.m_ineq;
    ctx.use_exact_hessian = use_exact_hessian;
    ctx.iter_callback = std::move(iter_callback);

    if (has_eq) {
        ctx.mults_buf.emplace_back(Eigen::VectorXd::Zero(ctx.m_eq));
    }
    if (has_ineq) {
        ctx.mults_buf.emplace_back(Eigen::VectorXd::Zero(ctx.m_ineq));
    }

    if (x0.size() != ctx.n || x_L.size() != ctx.n || x_U.size() != ctx.n) {
        throw std::invalid_argument("solve_ipopt_cpp: x0, x_L, or x_U dimension mismatch.");
    }
    if (g_L.size() != ctx.m || g_U.size() != ctx.m) {
        throw std::invalid_argument("solve_ipopt_cpp: g_L or g_U dimension mismatch.");
    }

    // Clamp infinite bounds to +/- 1e20 (IPOPT default nlp_lower/upper_bound_inf is +/- 1e19)
    Eigen::VectorXd xL_buf = x_L.cwiseMax(-1e20);
    Eigen::VectorXd xU_buf = x_U.cwiseMin(1e20);
    Eigen::VectorXd gL_buf = g_L.cwiseMax(-1e20);
    Eigen::VectorXd gU_buf = g_U.cwiseMin(1e20);

    build_sparsity_patterns(ctx);

    const ipindex n = static_cast<ipindex>(ctx.n);
    const ipindex m = static_cast<ipindex>(ctx.m);
    const ipindex nele_jac = static_cast<ipindex>(ctx.jac_rows.size());
    const ipindex nele_hess = static_cast<ipindex>(ctx.hess_rows.size());

    IpoptProblem nlp = api.CreateIpoptProblem(
        n,
        xL_buf.data(),
        xU_buf.data(),
        m,
        gL_buf.data(),
        gU_buf.data(),
        nele_jac,
        nele_hess,
        0, // C-style 0-based indexing
        &cb_eval_f,
        &cb_eval_g,
        &cb_eval_grad_f,
        &cb_eval_jac_g,
        &cb_eval_h);

    if (nlp == nullptr) {
        throw std::runtime_error("CreateIpoptProblem returned NULL (invalid NLP problem definition).");
    }

    std::string default_hess_mode = use_exact_hessian ? "exact" : "limited-memory";
    api.AddIpoptStrOption(nlp, const_cast<char*>("hessian_approximation"),
                          const_cast<char*>(default_hess_mode.c_str()));
    api.AddIpoptStrOption(nlp, const_cast<char*>("linear_solver"), const_cast<char*>("mumps"));
    api.AddIpoptIntOption(nlp, const_cast<char*>("print_level"), 0);
    api.AddIpoptStrOption(nlp, const_cast<char*>("sb"), const_cast<char*>("yes"));

    for (const auto& kv : str_options) {
        api.AddIpoptStrOption(nlp, const_cast<char*>(kv.first.c_str()),
                              const_cast<char*>(kv.second.c_str()));
    }
    for (const auto& kv : num_options) {
        api.AddIpoptNumOption(nlp, const_cast<char*>(kv.first.c_str()), kv.second);
    }
    for (const auto& kv : int_options) {
        api.AddIpoptIntOption(nlp, const_cast<char*>(kv.first.c_str()),
                              static_cast<ipindex>(kv.second));
    }

    if (api.SetIntermediateCallback != nullptr) {
        api.SetIntermediateCallback(nlp, &cb_intermediate);
    }

    IpoptResult res;
    res.x = x0.cwiseMax(xL_buf).cwiseMin(xU_buf);
    res.mult_g = Eigen::VectorXd::Zero(ctx.m);
    res.mult_x_L = Eigen::VectorXd::Zero(ctx.n);
    res.mult_x_U = Eigen::VectorXd::Zero(ctx.n);
    Eigen::VectorXd g_final = Eigen::VectorXd::Zero(ctx.m);

    ApplicationReturnStatus status = api.IpoptSolve(
        nlp,
        res.x.data(),
        ctx.m > 0 ? g_final.data() : nullptr,
        &res.obj_val,
        ctx.m > 0 ? res.mult_g.data() : nullptr,
        res.mult_x_L.data(),
        res.mult_x_U.data(),
        &ctx);

    api.FreeIpoptProblem(nlp);

    const auto t_end = std::chrono::steady_clock::now();
    res.status = static_cast<int>(status);
    res.success = (status == Solve_Succeeded || status == Solved_To_Acceptable_Level ||
                   status == Feasible_Point_Found);
    res.iterations = ctx.iter_count;
    res.elapsed_time = std::chrono::duration<double>(t_end - t_start).count();
    res.message = status_to_string(res.status);
    res.hist_iter = std::move(ctx.hist_iter);
    res.hist_obj = std::move(ctx.hist_obj);
    res.hist_inf_pr = std::move(ctx.hist_inf_pr);
    res.hist_inf_du = std::move(ctx.hist_inf_du);
    res.hist_mu = std::move(ctx.hist_mu);
    res.hist_alpha_pr = std::move(ctx.hist_alpha_pr);

    return res;
}

} // namespace fastopt
