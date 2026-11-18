#ifndef FASTOPT_IPOPT_SOLVER_HPP
#define FASTOPT_IPOPT_SOLVER_HPP

#include "fastopt/evaluator.hpp"
#include <Eigen/Dense>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace fastopt {

// ============================================================================
// IPOPT Solver Result Structure Returned to Python
// ============================================================================
struct IpoptResult {
    Eigen::VectorXd x;
    Eigen::VectorXd mult_g;
    Eigen::VectorXd mult_x_L;
    Eigen::VectorXd mult_x_U;
    double obj_val{0.0};
    int status{-1};
    bool success{false};
    int iterations{0};
    double elapsed_time{0.0};
    std::string message;

    // Per-iteration convergence history captured via IPOPT Intermediate_CB
    std::vector<int> hist_iter;
    std::vector<double> hist_obj;
    std::vector<double> hist_inf_pr;
    std::vector<double> hist_inf_du;
    std::vector<double> hist_mu;
    std::vector<double> hist_alpha_pr;
};

// Optional per-iteration callback: returns true to continue, false to request stop
using IpoptIterCallbackFn = std::function<bool(
    int iter, double obj, double inf_pr, double inf_du, double mu, double alpha_pr)>;

// ============================================================================
// Native C++17 IPOPT Solver Bridge (via Official IpStdCInterface C ABI)
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
    const std::string& lib_search_dir = "",
    IpoptIterCallbackFn iter_callback = nullptr);

} // namespace fastopt

#endif // FASTOPT_IPOPT_SOLVER_HPP
