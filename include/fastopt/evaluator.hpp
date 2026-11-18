#ifndef FASTOPT_EVALUATOR_HPP
#define FASTOPT_EVALUATOR_HPP

#include "fastopt/core.hpp"
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace fastopt {

// ============================================================================
// Graph-Colored Sparsity Plan for a Single Output Jacobian J (m x n)
// ============================================================================
struct JacobianSparsityPlan {
    Eigen::Index m{0};
    Eigen::Index n{0};
    // Sorted row-major 0-based COO indices of structural non-zeros
    Eigen::VectorXi rows;
    Eigen::VectorXi cols;
    // CSR row pointer into rows/cols (size m + 1)
    std::vector<int> row_ptr;
    // For column-mode extraction: list of (row, nz_idx) for each column c
    std::vector<std::vector<std::pair<int, int>>> col_nz_map;

    // Greedy graph coloring groups
    bool use_row_coloring{true};
    std::vector<std::vector<int>> row_color_groups;
    std::vector<std::vector<int>> col_color_groups;

    Eigen::Index nnz() const { return rows.size(); }
    int num_colors() const {
        return use_row_coloring ? static_cast<int>(row_color_groups.size())
                                : static_cast<int>(col_color_groups.size());
    }
};

// ============================================================================
// Graph-Colored Sparsity Plan for Symmetric Lagrangian Hessian H (n x n)
// ============================================================================
struct HessianSparsityPlan {
    Eigen::Index n{0};
    // Lower-triangular (row >= col) 0-based COO indices of structural non-zeros
    Eigen::VectorXi tril_rows;
    Eigen::VectorXi tril_cols;
    // Full-column non-zero map: for each column c, list of (row, tril_nz_idx_or_neg1)
    // and for each column c, list of (row, tril_nz_idx) where row >= c
    std::vector<std::vector<std::pair<int, int>>> col_tril_map;
    std::vector<std::vector<int>> col_full_rows;

    // Distance-2 / column-intersection greedy coloring groups
    std::vector<std::vector<int>> color_groups;

    Eigen::Index nnz_tril() const { return tril_rows.size(); }
    int num_colors() const { return static_cast<int>(color_groups.size()); }
};

// ============================================================================
// High-Performance Graph Evaluator (0th, 1st, 2nd Order AD, Sparsity & Batching)
// ============================================================================
class Evaluator {
public:
    Evaluator(std::shared_ptr<Graph> graph, std::vector<int> output_ids);

    // Number of decision variables (active un-frozen variables)
    Eigen::Index num_vars() const { return graph_->total_var_size(); }

    // Number of registered output expressions
    std::size_t num_outputs() const { return output_ids_.size(); }

    // Total scalar length of a specific output expression
    Eigen::Index output_size(std::size_t out_idx) const {
        return graph_->node(output_ids_.at(out_idx)).size();
    }

    Shape output_shape(std::size_t out_idx) const {
        return graph_->node(output_ids_.at(out_idx)).shape;
    }

    // Diagnostic counter for actual dynamic forward graph executions on main_ws_
    std::size_t forward_eval_count() const { return forward_eval_count_; }
    void reset_forward_eval_count() { forward_eval_count_ = 0; }
    void invalidate_forward_cache() { has_valid_forward_ = false; }

    // Single-point forward evaluation of all registered outputs
    std::vector<Eigen::MatrixXd> eval(const Eigen::Ref<const Eigen::VectorXd>& x);

    // Evaluate an arbitrary node ID at the given decision vector x
    Eigen::MatrixXd eval_node(const Eigen::Ref<const Eigen::VectorXd>& x, int node_id);

    // OpenMP parallel batch evaluation for evolutionary algorithms (e.g. pymoo)
    std::vector<Eigen::MatrixXd> eval_batch(
        const Eigen::Ref<const Eigen::MatrixXd>& X_pop,
        int num_threads = 0);

    // First-order reverse-mode gradient for a scalar output node
    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd>& x, std::size_t out_idx = 0);

    // First-order Jacobian matrix (out_size x n_vars) accelerated via Graph Coloring & OpenMP
    Eigen::MatrixXd jacobian(const Eigen::Ref<const Eigen::VectorXd>& x,
                             std::size_t out_idx = 0,
                             int num_threads = 0);

    // Sparse Jacobian COO pattern and compressed non-zero evaluation
    const JacobianSparsityPlan& jacobian_sparsity_plan(std::size_t out_idx = 0) {
        ensure_sparsity_plans();
        return jac_plans_.at(out_idx);
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXi> jacobian_sparsity(std::size_t out_idx = 0) {
        ensure_sparsity_plans();
        const auto& p = jac_plans_.at(out_idx);
        return {p.rows, p.cols};
    }
    int num_jacobian_colors(std::size_t out_idx = 0) {
        ensure_sparsity_plans();
        return jac_plans_.at(out_idx).num_colors();
    }
    Eigen::VectorXd sparse_jacobian_values(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        std::size_t out_idx = 0,
        int num_threads = 0);

    // Second-order exact Hessian-Vector Product (HVP): H_L(x, sigma, lambda) * v
    Eigen::VectorXd hessian_vec_prod(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        double obj_scale,
        const std::vector<Eigen::VectorXd>& constraint_multipliers,
        const Eigen::Ref<const Eigen::VectorXd>& v);

    // Exact Lagrangian Hessian matrix (n_vars x n_vars) accelerated via Graph Coloring & OpenMP
    Eigen::MatrixXd lagrangian_hessian(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        double obj_scale,
        const std::vector<Eigen::VectorXd>& constraint_multipliers,
        int num_threads = 0);

    // Convenience exact Hessian for a single scalar output expression
    Eigen::MatrixXd hessian(const Eigen::Ref<const Eigen::VectorXd>& x,
                            std::size_t out_idx = 0,
                            int num_threads = 0);

    // Sparse lower-triangular Lagrangian Hessian COO pattern and compressed non-zero evaluation
    const HessianSparsityPlan& hessian_sparsity_plan() {
        ensure_sparsity_plans();
        return hess_plan_;
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXi> hessian_sparsity() {
        ensure_sparsity_plans();
        return {hess_plan_.tril_rows, hess_plan_.tril_cols};
    }
    int num_hessian_colors() {
        ensure_sparsity_plans();
        return hess_plan_.num_colors();
    }
    Eigen::VectorXd sparse_hessian_values(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        double obj_scale,
        const std::vector<Eigen::VectorXd>& constraint_multipliers,
        int num_threads = 0);

    // Exact analytical 1st- and 2nd-order sensitivities w.r.t. frozen Parameter nodes:
    // Returns (dL_dp [n_p], d2L_dxp [n_vars x n_p], dout_dp [list of (m_k x n_p)])
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
    parameter_sensitivities(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        const std::vector<int>& param_ids,
        double obj_scale,
        const std::vector<Eigen::VectorXd>& constraint_multipliers);

    std::shared_ptr<Graph> graph() const { return graph_; }

private:
    std::shared_ptr<Graph> graph_;
    std::vector<int> output_ids_;
    Workspace main_ws_;
    std::vector<Workspace> thread_ws_;
    std::uint64_t synced_param_version_{0};

    // Same-point forward evaluation deduplication state for main_ws_
    Eigen::VectorXd last_forward_x_;
    std::uint64_t last_forward_param_version_{0};
    bool has_valid_forward_{false};
    mutable std::size_t forward_eval_count_{0};

    bool has_blackbox_{false};
    bool sparsity_ready_{false};

    std::vector<JacobianSparsityPlan> jac_plans_;
    HessianSparsityPlan hess_plan_;

    void ensure_static_synced();
    void ensure_forward_evaluated(const Eigen::Ref<const Eigen::VectorXd>& x);
    void ensure_thread_workspaces(int n_threads, bool need_tier1 = false, bool need_tier2 = false);
    void run_forward_dynamic(Workspace& ws, const Eigen::Ref<const Eigen::VectorXd>& x) const;
    void ensure_sparsity_plans();
    void build_sparsity_and_coloring_plans();
};

} // namespace fastopt

#endif // FASTOPT_EVALUATOR_HPP
