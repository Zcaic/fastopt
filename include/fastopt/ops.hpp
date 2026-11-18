#ifndef FASTOPT_OPS_HPP
#define FASTOPT_OPS_HPP

#include "fastopt/core.hpp"
#include <functional>

namespace fastopt {

// ============================================================================
// 1. Unary Element-Wise Operator Node
// ============================================================================
class UnaryOpNode : public Node {
public:
    UnaryOpNode(OpType op_type, Shape s, int in_id, std::string name = "")
        : Node(op_type, s, {in_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 2. Binary Element-Wise Operator Node (Supports Exact Shape or Scalar Broadcast)
// ============================================================================
class BinaryOpNode : public Node {
public:
    BinaryOpNode(OpType op_type, Shape s, int lhs_id, int rhs_id, std::string name = "")
        : Node(op_type, s, {lhs_id, rhs_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 3. Matrix Multiplication Node: C = A * B
// ============================================================================
class MatMulNode : public Node {
public:
    MatMulNode(Shape s, int a_id, int b_id, std::string name = "")
        : Node(OpType::MatMul, s, {a_id, b_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 3B. Symbolic Linear System Solver Node: Y = A^{-1} B (Exact 1st & 2nd AD)
// ============================================================================
class LinearSolveNode : public Node {
public:
    LinearSolveNode(Shape s, int a_id, int b_id, std::string name = "")
        : Node(OpType::LinearSolve, s, {a_id, b_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 4. Inner Product Node: c = dot(a, b) = sum(a_i * b_i) -> Scalar (1, 1)
// ============================================================================
class DotNode : public Node {
public:
    DotNode(int a_id, int b_id, std::string name = "")
        : Node(OpType::Dot, Shape(1, 1), {a_id, b_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 5. Reduction Sum Node (axis = -1: all elements, axis = 0: rows, axis = 1: cols)
// ============================================================================
class SumNode : public Node {
public:
    int axis{-1};

    SumNode(Shape s, int in_id, int reduce_axis = -1, std::string name = "")
        : Node(OpType::Sum, s, {in_id}, std::move(name)), axis(reduce_axis) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 5B. Kreisselmeier-Steinhauser (KS) & LogSumExp Smooth Envelope Node
//     f(x) = m + (1/rho) * log( sum_i exp(rho * (x_i - m)) ) [- (1/rho)*log(N)]
// ============================================================================
class KSLogSumExpNode : public Node {
public:
    double rho{50.0};
    int axis{-1};
    bool subtract_log_n{false};

    KSLogSumExpNode(Shape s, int in_id, double sharpness = 50.0,
                    int reduce_axis = -1, bool sub_log_n = false,
                    std::string name = "")
        : Node(OpType::KSLogSumExp, s, {in_id}, std::move(name)),
          rho(sharpness),
          axis(reduce_axis),
          subtract_log_n(sub_log_n) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 6. Matrix Transpose Node: B = A^T
// ============================================================================
class TransposeNode : public Node {
public:
    TransposeNode(Shape s, int in_id, std::string name = "")
        : Node(OpType::Transpose, s, {in_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 7. 2D Sub-Matrix / Vector Slice Node: X[r0:r1, c0:c1]
// ============================================================================
class SliceNode : public Node {
public:
    Eigen::Index row_start{0}, row_end{1}, col_start{0}, col_end{1};
    Eigen::Index r0{0}, r1{1}, c0{0}, c1{1};

    SliceNode(Shape s, int in_id, Eigen::Index r_start, Eigen::Index r_end,
              Eigen::Index c_start, Eigen::Index c_end, std::string name = "")
        : Node(OpType::Slice, s, {in_id}, std::move(name)),
          row_start(r_start), row_end(r_end), col_start(c_start), col_end(c_end),
          r0(r_start), r1(r_end), c0(c_start), c1(c_end) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 8. Tensor Concatenation Node (axis = 0: vertical stack, axis = 1: horizontal)
// ============================================================================
class ConcatNode : public Node {
public:
    int axis{0};

    ConcatNode(Shape s, std::vector<int> in_ids, int concat_axis = 0, std::string name = "")
        : Node(OpType::Concat, s, std::move(in_ids), std::move(name)), axis(concat_axis) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 9. Reshape Node (Preserves Column-Major Element Order)
// ============================================================================
class ReshapeNode : public Node {
public:
    ReshapeNode(Shape new_shape, int in_id, std::string name = "")
        : Node(OpType::Reshape, new_shape, {in_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 10. 1D Linear Interpolation Node (Differentiable w.r.t. x_query and y_data)
// ============================================================================
class LinearInterpNode : public Node {
public:
    Eigen::VectorXd x_grid;
    bool extrapolate{false};

    LinearInterpNode(Shape s, int x_query_id, int y_data_id,
                     Eigen::VectorXd grid, bool allow_extrap = false,
                     std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    Eigen::Index locate_interval(double xq, double& t, double& h) const;
};

// ============================================================================
// 11. 1D B-Spline Evaluation Node (Cox-de Boor, C2 Smooth, 1st & 2nd Order AD)
// ============================================================================
class BSplineEvalNode : public Node {
public:
    Eigen::VectorXd knots;
    int degree{3};
    bool extrapolate{false};

    BSplineEvalNode(Shape s, int u_query_id, int ctrl_pts_id,
                    Eigen::VectorXd knot_vec, int spline_deg = 3,
                    bool allow_extrap = false, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    int find_span(double u) const;
};

// ============================================================================
// 11B. 2D Bilinear Interpolation Node (Differentiable w.r.t. x_query, y_query & z_data)
// ============================================================================
class LinearInterp2DNode : public Node {
public:
    Eigen::VectorXd x_grid;
    Eigen::VectorXd y_grid;
    bool extrapolate{false};

    LinearInterp2DNode(Shape s, int x_query_id, int y_query_id, int z_data_id,
                       Eigen::VectorXd xg, Eigen::VectorXd yg,
                       bool allow_extrap = false, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 11C. 2D Tensor-Product B-Spline Surface Node (C2 Smooth, 1st & 2nd AD)
// ============================================================================
class BSplineEval2DNode : public Node {
public:
    Eigen::VectorXd knots_u;
    Eigen::VectorXd knots_v;
    int degree_u{3};
    int degree_v{3};
    bool extrapolate{false};

    BSplineEval2DNode(Shape s, int u_query_id, int v_query_id, int ctrl_pts_id,
                      Eigen::VectorXd ku, Eigen::VectorXd kv,
                      int deg_u = 3, int deg_v = 3,
                      bool allow_extrap = false, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 11D. 3D Trilinear Interpolation Node (Differentiable w.r.t. x, y, z & v_data)
// ============================================================================
class LinearInterp3DNode : public Node {
public:
    Eigen::VectorXd x_grid;
    Eigen::VectorXd y_grid;
    Eigen::VectorXd z_grid;
    bool extrapolate{false};

    LinearInterp3DNode(Shape s, int x_query_id, int y_query_id, int z_query_id, int v_data_id,
                       Eigen::VectorXd xg, Eigen::VectorXd yg, Eigen::VectorXd zg,
                       bool allow_extrap = false, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 11E. 3D Tensor-Product B-Spline Volume Node (C2 Smooth, Hierarchical Contraction)
// ============================================================================
class BSplineEval3DNode : public Node {
public:
    Eigen::VectorXd knots_u;
    Eigen::VectorXd knots_v;
    Eigen::VectorXd knots_w;
    int degree_u{3};
    int degree_v{3};
    int degree_w{3};
    bool extrapolate{false};

    BSplineEval3DNode(Shape s, int u_query_id, int v_query_id, int w_query_id, int ctrl_pts_id,
                      Eigen::VectorXd ku, Eigen::VectorXd kv, Eigen::VectorXd kw,
                      int deg_u = 3, int deg_v = 3, int deg_w = 3,
                      bool allow_extrap = false, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 12. Hard Element-Wise Conditional Branching Node (NaN-Safe Derivative Masking)
// ============================================================================
class WhereNode : public Node {
public:
    WhereNode(Shape s, int cond_id, int true_id, int false_id, std::string name = "")
        : Node(OpType::Where, s, {cond_id, true_id, false_id}, std::move(name)) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 13. Smooth Differentiable Branching Node: w = 0.5 * (1 + tanh(k * (x - x0)))
// ============================================================================
class SmoothWhereNode : public Node {
public:
    double threshold{0.0};
    double sharpness{10.0};

    SmoothWhereNode(Shape s, int x_id, int true_id, int false_id,
                    double thresh = 0.0, double sharp = 10.0, std::string name = "")
        : Node(OpType::SmoothWhere, s, {x_id, true_id, false_id}, std::move(name)),
          threshold(thresh), sharpness(sharp) {}

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 14. Short-Circuit Subgraph Conditional Node (Evaluates Only the Taken Branch)
// ============================================================================
class CondNode : public Node {
public:
    std::shared_ptr<Graph> true_graph;
    int true_out_id{-1};
    std::shared_ptr<Graph> false_graph;
    int false_out_id{-1};

    CondNode(Shape s, int cond_id, const std::vector<int>& arg_ids,
             std::shared_ptr<Graph> t_graph, int t_out,
             std::shared_ptr<Graph> f_graph, int f_out, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 15. Nested Implicit Solver Node: Solves R(x, y) = 0 with Exact 1st/2nd-Order IFT
// ============================================================================
class ImplicitSolverNode : public Node {
public:
    std::shared_ptr<Graph> residual_graph;
    int residual_out_id{-1};
    int state_var_id{-1};
    std::vector<int> param_input_ids;
    int max_iter{50};
    double tol{1e-11};

    ImplicitSolverNode(Shape s, const std::vector<int>& outer_input_ids,
                       std::shared_ptr<Graph> res_graph, int res_out_id,
                       int inner_state_var_id, std::vector<int> inner_param_ids,
                       int max_newton_iter = 50, double newton_tol = 1e-11,
                       std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    void populate_subworkspace(const Workspace& outer_ws, Workspace& sub_ws,
                               const Eigen::VectorXd& y_val) const;
    Eigen::MatrixXd compute_jy(Workspace& sub_ws, const Eigen::VectorXd& y_val) const;
};

// ============================================================================
// 16. Custom Black-Box Function Node (Analytical/FD 1st/2nd AD & Batch Eval)
// ============================================================================
class BlackBoxNode : public Node {
public:
    using EvalFn = std::function<Eigen::VectorXd(const Eigen::VectorXd& u)>;
    using BatchEvalFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd& U_batch)>;
    using JacFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd& u)>;
    using HvpFn = std::function<Eigen::VectorXd(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& y_bar,
        const Eigen::VectorXd& u_dot)>;

    EvalFn eval_fn;
    BatchEvalFn batch_eval_fn;
    JacFn jac_fn;
    HvpFn hvp_fn;
    double fd_step{1e-7};
    bool fd_hessian{true};
    bool is_thread_safe{false};
    Eigen::MatrixXi jac_sparsity_mask;

    BlackBoxNode(Shape out_shape,
                 std::vector<int> in_ids,
                 EvalFn f_cb,
                 JacFn j_cb = nullptr,
                 HvpFn h_cb = nullptr,
                 double finite_diff_step = 1e-7,
                 bool enable_fd_hessian = true,
                 std::string name = "",
                 BatchEvalFn batch_cb = nullptr,
                 bool thread_safe = false,
                 Eigen::MatrixXi jac_mask = Eigen::MatrixXi());

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

    void forward_batch_workspaces(std::vector<Workspace>& batch_ws, Eigen::Index active_pop = -1) const;

private:
    mutable Eigen::VectorXd last_u_jac_;
    mutable Eigen::MatrixXd cached_jac_;

    Eigen::Index total_input_size() const;
    Eigen::VectorXd gather_input_val(const Workspace& ws) const;
    Eigen::VectorXd gather_input_dot(const Workspace& ws) const;
    Eigen::MatrixXd compute_jacobian_uncached(const Eigen::VectorXd& u, double step_override = 0.0) const;
    Eigen::MatrixXd get_or_compute_jacobian(const Eigen::VectorXd& u) const;
    Eigen::VectorXd compute_hvp(const Eigen::VectorXd& u,
                                const Eigen::VectorXd& y_bar,
                                const Eigen::VectorXd& u_dot) const;
};

// ============================================================================
// 17. Independent Parallel Loop Node (MapNode / fo.vmap)
// ============================================================================
class MapNode : public Node {
public:
    std::shared_ptr<Graph> body_graph;
    int body_out_id{-1};
    std::vector<int> inner_var_ids;
    std::vector<int> in_axes;
    int out_axis{0};
    int num_batches{1};
    bool body_is_thread_safe{true};

    MapNode(Shape out_shape,
            const std::vector<int>& outer_in_ids,
            std::shared_ptr<Graph> sub_graph,
            int sub_out_id,
            std::vector<int> sub_var_ids,
            std::vector<int> axes_in,
            int axis_out,
            int n_batches,
            std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    mutable std::vector<std::vector<Workspace>> thread_tapes_;

    std::vector<Workspace>& ensure_tape(int outer_tid, bool need_tier1, bool need_tier2) const;
    void extract_slice(const Eigen::VectorXd& outer_buf, const Node& outer_n,
                       int axis, int b, const Shape& inner_s,
                       Eigen::Ref<Eigen::VectorXd> dst) const;
    void accumulate_slice(const Eigen::Ref<const Eigen::VectorXd>& src,
                          const Shape& inner_s, int axis, int b,
                          const Node& outer_n, Eigen::VectorXd& outer_buf) const;
    void write_output_slice(const Eigen::Ref<const Eigen::VectorXd>& src,
                            const Shape& slice_s, int b,
                            Eigen::VectorXd& outer_buf) const;
    void read_output_slice(const Eigen::VectorXd& outer_buf,
                           const Shape& slice_s, int b,
                           Eigen::Ref<Eigen::VectorXd> dst) const;
};

// ============================================================================
// 18. Sequential Recurrent Loop Node (ScanNode / fo.scan)
// ============================================================================
class ScanNode : public Node {
public:
    std::shared_ptr<Graph> step_graph;
    int inner_carry_id{-1};
    std::vector<int> inner_seq_ids;
    std::vector<int> seq_axes;
    std::vector<int> inner_shared_ids;
    int next_carry_out_id{-1};
    int step_y_out_id{-1};
    int num_steps{1};

    ScanNode(Shape out_shape,
             const std::vector<int>& outer_in_ids,
             std::shared_ptr<Graph> sub_graph,
             int sub_carry_id,
             std::vector<int> sub_seq_ids,
             std::vector<int> axes_seq,
             std::vector<int> sub_shared_ids,
             int sub_next_carry_id,
             int sub_y_id,
             int n_steps,
             std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    mutable std::vector<std::vector<Workspace>> thread_tapes_;

    std::vector<Workspace>& ensure_tape(int outer_tid, bool need_tier1, bool need_tier2) const;
    void extract_seq_slice(const Eigen::VectorXd& outer_buf, const Node& outer_n,
                           int axis, int t, const Shape& inner_s,
                           Eigen::Ref<Eigen::VectorXd> dst) const;
    void accumulate_seq_slice(const Eigen::Ref<const Eigen::VectorXd>& src,
                              const Shape& inner_s, int axis, int t,
                              const Node& outer_n, Eigen::VectorXd& outer_buf) const;
};

// ============================================================================
// 19. Graph Operator Construction Helpers
// ============================================================================
namespace ops {

int unary(Graph& g, OpType op, int a);
int binary(Graph& g, OpType op, int a, int b);
int matmul(Graph& g, int a, int b);
int linear_solve(Graph& g, int a, int b);
int dot(Graph& g, int a, int b);
int sum(Graph& g, int a, int axis = -1);
int ks_logsumexp(Graph& g, int a, double rho = 50.0, int axis = -1, bool subtract_log_n = false);
int transpose(Graph& g, int a);
int slice(Graph& g, int a, Eigen::Index r0, Eigen::Index r1, Eigen::Index c0, Eigen::Index c1);
int concat(Graph& g, const std::vector<int>& inputs, int axis = 0);
int reshape(Graph& g, int a, Shape new_shape);
int linear_interp(Graph& g, int x_query_id, int y_data_id,
                  const Eigen::VectorXd& x_grid, bool extrapolate = false);
int bspline_eval(Graph& g, int u_query_id, int ctrl_pts_id,
                 const Eigen::VectorXd& knots, int degree = 3, bool extrapolate = false);
int linear_interp_2d(Graph& g, int x_query_id, int y_query_id, int z_data_id,
                     const Eigen::VectorXd& x_grid, const Eigen::VectorXd& y_grid,
                     bool extrapolate = false);
int bspline_eval_2d(Graph& g, int u_query_id, int v_query_id, int ctrl_pts_id,
                    const Eigen::VectorXd& knots_u, const Eigen::VectorXd& knots_v,
                    int degree_u = 3, int degree_v = 3, bool extrapolate = false);
int linear_interp_3d(Graph& g, int x_query_id, int y_query_id, int z_query_id, int v_data_id,
                     const Eigen::VectorXd& x_grid, const Eigen::VectorXd& y_grid,
                     const Eigen::VectorXd& z_grid, bool extrapolate = false);
int bspline_eval_3d(Graph& g, int u_query_id, int v_query_id, int w_query_id, int ctrl_pts_id,
                    const Eigen::VectorXd& knots_u, const Eigen::VectorXd& knots_v,
                    const Eigen::VectorXd& knots_w,
                    int degree_u = 3, int degree_v = 3, int degree_w = 3,
                    bool extrapolate = false);
int where(Graph& g, int cond_id, int true_id, int false_id);
int smooth_where(Graph& g, int x_id, int true_id, int false_id, double threshold = 0.0, double sharpness = 10.0);
int cond_subgraph(Graph& g, int cond_id, const std::vector<int>& arg_ids,
                  std::shared_ptr<Graph> true_graph, int true_out_id,
                  std::shared_ptr<Graph> false_graph, int false_out_id);
int implicit_solve(Graph& g, const std::vector<int>& outer_input_ids,
                   std::shared_ptr<Graph> res_graph, int res_out_id,
                   int inner_state_var_id, const std::vector<int>& inner_param_ids,
                   int max_iter = 50, double tol = 1e-11);
int blackbox(Graph& g, Shape out_shape, const std::vector<int>& in_ids,
             BlackBoxNode::EvalFn eval_fn,
             BlackBoxNode::JacFn jac_fn = nullptr,
             BlackBoxNode::HvpFn hvp_fn = nullptr,
             double fd_step = 1e-7,
             bool fd_hessian = true,
             const std::string& name = "",
             BlackBoxNode::BatchEvalFn batch_eval_fn = nullptr,
             bool is_thread_safe = false,
             Eigen::MatrixXi jac_sparsity_mask = Eigen::MatrixXi());
int map_subgraph(Graph& g, Shape out_shape, const std::vector<int>& outer_in_ids,
                 std::shared_ptr<Graph> body_graph, int body_out_id,
                 const std::vector<int>& inner_var_ids,
                 const std::vector<int>& in_axes, int out_axis, int num_batches,
                 const std::string& name = "");
int scan_subgraph(Graph& g, Shape out_shape, const std::vector<int>& outer_in_ids,
                  std::shared_ptr<Graph> step_graph, int inner_carry_id,
                  const std::vector<int>& inner_seq_ids,
                  const std::vector<int>& seq_axes,
                  const std::vector<int>& inner_shared_ids,
                  int next_carry_out_id, int step_y_out_id, int num_steps,
                  const std::string& name = "");

} // namespace ops

} // namespace fastopt

#endif // FASTOPT_OPS_HPP
