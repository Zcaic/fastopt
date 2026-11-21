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
// 19. Fused CPU Neural Network Surrogate Operator (MLP / ResNet / LayerNorm)
// ============================================================================
enum class NNActivation {
    Identity = 0,
    Tanh = 1,
    SiLU = 2,
    GELU = 3,
    ReLU = 4,
    Softplus = 5,
    ELU = 6,
    Sin = 7,
    Sigmoid = 8,
    LeakyReLU = 9,
    Mish = 10,
    SELU = 11,
    HardSwish = 12
};

struct MLPLayerSpec {
    Eigen::MatrixXd W;
    Eigen::VectorXd b;
    NNActivation activation = NNActivation::Tanh;
    bool residual = false;
    bool layer_norm = false;
    Eigen::VectorXd ln_gamma;
    Eigen::VectorXd ln_beta;
    double ln_eps = 1e-5;
};

// Immutable shared weight container (supports zero-copy sharing across nodes
// and optional float32 mixed-precision storage for 2x AVX2/FMA SIMD throughput).
struct MLPEngineData {
    struct LayerD {
        Eigen::MatrixXd W;
        Eigen::VectorXd b;
        NNActivation activation = NNActivation::Tanh;
        bool residual = false;
        bool layer_norm = false;
        Eigen::VectorXd ln_gamma;
        Eigen::VectorXd ln_beta;
        double ln_eps = 1e-5;
        Eigen::Index d_in = 0;
        Eigen::Index d_out = 0;
    };
    struct LayerF {
        Eigen::MatrixXf W;
        Eigen::VectorXf b;
        NNActivation activation = NNActivation::Tanh;
        bool residual = false;
        bool layer_norm = false;
        Eigen::VectorXf ln_gamma;
        Eigen::VectorXf ln_beta;
        float ln_eps = 1e-5f;
        Eigen::Index d_in = 0;
        Eigen::Index d_out = 0;
    };

    bool use_float32 = false;
    Eigen::Index d_in = 0;
    Eigen::Index d_out = 0;
    bool has_in_norm = false;
    bool has_out_norm = false;

    Eigen::VectorXd in_mean_d, in_inv_std_d, out_mean_d, out_std_d;
    Eigen::VectorXf in_mean_f, in_inv_std_f, out_mean_f, out_std_f;

    std::vector<LayerD> layers_d;
    std::vector<LayerF> layers_f;

    static std::shared_ptr<const MLPEngineData> create(
        const std::vector<MLPLayerSpec>& layers,
        bool use_float32 = false,
        const Eigen::VectorXd& in_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& in_inv_std = Eigen::VectorXd(),
        const Eigen::VectorXd& out_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& out_std = Eigen::VectorXd());
};

// Standalone compiled MLP handle for direct numeric evaluation and zero-copy
// insertion into one or more fastopt AD graphs.
class CompiledMLP {
public:
    std::shared_ptr<const MLPEngineData> data;
    std::string name;

    CompiledMLP(const std::vector<MLPLayerSpec>& layers,
                bool use_float32 = false,
                const Eigen::VectorXd& in_mean = Eigen::VectorXd(),
                const Eigen::VectorXd& in_inv_std = Eigen::VectorXd(),
                const Eigen::VectorXd& out_mean = Eigen::VectorXd(),
                const Eigen::VectorXd& out_std = Eigen::VectorXd(),
                std::string name_ = "");

    Eigen::Index d_in() const { return data->d_in; }
    Eigen::Index d_out() const { return data->d_out; }
    int num_layers() const {
        return static_cast<int>(data->use_float32 ? data->layers_f.size() : data->layers_d.size());
    }
    bool use_float32() const { return data->use_float32; }

    // Direct numeric evaluation for arbitrary batch matrix X of shape (d_in, B).
    // Thread-safe and GIL-free; automatically uses OpenMP column blocking for large B.
    Eigen::MatrixXd eval(const Eigen::MatrixXd& X, int num_threads = 0) const;
};

class MLPNode : public Node {
public:
    std::shared_ptr<const MLPEngineData> data;
    Eigen::Index batch_cols = 1;

    MLPNode(Shape out_shape, int x_id,
            std::shared_ptr<const MLPEngineData> engine_data,
            std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    struct ScratchD;
    struct ScratchF;
    mutable std::vector<std::shared_ptr<ScratchD>> thread_scratch_d_;
    mutable std::vector<std::shared_ptr<ScratchF>> thread_scratch_f_;

    ScratchD& ensure_scratch_d(int tid) const;
    ScratchF& ensure_scratch_f(int tid) const;
};

// Fused 1D Convolution (im2col + BLAS3 GEMM + fused activation, fp32/fp64, 0/1/2 AD)
struct Conv1DEngineData {
    bool use_float32 = false;
    Eigen::Index c_in = 0;
    Eigen::Index c_out = 0;
    int kernel_size = 1;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    NNActivation activation = NNActivation::Identity;

    // W has shape (c_out, c_in * kernel_size)
    Eigen::MatrixXd W_d;
    Eigen::VectorXd b_d;
    Eigen::MatrixXf W_f;
    Eigen::VectorXf b_f;

    Eigen::Index out_length(Eigen::Index l_in) const {
        const Eigen::Index eff_k = static_cast<Eigen::Index>(dilation) * (kernel_size - 1) + 1;
        const Eigen::Index num = l_in + 2 * static_cast<Eigen::Index>(padding) - eff_k;
        if (num < 0) {
            throw std::invalid_argument("Conv1D: input length too small for kernel_size/dilation/padding.");
        }
        return num / static_cast<Eigen::Index>(stride) + 1;
    }

    static std::shared_ptr<const Conv1DEngineData> create(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& b,
        int c_in,
        int kernel_size,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        NNActivation activation = NNActivation::Identity,
        bool use_float32 = false);
};

class CompiledConv1D {
public:
    std::shared_ptr<const Conv1DEngineData> data;
    std::string name;

    CompiledConv1D(const Eigen::MatrixXd& W,
                   const Eigen::VectorXd& b,
                   int c_in,
                   int kernel_size,
                   int stride = 1,
                   int padding = 0,
                   int dilation = 1,
                   NNActivation activation = NNActivation::Identity,
                   bool use_float32 = false,
                   std::string name_ = "");

    Eigen::Index c_in() const { return data->c_in; }
    Eigen::Index c_out() const { return data->c_out; }
    int kernel_size() const { return data->kernel_size; }
    int stride() const { return data->stride; }
    int padding() const { return data->padding; }
    int dilation() const { return data->dilation; }
    bool use_float32() const { return data->use_float32; }

    Eigen::MatrixXd eval(const Eigen::MatrixXd& X) const;
};

class Conv1DNode : public Node {
public:
    std::shared_ptr<const Conv1DEngineData> data;
    Eigen::Index l_in = 0;
    Eigen::Index l_out = 0;

    Conv1DNode(Shape out_shape, int x_id, Eigen::Index l_in_,
               std::shared_ptr<const Conv1DEngineData> engine_data,
               std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    struct ScratchD;
    struct ScratchF;
    mutable std::vector<std::shared_ptr<ScratchD>> thread_scratch_d_;
    mutable std::vector<std::shared_ptr<ScratchF>> thread_scratch_f_;

    ScratchD& ensure_scratch_d(int tid) const;
    ScratchF& ensure_scratch_f(int tid) const;
};

// Fused Multi-Head Self-Attention (Q/K/V/O linear projections + scaled dot-product softmax
// + optional causal mask & residual skip, fp32/fp64, exact 0/1/2 AD)
struct AttentionEngineData {
    bool use_float32 = false;
    Eigen::Index d_model = 0;
    int num_heads = 1;
    Eigen::Index d_head = 0;
    double scale = 1.0;
    bool causal = false;
    bool residual = false;

    Eigen::MatrixXd Wq_d, Wk_d, Wv_d, Wo_d;
    Eigen::VectorXd bq_d, bk_d, bv_d, bo_d;
    Eigen::MatrixXf Wq_f, Wk_f, Wv_f, Wo_f;
    Eigen::VectorXf bq_f, bk_f, bv_f, bo_f;

    static std::shared_ptr<const AttentionEngineData> create(
        const Eigen::MatrixXd& Wq, const Eigen::VectorXd& bq,
        const Eigen::MatrixXd& Wk, const Eigen::VectorXd& bk,
        const Eigen::MatrixXd& Wv, const Eigen::VectorXd& bv,
        const Eigen::MatrixXd& Wo, const Eigen::VectorXd& bo,
        int num_heads,
        bool causal = false,
        bool residual = false,
        bool use_float32 = false);
};

class CompiledAttention {
public:
    std::shared_ptr<const AttentionEngineData> data;
    std::string name;

    CompiledAttention(const Eigen::MatrixXd& Wq, const Eigen::VectorXd& bq,
                      const Eigen::MatrixXd& Wk, const Eigen::VectorXd& bk,
                      const Eigen::MatrixXd& Wv, const Eigen::VectorXd& bv,
                      const Eigen::MatrixXd& Wo, const Eigen::VectorXd& bo,
                      int num_heads,
                      bool causal = false,
                      bool residual = false,
                      bool use_float32 = false,
                      std::string name_ = "");

    Eigen::Index d_model() const { return data->d_model; }
    int num_heads() const { return data->num_heads; }
    bool causal() const { return data->causal; }
    bool residual() const { return data->residual; }
    bool use_float32() const { return data->use_float32; }

    Eigen::MatrixXd eval(const Eigen::MatrixXd& X) const;
};

class AttentionNode : public Node {
public:
    std::shared_ptr<const AttentionEngineData> data;
    Eigen::Index seq_len = 0;

    AttentionNode(Shape out_shape, int x_id,
                  std::shared_ptr<const AttentionEngineData> engine_data,
                  std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    struct ScratchD;
    struct ScratchF;
    mutable std::vector<std::shared_ptr<ScratchD>> thread_scratch_d_;
    mutable std::vector<std::shared_ptr<ScratchF>> thread_scratch_f_;

    ScratchD& ensure_scratch_d(int tid) const;
    ScratchF& ensure_scratch_f(int tid) const;
};

// Fused 2D Convolution (im2col_2d + BLAS3 GEMM + fused activation, fp32/fp64, 0/1/2 AD)
// Spatial layout: input (C_in, H_in * W_in) -> output (C_out, H_out * W_out).
struct Conv2DEngineData {
    bool use_float32 = false;
    Eigen::Index c_in = 0;
    Eigen::Index c_out = 0;
    int kh = 1, kw = 1;
    int sh = 1, sw = 1;
    int ph = 0, pw = 0;
    int dh = 1, dw = 1;
    NNActivation activation = NNActivation::Identity;

    // W has shape (c_out, c_in * kh * kw)
    Eigen::MatrixXd W_d;
    Eigen::VectorXd b_d;
    Eigen::MatrixXf W_f;
    Eigen::VectorXf b_f;

    std::pair<int, int> out_hw(int h_in, int w_in) const {
        const int eff_kh = dh * (kh - 1) + 1;
        const int eff_kw = dw * (kw - 1) + 1;
        const int nh = h_in + 2 * ph - eff_kh;
        const int nw = w_in + 2 * pw - eff_kw;
        if (nh < 0 || nw < 0) {
            throw std::invalid_argument("Conv2D: input (H, W) too small for kernel/dilation/padding.");
        }
        return {nh / sh + 1, nw / sw + 1};
    }

    static std::shared_ptr<const Conv2DEngineData> create(
        const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
        int c_in, int kh, int kw,
        int sh = 1, int sw = 1, int ph = 0, int pw = 0, int dh = 1, int dw = 1,
        NNActivation activation = NNActivation::Identity,
        bool use_float32 = false);
};

class CompiledConv2D {
public:
    std::shared_ptr<const Conv2DEngineData> data;
    std::string name;

    CompiledConv2D(const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
                   int c_in, int kh, int kw,
                   int sh = 1, int sw = 1, int ph = 0, int pw = 0, int dh = 1, int dw = 1,
                   NNActivation activation = NNActivation::Identity,
                   bool use_float32 = false, std::string name_ = "");

    Eigen::Index c_in() const { return data->c_in; }
    Eigen::Index c_out() const { return data->c_out; }
    std::pair<int, int> out_hw(int h_in, int w_in) const { return data->out_hw(h_in, w_in); }
    Eigen::MatrixXd eval(const Eigen::MatrixXd& X, int h_in, int w_in) const;
};

class Conv2DNode : public Node {
public:
    std::shared_ptr<const Conv2DEngineData> data;
    int h_in = 0, w_in = 0, h_out = 0, w_out = 0;

    Conv2DNode(Shape out_shape, int x_id, int h_in_, int w_in_, int h_out_, int w_out_,
               std::shared_ptr<const Conv2DEngineData> engine_data, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    struct ScratchD;
    struct ScratchF;
    mutable std::vector<std::shared_ptr<ScratchD>> thread_scratch_d_;
    mutable std::vector<std::shared_ptr<ScratchF>> thread_scratch_f_;
    ScratchD& ensure_scratch_d(int tid) const;
    ScratchF& ensure_scratch_f(int tid) const;
};

// 2D Spatial Pooling (AvgPool2D / MaxPool2D, exact 0/1/2 AD & local window sparsity)
enum class Pool2DMode { Avg = 0, Max = 1 };

class Pool2DNode : public Node {
public:
    Pool2DMode mode = Pool2DMode::Avg;
    Eigen::Index channels = 0;
    int h_in = 0, w_in = 0, h_out = 0, w_out = 0;
    int kh = 2, kw = 2, sh = 2, sw = 2, ph = 0, pw = 0;

    Pool2DNode(Shape out_shape, int x_id, Pool2DMode mode_,
               Eigen::Index channels_, int h_in_, int w_in_, int h_out_, int w_out_,
               int kh_, int kw_, int sh_, int sw_, int ph_, int pw_,
               std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;

private:
    mutable std::vector<std::vector<Eigen::Index>> thread_argmax_;
    std::vector<Eigen::Index>& ensure_argmax(int tid) const;
};

// Native C++ RMSNorm (column-wise root-mean-square normalization with affine gamma, exact 0/1/2 AD)
class RMSNormNode : public Node {
public:
    Eigen::VectorXd gamma;
    double eps = 1e-6;
    Eigen::Index dim = 0;
    Eigen::Index cols = 1;

    RMSNormNode(Shape out_shape, int x_id, Eigen::VectorXd gamma_, double eps_ = 1e-6, std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};


// Native C++17 Kernel Surrogate Engine (RBF, Ordinary/Universal Kriging, KPLS, GEK, GEKPLS, Co-Kriging)
// Supports 7 radial/anisotropic kernels, polynomial trend (deg -1/0/1/2), value + gradient basis,
// and exact 0th, 1st (VJP/JVP), and 2nd-order (Forward-over-Reverse HVP) AD for both Mean and Variance.
enum class SurrogateKernel {
    Gaussian = 0,
    Multiquadric = 1,
    InvMultiquadric = 2,
    Cubic = 3,
    ThinPlateSpline = 4,
    Matern32 = 5,
    Matern52 = 6
};

enum class SurrogateOutputMode {
    Mean = 0,
    Variance = 1
};

struct SurrogateEngineData {
    SurrogateKernel kernel = SurrogateKernel::Gaussian;
    int poly_degree = 0; // -1: none, 0: const, 1: linear, 2: quadratic
    Eigen::Index d_in = 0;
    Eigen::Index d_out = 0;
    Eigen::Index n_centers = 0;
    Eigen::Index n_poly = 0;
    bool has_grad_basis = false;
    bool has_variance = false;
    double r_eps = 1e-14;

    Eigen::MatrixXd X_centers;    // (d_in, n_centers) normalized training centers
    Eigen::VectorXd theta;        // (d_in,) positive inverse squared lengthscales
    Eigen::VectorXd in_mean;      // (d_in,)
    Eigen::VectorXd in_inv_std;   // (d_in,)
    Eigen::VectorXd out_mean;     // (d_out,)
    Eigen::VectorXd out_std;      // (d_out,)

    Eigen::MatrixXd alpha;        // (n_centers, d_out) value basis weights
    Eigen::MatrixXd beta_scaled;  // (d_in * n_centers, d_out) pre-multiplied by theta (empty if !has_grad_basis)
    Eigen::MatrixXd poly_coef;    // (n_poly, d_out) polynomial trend weights

    // Variance quadratic form in normalized space:
    // V(x) = 1 - k^T M_kk k - 2 k^T M_kp p + p^T M_pp p
    Eigen::VectorXd sigma2;       // (d_out,) process variance in normalized space
    Eigen::MatrixXd M_kk;         // (M_basis, M_basis)
    Eigen::MatrixXd M_kp;         // (M_basis, n_poly)
    Eigen::MatrixXd M_pp;         // (n_poly, n_poly)

    static Eigen::Index poly_dim(Eigen::Index d, int deg) {
        if (deg < 0) return 0;
        if (deg == 0) return 1;
        if (deg == 1) return 1 + d;
        return 1 + d + (d * (d + 1)) / 2;
    }

    static std::shared_ptr<const SurrogateEngineData> create(
        SurrogateKernel kernel,
        int poly_degree,
        const Eigen::MatrixXd& X_centers,
        const Eigen::VectorXd& theta,
        const Eigen::MatrixXd& alpha,
        const Eigen::MatrixXd& beta_grad = Eigen::MatrixXd(),
        const Eigen::MatrixXd& poly_coef = Eigen::MatrixXd(),
        const Eigen::VectorXd& in_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& in_inv_std = Eigen::VectorXd(),
        const Eigen::VectorXd& out_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& out_std = Eigen::VectorXd(),
        const Eigen::VectorXd& sigma2 = Eigen::VectorXd(),
        const Eigen::MatrixXd& M_kk = Eigen::MatrixXd(),
        const Eigen::MatrixXd& M_kp = Eigen::MatrixXd(),
        const Eigen::MatrixXd& M_pp = Eigen::MatrixXd(),
        double r_eps = 1e-14);
};

class CompiledSurrogate {
public:
    std::shared_ptr<const SurrogateEngineData> data;
    std::string name;

    CompiledSurrogate(
        SurrogateKernel kernel,
        int poly_degree,
        const Eigen::MatrixXd& X_centers,
        const Eigen::VectorXd& theta,
        const Eigen::MatrixXd& alpha,
        const Eigen::MatrixXd& beta_grad = Eigen::MatrixXd(),
        const Eigen::MatrixXd& poly_coef = Eigen::MatrixXd(),
        const Eigen::VectorXd& in_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& in_inv_std = Eigen::VectorXd(),
        const Eigen::VectorXd& out_mean = Eigen::VectorXd(),
        const Eigen::VectorXd& out_std = Eigen::VectorXd(),
        const Eigen::VectorXd& sigma2 = Eigen::VectorXd(),
        const Eigen::MatrixXd& M_kk = Eigen::MatrixXd(),
        const Eigen::MatrixXd& M_kp = Eigen::MatrixXd(),
        const Eigen::MatrixXd& M_pp = Eigen::MatrixXd(),
        double r_eps = 1e-14,
        std::string name_ = "");

    Eigen::Index d_in() const { return data->d_in; }
    Eigen::Index d_out() const { return data->d_out; }
    Eigen::Index n_centers() const { return data->n_centers; }
    bool has_grad_basis() const { return data->has_grad_basis; }
    bool has_variance() const { return data->has_variance; }

    Eigen::MatrixXd eval(const Eigen::MatrixXd& X, int num_threads = 0) const;
    Eigen::MatrixXd eval_variance(const Eigen::MatrixXd& X, int num_threads = 0) const;
};

class SurrogateNode : public Node {
public:
    std::shared_ptr<const SurrogateEngineData> data;
    SurrogateOutputMode mode = SurrogateOutputMode::Mean;

    SurrogateNode(Shape out_shape, int x_id,
                  std::shared_ptr<const SurrogateEngineData> engine_data,
                  SurrogateOutputMode mode_ = SurrogateOutputMode::Mean,
                  std::string name = "");

    void forward(Workspace& ws) const override;
    void backward(Workspace& ws) const override;
    void forward_tangent(Workspace& ws) const override;
    void backward_hessian(Workspace& ws) const override;
};

// ============================================================================
// 20. Graph Operator Construction Helpers
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
int mlp(Graph& g, int x_id, std::shared_ptr<const MLPEngineData> engine_data,
        const std::string& name = "");
int conv1d(Graph& g, int x_id, std::shared_ptr<const Conv1DEngineData> engine_data,
           const std::string& name = "");
int attention(Graph& g, int x_id, std::shared_ptr<const AttentionEngineData> engine_data,
              const std::string& name = "");
int conv2d(Graph& g, int x_id, int h_in, int w_in,
           std::shared_ptr<const Conv2DEngineData> engine_data,
           const std::string& name = "");
int pool2d(Graph& g, int x_id, Pool2DMode mode, int h_in, int w_in,
           int kh, int kw, int sh = 2, int sw = 2, int ph = 0, int pw = 0,
           const std::string& name = "");
int rms_norm(Graph& g, int x_id, const Eigen::VectorXd& gamma,
             double eps = 1e-6, const std::string& name = "");
int surrogate(Graph& g, int x_id, std::shared_ptr<const SurrogateEngineData> engine_data,
              SurrogateOutputMode mode = SurrogateOutputMode::Mean, const std::string& name = "");

} // namespace ops

} // namespace fastopt

#endif // FASTOPT_OPS_HPP
