#ifndef FASTOPT_CORE_HPP
#define FASTOPT_CORE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastopt {

// ============================================================================
// 1. Tensor Shape Metadata (2D Matrix / Column Vector / Scalar)
// ============================================================================
struct Shape {
    Eigen::Index rows{1};
    Eigen::Index cols{1};

    constexpr Shape() = default;
    constexpr Shape(Eigen::Index r, Eigen::Index c) : rows(r), cols(c) {}

    constexpr Eigen::Index size() const { return rows * cols; }
    constexpr bool is_scalar() const { return rows == 1 && cols == 1; }
    constexpr bool is_vector() const { return cols == 1 || rows == 1; }

    constexpr bool operator==(const Shape& other) const {
        return rows == other.rows && cols == other.cols;
    }
    constexpr bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    }
};

// ============================================================================
// 2. Operator Type Enumeration
// ============================================================================
enum class OpType {
    // Leaf nodes
    Variable,
    Parameter,
    Constant,
    ImplicitState,

    // Unary element-wise operators
    Neg,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Abs,
    Square,

    // Binary element-wise operators (with broadcasting)
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,

    // Comparison operators (produce 0.0 / 1.0 condition masks)
    Lt,
    Le,
    Gt,
    Ge,
    Eq,

    // Linear algebra, tensor reduction, and smooth envelope operators
    MatMul,
    LinearSolve,
    Dot,
    Sum,
    KSLogSumExp,
    Transpose,

    // Shape and indexing operators
    Slice,
    Concat,
    Reshape,

    // 1D, 2D, and 3D interpolation operators
    LinearInterp,
    BSplineEval,
    LinearInterp2D,
    BSplineEval2D,
    LinearInterp3D,
    BSplineEval3D,

    // Conditional branching operators
    Where,
    SmoothWhere,
    Cond,

    // Nested implicit equation solver (R(x, y) = 0 with IFT derivatives)
    ImplicitSolver,

    // Custom black-box operator node (with analytical or FD 1st/2nd derivatives & batch eval)
    BlackBox,

    // Structured loop operators (subgraph reuse for independent vmap and sequential scan)
    Map,
    Scan,

    // Fused CPU neural network surrogate operators (MLP / Conv1D / MultiHeadAttention, fp32/fp64)
    MLP,
    Conv1D,
    Attention,
    Conv2D,
    Pool2D,
    RMSNorm,
    Surrogate
};

// ============================================================================
// 3. Thread-Local Numerical Workspace (Tiered On-Demand Buffers)
// ============================================================================
class Workspace {
public:
    // Tier 0: Forward values buffer
    Eigen::VectorXd val;

    // Tier 1: First-order reverse adjoints buffer
    Eigen::VectorXd bar;

    // Tier 2: Forward tangents and second-order reverse adjoints (for HVP)
    Eigen::VectorXd dot;
    Eigen::VectorXd dot_bar;

    Workspace() = default;
    explicit Workspace(Eigen::Index total_size) { allocate_tier0(total_size); }

    void allocate_tier0(Eigen::Index total_size) {
        if (val.size() != total_size) {
            val.setZero(total_size);
        }
    }

    void allocate_tier1(Eigen::Index total_size) {
        if (bar.size() != total_size) {
            bar.resize(total_size);
        }
        bar.setZero();
    }

    void allocate_tier2(Eigen::Index total_size) {
        if (dot.size() != total_size) {
            dot.resize(total_size);
        }
        if (dot_bar.size() != total_size) {
            dot_bar.resize(total_size);
        }
        dot.setZero();
        dot_bar.setZero();
    }

    // Matrix views into the flat buffers
    Eigen::Map<Eigen::MatrixXd> val_mat(Eigen::Index offset, const Shape& s) {
        return Eigen::Map<Eigen::MatrixXd>(val.data() + offset, s.rows, s.cols);
    }
    Eigen::Map<const Eigen::MatrixXd> val_mat(Eigen::Index offset, const Shape& s) const {
        return Eigen::Map<const Eigen::MatrixXd>(val.data() + offset, s.rows, s.cols);
    }

    Eigen::Map<Eigen::MatrixXd> bar_mat(Eigen::Index offset, const Shape& s) {
        return Eigen::Map<Eigen::MatrixXd>(bar.data() + offset, s.rows, s.cols);
    }
    Eigen::Map<const Eigen::MatrixXd> bar_mat(Eigen::Index offset, const Shape& s) const {
        return Eigen::Map<const Eigen::MatrixXd>(bar.data() + offset, s.rows, s.cols);
    }

    Eigen::Map<Eigen::MatrixXd> dot_mat(Eigen::Index offset, const Shape& s) {
        return Eigen::Map<Eigen::MatrixXd>(dot.data() + offset, s.rows, s.cols);
    }
    Eigen::Map<const Eigen::MatrixXd> dot_mat(Eigen::Index offset, const Shape& s) const {
        return Eigen::Map<const Eigen::MatrixXd>(dot.data() + offset, s.rows, s.cols);
    }

    Eigen::Map<Eigen::MatrixXd> dot_bar_mat(Eigen::Index offset, const Shape& s) {
        return Eigen::Map<Eigen::MatrixXd>(dot_bar.data() + offset, s.rows, s.cols);
    }
    Eigen::Map<const Eigen::MatrixXd> dot_bar_mat(Eigen::Index offset, const Shape& s) const {
        return Eigen::Map<const Eigen::MatrixXd>(dot_bar.data() + offset, s.rows, s.cols);
    }

    // 1D Array views for fast SIMD element-wise operations
    Eigen::Map<Eigen::ArrayXd> val_arr(Eigen::Index offset, const Shape& s_len) {
        return Eigen::Map<Eigen::ArrayXd>(val.data() + offset, s_len.size());
    }
    Eigen::Map<Eigen::ArrayXd> val_arr(Eigen::Index offset, Eigen::Index len) {
        return Eigen::Map<Eigen::ArrayXd>(val.data() + offset, len);
    }
    Eigen::Map<const Eigen::ArrayXd> val_arr(Eigen::Index offset, Eigen::Index len) const {
        return Eigen::Map<const Eigen::ArrayXd>(val.data() + offset, len);
    }

    Eigen::Map<Eigen::ArrayXd> bar_arr(Eigen::Index offset, Eigen::Index len) {
        return Eigen::Map<Eigen::ArrayXd>(bar.data() + offset, len);
    }
    Eigen::Map<const Eigen::ArrayXd> bar_arr(Eigen::Index offset, Eigen::Index len) const {
        return Eigen::Map<const Eigen::ArrayXd>(bar.data() + offset, len);
    }

    Eigen::Map<Eigen::ArrayXd> dot_arr(Eigen::Index offset, Eigen::Index len) {
        return Eigen::Map<Eigen::ArrayXd>(dot.data() + offset, len);
    }
    Eigen::Map<const Eigen::ArrayXd> dot_arr(Eigen::Index offset, Eigen::Index len) const {
        return Eigen::Map<const Eigen::ArrayXd>(dot.data() + offset, len);
    }

    Eigen::Map<Eigen::ArrayXd> dot_bar_arr(Eigen::Index offset, Eigen::Index len) {
        return Eigen::Map<Eigen::ArrayXd>(dot_bar.data() + offset, len);
    }
    Eigen::Map<const Eigen::ArrayXd> dot_bar_arr(Eigen::Index offset, Eigen::Index len) const {
        return Eigen::Map<const Eigen::ArrayXd>(dot_bar.data() + offset, len);
    }
};

class Graph;

// ============================================================================
// 4. Stateless Computation Graph Node Base Class
// ============================================================================
class Node {
public:
    int id{-1};
    OpType op{OpType::Constant};
    Shape shape{1, 1};
    Eigen::Index offset{0};
    std::vector<int> inputs;
    bool requires_grad{false};
    bool is_param_only{true};
    std::string name;
    const Graph* graph{nullptr};

    Node(OpType op_type, Shape s, std::vector<int> in_ids = {}, std::string node_name = "")
        : op(op_type), shape(s), inputs(std::move(in_ids)), name(std::move(node_name)) {}

    virtual ~Node() = default;

    Eigen::Index size() const { return shape.size(); }

    // Four-channel execution interface operating strictly on Workspace buffers
    virtual void forward(Workspace& ws) const = 0;
    virtual void backward(Workspace& ws) const = 0;
    virtual void forward_tangent(Workspace& ws) const = 0;
    virtual void backward_hessian(Workspace& ws) const = 0;
};

// ============================================================================
// 5. Leaf Nodes: VariableNode, ParameterNode, ConstantNode, ImplicitStateNode
// ============================================================================
class VariableNode : public Node {
public:
    Eigen::VectorXd init_val;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    VariableNode(Shape s, const Eigen::VectorXd& init, const Eigen::VectorXd& lower,
                 const Eigen::VectorXd& upper, std::string var_name = "");

    void forward(Workspace& /*ws*/) const override {}
    void backward(Workspace& /*ws*/) const override {}
    void forward_tangent(Workspace& /*ws*/) const override {}
    void backward_hessian(Workspace& /*ws*/) const override {}
};

class ParameterNode : public Node {
public:
    Eigen::VectorXd current_val;

    ParameterNode(Shape s, const Eigen::VectorXd& val, std::string param_name = "");

    void forward(Workspace& ws) const override {
        ws.val.segment(offset, size()) = current_val;
    }
    void backward(Workspace& /*ws*/) const override {}
    void forward_tangent(Workspace& ws) const override {
        ws.dot.segment(offset, size()).setZero();
    }
    void backward_hessian(Workspace& /*ws*/) const override {}
};

class ConstantNode : public Node {
public:
    Eigen::VectorXd const_val;

    ConstantNode(Shape s, const Eigen::VectorXd& val, std::string const_name = "");

    void forward(Workspace& ws) const override {
        ws.val.segment(offset, size()) = const_val;
    }
    void backward(Workspace& /*ws*/) const override {}
    void forward_tangent(Workspace& ws) const override {
        ws.dot.segment(offset, size()).setZero();
    }
    void backward_hessian(Workspace& /*ws*/) const override {}
};

class ImplicitStateNode : public Node {
public:
    Eigen::VectorXd init_guess;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    ImplicitStateNode(Shape s, const Eigen::VectorXd& guess, const Eigen::VectorXd& lower,
                      const Eigen::VectorXd& upper, std::string state_name = "");

    void forward(Workspace& /*ws*/) const override {}
    void backward(Workspace& /*ws*/) const override {}
    void forward_tangent(Workspace& /*ws*/) const override {}
    void backward_hessian(Workspace& /*ws*/) const override {}
};

// ============================================================================
// 6. Computation Graph Container (Three-Segment Memory Layout & CSE)
// ============================================================================
class Graph {
public:
    Graph() = default;

    // Add a constructed node to the graph and return its assigned ID
    int add_node(std::shared_ptr<Node> node);

    // Leaf creation helpers
    int add_variable(Shape s, const Eigen::VectorXd& init, const Eigen::VectorXd& lb,
                     const Eigen::VectorXd& ub, const std::string& name = "");
    int add_parameter(Shape s, const Eigen::VectorXd& val, const std::string& name = "");
    int add_constant(Shape s, const Eigen::VectorXd& val, const std::string& name = "");
    int add_scalar_constant(double val);
    int add_implicit_state(Shape s, const Eigen::VectorXd& guess, const Eigen::VectorXd& lb,
                           const Eigen::VectorXd& ub, const std::string& name = "");

    // Mutate parameter value at runtime and invalidate parameter folding cache
    void set_parameter_value(int node_id, const Eigen::VectorXd& val);
    Eigen::VectorXd get_parameter_value(int node_id) const;

    // Mutate variable initial value or bounds
    void set_variable_init(int node_id, const Eigen::VectorXd& val);
    Eigen::VectorXd get_variable_init(int node_id) const;

    // Compile topological layout and three-segment contiguous memory offsets
    void compile(const std::vector<int>& output_ids = {});

    // Common Subexpression Elimination (CSE) & Constant Folding controls
    void set_cse_enabled(bool enabled) { cse_enabled_ = enabled; }
    bool cse_enabled() const { return cse_enabled_; }
    std::size_t cse_hit_count() const { return cse_hit_count_; }
    int find_cse_node(const std::string& key);
    void register_cse_node(const std::string& key, int node_id);
    void record_cse_hit() { ++cse_hit_count_; }

    // Recursive thread-safety check across this graph and any nested subgraphs
    bool has_unsafe_blackbox() const;

    // Accessors
    const Node& node(int id) const { return *nodes_.at(static_cast<std::size_t>(id)); }
    Node& node_mut(int id) { return *nodes_.at(static_cast<std::size_t>(id)); }
    std::size_t num_nodes() const { return nodes_.size(); }

    const std::vector<int>& variable_ids() const { return var_ids_; }
    const std::vector<int>& parameter_ids() const { return param_ids_; }
    const std::vector<int>& constant_ids() const { return const_ids_; }
    const std::vector<int>& param_only_exec_order() const { return param_only_order_; }
    const std::vector<int>& dynamic_exec_order() const { return dynamic_order_; }
    const std::vector<int>& all_exec_order() const { return all_exec_order_; }

    Eigen::Index total_var_size() const { return total_var_size_; }
    Eigen::Index total_memory_size() const { return total_memory_size_; }
    bool is_compiled() const { return is_compiled_; }
    std::uint64_t param_version() const { return param_version_; }

    // Gather initial decision vector and bounds across all VariableNodes
    Eigen::VectorXd initial_decision_vector() const;
    Eigen::VectorXd lower_bounds_vector() const;
    Eigen::VectorXd upper_bounds_vector() const;

    // Initialize static segments (constants and parameters) into a Workspace
    void initialize_static_workspace(Workspace& ws) const;

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<int> var_ids_;
    std::vector<int> param_ids_;
    std::vector<int> const_ids_;

    // Execution orders populated by compile()
    std::vector<int> param_only_order_;
    std::vector<int> dynamic_order_;
    std::vector<int> all_exec_order_;

    Eigen::Index total_var_size_{0};
    Eigen::Index total_memory_size_{0};
    bool is_compiled_{false};
    std::uint64_t param_version_{1};

    // CSE & scalar constant deduplication tables
    bool cse_enabled_{true};
    std::size_t cse_hit_count_{0};
    std::unordered_map<std::uint64_t, int> scalar_const_cache_;
    std::unordered_map<std::string, int> cse_table_;
};

} // namespace fastopt

#endif // FASTOPT_CORE_HPP
