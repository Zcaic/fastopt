#include "fastopt/core.hpp"
#include "fastopt/ops.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace fastopt {

namespace {

inline std::uint64_t double_to_bits(double v) {
    if (v == 0.0) {
        v = 0.0; // Canonicalize -0.0 to +0.0
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(double));
    return bits;
}

} // namespace

// ============================================================================
// Leaf Node Constructors
// ============================================================================
VariableNode::VariableNode(Shape s, const Eigen::VectorXd& init,
                           const Eigen::VectorXd& lower,
                           const Eigen::VectorXd& upper,
                           std::string var_name)
    : Node(OpType::Variable, s, {}, std::move(var_name)),
      init_val(init),
      lb(lower),
      ub(upper) {
    requires_grad = true;
    is_param_only = false;
    if (init_val.size() != s.size() || lb.size() != s.size() || ub.size() != s.size()) {
        throw std::invalid_argument("VariableNode: init, lb, or ub size does not match shape.");
    }
}

ParameterNode::ParameterNode(Shape s, const Eigen::VectorXd& val, std::string param_name)
    : Node(OpType::Parameter, s, {}, std::move(param_name)), current_val(val) {
    requires_grad = false;
    is_param_only = true;
    if (current_val.size() != s.size()) {
        throw std::invalid_argument("ParameterNode: value size does not match shape.");
    }
}

ConstantNode::ConstantNode(Shape s, const Eigen::VectorXd& val, std::string const_name)
    : Node(OpType::Constant, s, {}, std::move(const_name)), const_val(val) {
    requires_grad = false;
    is_param_only = true;
    if (const_val.size() != s.size()) {
        throw std::invalid_argument("ConstantNode: value size does not match shape.");
    }
}

ImplicitStateNode::ImplicitStateNode(Shape s, const Eigen::VectorXd& guess,
                                     const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper,
                                     std::string state_name)
    : Node(OpType::ImplicitState, s, {}, std::move(state_name)),
      init_guess(guess),
      lb(lower),
      ub(upper) {
    requires_grad = true;
    is_param_only = false;
    if (init_guess.size() != s.size() || lb.size() != s.size() || ub.size() != s.size()) {
        throw std::invalid_argument("ImplicitStateNode: guess, lb, or ub size does not match shape.");
    }
}

// ============================================================================
// Graph Implementation
// ============================================================================
int Graph::add_node(std::shared_ptr<Node> node) {
    if (!node) {
        throw std::invalid_argument("Graph::add_node received null node pointer.");
    }
    int id = static_cast<int>(nodes_.size());
    node->id = id;
    node->graph = this;

    // Propagate requires_grad and is_param_only flags from parent nodes for non-leaf ops
    if (node->op != OpType::Variable && node->op != OpType::Parameter &&
        node->op != OpType::Constant && node->op != OpType::ImplicitState) {
        bool grad = false;
        bool param_only = true;
        for (int parent_id : node->inputs) {
            if (parent_id < 0 || parent_id >= id) {
                throw std::out_of_range("Graph::add_node: invalid parent node ID.");
            }
            const Node& p = *nodes_[static_cast<std::size_t>(parent_id)];
            if (p.requires_grad) grad = true;
            if (!p.is_param_only) param_only = false;
        }
        node->requires_grad = grad;
        node->is_param_only = param_only;
    }

    nodes_.push_back(std::move(node));
    is_compiled_ = false;
    return id;
}

int Graph::add_variable(Shape s, const Eigen::VectorXd& init, const Eigen::VectorXd& lb,
                        const Eigen::VectorXd& ub, const std::string& name) {
    auto node = std::make_shared<VariableNode>(s, init, lb, ub, name);
    node->offset = total_var_size_;
    total_var_size_ += s.size();
    int id = add_node(node);
    var_ids_.push_back(id);
    return id;
}

int Graph::add_parameter(Shape s, const Eigen::VectorXd& val, const std::string& name) {
    auto node = std::make_shared<ParameterNode>(s, val, name);
    int id = add_node(node);
    param_ids_.push_back(id);
    return id;
}

int Graph::add_constant(Shape s, const Eigen::VectorXd& val, const std::string& name) {
    if (cse_enabled_ && s.is_scalar() && val.size() == 1 && name.empty()) {
        const std::uint64_t key = double_to_bits(val(0));
        auto it = scalar_const_cache_.find(key);
        if (it != scalar_const_cache_.end()) {
            ++cse_hit_count_;
            return it->second;
        }
        auto node = std::make_shared<ConstantNode>(s, val, name);
        int id = add_node(node);
        const_ids_.push_back(id);
        scalar_const_cache_[key] = id;
        return id;
    }
    auto node = std::make_shared<ConstantNode>(s, val, name);
    int id = add_node(node);
    const_ids_.push_back(id);
    return id;
}

int Graph::add_scalar_constant(double val) {
    if (cse_enabled_) {
        const std::uint64_t key = double_to_bits(val);
        auto it = scalar_const_cache_.find(key);
        if (it != scalar_const_cache_.end()) {
            ++cse_hit_count_;
            return it->second;
        }
        Eigen::VectorXd v(1);
        v(0) = val;
        auto node = std::make_shared<ConstantNode>(Shape(1, 1), v, "");
        int id = add_node(node);
        const_ids_.push_back(id);
        scalar_const_cache_[key] = id;
        return id;
    }
    Eigen::VectorXd v(1);
    v(0) = val;
    return add_constant(Shape(1, 1), v);
}

int Graph::find_cse_node(const std::string& key) {
    if (!cse_enabled_) return -1;
    auto it = cse_table_.find(key);
    if (it != cse_table_.end()) {
        ++cse_hit_count_;
        return it->second;
    }
    return -1;
}

void Graph::register_cse_node(const std::string& key, int node_id) {
    if (cse_enabled_) {
        cse_table_[key] = node_id;
    }
}

bool Graph::has_unsafe_blackbox() const {
    for (const auto& n_ptr : nodes_) {
        if (!n_ptr) continue;
        const Node& nd = *n_ptr;
        if (nd.op == OpType::BlackBox) {
            const auto& bb = static_cast<const BlackBoxNode&>(nd);
            if (!bb.is_thread_safe) {
                return true;
            }
        } else if (nd.op == OpType::Map) {
            const auto& mn = static_cast<const MapNode&>(nd);
            if (mn.body_graph && mn.body_graph->has_unsafe_blackbox()) {
                return true;
            }
        } else if (nd.op == OpType::Scan) {
            const auto& sn = static_cast<const ScanNode&>(nd);
            if (sn.step_graph && sn.step_graph->has_unsafe_blackbox()) {
                return true;
            }
        } else if (nd.op == OpType::Cond) {
            const auto& cn = static_cast<const CondNode&>(nd);
            if ((cn.true_graph && cn.true_graph->has_unsafe_blackbox()) ||
                (cn.false_graph && cn.false_graph->has_unsafe_blackbox())) {
                return true;
            }
        } else if (nd.op == OpType::ImplicitSolver) {
            const auto& in_sol = static_cast<const ImplicitSolverNode&>(nd);
            if (in_sol.residual_graph && in_sol.residual_graph->has_unsafe_blackbox()) {
                return true;
            }
        }
    }
    return false;
}

int Graph::add_implicit_state(Shape s, const Eigen::VectorXd& guess, const Eigen::VectorXd& lb,
                              const Eigen::VectorXd& ub, const std::string& name) {
    auto node = std::make_shared<ImplicitStateNode>(s, guess, lb, ub, name);
    node->offset = total_var_size_;
    total_var_size_ += s.size();
    int id = add_node(node);
    var_ids_.push_back(id);
    return id;
}

void Graph::set_parameter_value(int node_id, const Eigen::VectorXd& val) {
    Node& n = node_mut(node_id);
    if (n.op != OpType::Parameter) {
        throw std::invalid_argument("set_parameter_value called on non-Parameter node.");
    }
    auto& p = static_cast<ParameterNode&>(n);
    if (val.size() != p.size()) {
        throw std::invalid_argument("set_parameter_value: dimension mismatch.");
    }
    p.current_val = val;
    ++param_version_;
}

Eigen::VectorXd Graph::get_parameter_value(int node_id) const {
    const Node& n = node(node_id);
    if (n.op != OpType::Parameter) {
        throw std::invalid_argument("get_parameter_value called on non-Parameter node.");
    }
    return static_cast<const ParameterNode&>(n).current_val;
}

void Graph::set_variable_init(int node_id, const Eigen::VectorXd& val) {
    Node& n = node_mut(node_id);
    if (n.op == OpType::Variable) {
        auto& v = static_cast<VariableNode&>(n);
        if (val.size() != v.size()) {
            throw std::invalid_argument("set_variable_init: dimension mismatch.");
        }
        v.init_val = val;
    } else if (n.op == OpType::ImplicitState) {
        auto& v = static_cast<ImplicitStateNode&>(n);
        if (val.size() != v.size()) {
            throw std::invalid_argument("set_variable_init: dimension mismatch.");
        }
        v.init_guess = val;
    } else {
        throw std::invalid_argument("set_variable_init called on non-Variable node.");
    }
}

Eigen::VectorXd Graph::get_variable_init(int node_id) const {
    const Node& n = node(node_id);
    if (n.op == OpType::Variable) {
        return static_cast<const VariableNode&>(n).init_val;
    } else if (n.op == OpType::ImplicitState) {
        return static_cast<const ImplicitStateNode&>(n).init_guess;
    }
    throw std::invalid_argument("get_variable_init called on non-Variable node.");
}

void Graph::compile(const std::vector<int>& output_ids) {
    const std::size_t n = nodes_.size();
    std::vector<char> reachable(n, 0);

    if (output_ids.empty()) {
        std::fill(reachable.begin(), reachable.end(), 1);
    } else {
        for (int out_id : output_ids) {
            if (out_id < 0 || static_cast<std::size_t>(out_id) >= n) {
                throw std::out_of_range("Graph::compile: invalid output node ID.");
            }
            reachable[static_cast<std::size_t>(out_id)] = 1;
        }
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            if (reachable[static_cast<std::size_t>(i)]) {
                for (int p_id : nodes_[static_cast<std::size_t>(i)]->inputs) {
                    reachable[static_cast<std::size_t>(p_id)] = 1;
                }
            }
        }
    }

    // Three-segment contiguous memory offset allocation:
    // Segment 1: All Variable / ImplicitState leaf nodes in creation order
    Eigen::Index current_offset = 0;
    for (int vid : var_ids_) {
        Node& vnode = *nodes_[static_cast<std::size_t>(vid)];
        vnode.offset = current_offset;
        current_offset += vnode.size();
    }
    total_var_size_ = current_offset;

    // Segment 2: Parameter and Constant leaf nodes
    for (int pid : param_ids_) {
        Node& pnode = *nodes_[static_cast<std::size_t>(pid)];
        pnode.offset = current_offset;
        current_offset += pnode.size();
    }
    for (int cid : const_ids_) {
        Node& cnode = *nodes_[static_cast<std::size_t>(cid)];
        cnode.offset = current_offset;
        current_offset += cnode.size();
    }

    // Segment 3: Intermediate operator nodes (separated into param_only and dynamic orders)
    param_only_order_.clear();
    dynamic_order_.clear();
    all_exec_order_.clear();

    for (std::size_t i = 0; i < n; ++i) {
        Node& nd = *nodes_[i];
        if (nd.op == OpType::Variable || nd.op == OpType::Parameter ||
            nd.op == OpType::Constant || nd.op == OpType::ImplicitState) {
            continue;
        }
        nd.offset = current_offset;
        current_offset += nd.size();

        if (!reachable[i]) continue;

        all_exec_order_.push_back(static_cast<int>(i));
        if (nd.is_param_only) {
            param_only_order_.push_back(static_cast<int>(i));
        } else {
            dynamic_order_.push_back(static_cast<int>(i));
        }
    }

    total_memory_size_ = current_offset;
    is_compiled_ = true;
    ++param_version_;
}

Eigen::VectorXd Graph::initial_decision_vector() const {
    Eigen::VectorXd x0(total_var_size_);
    for (int vid : var_ids_) {
        const Node& n = *nodes_[static_cast<std::size_t>(vid)];
        if (n.op == OpType::Variable) {
            x0.segment(n.offset, n.size()) = static_cast<const VariableNode&>(n).init_val;
        } else if (n.op == OpType::ImplicitState) {
            x0.segment(n.offset, n.size()) = static_cast<const ImplicitStateNode&>(n).init_guess;
        }
    }
    return x0;
}

Eigen::VectorXd Graph::lower_bounds_vector() const {
    Eigen::VectorXd lb(total_var_size_);
    for (int vid : var_ids_) {
        const Node& n = *nodes_[static_cast<std::size_t>(vid)];
        if (n.op == OpType::Variable) {
            lb.segment(n.offset, n.size()) = static_cast<const VariableNode&>(n).lb;
        } else if (n.op == OpType::ImplicitState) {
            lb.segment(n.offset, n.size()) = static_cast<const ImplicitStateNode&>(n).lb;
        }
    }
    return lb;
}

Eigen::VectorXd Graph::upper_bounds_vector() const {
    Eigen::VectorXd ub(total_var_size_);
    for (int vid : var_ids_) {
        const Node& n = *nodes_[static_cast<std::size_t>(vid)];
        if (n.op == OpType::Variable) {
            ub.segment(n.offset, n.size()) = static_cast<const VariableNode&>(n).ub;
        } else if (n.op == OpType::ImplicitState) {
            ub.segment(n.offset, n.size()) = static_cast<const ImplicitStateNode&>(n).ub;
        }
    }
    return ub;
}

void Graph::initialize_static_workspace(Workspace& ws) const {
    ws.allocate_tier0(total_memory_size_);
    for (int cid : const_ids_) {
        nodes_[static_cast<std::size_t>(cid)]->forward(ws);
    }
    for (int pid : param_ids_) {
        nodes_[static_cast<std::size_t>(pid)]->forward(ws);
    }
    for (int id : param_only_order_) {
        nodes_[static_cast<std::size_t>(id)]->forward(ws);
    }
}

} // namespace fastopt
