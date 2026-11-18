#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "fastopt/c_abi.h"
#include "fastopt/core.hpp"
#include "fastopt/evaluator.hpp"
#include "fastopt/ipopt_solver.hpp"
#include "fastopt/ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace fastopt;

namespace {

// Holds a Python callable and ensures GIL is acquired on destruction
struct GilSafeCallable {
    nb::object fn;
    explicit GilSafeCallable(nb::object f) : fn(std::move(f)) {}
    ~GilSafeCallable() {
        if (fn.is_valid()) {
            if (Py_IsInitialized()) {
                nb::gil_scoped_acquire gil;
                fn.reset();
            } else {
                fn.release();
            }
        }
    }
};

int create_c_abi_blackbox_node(
    Graph& g,
    Shape out_shape,
    const std::vector<int>& in_ids,
    FastOptCEvalFn eval_fn,
    FastOptCJacFn jac_fn,
    FastOptCHvpFn hvp_fn,
    FastOptCBatchEvalFn batch_fn,
    void* user_data,
    bool is_thread_safe,
    double fd_step,
    bool fd_hessian,
    Eigen::MatrixXi jac_sparsity_mask,
    const std::string& name,
    std::shared_ptr<GilSafeCallable> lifetime_holder = nullptr) {
    if (!eval_fn) {
        throw std::invalid_argument("C-ABI BlackBox: eval function pointer cannot be NULL.");
    }
    const Eigen::Index m_out = out_shape.size();

    BlackBoxNode::EvalFn eval_cb =
        [eval_fn, user_data, m_out, lifetime_holder](const Eigen::VectorXd& u) -> Eigen::VectorXd {
        Eigen::VectorXd y = Eigen::VectorXd::Zero(m_out);
        const int rc = eval_fn(
            u.data(), static_cast<int32_t>(u.size()), y.data(), static_cast<int32_t>(m_out), user_data);
        if (rc != 0) {
            throw std::runtime_error("C-ABI BlackBox eval returned non-zero status: " + std::to_string(rc));
        }
        return y;
    };

    BlackBoxNode::JacFn jac_cb = nullptr;
    if (jac_fn != nullptr) {
        jac_cb = [jac_fn, user_data, m_out, lifetime_holder](const Eigen::VectorXd& u) -> Eigen::MatrixXd {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m_out, u.size());
            const int rc = jac_fn(
                u.data(), static_cast<int32_t>(u.size()), J.data(), static_cast<int32_t>(m_out), user_data);
            if (rc != 0) {
                throw std::runtime_error("C-ABI BlackBox jac returned non-zero status: " + std::to_string(rc));
            }
            return J;
        };
    }

    BlackBoxNode::HvpFn hvp_cb = nullptr;
    if (hvp_fn != nullptr) {
        hvp_cb = [hvp_fn, user_data, m_out, lifetime_holder](
                     const Eigen::VectorXd& u,
                     const Eigen::VectorXd& y_bar,
                     const Eigen::VectorXd& u_dot) -> Eigen::VectorXd {
            Eigen::VectorXd hvp_out = Eigen::VectorXd::Zero(u.size());
            const int rc = hvp_fn(
                u.data(), y_bar.data(), u_dot.data(),
                static_cast<int32_t>(u.size()), static_cast<int32_t>(m_out),
                hvp_out.data(), user_data);
            if (rc != 0) {
                throw std::runtime_error("C-ABI BlackBox hvp returned non-zero status: " + std::to_string(rc));
            }
            return hvp_out;
        };
    }

    BlackBoxNode::BatchEvalFn batch_cb = nullptr;
    if (batch_fn != nullptr) {
        batch_cb = [batch_fn, user_data, m_out, lifetime_holder](
                       const Eigen::MatrixXd& U_batch) -> Eigen::MatrixXd {
            Eigen::MatrixXd Y_batch = Eigen::MatrixXd::Zero(U_batch.rows(), m_out);
            const int rc = batch_fn(
                U_batch.data(), static_cast<int32_t>(U_batch.rows()), static_cast<int32_t>(U_batch.cols()),
                Y_batch.data(), static_cast<int32_t>(m_out), user_data);
            if (rc != 0) {
                throw std::runtime_error("C-ABI BlackBox eval_batch returned non-zero status: " + std::to_string(rc));
            }
            return Y_batch;
        };
    }

    return ops::blackbox(
        g, out_shape, in_ids, std::move(eval_cb), std::move(jac_cb), std::move(hvp_cb),
        fd_step, fd_hessian, name, std::move(batch_cb), is_thread_safe, std::move(jac_sparsity_mask));
}

} // namespace

NB_MODULE(_fastopt_impl, m) {
    m.doc() = "C++17 Tensor-Level Automatic Differentiation and Optimization Core (fastopt)";

    // 1. Shape metadata binding
    nb::class_<Shape>(m, "Shape")
        .def(nb::init<>())
        .def(nb::init<Eigen::Index, Eigen::Index>(), "rows"_a, "cols"_a = 1)
        .def_rw("rows", &Shape::rows)
        .def_rw("cols", &Shape::cols)
        .def("size", &Shape::size)
        .def("is_scalar", &Shape::is_scalar)
        .def("is_vector", &Shape::is_vector)
        .def("__repr__", &Shape::to_string);

    // 2. OpType enum binding
    nb::enum_<OpType>(m, "OpType")
        .value("Variable", OpType::Variable)
        .value("Parameter", OpType::Parameter)
        .value("Constant", OpType::Constant)
        .value("ImplicitState", OpType::ImplicitState)
        .value("Neg", OpType::Neg)
        .value("Sin", OpType::Sin)
        .value("Cos", OpType::Cos)
        .value("Tan", OpType::Tan)
        .value("Asin", OpType::Asin)
        .value("Acos", OpType::Acos)
        .value("Atan", OpType::Atan)
        .value("Sinh", OpType::Sinh)
        .value("Cosh", OpType::Cosh)
        .value("Tanh", OpType::Tanh)
        .value("Exp", OpType::Exp)
        .value("Log", OpType::Log)
        .value("Sqrt", OpType::Sqrt)
        .value("Abs", OpType::Abs)
        .value("Square", OpType::Square)
        .value("Add", OpType::Add)
        .value("Sub", OpType::Sub)
        .value("Mul", OpType::Mul)
        .value("Div", OpType::Div)
        .value("Pow", OpType::Pow)
        .value("Atan2", OpType::Atan2)
        .value("Lt", OpType::Lt)
        .value("Le", OpType::Le)
        .value("Gt", OpType::Gt)
        .value("Ge", OpType::Ge)
        .value("Eq", OpType::Eq)
        .value("MatMul", OpType::MatMul)
        .value("LinearSolve", OpType::LinearSolve)
        .value("Dot", OpType::Dot)
        .value("Sum", OpType::Sum)
        .value("KSLogSumExp", OpType::KSLogSumExp)
        .value("Transpose", OpType::Transpose)
        .value("Slice", OpType::Slice)
        .value("Concat", OpType::Concat)
        .value("Reshape", OpType::Reshape)
        .value("LinearInterp", OpType::LinearInterp)
        .value("BSplineEval", OpType::BSplineEval)
        .value("LinearInterp2D", OpType::LinearInterp2D)
        .value("BSplineEval2D", OpType::BSplineEval2D)
        .value("LinearInterp3D", OpType::LinearInterp3D)
        .value("BSplineEval3D", OpType::BSplineEval3D)
        .value("Where", OpType::Where)
        .value("SmoothWhere", OpType::SmoothWhere)
        .value("Cond", OpType::Cond)
        .value("ImplicitSolver", OpType::ImplicitSolver)
        .value("BlackBox", OpType::BlackBox)
        .value("Map", OpType::Map)
        .value("Scan", OpType::Scan);

    // 3. Computation Graph container binding
    nb::class_<Graph>(m, "Graph")
        .def(nb::init<>())
        .def("add_variable", &Graph::add_variable,
             "shape"_a, "init"_a, "lb"_a, "ub"_a, "name"_a = "")
        .def("add_parameter", &Graph::add_parameter,
             "shape"_a, "val"_a, "name"_a = "")
        .def("add_constant", &Graph::add_constant,
             "shape"_a, "val"_a, "name"_a = "")
        .def("add_scalar_constant", &Graph::add_scalar_constant, "val"_a)
        .def("add_implicit_state", &Graph::add_implicit_state,
             "shape"_a, "guess"_a, "lb"_a, "ub"_a, "name"_a = "")
        .def("set_parameter_value", &Graph::set_parameter_value, "node_id"_a, "val"_a)
        .def("get_parameter_value", &Graph::get_parameter_value, "node_id"_a)
        .def("set_variable_init", &Graph::set_variable_init, "node_id"_a, "val"_a)
        .def("get_variable_init", &Graph::get_variable_init, "node_id"_a)
        .def("compile", &Graph::compile, "output_ids"_a = std::vector<int>{})
        .def("num_nodes", &Graph::num_nodes)
        .def("total_var_size", &Graph::total_var_size)
        .def("total_memory_size", &Graph::total_memory_size)
        .def("is_compiled", &Graph::is_compiled)
        .def("set_cse_enabled", &Graph::set_cse_enabled, "enabled"_a)
        .def("cse_enabled", &Graph::cse_enabled)
        .def("cse_hit_count", &Graph::cse_hit_count)
        .def("has_unsafe_blackbox", &Graph::has_unsafe_blackbox)
        .def("variable_ids", &Graph::variable_ids)
        .def("parameter_ids", &Graph::parameter_ids)
        .def("initial_decision_vector", &Graph::initial_decision_vector)
        .def("lower_bounds_vector", &Graph::lower_bounds_vector)
        .def("upper_bounds_vector", &Graph::upper_bounds_vector)
        .def("node_shape", [](const Graph& g, int id) { return g.node(id).shape; }, "id"_a)
        .def("node_op", [](const Graph& g, int id) { return g.node(id).op; }, "id"_a)
        .def("node_requires_grad", [](const Graph& g, int id) { return g.node(id).requires_grad; }, "id"_a)
        .def("node_is_param_only", [](const Graph& g, int id) { return g.node(id).is_param_only; }, "id"_a)
        .def("node_offset", [](const Graph& g, int id) { return g.node(id).offset; }, "id"_a);

    // 4. Operator construction functions
    m.def("op_unary", &ops::unary, "graph"_a, "op"_a, "a"_a);
    m.def("op_binary", &ops::binary, "graph"_a, "op"_a, "a"_a, "b"_a);
    m.def("op_matmul", &ops::matmul, "graph"_a, "a"_a, "b"_a);
    m.def("op_linear_solve", &ops::linear_solve, "graph"_a, "a"_a, "b"_a);
    m.def("op_dot", &ops::dot, "graph"_a, "a"_a, "b"_a);
    m.def("op_sum", &ops::sum, "graph"_a, "a"_a, "axis"_a = -1);
    m.def("op_ks_logsumexp", &ops::ks_logsumexp, "graph"_a, "a"_a, "rho"_a = 50.0, "axis"_a = -1, "subtract_log_n"_a = false);
    m.def("op_transpose", &ops::transpose, "graph"_a, "a"_a);
    m.def("op_slice", &ops::slice, "graph"_a, "a"_a, "r0"_a, "r1"_a, "c0"_a, "c1"_a);
    m.def("op_concat", &ops::concat, "graph"_a, "inputs"_a, "axis"_a = 0);
    m.def("op_reshape", &ops::reshape, "graph"_a, "a"_a, "new_shape"_a);
    m.def("op_linear_interp", &ops::linear_interp,
          "graph"_a, "x_query_id"_a, "y_data_id"_a, "x_grid"_a, "extrapolate"_a = false);
    m.def("op_bspline_eval", &ops::bspline_eval,
          "graph"_a, "u_query_id"_a, "ctrl_pts_id"_a, "knots"_a,
          "degree"_a = 3, "extrapolate"_a = false);
    m.def("op_linear_interp_2d", &ops::linear_interp_2d,
          "graph"_a, "x_query_id"_a, "y_query_id"_a, "z_data_id"_a,
          "x_grid"_a, "y_grid"_a, "extrapolate"_a = false);
    m.def("op_bspline_eval_2d", &ops::bspline_eval_2d,
          "graph"_a, "u_query_id"_a, "v_query_id"_a, "ctrl_pts_id"_a,
          "knots_u"_a, "knots_v"_a, "degree_u"_a = 3, "degree_v"_a = 3,
          "extrapolate"_a = false);
    m.def("op_linear_interp_3d", &ops::linear_interp_3d,
          "graph"_a, "x_query_id"_a, "y_query_id"_a, "z_query_id"_a, "v_data_id"_a,
          "x_grid"_a, "y_grid"_a, "z_grid"_a, "extrapolate"_a = false);
    m.def("op_bspline_eval_3d", &ops::bspline_eval_3d,
          "graph"_a, "u_query_id"_a, "v_query_id"_a, "w_query_id"_a, "ctrl_pts_id"_a,
          "knots_u"_a, "knots_v"_a, "knots_w"_a,
          "degree_u"_a = 3, "degree_v"_a = 3, "degree_w"_a = 3,
          "extrapolate"_a = false);
    m.def("op_where", &ops::where, "graph"_a, "cond_id"_a, "true_id"_a, "false_id"_a);
    m.def("op_smooth_where", &ops::smooth_where,
          "graph"_a, "x_id"_a, "true_id"_a, "false_id"_a,
          "threshold"_a = 0.0, "sharpness"_a = 10.0);
    m.def("op_cond_subgraph", &ops::cond_subgraph,
          "graph"_a, "cond_id"_a, "arg_ids"_a,
          "true_graph"_a, "true_out_id"_a,
          "false_graph"_a, "false_out_id"_a);
    m.def("op_implicit_solve", &ops::implicit_solve,
          "graph"_a, "outer_input_ids"_a,
          "res_graph"_a, "res_out_id"_a,
          "inner_state_var_id"_a, "inner_param_ids"_a,
          "max_iter"_a = 50, "tol"_a = 1e-11);

    m.def("op_blackbox",
          [](Graph& g, Shape out_shape, const std::vector<int>& in_ids,
             nb::object py_eval, nb::object py_jac, nb::object py_hvp,
             double fd_step, bool fd_hessian, const std::string& name,
             nb::object py_batch_eval, Eigen::MatrixXi jac_sparsity_mask) {
              if (py_eval.is_none()) {
                  throw std::invalid_argument("op_blackbox: eval callback cannot be None.");
              }
              auto eval_holder = std::make_shared<GilSafeCallable>(std::move(py_eval));
              BlackBoxNode::EvalFn eval_cb = [eval_holder](const Eigen::VectorXd& u) -> Eigen::VectorXd {
                  nb::gil_scoped_acquire gil;
                  nb::object res = eval_holder->fn(u);
                  return nb::cast<Eigen::VectorXd>(res);
              };

              BlackBoxNode::JacFn jac_cb = nullptr;
              if (!py_jac.is_none()) {
                  auto jac_holder = std::make_shared<GilSafeCallable>(std::move(py_jac));
                  jac_cb = [jac_holder](const Eigen::VectorXd& u) -> Eigen::MatrixXd {
                      nb::gil_scoped_acquire gil;
                      nb::object res = jac_holder->fn(u);
                      return nb::cast<Eigen::MatrixXd>(res);
                  };
              }

              BlackBoxNode::HvpFn hvp_cb = nullptr;
              if (!py_hvp.is_none()) {
                  auto hvp_holder = std::make_shared<GilSafeCallable>(std::move(py_hvp));
                  hvp_cb = [hvp_holder](const Eigen::VectorXd& u,
                                        const Eigen::VectorXd& y_bar,
                                        const Eigen::VectorXd& u_dot) -> Eigen::VectorXd {
                      nb::gil_scoped_acquire gil;
                      nb::object res = hvp_holder->fn(u, y_bar, u_dot);
                      return nb::cast<Eigen::VectorXd>(res);
                  };
              }

              BlackBoxNode::BatchEvalFn batch_cb = nullptr;
              if (!py_batch_eval.is_none()) {
                  auto batch_holder = std::make_shared<GilSafeCallable>(std::move(py_batch_eval));
                  batch_cb = [batch_holder](const Eigen::MatrixXd& U_batch) -> Eigen::MatrixXd {
                      nb::gil_scoped_acquire gil;
                      nb::object res = batch_holder->fn(U_batch);
                      return nb::cast<Eigen::MatrixXd>(res);
                  };
              }

              return ops::blackbox(g, out_shape, in_ids, std::move(eval_cb),
                                   std::move(jac_cb), std::move(hvp_cb),
                                   fd_step, fd_hessian, name, std::move(batch_cb),
                                   false, std::move(jac_sparsity_mask));
          },
          "graph"_a, "out_shape"_a, "in_ids"_a,
          "eval_fn"_a, "jac_fn"_a = nb::none(), "hvp_fn"_a = nb::none(),
          "fd_step"_a = 1e-7, "fd_hessian"_a = true, "name"_a = "",
          "batch_eval_fn"_a = nb::none(),
          "jac_sparsity_mask"_a = Eigen::MatrixXi());

    // Zero-GIL C-ABI BlackBox from PyCapsule (`FastOptCBlackBox*`)
    m.def("op_c_blackbox_capsule",
          [](Graph& g, Shape out_shape, const std::vector<int>& in_ids,
             nb::object capsule_obj, double fd_step, bool fd_hessian,
             Eigen::MatrixXi jac_sparsity_mask, const std::string& name) {
              PyObject* py_ptr = capsule_obj.ptr();
              if (!PyCapsule_CheckExact(py_ptr)) {
                  throw std::invalid_argument("op_c_blackbox_capsule expects a valid PyCapsule object.");
              }
              const char* cap_name = PyCapsule_GetName(py_ptr);
              void* raw_ptr = PyCapsule_GetPointer(py_ptr, cap_name);
              if (raw_ptr == nullptr) {
                  PyErr_Clear();
                  throw std::invalid_argument("op_c_blackbox_capsule: PyCapsule pointer is NULL.");
              }
              const auto* desc = static_cast<const FastOptCBlackBox*>(raw_ptr);
              if (desc->abi_version != FASTOPT_C_ABI_VERSION) {
                  throw std::invalid_argument(
                      "op_c_blackbox_capsule: incompatible FastOptCBlackBox abi_version.");
              }
              auto holder = std::make_shared<GilSafeCallable>(std::move(capsule_obj));
              return create_c_abi_blackbox_node(
                  g, out_shape, in_ids, desc->eval, desc->jac, desc->hvp, desc->eval_batch,
                  desc->user_data, desc->is_thread_safe != 0, fd_step, fd_hessian,
                  std::move(jac_sparsity_mask), name, std::move(holder));
          },
          "graph"_a, "out_shape"_a, "in_ids"_a, "capsule"_a,
          "fd_step"_a = 1e-7, "fd_hessian"_a = true,
          "jac_sparsity_mask"_a = Eigen::MatrixXi(), "name"_a = "");

    // Zero-GIL C-ABI BlackBox from raw function pointers (ctypes / DLL / C++ uintptr_t)
    m.def("op_c_blackbox_raw",
          [](Graph& g, Shape out_shape, const std::vector<int>& in_ids,
             std::uintptr_t eval_ptr, std::uintptr_t jac_ptr,
             std::uintptr_t hvp_ptr, std::uintptr_t batch_ptr,
             std::uintptr_t user_data_ptr, bool is_thread_safe,
             double fd_step, bool fd_hessian,
             Eigen::MatrixXi jac_sparsity_mask, const std::string& name,
             nb::object keep_alive) {
              auto eval_fn = reinterpret_cast<FastOptCEvalFn>(eval_ptr);
              auto jac_fn = reinterpret_cast<FastOptCJacFn>(jac_ptr);
              auto hvp_fn = reinterpret_cast<FastOptCHvpFn>(hvp_ptr);
              auto batch_fn = reinterpret_cast<FastOptCBatchEvalFn>(batch_ptr);
              void* user_data = reinterpret_cast<void*>(user_data_ptr);
              std::shared_ptr<GilSafeCallable> holder = nullptr;
              if (!keep_alive.is_none()) {
                  holder = std::make_shared<GilSafeCallable>(std::move(keep_alive));
              }
              return create_c_abi_blackbox_node(
                  g, out_shape, in_ids, eval_fn, jac_fn, hvp_fn, batch_fn,
                  user_data, is_thread_safe, fd_step, fd_hessian,
                  std::move(jac_sparsity_mask), name, std::move(holder));
          },
          "graph"_a, "out_shape"_a, "in_ids"_a,
          "eval_ptr"_a, "jac_ptr"_a = 0, "hvp_ptr"_a = 0, "batch_ptr"_a = 0,
          "user_data_ptr"_a = 0, "is_thread_safe"_a = true,
          "fd_step"_a = 1e-7, "fd_hessian"_a = true,
          "jac_sparsity_mask"_a = Eigen::MatrixXi(), "name"_a = "",
          "keep_alive"_a = nb::none());

    m.def("op_map_subgraph", &ops::map_subgraph,
          "graph"_a, "out_shape"_a, "outer_in_ids"_a,
          "body_graph"_a, "body_out_id"_a,
          "inner_var_ids"_a, "in_axes"_a, "out_axis"_a, "num_batches"_a,
          "name"_a = "");

    m.def("op_scan_subgraph", &ops::scan_subgraph,
          "graph"_a, "out_shape"_a, "outer_in_ids"_a,
          "step_graph"_a, "inner_carry_id"_a,
          "inner_seq_ids"_a, "seq_axes"_a, "inner_shared_ids"_a,
          "next_carry_out_id"_a, "step_y_out_id"_a, "num_steps"_a,
          "name"_a = "");

    // 5. High-performance Evaluator with GIL release on compute-intensive methods
    nb::class_<Evaluator>(m, "Evaluator")
        .def(nb::init<std::shared_ptr<Graph>, std::vector<int>>(), "graph"_a, "output_ids"_a)
        .def("num_vars", &Evaluator::num_vars)
        .def("num_outputs", &Evaluator::num_outputs)
        .def("output_size", &Evaluator::output_size, "out_idx"_a)
        .def("output_shape", &Evaluator::output_shape, "out_idx"_a)
        .def("forward_eval_count", &Evaluator::forward_eval_count)
        .def("reset_forward_eval_count", &Evaluator::reset_forward_eval_count)
        .def("invalidate_forward_cache", &Evaluator::invalidate_forward_cache)
        .def("eval", &Evaluator::eval, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_node", &Evaluator::eval_node, "x"_a, "node_id"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_batch", &Evaluator::eval_batch, "X_pop"_a, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("gradient", &Evaluator::gradient, "x"_a, "out_idx"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("jacobian", &Evaluator::jacobian, "x"_a, "out_idx"_a = 0, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("jacobian_sparsity", &Evaluator::jacobian_sparsity, "out_idx"_a = 0)
        .def("num_jacobian_colors", &Evaluator::num_jacobian_colors, "out_idx"_a = 0)
        .def("sparse_jacobian_values", &Evaluator::sparse_jacobian_values,
             "x"_a, "out_idx"_a = 0, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("hessian_vec_prod", &Evaluator::hessian_vec_prod,
             "x"_a, "obj_scale"_a, "constraint_multipliers"_a, "v"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("lagrangian_hessian", &Evaluator::lagrangian_hessian,
             "x"_a, "obj_scale"_a, "constraint_multipliers"_a, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("hessian", &Evaluator::hessian, "x"_a, "out_idx"_a = 0, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("hessian_sparsity", &Evaluator::hessian_sparsity)
        .def("num_hessian_colors", &Evaluator::num_hessian_colors)
        .def("sparse_hessian_values", &Evaluator::sparse_hessian_values,
             "x"_a, "obj_scale"_a, "constraint_multipliers"_a, "num_threads"_a = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("parameter_sensitivities", &Evaluator::parameter_sensitivities,
             "x"_a, "param_ids"_a, "obj_scale"_a, "constraint_multipliers"_a,
             nb::call_guard<nb::gil_scoped_release>());

    // 6. Native C++ IPOPT Solver result and entry point
    nb::class_<IpoptResult>(m, "IpoptResult")
        .def_ro("x", &IpoptResult::x)
        .def_ro("mult_g", &IpoptResult::mult_g)
        .def_ro("mult_x_L", &IpoptResult::mult_x_L)
        .def_ro("mult_x_U", &IpoptResult::mult_x_U)
        .def_ro("obj_val", &IpoptResult::obj_val)
        .def_ro("status", &IpoptResult::status)
        .def_ro("success", &IpoptResult::success)
        .def_ro("iterations", &IpoptResult::iterations)
        .def_ro("elapsed_time", &IpoptResult::elapsed_time)
        .def_ro("message", &IpoptResult::message)
        .def_ro("hist_iter", &IpoptResult::hist_iter)
        .def_ro("hist_obj", &IpoptResult::hist_obj)
        .def_ro("hist_inf_pr", &IpoptResult::hist_inf_pr)
        .def_ro("hist_inf_du", &IpoptResult::hist_inf_du)
        .def_ro("hist_mu", &IpoptResult::hist_mu)
        .def_ro("hist_alpha_pr", &IpoptResult::hist_alpha_pr);

    m.def(
        "solve_ipopt_cpp",
        [](Evaluator& evaluator,
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
           nb::object py_iter_cb) {
            IpoptIterCallbackFn cb_fn = nullptr;
            if (!py_iter_cb.is_none()) {
                auto cb_holder = std::make_shared<GilSafeCallable>(std::move(py_iter_cb));
                cb_fn = [cb_holder](int iter, double obj, double inf_pr, double inf_du, double mu, double alpha_pr) -> bool {
                    nb::gil_scoped_acquire gil;
                    nb::object ret = cb_holder->fn(iter, obj, inf_pr, inf_du, mu, alpha_pr);
                    if (ret.is_none()) return true;
                    return nb::cast<bool>(ret);
                };
            }
            nb::gil_scoped_release release;
            return solve_ipopt_cpp(
                evaluator, x0, x_L, x_U, g_L, g_U, has_eq, has_ineq, use_exact_hessian,
                str_options, num_options, int_options, lib_search_dir, std::move(cb_fn));
        },
        "evaluator"_a, "x0"_a, "x_L"_a, "x_U"_a, "g_L"_a, "g_U"_a,
        "has_eq"_a, "has_ineq"_a, "use_exact_hessian"_a = true,
        "str_options"_a = std::map<std::string, std::string>{},
        "num_options"_a = std::map<std::string, double>{},
        "int_options"_a = std::map<std::string, int>{},
        "lib_search_dir"_a = "",
        "iter_callback"_a = nb::none());
}
