#include "fastopt/ops.hpp"
#include <cmath>
#include <limits>
#include <omp.h>
#include <stdexcept>

namespace fastopt {

namespace {

// Broadcast shape inference helper
Shape broadcast_shape(const Shape& a, const Shape& b) {
    if (a == b) return a;
    if (a.is_scalar()) return b;
    if (b.is_scalar()) return a;
    if ((a.rows == b.rows || a.rows == 1 || b.rows == 1) &&
        (a.cols == b.cols || a.cols == 1 || b.cols == 1)) {
        return Shape(std::max(a.rows, b.rows), std::max(a.cols, b.cols));
    }
    throw std::invalid_argument("Incompatible shapes for broadcasting: " +
                                a.to_string() + " vs " + b.to_string());
}

// Compute 1st and 2nd derivatives of scalar unary operators: f(x), f'(x), f''(x)
inline void unary_scalar_jet(OpType op, double x, double z, double& df, double& d2f) {
    switch (op) {
        case OpType::Neg:
            df = -1.0;
            d2f = 0.0;
            break;
        case OpType::Sin:
            df = std::cos(x);
            d2f = -z;
            break;
        case OpType::Cos:
            df = -std::sin(x);
            d2f = -z;
            break;
        case OpType::Tan: {
            double sec2 = 1.0 + z * z;
            df = sec2;
            d2f = 2.0 * z * sec2;
            break;
        }
        case OpType::Tanh: {
            double sech2 = 1.0 - z * z;
            df = sech2;
            d2f = -2.0 * z * sech2;
            break;
        }
        case OpType::Exp:
            df = z;
            d2f = z;
            break;
        case OpType::Log: {
            double inv = 1.0 / x;
            df = inv;
            d2f = -inv * inv;
            break;
        }
        case OpType::Sqrt: {
            double inv_2sqrt = 0.5 / z;
            df = inv_2sqrt;
            d2f = -inv_2sqrt / (2.0 * x);
            break;
        }
        case OpType::Abs:
            df = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
            d2f = 0.0;
            break;
        case OpType::Square:
            df = 2.0 * x;
            d2f = 2.0;
            break;
        case OpType::Asin: {
            double one_minus_x2 = 1.0 - x * x;
            double inv_sqrt = 1.0 / std::sqrt(one_minus_x2);
            df = inv_sqrt;
            d2f = x * inv_sqrt / one_minus_x2;
            break;
        }
        case OpType::Acos: {
            double one_minus_x2 = 1.0 - x * x;
            double inv_sqrt = 1.0 / std::sqrt(one_minus_x2);
            df = -inv_sqrt;
            d2f = -x * inv_sqrt / one_minus_x2;
            break;
        }
        case OpType::Atan: {
            double inv = 1.0 / (1.0 + x * x);
            df = inv;
            d2f = -2.0 * x * inv * inv;
            break;
        }
        case OpType::Sinh:
            df = std::cosh(x);
            d2f = z;
            break;
        case OpType::Cosh:
            df = std::sinh(x);
            d2f = z;
            break;
        default:
            df = 0.0;
            d2f = 0.0;
            break;
    }
}

// Compute 1st and 2nd partial derivatives of binary scalar operators:
// z = phi(a, b), fa = dz/da, fb = dz/db, faa = d2z/da2, fab = d2z/dadb, fbb = d2z/db2
inline void binary_scalar_jet(OpType op, double a, double b, double z,
                              double& fa, double& fb,
                              double& faa, double& fab, double& fbb) {
    fa = 0.0; fb = 0.0; faa = 0.0; fab = 0.0; fbb = 0.0;
    switch (op) {
        case OpType::Add:
            fa = 1.0;
            fb = 1.0;
            break;
        case OpType::Sub:
            fa = 1.0;
            fb = -1.0;
            break;
        case OpType::Mul:
            fa = b;
            fb = a;
            fab = 1.0;
            break;
        case OpType::Div: {
            double inv_b = 1.0 / b;
            double inv_b2 = inv_b * inv_b;
            fa = inv_b;
            fb = -a * inv_b2;
            fab = -inv_b2;
            fbb = 2.0 * a * inv_b2 * inv_b;
            break;
        }
        case OpType::Pow: {
            if (b == 2.0) {
                fa = 2.0 * a;
                faa = 2.0;
            } else if (b == 1.0) {
                fa = 1.0;
            } else if (b == 0.0) {
                fa = 0.0;
            } else {
                double p1 = std::pow(a, b - 1.0);
                double p2 = std::pow(a, b - 2.0);
                fa = b * p1;
                faa = b * (b - 1.0) * p2;
            }
            if (a > 0.0) {
                double lna = std::log(a);
                fb = z * lna;
                fab = std::pow(a, b - 1.0) * (1.0 + b * lna);
                fbb = z * lna * lna;
            }
            break;
        }
        case OpType::Atan2: {
            double r2 = a * a + b * b;
            if (r2 > 0.0) {
                double inv_r2 = 1.0 / r2;
                double inv_r4 = inv_r2 * inv_r2;
                fa = b * inv_r2;
                fb = -a * inv_r2;
                faa = -2.0 * a * b * inv_r4;
                fab = (a * a - b * b) * inv_r4;
                fbb = 2.0 * a * b * inv_r4;
            }
            break;
        }
        default:
            break;
    }
}

inline double eval_binary_scalar(OpType op, double a, double b) {
    switch (op) {
        case OpType::Add: return a + b;
        case OpType::Sub: return a - b;
        case OpType::Mul: return a * b;
        case OpType::Div: return a / b;
        case OpType::Pow:   return (b == 2.0) ? (a * a) : std::pow(a, b);
        case OpType::Atan2: return std::atan2(a, b);
        case OpType::Lt:  return (a < b) ? 1.0 : 0.0;
        case OpType::Le:  return (a <= b) ? 1.0 : 0.0;
        case OpType::Gt:  return (a > b) ? 1.0 : 0.0;
        case OpType::Ge:  return (a >= b) ? 1.0 : 0.0;
        case OpType::Eq:  return (a == b) ? 1.0 : 0.0;
        default: return 0.0;
    }
}

} // namespace

// ============================================================================
// 1. UnaryOpNode Implementation
// ============================================================================
void UnaryOpNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const Eigen::Index n = size();
    auto x = ws.val_arr(in.offset, n);
    auto z = ws.val_arr(offset, n);

    switch (op) {
        case OpType::Neg:    z = -x; break;
        case OpType::Sin:    z = x.sin(); break;
        case OpType::Cos:    z = x.cos(); break;
        case OpType::Tan:    z = x.tan(); break;
        case OpType::Tanh:   z = x.tanh(); break;
        case OpType::Exp:    z = x.exp(); break;
        case OpType::Log:    z = x.log(); break;
        case OpType::Sqrt:   z = x.sqrt(); break;
        case OpType::Abs:    z = x.abs(); break;
        case OpType::Square: z = x.square(); break;
        case OpType::Asin:   z = x.asin(); break;
        case OpType::Acos:   z = x.acos(); break;
        case OpType::Atan:   z = x.atan(); break;
        case OpType::Sinh:   z = x.sinh(); break;
        case OpType::Cosh:   z = x.cosh(); break;
        default:
            throw std::runtime_error("Unsupported UnaryOpNode in forward().");
    }
}

void UnaryOpNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;

    const Eigen::Index n = size();
    const double* x_ptr = ws.val.data() + in.offset;
    const double* z_ptr = ws.val.data() + offset;
    const double* zb_ptr = ws.bar.data() + offset;
    double* xb_ptr = ws.bar.data() + in.offset;

    for (Eigen::Index i = 0; i < n; ++i) {
        const double zb = zb_ptr[i];
        // Short-circuit zero upstream adjoints to prevent 0.0 * NaN contamination
        if (zb == 0.0) continue;
        double df = 0.0, d2f = 0.0;
        unary_scalar_jet(op, x_ptr[i], z_ptr[i], df, d2f);
        xb_ptr[i] += zb * df;
    }
}

void UnaryOpNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const Eigen::Index n = size();
    if (!in.requires_grad) {
        ws.dot.segment(offset, n).setZero();
        return;
    }

    const double* x_ptr = ws.val.data() + in.offset;
    const double* z_ptr = ws.val.data() + offset;
    const double* xd_ptr = ws.dot.data() + in.offset;
    double* zd_ptr = ws.dot.data() + offset;

    for (Eigen::Index i = 0; i < n; ++i) {
        const double xd = xd_ptr[i];
        if (xd == 0.0 || !std::isfinite(z_ptr[i])) {
            zd_ptr[i] = 0.0;
            continue;
        }
        double df = 0.0, d2f = 0.0;
        unary_scalar_jet(op, x_ptr[i], z_ptr[i], df, d2f);
        zd_ptr[i] = df * xd;
    }
}

void UnaryOpNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;

    const Eigen::Index n = size();
    const double* x_ptr = ws.val.data() + in.offset;
    const double* z_ptr = ws.val.data() + offset;
    const double* zb_ptr = ws.bar.data() + offset;
    const double* xd_ptr = ws.dot.data() + in.offset;
    const double* zdb_ptr = ws.dot_bar.data() + offset;
    double* xdb_ptr = ws.dot_bar.data() + in.offset;

    for (Eigen::Index i = 0; i < n; ++i) {
        const double zb = zb_ptr[i];
        const double zdb = zdb_ptr[i];
        if (zb == 0.0 && zdb == 0.0) continue;
        double df = 0.0, d2f = 0.0;
        unary_scalar_jet(op, x_ptr[i], z_ptr[i], df, d2f);
        double acc = 0.0;
        if (zdb != 0.0) acc += zdb * df;
        if (zb != 0.0 && xd_ptr[i] != 0.0) acc += zb * d2f * xd_ptr[i];
        xdb_ptr[i] += acc;
    }
}

// ============================================================================
// 2. BinaryOpNode Implementation (Supports SIMD Fast-Paths & Full Broadcasting)
// ============================================================================
void BinaryOpNode::forward(Workspace& ws) const {
    const Node& lhs = graph->node(inputs[0]);
    const Node& rhs = graph->node(inputs[1]);
    const Eigen::Index n = size();

    // Fast-path 1: Same shape (contiguous 1D SIMD vectorization)
    if (lhs.shape == rhs.shape) {
        auto z = ws.val_arr(offset, n);
        auto a = ws.val_arr(lhs.offset, n);
        auto b = ws.val_arr(rhs.offset, n);
        switch (op) {
            case OpType::Add: z = a + b; return;
            case OpType::Sub: z = a - b; return;
            case OpType::Mul: z = a * b; return;
            case OpType::Div: z = a / b; return;
            default: break;
        }
    } else if (rhs.shape.is_scalar()) {
        // Fast-path 2: Tensor op Scalar
        auto z = ws.val_arr(offset, n);
        auto a = ws.val_arr(lhs.offset, n);
        const double b0 = ws.val(rhs.offset);
        switch (op) {
            case OpType::Add: z = a + b0; return;
            case OpType::Sub: z = a - b0; return;
            case OpType::Mul: z = a * b0; return;
            case OpType::Div: z = a * (1.0 / b0); return;
            case OpType::Pow:
                if (b0 == 2.0) {
                    z = a.square();
                } else {
                    z = a.pow(b0);
                }
                return;
            default: break;
        }
    } else if (lhs.shape.is_scalar()) {
        // Fast-path 3: Scalar op Tensor
        auto z = ws.val_arr(offset, n);
        const double a0 = ws.val(lhs.offset);
        auto b = ws.val_arr(rhs.offset, n);
        switch (op) {
            case OpType::Add: z = a0 + b; return;
            case OpType::Sub: z = a0 - b; return;
            case OpType::Mul: z = a0 * b; return;
            case OpType::Div: z = a0 / b; return;
            default: break;
        }
    }

    auto Z = ws.val_mat(offset, shape);
    auto A = ws.val_mat(lhs.offset, lhs.shape);
    auto B = ws.val_mat(rhs.offset, rhs.shape);

    for (Eigen::Index c = 0; c < shape.cols; ++c) {
        const Eigen::Index ca = (lhs.shape.cols == 1) ? 0 : c;
        const Eigen::Index cb = (rhs.shape.cols == 1) ? 0 : c;
        for (Eigen::Index r = 0; r < shape.rows; ++r) {
            const Eigen::Index ra = (lhs.shape.rows == 1) ? 0 : r;
            const Eigen::Index rb = (rhs.shape.rows == 1) ? 0 : r;
            Z(r, c) = eval_binary_scalar(op, A(ra, ca), B(rb, cb));
        }
    }
}

void BinaryOpNode::backward(Workspace& ws) const {
    if (op == OpType::Lt || op == OpType::Le || op == OpType::Gt ||
        op == OpType::Ge || op == OpType::Eq) {
        return;
    }
    const Node& lhs = graph->node(inputs[0]);
    const Node& rhs = graph->node(inputs[1]);
    if (!lhs.requires_grad && !rhs.requires_grad) return;
    const Eigen::Index n = size();

    // SIMD fast-paths for Add / Sub / Mul (when finite)
    if (op == OpType::Add) {
        auto zb = ws.bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb;
            if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) += zb;
            return;
        } else if (rhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb;
            if (rhs.requires_grad) ws.bar(rhs.offset) += zb.sum();
            return;
        } else if (lhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.bar(lhs.offset) += zb.sum();
            if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) += zb;
            return;
        }
    } else if (op == OpType::Sub) {
        auto zb = ws.bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb;
            if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) -= zb;
            return;
        } else if (rhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb;
            if (rhs.requires_grad) ws.bar(rhs.offset) -= zb.sum();
            return;
        } else if (lhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.bar(lhs.offset) += zb.sum();
            if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) -= zb;
            return;
        }
    } else if (op == OpType::Mul) {
        auto zb = ws.bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            auto a = ws.val_arr(lhs.offset, n);
            auto b = ws.val_arr(rhs.offset, n);
            if (a.allFinite() && b.allFinite()) {
                if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb * b;
                if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) += zb * a;
                return;
            }
        } else if (rhs.shape.is_scalar()) {
            auto a = ws.val_arr(lhs.offset, n);
            const double b0 = ws.val(rhs.offset);
            if (std::isfinite(b0) && a.allFinite()) {
                if (lhs.requires_grad) ws.bar_arr(lhs.offset, n) += zb * b0;
                if (rhs.requires_grad) ws.bar(rhs.offset) += (zb * a).sum();
                return;
            }
        } else if (lhs.shape.is_scalar()) {
            const double a0 = ws.val(lhs.offset);
            auto b = ws.val_arr(rhs.offset, n);
            if (std::isfinite(a0) && b.allFinite()) {
                if (lhs.requires_grad) ws.bar(lhs.offset) += (zb * b).sum();
                if (rhs.requires_grad) ws.bar_arr(rhs.offset, n) += zb * a0;
                return;
            }
        }
    }

    auto Z = ws.val_mat(offset, shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto A = ws.val_mat(lhs.offset, lhs.shape);
    auto B = ws.val_mat(rhs.offset, rhs.shape);
    auto Ab = ws.bar_mat(lhs.offset, lhs.shape);
    auto Bb = ws.bar_mat(rhs.offset, rhs.shape);

    for (Eigen::Index c = 0; c < shape.cols; ++c) {
        const Eigen::Index ca = (lhs.shape.cols == 1) ? 0 : c;
        const Eigen::Index cb = (rhs.shape.cols == 1) ? 0 : c;
        for (Eigen::Index r = 0; r < shape.rows; ++r) {
            const double zb = Zb(r, c);
            if (zb == 0.0) continue;
            const Eigen::Index ra = (lhs.shape.rows == 1) ? 0 : r;
            const Eigen::Index rb = (rhs.shape.rows == 1) ? 0 : r;
            double fa, fb, faa, fab, fbb;
            binary_scalar_jet(op, A(ra, ca), B(rb, cb), Z(r, c), fa, fb, faa, fab, fbb);
            if (lhs.requires_grad) Ab(ra, ca) += zb * fa;
            if (rhs.requires_grad) Bb(rb, cb) += zb * fb;
        }
    }
}

void BinaryOpNode::forward_tangent(Workspace& ws) const {
    auto Zd = ws.dot_mat(offset, shape);
    if (op == OpType::Lt || op == OpType::Le || op == OpType::Gt ||
        op == OpType::Ge || op == OpType::Eq) {
        Zd.setZero();
        return;
    }
    const Node& lhs = graph->node(inputs[0]);
    const Node& rhs = graph->node(inputs[1]);
    if (!lhs.requires_grad && !rhs.requires_grad) {
        Zd.setZero();
        return;
    }
    const Eigen::Index n = size();

    if (op == OpType::Add) {
        auto zd = ws.dot_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad && rhs.requires_grad) {
                zd = ws.dot_arr(lhs.offset, n) + ws.dot_arr(rhs.offset, n);
            } else if (lhs.requires_grad) {
                zd = ws.dot_arr(lhs.offset, n);
            } else {
                zd = ws.dot_arr(rhs.offset, n);
            }
            return;
        }
    } else if (op == OpType::Sub) {
        auto zd = ws.dot_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad && rhs.requires_grad) {
                zd = ws.dot_arr(lhs.offset, n) - ws.dot_arr(rhs.offset, n);
            } else if (lhs.requires_grad) {
                zd = ws.dot_arr(lhs.offset, n);
            } else {
                zd = -ws.dot_arr(rhs.offset, n);
            }
            return;
        }
    } else if (op == OpType::Mul && ws.val_arr(offset, n).allFinite()) {
        auto zd = ws.dot_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            auto a = ws.val_arr(lhs.offset, n);
            auto b = ws.val_arr(rhs.offset, n);
            if (lhs.requires_grad && rhs.requires_grad) {
                zd = b * ws.dot_arr(lhs.offset, n) + a * ws.dot_arr(rhs.offset, n);
            } else if (lhs.requires_grad) {
                zd = b * ws.dot_arr(lhs.offset, n);
            } else {
                zd = a * ws.dot_arr(rhs.offset, n);
            }
            return;
        } else if (rhs.shape.is_scalar()) {
            const double b0 = ws.val(rhs.offset);
            const double bd0 = rhs.requires_grad ? ws.dot(rhs.offset) : 0.0;
            if (lhs.requires_grad && rhs.requires_grad) {
                zd = b0 * ws.dot_arr(lhs.offset, n) + ws.val_arr(lhs.offset, n) * bd0;
            } else if (lhs.requires_grad) {
                zd = b0 * ws.dot_arr(lhs.offset, n);
            } else {
                zd = ws.val_arr(lhs.offset, n) * bd0;
            }
            return;
        } else if (lhs.shape.is_scalar()) {
            const double a0 = ws.val(lhs.offset);
            const double ad0 = lhs.requires_grad ? ws.dot(lhs.offset) : 0.0;
            if (lhs.requires_grad && rhs.requires_grad) {
                zd = ws.val_arr(rhs.offset, n) * ad0 + a0 * ws.dot_arr(rhs.offset, n);
            } else if (lhs.requires_grad) {
                zd = ws.val_arr(rhs.offset, n) * ad0;
            } else {
                zd = a0 * ws.dot_arr(rhs.offset, n);
            }
            return;
        }
    }

    auto Z = ws.val_mat(offset, shape);
    auto A = ws.val_mat(lhs.offset, lhs.shape);
    auto B = ws.val_mat(rhs.offset, rhs.shape);
    auto Ad = ws.dot_mat(lhs.offset, lhs.shape);
    auto Bd = ws.dot_mat(rhs.offset, rhs.shape);

    for (Eigen::Index c = 0; c < shape.cols; ++c) {
        const Eigen::Index ca = (lhs.shape.cols == 1) ? 0 : c;
        const Eigen::Index cb = (rhs.shape.cols == 1) ? 0 : c;
        for (Eigen::Index r = 0; r < shape.rows; ++r) {
            const Eigen::Index ra = (lhs.shape.rows == 1) ? 0 : r;
            const Eigen::Index rb = (rhs.shape.rows == 1) ? 0 : r;
            const double ad = lhs.requires_grad ? Ad(ra, ca) : 0.0;
            const double bd = rhs.requires_grad ? Bd(rb, cb) : 0.0;
            if ((ad == 0.0 && bd == 0.0) || !std::isfinite(Z(r, c))) {
                Zd(r, c) = 0.0;
                continue;
            }
            double fa, fb, faa, fab, fbb;
            binary_scalar_jet(op, A(ra, ca), B(rb, cb), Z(r, c), fa, fb, faa, fab, fbb);
            Zd(r, c) = fa * ad + fb * bd;
        }
    }
}

void BinaryOpNode::backward_hessian(Workspace& ws) const {
    if (op == OpType::Lt || op == OpType::Le || op == OpType::Gt ||
        op == OpType::Ge || op == OpType::Eq) {
        return;
    }
    const Node& lhs = graph->node(inputs[0]);
    const Node& rhs = graph->node(inputs[1]);
    if (!lhs.requires_grad && !rhs.requires_grad) return;
    const Eigen::Index n = size();

    if (op == OpType::Add) {
        auto zdb = ws.dot_bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad) ws.dot_bar_arr(lhs.offset, n) += zdb;
            if (rhs.requires_grad) ws.dot_bar_arr(rhs.offset, n) += zdb;
            return;
        } else if (rhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.dot_bar_arr(lhs.offset, n) += zdb;
            if (rhs.requires_grad) ws.dot_bar(rhs.offset) += zdb.sum();
            return;
        } else if (lhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.dot_bar(lhs.offset) += zdb.sum();
            if (rhs.requires_grad) ws.dot_bar_arr(rhs.offset, n) += zdb;
            return;
        }
    } else if (op == OpType::Sub) {
        auto zdb = ws.dot_bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            if (lhs.requires_grad) ws.dot_bar_arr(lhs.offset, n) += zdb;
            if (rhs.requires_grad) ws.dot_bar_arr(rhs.offset, n) -= zdb;
            return;
        } else if (rhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.dot_bar_arr(lhs.offset, n) += zdb;
            if (rhs.requires_grad) ws.dot_bar(rhs.offset) -= zdb.sum();
            return;
        } else if (lhs.shape.is_scalar()) {
            if (lhs.requires_grad) ws.dot_bar(lhs.offset) += zdb.sum();
            if (rhs.requires_grad) ws.dot_bar_arr(rhs.offset, n) -= zdb;
            return;
        }
    } else if (op == OpType::Mul && ws.val_arr(offset, n).allFinite()) {
        auto zb = ws.bar_arr(offset, n);
        auto zdb = ws.dot_bar_arr(offset, n);
        if (lhs.shape == rhs.shape) {
            auto a = ws.val_arr(lhs.offset, n);
            auto b = ws.val_arr(rhs.offset, n);
            if (lhs.requires_grad) {
                if (rhs.requires_grad) {
                    ws.dot_bar_arr(lhs.offset, n) += zdb * b + zb * ws.dot_arr(rhs.offset, n);
                } else {
                    ws.dot_bar_arr(lhs.offset, n) += zdb * b;
                }
            }
            if (rhs.requires_grad) {
                if (lhs.requires_grad) {
                    ws.dot_bar_arr(rhs.offset, n) += zdb * a + zb * ws.dot_arr(lhs.offset, n);
                } else {
                    ws.dot_bar_arr(rhs.offset, n) += zdb * a;
                }
            }
            return;
        }
    }

    auto Z = ws.val_mat(offset, shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);
    auto A = ws.val_mat(lhs.offset, lhs.shape);
    auto B = ws.val_mat(rhs.offset, rhs.shape);
    auto Ad = ws.dot_mat(lhs.offset, lhs.shape);
    auto Bd = ws.dot_mat(rhs.offset, rhs.shape);
    auto Adb = ws.dot_bar_mat(lhs.offset, lhs.shape);
    auto Bdb = ws.dot_bar_mat(rhs.offset, rhs.shape);

    for (Eigen::Index c = 0; c < shape.cols; ++c) {
        const Eigen::Index ca = (lhs.shape.cols == 1) ? 0 : c;
        const Eigen::Index cb = (rhs.shape.cols == 1) ? 0 : c;
        for (Eigen::Index r = 0; r < shape.rows; ++r) {
            const double zb = Zb(r, c);
            const double zdb = Zdb(r, c);
            if (zb == 0.0 && zdb == 0.0) continue;
            const Eigen::Index ra = (lhs.shape.rows == 1) ? 0 : r;
            const Eigen::Index rb = (rhs.shape.rows == 1) ? 0 : r;
            double fa, fb, faa, fab, fbb;
            binary_scalar_jet(op, A(ra, ca), B(rb, cb), Z(r, c), fa, fb, faa, fab, fbb);
            const double ad = lhs.requires_grad ? Ad(ra, ca) : 0.0;
            const double bd = rhs.requires_grad ? Bd(rb, cb) : 0.0;
            if (lhs.requires_grad) {
                double acc_a = 0.0;
                if (zdb != 0.0) acc_a += zdb * fa;
                if (zb != 0.0) {
                    if (ad != 0.0) acc_a += zb * faa * ad;
                    if (bd != 0.0) acc_a += zb * fab * bd;
                }
                Adb(ra, ca) += acc_a;
            }
            if (rhs.requires_grad) {
                double acc_b = 0.0;
                if (zdb != 0.0) acc_b += zdb * fb;
                if (zb != 0.0) {
                    if (ad != 0.0) acc_b += zb * fab * ad;
                    if (bd != 0.0) acc_b += zb * fbb * bd;
                }
                Bdb(rb, cb) += acc_b;
            }
        }
    }
}

// ============================================================================
// 3. MatMulNode Implementation (C = A * B)
// ============================================================================
void MatMulNode::forward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_mat(a.offset, a.shape);
    auto B = ws.val_mat(b.offset, b.shape);
    auto C = ws.val_mat(offset, shape);
    C.noalias() = A * B;
}

void MatMulNode::backward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_mat(a.offset, a.shape);
    auto B = ws.val_mat(b.offset, b.shape);
    auto Cb = ws.bar_mat(offset, shape);

    if (a.requires_grad) {
        auto Ab = ws.bar_mat(a.offset, a.shape);
        Ab.noalias() += Cb * B.transpose();
    }
    if (b.requires_grad) {
        auto Bb = ws.bar_mat(b.offset, b.shape);
        Bb.noalias() += A.transpose() * Cb;
    }
}

void MatMulNode::forward_tangent(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_mat(a.offset, a.shape);
    auto B = ws.val_mat(b.offset, b.shape);
    auto Ad = ws.dot_mat(a.offset, a.shape);
    auto Bd = ws.dot_mat(b.offset, b.shape);
    auto Cd = ws.dot_mat(offset, shape);

    Cd.setZero();
    if (a.requires_grad) Cd.noalias() += Ad * B;
    if (b.requires_grad) Cd.noalias() += A * Bd;
}

void MatMulNode::backward_hessian(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_mat(a.offset, a.shape);
    auto B = ws.val_mat(b.offset, b.shape);
    auto Ad = ws.dot_mat(a.offset, a.shape);
    auto Bd = ws.dot_mat(b.offset, b.shape);
    auto Cb = ws.bar_mat(offset, shape);
    auto Cdb = ws.dot_bar_mat(offset, shape);

    if (a.requires_grad) {
        auto Adb = ws.dot_bar_mat(a.offset, a.shape);
        Adb.noalias() += Cdb * B.transpose();
        if (b.requires_grad) {
            Adb.noalias() += Cb * Bd.transpose();
        }
    }
    if (b.requires_grad) {
        auto Bdb = ws.dot_bar_mat(b.offset, b.shape);
        Bdb.noalias() += A.transpose() * Cdb;
        if (a.requires_grad) {
            Bdb.noalias() += Ad.transpose() * Cb;
        }
    }
}

// ============================================================================
// 3B. LinearSolveNode Implementation: Y = A^{-1} B (Exact 1st & 2nd Order AD)
// ============================================================================
void LinearSolveNode::forward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_mat(a.offset, a.shape);
    auto B = ws.val_mat(b.offset, b.shape);
    auto Y = ws.val_mat(offset, shape);
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Y = lu.solve(B);
}

void LinearSolveNode::backward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    if (!a.requires_grad && !b.requires_grad) return;

    auto Yb = ws.bar_mat(offset, shape);
    if (Yb.squaredNorm() == 0.0) return;

    auto A = ws.val_mat(a.offset, a.shape);
    auto Y = ws.val_mat(offset, shape);

    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Eigen::MatrixXd V = lu.transpose().solve(Yb); // V = A^{-T} * Y_bar

    if (b.requires_grad) {
        ws.bar_mat(b.offset, b.shape).noalias() += V;
    }
    if (a.requires_grad) {
        ws.bar_mat(a.offset, a.shape).noalias() -= V * Y.transpose();
    }
}

void LinearSolveNode::forward_tangent(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto Yd = ws.dot_mat(offset, shape);
    if (!a.requires_grad && !b.requires_grad) {
        Yd.setZero();
        return;
    }

    auto A = ws.val_mat(a.offset, a.shape);
    auto Y = ws.val_mat(offset, shape);

    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(shape.rows, shape.cols);
    if (b.requires_grad) {
        rhs += ws.dot_mat(b.offset, b.shape);
    }
    if (a.requires_grad) {
        rhs.noalias() -= ws.dot_mat(a.offset, a.shape) * Y;
    }
    if (rhs.squaredNorm() == 0.0) {
        Yd.setZero();
        return;
    }

    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Yd = lu.solve(rhs);
}

void LinearSolveNode::backward_hessian(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    if (!a.requires_grad && !b.requires_grad) return;

    auto Yb = ws.bar_mat(offset, shape);
    auto Ydb = ws.dot_bar_mat(offset, shape);
    if (Yb.squaredNorm() == 0.0 && Ydb.squaredNorm() == 0.0) return;

    auto A = ws.val_mat(a.offset, a.shape);
    auto Y = ws.val_mat(offset, shape);
    auto Yd = ws.dot_mat(offset, shape);

    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Eigen::MatrixXd V = lu.transpose().solve(Yb); // V = A^{-T} * Y_bar

    Eigen::MatrixXd rhs_vd = Ydb;
    if (a.requires_grad) {
        auto Ad = ws.dot_mat(a.offset, a.shape);
        rhs_vd.noalias() -= Ad.transpose() * V;
    }
    Eigen::MatrixXd Vd = lu.transpose().solve(rhs_vd); // V_dot = A^{-T} * (Y_dot_bar - A_dot^T * V)

    if (b.requires_grad) {
        ws.dot_bar_mat(b.offset, b.shape).noalias() += Vd;
    }
    if (a.requires_grad) {
        auto Adb = ws.dot_bar_mat(a.offset, a.shape);
        Adb.noalias() -= Vd * Y.transpose();
        Adb.noalias() -= V * Yd.transpose();
    }
}

// ============================================================================
// 4. DotNode Implementation: z = sum(A .* B)
// ============================================================================
void DotNode::forward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    auto A = ws.val_arr(a.offset, a.size());
    auto B = ws.val_arr(b.offset, b.size());
    ws.val(offset) = (A * B).sum();
}

void DotNode::backward(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    const double zb = ws.bar(offset);
    if (zb == 0.0) return;
    if (a.requires_grad) {
        ws.bar_arr(a.offset, a.size()) += zb * ws.val_arr(b.offset, b.size());
    }
    if (b.requires_grad) {
        ws.bar_arr(b.offset, b.size()) += zb * ws.val_arr(a.offset, a.size());
    }
}

void DotNode::forward_tangent(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    double zd = 0.0;
    if (a.requires_grad) {
        zd += (ws.dot_arr(a.offset, a.size()) * ws.val_arr(b.offset, b.size())).sum();
    }
    if (b.requires_grad) {
        zd += (ws.val_arr(a.offset, a.size()) * ws.dot_arr(b.offset, b.size())).sum();
    }
    ws.dot(offset) = zd;
}

void DotNode::backward_hessian(Workspace& ws) const {
    const Node& a = graph->node(inputs[0]);
    const Node& b = graph->node(inputs[1]);
    const double zb = ws.bar(offset);
    const double zdb = ws.dot_bar(offset);
    if (zb == 0.0 && zdb == 0.0) return;

    if (a.requires_grad) {
        ws.dot_bar_arr(a.offset, a.size()) += zdb * ws.val_arr(b.offset, b.size());
        if (b.requires_grad) {
            ws.dot_bar_arr(a.offset, a.size()) += zb * ws.dot_arr(b.offset, b.size());
        }
    }
    if (b.requires_grad) {
        ws.dot_bar_arr(b.offset, b.size()) += zdb * ws.val_arr(a.offset, a.size());
        if (a.requires_grad) {
            ws.dot_bar_arr(b.offset, b.size()) += zb * ws.dot_arr(a.offset, a.size());
        }
    }
}

// ============================================================================
// 5. SumNode Implementation
// ============================================================================
void SumNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    auto X = ws.val_mat(in.offset, in.shape);
    auto Z = ws.val_mat(offset, shape);
    if (axis == -1) {
        Z(0, 0) = X.sum();
    } else if (axis == 0) {
        Z = X.colwise().sum();
    } else if (axis == 1) {
        Z = X.rowwise().sum();
    }
}

void SumNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    auto Xb = ws.bar_mat(in.offset, in.shape);
    auto Zb = ws.bar_mat(offset, shape);
    if (axis == -1) {
        Xb.array() += Zb(0, 0);
    } else if (axis == 0) {
        Xb.rowwise() += Zb.row(0);
    } else if (axis == 1) {
        Xb.colwise() += Zb.col(0);
    }
}

void SumNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    auto Zd = ws.dot_mat(offset, shape);
    if (!in.requires_grad) {
        Zd.setZero();
        return;
    }
    auto Xd = ws.dot_mat(in.offset, in.shape);
    if (axis == -1) {
        Zd(0, 0) = Xd.sum();
    } else if (axis == 0) {
        Zd = Xd.colwise().sum();
    } else if (axis == 1) {
        Zd = Xd.rowwise().sum();
    }
}

void SumNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    auto Xdb = ws.dot_bar_mat(in.offset, in.shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);
    if (axis == -1) {
        Xdb.array() += Zdb(0, 0);
    } else if (axis == 0) {
        Xdb.rowwise() += Zdb.row(0);
    } else if (axis == 1) {
        Xdb.colwise() += Zdb.col(0);
    }
}

// ============================================================================
// 5b. KSLogSumExpNode Implementation (Overflow-Safe KS / LogSumExp Reduction)
// ============================================================================
namespace {

inline double ks_slice_forward(const double* x_ptr, Eigen::Index n, Eigen::Index stride,
                               double rho, bool subtract_log_n) {
    if (n <= 0) return 0.0;
    double max_s = -std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < n; ++i) {
        const double s = rho * x_ptr[i * stride];
        if (s > max_s) max_s = s;
    }
    double sum_exp = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        sum_exp += std::exp(rho * x_ptr[i * stride] - max_s);
    }
    double lse = max_s + std::log(sum_exp);
    if (subtract_log_n) {
        lse -= std::log(static_cast<double>(n));
    }
    return lse / rho;
}

inline void ks_slice_backward(const double* x_ptr, double* xb_ptr,
                              Eigen::Index n, Eigen::Index stride,
                              double rho, double zb) {
    if (zb == 0.0 || n <= 0) return;
    double max_s = -std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < n; ++i) {
        const double s = rho * x_ptr[i * stride];
        if (s > max_s) max_s = s;
    }
    double sum_exp = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        sum_exp += std::exp(rho * x_ptr[i * stride] - max_s);
    }
    const double inv_sum = 1.0 / sum_exp;
    for (Eigen::Index i = 0; i < n; ++i) {
        const double p_i = std::exp(rho * x_ptr[i * stride] - max_s) * inv_sum;
        xb_ptr[i * stride] += zb * p_i;
    }
}

inline double ks_slice_forward_tangent(const double* x_ptr, const double* xd_ptr,
                                       Eigen::Index n, Eigen::Index stride, double rho) {
    if (n <= 0) return 0.0;
    double max_s = -std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < n; ++i) {
        const double s = rho * x_ptr[i * stride];
        if (s > max_s) max_s = s;
    }
    double sum_exp = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        sum_exp += std::exp(rho * x_ptr[i * stride] - max_s);
    }
    const double inv_sum = 1.0 / sum_exp;
    double zd = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const double p_i = std::exp(rho * x_ptr[i * stride] - max_s) * inv_sum;
        zd += p_i * xd_ptr[i * stride];
    }
    return zd;
}

inline void ks_slice_backward_hessian(const double* x_ptr, const double* xd_ptr, double* xdb_ptr,
                                      Eigen::Index n, Eigen::Index stride,
                                      double rho, double zb, double zdb) {
    if ((zb == 0.0 && zdb == 0.0) || n <= 0) return;
    double max_s = -std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < n; ++i) {
        const double s = rho * x_ptr[i * stride];
        if (s > max_s) max_s = s;
    }
    double sum_exp = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        sum_exp += std::exp(rho * x_ptr[i * stride] - max_s);
    }
    const double inv_sum = 1.0 / sum_exp;
    double zd = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const double p_i = std::exp(rho * x_ptr[i * stride] - max_s) * inv_sum;
        zd += p_i * xd_ptr[i * stride];
    }
    for (Eigen::Index i = 0; i < n; ++i) {
        const double p_i = std::exp(rho * x_ptr[i * stride] - max_s) * inv_sum;
        xdb_ptr[i * stride] += zdb * p_i + zb * rho * p_i * (xd_ptr[i * stride] - zd);
    }
}

} // namespace

void KSLogSumExpNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const double* x_ptr = ws.val.data() + in.offset;
    auto Z = ws.val_mat(offset, shape);
    const Eigen::Index R = in.shape.rows;
    const Eigen::Index C = in.shape.cols;

    if (axis == -1) {
        Z(0, 0) = ks_slice_forward(x_ptr, in.size(), 1, rho, subtract_log_n);
    } else if (axis == 0) {
        for (Eigen::Index c = 0; c < C; ++c) {
            Z(0, c) = ks_slice_forward(x_ptr + c * R, R, 1, rho, subtract_log_n);
        }
    } else if (axis == 1) {
        for (Eigen::Index r = 0; r < R; ++r) {
            Z(r, 0) = ks_slice_forward(x_ptr + r, C, R, rho, subtract_log_n);
        }
    }
}

void KSLogSumExpNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double* x_ptr = ws.val.data() + in.offset;
    double* xb_ptr = ws.bar.data() + in.offset;
    auto Zb = ws.bar_mat(offset, shape);
    const Eigen::Index R = in.shape.rows;
    const Eigen::Index C = in.shape.cols;

    if (axis == -1) {
        ks_slice_backward(x_ptr, xb_ptr, in.size(), 1, rho, Zb(0, 0));
    } else if (axis == 0) {
        for (Eigen::Index c = 0; c < C; ++c) {
            ks_slice_backward(x_ptr + c * R, xb_ptr + c * R, R, 1, rho, Zb(0, c));
        }
    } else if (axis == 1) {
        for (Eigen::Index r = 0; r < R; ++r) {
            ks_slice_backward(x_ptr + r, xb_ptr + r, C, R, rho, Zb(r, 0));
        }
    }
}

void KSLogSumExpNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    auto Zd = ws.dot_mat(offset, shape);
    if (!in.requires_grad) {
        Zd.setZero();
        return;
    }
    const double* x_ptr = ws.val.data() + in.offset;
    const double* xd_ptr = ws.dot.data() + in.offset;
    const Eigen::Index R = in.shape.rows;
    const Eigen::Index C = in.shape.cols;

    if (axis == -1) {
        Zd(0, 0) = ks_slice_forward_tangent(x_ptr, xd_ptr, in.size(), 1, rho);
    } else if (axis == 0) {
        for (Eigen::Index c = 0; c < C; ++c) {
            Zd(0, c) = ks_slice_forward_tangent(x_ptr + c * R, xd_ptr + c * R, R, 1, rho);
        }
    } else if (axis == 1) {
        for (Eigen::Index r = 0; r < R; ++r) {
            Zd(r, 0) = ks_slice_forward_tangent(x_ptr + r, xd_ptr + r, C, R, rho);
        }
    }
}

void KSLogSumExpNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double* x_ptr = ws.val.data() + in.offset;
    const double* xd_ptr = ws.dot.data() + in.offset;
    double* xdb_ptr = ws.dot_bar.data() + in.offset;
    auto Zb = ws.bar_mat(offset, shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);
    const Eigen::Index R = in.shape.rows;
    const Eigen::Index C = in.shape.cols;

    if (axis == -1) {
        ks_slice_backward_hessian(x_ptr, xd_ptr, xdb_ptr, in.size(), 1, rho, Zb(0, 0), Zdb(0, 0));
    } else if (axis == 0) {
        for (Eigen::Index c = 0; c < C; ++c) {
            ks_slice_backward_hessian(x_ptr + c * R, xd_ptr + c * R, xdb_ptr + c * R, R, 1,
                                      rho, Zb(0, c), Zdb(0, c));
        }
    } else if (axis == 1) {
        for (Eigen::Index r = 0; r < R; ++r) {
            ks_slice_backward_hessian(x_ptr + r, xd_ptr + r, xdb_ptr + r, C, R,
                                      rho, Zb(r, 0), Zdb(r, 0));
        }
    }
}

// ============================================================================
// 6. TransposeNode Implementation
// ============================================================================
void TransposeNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    ws.val_mat(offset, shape) = ws.val_mat(in.offset, in.shape).transpose();
}

void TransposeNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.bar_mat(in.offset, in.shape) += ws.bar_mat(offset, shape).transpose();
}

void TransposeNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) {
        ws.dot_mat(offset, shape).setZero();
        return;
    }
    ws.dot_mat(offset, shape) = ws.dot_mat(in.offset, in.shape).transpose();
}

void TransposeNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.dot_bar_mat(in.offset, in.shape) += ws.dot_bar_mat(offset, shape).transpose();
}

// ============================================================================
// 7. SliceNode Implementation
// ============================================================================
void SliceNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    ws.val_mat(offset, shape) =
        ws.val_mat(in.offset, in.shape).block(row_start, col_start, shape.rows, shape.cols);
}

void SliceNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.bar_mat(in.offset, in.shape).block(row_start, col_start, shape.rows, shape.cols) +=
        ws.bar_mat(offset, shape);
}

void SliceNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) {
        ws.dot_mat(offset, shape).setZero();
        return;
    }
    ws.dot_mat(offset, shape) =
        ws.dot_mat(in.offset, in.shape).block(row_start, col_start, shape.rows, shape.cols);
}

void SliceNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.dot_bar_mat(in.offset, in.shape).block(row_start, col_start, shape.rows, shape.cols) +=
        ws.dot_bar_mat(offset, shape);
}

// ============================================================================
// 8. ConcatNode Implementation
// ============================================================================
void ConcatNode::forward(Workspace& ws) const {
    auto Z = ws.val_mat(offset, shape);
    Eigen::Index pos = 0;
    for (int id : inputs) {
        const Node& in = graph->node(id);
        auto X = ws.val_mat(in.offset, in.shape);
        if (axis == 0) {
            Z.middleRows(pos, in.shape.rows) = X;
            pos += in.shape.rows;
        } else {
            Z.middleCols(pos, in.shape.cols) = X;
            pos += in.shape.cols;
        }
    }
}

void ConcatNode::backward(Workspace& ws) const {
    auto Zb = ws.bar_mat(offset, shape);
    Eigen::Index pos = 0;
    for (int id : inputs) {
        const Node& in = graph->node(id);
        if (in.requires_grad) {
            auto Xb = ws.bar_mat(in.offset, in.shape);
            if (axis == 0) {
                Xb += Zb.middleRows(pos, in.shape.rows);
            } else {
                Xb += Zb.middleCols(pos, in.shape.cols);
            }
        }
        pos += (axis == 0) ? in.shape.rows : in.shape.cols;
    }
}

void ConcatNode::forward_tangent(Workspace& ws) const {
    auto Zd = ws.dot_mat(offset, shape);
    Eigen::Index pos = 0;
    for (int id : inputs) {
        const Node& in = graph->node(id);
        if (in.requires_grad) {
            auto Xd = ws.dot_mat(in.offset, in.shape);
            if (axis == 0) {
                Zd.middleRows(pos, in.shape.rows) = Xd;
            } else {
                Zd.middleCols(pos, in.shape.cols) = Xd;
            }
        } else {
            if (axis == 0) {
                Zd.middleRows(pos, in.shape.rows).setZero();
            } else {
                Zd.middleCols(pos, in.shape.cols).setZero();
            }
        }
        pos += (axis == 0) ? in.shape.rows : in.shape.cols;
    }
}

void ConcatNode::backward_hessian(Workspace& ws) const {
    auto Zdb = ws.dot_bar_mat(offset, shape);
    Eigen::Index pos = 0;
    for (int id : inputs) {
        const Node& in = graph->node(id);
        if (in.requires_grad) {
            auto Xdb = ws.dot_bar_mat(in.offset, in.shape);
            if (axis == 0) {
                Xdb += Zdb.middleRows(pos, in.shape.rows);
            } else {
                Xdb += Zdb.middleCols(pos, in.shape.cols);
            }
        }
        pos += (axis == 0) ? in.shape.rows : in.shape.cols;
    }
}

// ============================================================================
// 9. ReshapeNode Implementation
// ============================================================================
void ReshapeNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    ws.val.segment(offset, size()) = ws.val.segment(in.offset, size());
}

void ReshapeNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.bar.segment(in.offset, size()) += ws.bar.segment(offset, size());
}

void ReshapeNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) {
        ws.dot.segment(offset, size()).setZero();
        return;
    }
    ws.dot.segment(offset, size()) = ws.dot.segment(in.offset, size());
}

void ReshapeNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    ws.dot_bar.segment(in.offset, size()) += ws.dot_bar.segment(offset, size());
}

// ============================================================================
// 10. WhereNode Implementation (Strict Branch Selection for NaN-Safety)
// ============================================================================
void WhereNode::forward(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);

    auto C = ws.val_mat(c_node.offset, c_node.shape);
    auto T = ws.val_mat(t_node.offset, t_node.shape);
    auto F = ws.val_mat(f_node.offset, f_node.shape);
    auto Z = ws.val_mat(offset, shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cc = (c_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rc = (c_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;
            Z(row, col) = (C(rc, cc) > 0.5) ? T(rt, ct) : F(rf, cf);
        }
    }
}

void WhereNode::backward(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);
    if (!t_node.requires_grad && !f_node.requires_grad) return;

    auto C = ws.val_mat(c_node.offset, c_node.shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Tb = ws.bar_mat(t_node.offset, t_node.shape);
    auto Fb = ws.bar_mat(f_node.offset, f_node.shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cc = (c_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rc = (c_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;
            if (C(rc, cc) > 0.5) {
                if (t_node.requires_grad) Tb(rt, ct) += Zb(row, col);
            } else {
                if (f_node.requires_grad) Fb(rf, cf) += Zb(row, col);
            }
        }
    }
}

void WhereNode::forward_tangent(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);

    auto C = ws.val_mat(c_node.offset, c_node.shape);
    auto Td = ws.dot_mat(t_node.offset, t_node.shape);
    auto Fd = ws.dot_mat(f_node.offset, f_node.shape);
    auto Zd = ws.dot_mat(offset, shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cc = (c_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rc = (c_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;
            if (C(rc, cc) > 0.5) {
                Zd(row, col) = t_node.requires_grad ? Td(rt, ct) : 0.0;
            } else {
                Zd(row, col) = f_node.requires_grad ? Fd(rf, cf) : 0.0;
            }
        }
    }
}

void WhereNode::backward_hessian(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);
    if (!t_node.requires_grad && !f_node.requires_grad) return;

    auto C = ws.val_mat(c_node.offset, c_node.shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);
    auto Tdb = ws.dot_bar_mat(t_node.offset, t_node.shape);
    auto Fdb = ws.dot_bar_mat(f_node.offset, f_node.shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cc = (c_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rc = (c_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;
            if (C(rc, cc) > 0.5) {
                if (t_node.requires_grad) Tdb(rt, ct) += Zdb(row, col);
            } else {
                if (f_node.requires_grad) Fdb(rf, cf) += Zdb(row, col);
            }
        }
    }
}

// ============================================================================
// 11. SmoothWhereNode Implementation: z = f + w(x) * (t - f)
//     w(x) = 0.5 * (1 + tanh(k * (x - x0)))
// ============================================================================
void SmoothWhereNode::forward(Workspace& ws) const {
    const Node& x_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);

    auto X = ws.val_mat(x_node.offset, x_node.shape);
    auto T = ws.val_mat(t_node.offset, t_node.shape);
    auto F = ws.val_mat(f_node.offset, f_node.shape);
    auto Z = ws.val_mat(offset, shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cx = (x_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rx = (x_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;
            const double w = 0.5 * (1.0 + std::tanh(sharpness * (X(rx, cx) - threshold)));
            Z(row, col) = F(rf, cf) + w * (T(rt, ct) - F(rf, cf));
        }
    }
}

void SmoothWhereNode::backward(Workspace& ws) const {
    const Node& x_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);
    if (!x_node.requires_grad && !t_node.requires_grad && !f_node.requires_grad) return;

    auto X = ws.val_mat(x_node.offset, x_node.shape);
    auto T = ws.val_mat(t_node.offset, t_node.shape);
    auto F = ws.val_mat(f_node.offset, f_node.shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Xb = ws.bar_mat(x_node.offset, x_node.shape);
    auto Tb = ws.bar_mat(t_node.offset, t_node.shape);
    auto Fb = ws.bar_mat(f_node.offset, f_node.shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cx = (x_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const double zb = Zb(row, col);
            if (zb == 0.0) continue;
            const Eigen::Index rx = (x_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;

            const double th = std::tanh(sharpness * (X(rx, cx) - threshold));
            const double w = 0.5 * (1.0 + th);
            const double dw = 0.5 * sharpness * (1.0 - th * th);
            const double diff = T(rt, ct) - F(rf, cf);

            if (x_node.requires_grad) Xb(rx, cx) += zb * dw * diff;
            if (t_node.requires_grad) Tb(rt, ct) += zb * w;
            if (f_node.requires_grad) Fb(rf, cf) += zb * (1.0 - w);
        }
    }
}

void SmoothWhereNode::forward_tangent(Workspace& ws) const {
    const Node& x_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);

    auto X = ws.val_mat(x_node.offset, x_node.shape);
    auto T = ws.val_mat(t_node.offset, t_node.shape);
    auto F = ws.val_mat(f_node.offset, f_node.shape);
    auto Xd = ws.dot_mat(x_node.offset, x_node.shape);
    auto Td = ws.dot_mat(t_node.offset, t_node.shape);
    auto Fd = ws.dot_mat(f_node.offset, f_node.shape);
    auto Zd = ws.dot_mat(offset, shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cx = (x_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const Eigen::Index rx = (x_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;

            const double th = std::tanh(sharpness * (X(rx, cx) - threshold));
            const double w = 0.5 * (1.0 + th);
            const double dw = 0.5 * sharpness * (1.0 - th * th);
            const double diff = T(rt, ct) - F(rf, cf);

            const double xd = x_node.requires_grad ? Xd(rx, cx) : 0.0;
            const double td = t_node.requires_grad ? Td(rt, ct) : 0.0;
            const double fd = f_node.requires_grad ? Fd(rf, cf) : 0.0;

            Zd(row, col) = dw * diff * xd + w * td + (1.0 - w) * fd;
        }
    }
}

void SmoothWhereNode::backward_hessian(Workspace& ws) const {
    const Node& x_node = graph->node(inputs[0]);
    const Node& t_node = graph->node(inputs[1]);
    const Node& f_node = graph->node(inputs[2]);
    if (!x_node.requires_grad && !t_node.requires_grad && !f_node.requires_grad) return;

    auto X = ws.val_mat(x_node.offset, x_node.shape);
    auto T = ws.val_mat(t_node.offset, t_node.shape);
    auto F = ws.val_mat(f_node.offset, f_node.shape);
    auto Xd = ws.dot_mat(x_node.offset, x_node.shape);
    auto Td = ws.dot_mat(t_node.offset, t_node.shape);
    auto Fd = ws.dot_mat(f_node.offset, f_node.shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);
    auto Xdb = ws.dot_bar_mat(x_node.offset, x_node.shape);
    auto Tdb = ws.dot_bar_mat(t_node.offset, t_node.shape);
    auto Fdb = ws.dot_bar_mat(f_node.offset, f_node.shape);

    for (Eigen::Index col = 0; col < shape.cols; ++col) {
        const Eigen::Index cx = (x_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index ct = (t_node.shape.cols == 1) ? 0 : col;
        const Eigen::Index cf = (f_node.shape.cols == 1) ? 0 : col;
        for (Eigen::Index row = 0; row < shape.rows; ++row) {
            const double zb = Zb(row, col);
            const double zdb = Zdb(row, col);
            if (zb == 0.0 && zdb == 0.0) continue;
            const Eigen::Index rx = (x_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rt = (t_node.shape.rows == 1) ? 0 : row;
            const Eigen::Index rf = (f_node.shape.rows == 1) ? 0 : row;

            const double th = std::tanh(sharpness * (X(rx, cx) - threshold));
            const double w = 0.5 * (1.0 + th);
            const double sech2 = 1.0 - th * th;
            const double dw = 0.5 * sharpness * sech2;
            const double d2w = -sharpness * sharpness * th * sech2;
            const double diff = T(rt, ct) - F(rf, cf);

            const double xd = x_node.requires_grad ? Xd(rx, cx) : 0.0;
            const double td = t_node.requires_grad ? Td(rt, ct) : 0.0;
            const double fd = f_node.requires_grad ? Fd(rf, cf) : 0.0;

            if (x_node.requires_grad) {
                Xdb(rx, cx) += zdb * (dw * diff) + zb * (d2w * diff * xd + dw * (td - fd));
            }
            if (t_node.requires_grad) {
                Tdb(rt, ct) += zdb * w + zb * (dw * xd);
            }
            if (f_node.requires_grad) {
                Fdb(rf, cf) += zdb * (1.0 - w) - zb * (dw * xd);
            }
        }
    }
}

// ============================================================================
// 12. CondNode Implementation (Short-Circuit Subgraph Conditional Execution)
// ============================================================================
namespace {

std::vector<int> build_ParentList(int cond_id, const std::vector<int>& arg_ids) {
    std::vector<int> all;
    all.reserve(1 + arg_ids.size());
    all.push_back(cond_id);
    all.insert(all.end(), arg_ids.begin(), arg_ids.end());
    return all;
}

} // namespace

CondNode::CondNode(Shape s, int cond_id, const std::vector<int>& arg_ids,
                   std::shared_ptr<Graph> t_graph, int t_out,
                   std::shared_ptr<Graph> f_graph, int f_out, std::string name)
    : Node(OpType::Cond, s, build_ParentList(cond_id, arg_ids), std::move(name)),
      true_graph(std::move(t_graph)),
      true_out_id(t_out),
      false_graph(std::move(f_graph)),
      false_out_id(f_out) {
    if (!true_graph->is_compiled()) true_graph->compile({true_out_id});
    if (!false_graph->is_compiled()) false_graph->compile({false_out_id});
}

void CondNode::forward(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const bool take_true = (ws.val(c_node.offset) > 0.5);
    const Graph& sub_g = take_true ? *true_graph : *false_graph;
    const int sub_out = take_true ? true_out_id : false_out_id;

    Workspace sub_ws(sub_g.total_memory_size());
    sub_g.initialize_static_workspace(sub_ws);

    const auto& sub_vars = sub_g.variable_ids();
    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        const Node& inner_v = sub_g.node(sub_vars[i]);
        sub_ws.val.segment(inner_v.offset, inner_v.size()) =
            ws.val.segment(outer_in.offset, outer_in.size());
    }
    for (int id : sub_g.dynamic_exec_order()) {
        sub_g.node(id).forward(sub_ws);
    }
    const Node& out_n = sub_g.node(sub_out);
    ws.val.segment(offset, size()) = sub_ws.val.segment(out_n.offset, size());
}

void CondNode::backward(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const bool take_true = (ws.val(c_node.offset) > 0.5);
    const Graph& sub_g = take_true ? *true_graph : *false_graph;
    const int sub_out = take_true ? true_out_id : false_out_id;

    Workspace sub_ws(sub_g.total_memory_size());
    sub_g.initialize_static_workspace(sub_ws);
    sub_ws.allocate_tier1(sub_g.total_memory_size());

    const auto& sub_vars = sub_g.variable_ids();
    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        const Node& inner_v = sub_g.node(sub_vars[i]);
        sub_ws.val.segment(inner_v.offset, inner_v.size()) =
            ws.val.segment(outer_in.offset, outer_in.size());
    }
    for (int id : sub_g.dynamic_exec_order()) {
        sub_g.node(id).forward(sub_ws);
    }

    const Node& out_n = sub_g.node(sub_out);
    sub_ws.bar.segment(out_n.offset, size()) = ws.bar.segment(offset, size());

    const auto& dyn = sub_g.dynamic_exec_order();
    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        if (sub_g.node(*it).requires_grad) {
            sub_g.node(*it).backward(sub_ws);
        }
    }

    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        if (outer_in.requires_grad) {
            const Node& inner_v = sub_g.node(sub_vars[i]);
            ws.bar.segment(outer_in.offset, outer_in.size()) +=
                sub_ws.bar.segment(inner_v.offset, inner_v.size());
        }
    }
}

void CondNode::forward_tangent(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const bool take_true = (ws.val(c_node.offset) > 0.5);
    const Graph& sub_g = take_true ? *true_graph : *false_graph;
    const int sub_out = take_true ? true_out_id : false_out_id;

    Workspace sub_ws(sub_g.total_memory_size());
    sub_g.initialize_static_workspace(sub_ws);
    sub_ws.allocate_tier2(sub_g.total_memory_size());

    const auto& sub_vars = sub_g.variable_ids();
    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        const Node& inner_v = sub_g.node(sub_vars[i]);
        sub_ws.val.segment(inner_v.offset, inner_v.size()) =
            ws.val.segment(outer_in.offset, outer_in.size());
        if (outer_in.requires_grad) {
            sub_ws.dot.segment(inner_v.offset, inner_v.size()) =
                ws.dot.segment(outer_in.offset, outer_in.size());
        }
    }
    for (int id : sub_g.dynamic_exec_order()) {
        sub_g.node(id).forward(sub_ws);
        sub_g.node(id).forward_tangent(sub_ws);
    }

    const Node& out_n = sub_g.node(sub_out);
    ws.dot.segment(offset, size()) = sub_ws.dot.segment(out_n.offset, size());
}

void CondNode::backward_hessian(Workspace& ws) const {
    const Node& c_node = graph->node(inputs[0]);
    const bool take_true = (ws.val(c_node.offset) > 0.5);
    const Graph& sub_g = take_true ? *true_graph : *false_graph;
    const int sub_out = take_true ? true_out_id : false_out_id;

    Workspace sub_ws(sub_g.total_memory_size());
    sub_g.initialize_static_workspace(sub_ws);
    sub_ws.allocate_tier1(sub_g.total_memory_size());
    sub_ws.allocate_tier2(sub_g.total_memory_size());

    const auto& sub_vars = sub_g.variable_ids();
    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        const Node& inner_v = sub_g.node(sub_vars[i]);
        sub_ws.val.segment(inner_v.offset, inner_v.size()) =
            ws.val.segment(outer_in.offset, outer_in.size());
        if (outer_in.requires_grad) {
            sub_ws.dot.segment(inner_v.offset, inner_v.size()) =
                ws.dot.segment(outer_in.offset, outer_in.size());
        }
    }
    for (int id : sub_g.dynamic_exec_order()) {
        sub_g.node(id).forward(sub_ws);
        sub_g.node(id).forward_tangent(sub_ws);
    }

    const Node& out_n = sub_g.node(sub_out);
    sub_ws.bar.segment(out_n.offset, size()) = ws.bar.segment(offset, size());
    sub_ws.dot_bar.segment(out_n.offset, size()) = ws.dot_bar.segment(offset, size());

    const auto& dyn = sub_g.dynamic_exec_order();
    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        if (sub_g.node(*it).requires_grad) {
            sub_g.node(*it).backward(sub_ws);
            sub_g.node(*it).backward_hessian(sub_ws);
        }
    }

    for (std::size_t i = 0; i < sub_vars.size(); ++i) {
        const Node& outer_in = graph->node(inputs[i + 1]);
        if (outer_in.requires_grad) {
            const Node& inner_v = sub_g.node(sub_vars[i]);
            ws.dot_bar.segment(outer_in.offset, outer_in.size()) +=
                sub_ws.dot_bar.segment(inner_v.offset, inner_v.size());
        }
    }
}

// ============================================================================
// 13. ImplicitSolverNode Implementation (Damped Newton + 1st & 2nd Order IFT)
// ============================================================================
ImplicitSolverNode::ImplicitSolverNode(
    Shape s, const std::vector<int>& outer_input_ids,
    std::shared_ptr<Graph> res_graph, int res_out_id,
    int inner_state_var_id, std::vector<int> inner_param_ids,
    int max_newton_iter, double newton_tol, std::string name)
    : Node(OpType::ImplicitSolver, s, outer_input_ids, std::move(name)),
      residual_graph(std::move(res_graph)),
      residual_out_id(res_out_id),
      state_var_id(inner_state_var_id),
      param_input_ids(std::move(inner_param_ids)),
      max_iter(max_newton_iter),
      tol(newton_tol) {
    if (inputs.size() != param_input_ids.size()) {
        throw std::invalid_argument("ImplicitSolverNode: outer_input_ids and inner_param_ids length mismatch.");
    }
    if (!residual_graph->is_compiled()) {
        residual_graph->compile({residual_out_id});
    }
}

void ImplicitSolverNode::populate_subworkspace(
    const Workspace& outer_ws, Workspace& sub_ws, const Eigen::VectorXd& y_val) const {
    sub_ws.allocate_tier0(residual_graph->total_memory_size());
    residual_graph->initialize_static_workspace(sub_ws);

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Node& outer_n = graph->node(inputs[i]);
        const Node& inner_n = residual_graph->node(param_input_ids[i]);
        sub_ws.val.segment(inner_n.offset, inner_n.size()) =
            outer_ws.val.segment(outer_n.offset, outer_n.size());
    }

    const Node& y_node = residual_graph->node(state_var_id);
    sub_ws.val.segment(y_node.offset, y_node.size()) = y_val;

    for (int id : residual_graph->all_exec_order()) {
        residual_graph->node(id).forward(sub_ws);
    }
}

Eigen::MatrixXd ImplicitSolverNode::compute_jy(
    Workspace& sub_ws, const Eigen::VectorXd& y_val) const {
    const Node& y_node = residual_graph->node(state_var_id);
    const Node& r_node = residual_graph->node(residual_out_id);
    const Eigen::Index m = y_node.size();

    sub_ws.val.segment(y_node.offset, m) = y_val;
    for (int id : residual_graph->all_exec_order()) {
        residual_graph->node(id).forward(sub_ws);
    }

    Eigen::MatrixXd Jy(m, m);
    sub_ws.allocate_tier2(residual_graph->total_memory_size());

    for (Eigen::Index j = 0; j < m; ++j) {
        sub_ws.dot.setZero();
        sub_ws.dot(y_node.offset + j) = 1.0;
        for (int id : residual_graph->all_exec_order()) {
            residual_graph->node(id).forward_tangent(sub_ws);
        }
        Jy.col(j) = sub_ws.dot.segment(r_node.offset, m);
    }
    return Jy;
}

void ImplicitSolverNode::forward(Workspace& ws) const {
    const Node& y_node = residual_graph->node(state_var_id);
    const Node& r_node = residual_graph->node(residual_out_id);
    const Eigen::Index m = y_node.size();

    Eigen::VectorXd y = residual_graph->get_variable_init(state_var_id);
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(m, -1e20);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(m, 1e20);
    if (y_node.op == OpType::Variable) {
        lb = static_cast<const VariableNode&>(y_node).lb;
        ub = static_cast<const VariableNode&>(y_node).ub;
    } else if (y_node.op == OpType::ImplicitState) {
        lb = static_cast<const ImplicitStateNode&>(y_node).lb;
        ub = static_cast<const ImplicitStateNode&>(y_node).ub;
    }

    y = y.cwiseMax(lb).cwiseMin(ub);

    Workspace sub_ws(residual_graph->total_memory_size());
    populate_subworkspace(ws, sub_ws, y);

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd R = sub_ws.val.segment(r_node.offset, m);
        double res_norm = R.lpNorm<Eigen::Infinity>();
        if (res_norm <= tol) break;

        Eigen::MatrixXd Jy = compute_jy(sub_ws, y);
        Eigen::VectorXd dy = Jy.colPivHouseholderQr().solve(-R);

        double alpha = 1.0;
        const double r0_norm = R.norm();
        Eigen::VectorXd y_next = y;
        for (int ls = 0; ls < 10; ++ls) {
            y_next = (y + alpha * dy).cwiseMax(lb).cwiseMin(ub);
            sub_ws.val.segment(y_node.offset, m) = y_next;
            for (int id : residual_graph->all_exec_order()) {
                residual_graph->node(id).forward(sub_ws);
            }
            double r_next = sub_ws.val.segment(r_node.offset, m).norm();
            if (r_next < r0_norm || ls == 9) {
                y = y_next;
                break;
            }
            alpha *= 0.5;
        }
    }

    ws.val.segment(offset, m) = y;
}

void ImplicitSolverNode::backward(Workspace& ws) const {
    const Node& y_node = residual_graph->node(state_var_id);
    const Node& r_node = residual_graph->node(residual_out_id);
    const Eigen::Index m = y_node.size();

    Eigen::VectorXd y_star = ws.val.segment(offset, m);
    Eigen::VectorXd y_bar = ws.bar.segment(offset, m);
    if (y_bar.squaredNorm() == 0.0) return;

    Workspace sub_ws(residual_graph->total_memory_size());
    populate_subworkspace(ws, sub_ws, y_star);

    Eigen::MatrixXd Jy = compute_jy(sub_ws, y_star);
    Eigen::VectorXd lambda_ift = Jy.transpose().colPivHouseholderQr().solve(-y_bar);

    sub_ws.allocate_tier1(residual_graph->total_memory_size());
    sub_ws.bar.segment(r_node.offset, m) = lambda_ift;

    const auto& exec = residual_graph->all_exec_order();
    for (auto it = exec.rbegin(); it != exec.rend(); ++it) {
        residual_graph->node(*it).backward(sub_ws);
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Node& outer_n = graph->node(inputs[i]);
        if (outer_n.requires_grad) {
            const Node& inner_n = residual_graph->node(param_input_ids[i]);
            ws.bar.segment(outer_n.offset, outer_n.size()) +=
                sub_ws.bar.segment(inner_n.offset, inner_n.size());
        }
    }
}

void ImplicitSolverNode::forward_tangent(Workspace& ws) const {
    const Node& y_node = residual_graph->node(state_var_id);
    const Node& r_node = residual_graph->node(residual_out_id);
    const Eigen::Index m = y_node.size();

    Eigen::VectorXd y_star = ws.val.segment(offset, m);
    Workspace sub_ws(residual_graph->total_memory_size());
    populate_subworkspace(ws, sub_ws, y_star);

    Eigen::MatrixXd Jy = compute_jy(sub_ws, y_star);

    sub_ws.allocate_tier2(residual_graph->total_memory_size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Node& outer_n = graph->node(inputs[i]);
        const Node& inner_n = residual_graph->node(param_input_ids[i]);
        if (outer_n.requires_grad) {
            sub_ws.dot.segment(inner_n.offset, inner_n.size()) =
                ws.dot.segment(outer_n.offset, outer_n.size());
        }
    }
    sub_ws.dot.segment(y_node.offset, m).setZero();
    for (int id : residual_graph->all_exec_order()) {
        residual_graph->node(id).forward_tangent(sub_ws);
    }
    Eigen::VectorXd r_dot_x = sub_ws.dot.segment(r_node.offset, m);

    ws.dot.segment(offset, m) = Jy.colPivHouseholderQr().solve(-r_dot_x);
}

void ImplicitSolverNode::backward_hessian(Workspace& ws) const {
    const Node& y_node = residual_graph->node(state_var_id);
    const Node& r_node = residual_graph->node(residual_out_id);
    const Eigen::Index m = y_node.size();

    Eigen::VectorXd y_star = ws.val.segment(offset, m);
    Eigen::VectorXd y_bar = ws.bar.segment(offset, m);
    Eigen::VectorXd y_dot = ws.dot.segment(offset, m);
    Eigen::VectorXd y_dot_bar = ws.dot_bar.segment(offset, m);

    Workspace sub_ws(residual_graph->total_memory_size());
    populate_subworkspace(ws, sub_ws, y_star);
    Eigen::MatrixXd Jy = compute_jy(sub_ws, y_star);
    auto jy_T_qr = Jy.transpose().colPivHouseholderQr();

    Eigen::VectorXd lambda_ift = jy_T_qr.solve(-y_bar);

    sub_ws.allocate_tier1(residual_graph->total_memory_size());
    sub_ws.allocate_tier2(residual_graph->total_memory_size());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Node& outer_n = graph->node(inputs[i]);
        const Node& inner_n = residual_graph->node(param_input_ids[i]);
        if (outer_n.requires_grad) {
            sub_ws.dot.segment(inner_n.offset, inner_n.size()) =
                ws.dot.segment(outer_n.offset, outer_n.size());
        }
    }
    sub_ws.dot.segment(y_node.offset, m) = y_dot;

    for (int id : residual_graph->all_exec_order()) {
        residual_graph->node(id).forward_tangent(sub_ws);
    }

    sub_ws.bar.segment(r_node.offset, m) = lambda_ift;
    const auto& exec = residual_graph->all_exec_order();
    for (auto it = exec.rbegin(); it != exec.rend(); ++it) {
        residual_graph->node(*it).backward(sub_ws);
        residual_graph->node(*it).backward_hessian(sub_ws);
    }

    Eigen::VectorXd w_y = sub_ws.dot_bar.segment(y_node.offset, m);
    Eigen::VectorXd d_lambda = jy_T_qr.solve(-(y_dot_bar + w_y));

    Workspace sub_ws2(residual_graph->total_memory_size());
    populate_subworkspace(ws, sub_ws2, y_star);
    sub_ws2.allocate_tier1(residual_graph->total_memory_size());
    sub_ws2.bar.segment(r_node.offset, m) = d_lambda;
    for (auto it = exec.rbegin(); it != exec.rend(); ++it) {
        residual_graph->node(*it).backward(sub_ws2);
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Node& outer_n = graph->node(inputs[i]);
        if (outer_n.requires_grad) {
            const Node& inner_n = residual_graph->node(param_input_ids[i]);
            ws.dot_bar.segment(outer_n.offset, outer_n.size()) +=
                sub_ws.dot_bar.segment(inner_n.offset, inner_n.size()) +
                sub_ws2.bar.segment(inner_n.offset, inner_n.size());
        }
    }
}

// ============================================================================
// 14. BlackBoxNode Implementation (Analytical/FD 1st/2nd AD & Batch Eval)
// ============================================================================
BlackBoxNode::BlackBoxNode(
    Shape out_shape,
    std::vector<int> in_ids,
    EvalFn f_cb,
    JacFn j_cb,
    HvpFn h_cb,
    double finite_diff_step,
    bool enable_fd_hessian,
    std::string name,
    BatchEvalFn batch_cb,
    bool thread_safe,
    Eigen::MatrixXi jac_mask)
    : Node(OpType::BlackBox, out_shape, std::move(in_ids), std::move(name)),
      eval_fn(std::move(f_cb)),
      batch_eval_fn(std::move(batch_cb)),
      jac_fn(std::move(j_cb)),
      hvp_fn(std::move(h_cb)),
      fd_step(finite_diff_step),
      fd_hessian(enable_fd_hessian),
      is_thread_safe(thread_safe),
      jac_sparsity_mask(std::move(jac_mask)) {
    if (!eval_fn) {
        throw std::invalid_argument("BlackBoxNode: eval_fn callback cannot be empty.");
    }
    if (fd_step <= 0.0) {
        fd_step = 1e-7;
    }
}

Eigen::Index BlackBoxNode::total_input_size() const {
    Eigen::Index total = 0;
    for (int in_id : inputs) {
        total += graph->node(in_id).size();
    }
    return total;
}

Eigen::VectorXd BlackBoxNode::gather_input_val(const Workspace& ws) const {
    Eigen::VectorXd u(total_input_size());
    Eigen::Index off = 0;
    for (int in_id : inputs) {
        const Node& in_n = graph->node(in_id);
        const Eigen::Index sz = in_n.size();
        u.segment(off, sz) = ws.val.segment(in_n.offset, sz);
        off += sz;
    }
    return u;
}

Eigen::VectorXd BlackBoxNode::gather_input_dot(const Workspace& ws) const {
    Eigen::VectorXd u_dot = Eigen::VectorXd::Zero(total_input_size());
    Eigen::Index off = 0;
    for (int in_id : inputs) {
        const Node& in_n = graph->node(in_id);
        const Eigen::Index sz = in_n.size();
        if (in_n.requires_grad) {
            u_dot.segment(off, sz) = ws.dot.segment(in_n.offset, sz);
        }
        off += sz;
    }
    return u_dot;
}

Eigen::MatrixXd BlackBoxNode::compute_jacobian_uncached(
    const Eigen::VectorXd& u, double step_override) const {
    const Eigen::Index m_out = size();
    const Eigen::Index n_in = u.size();
    if (jac_fn) {
        Eigen::MatrixXd J = jac_fn(u);
        if (J.rows() != m_out || J.cols() != n_in) {
            throw std::runtime_error("BlackBoxNode: jac_fn returned matrix with mismatched dimensions.");
        }
        return J;
    }

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m_out, n_in);
    const double base_step = (step_override > 0.0) ? step_override : fd_step;
    Eigen::VectorXd u_pert = u;

    Eigen::Index col_off = 0;
    for (int in_id : inputs) {
        const Node& in_n = graph->node(in_id);
        const Eigen::Index sz = in_n.size();
        const bool has_mask = (jac_sparsity_mask.rows() == m_out && jac_sparsity_mask.cols() == n_in);
        if (in_n.requires_grad) {
            for (Eigen::Index k = 0; k < sz; ++k) {
                const Eigen::Index j = col_off + k;
                if (has_mask && (jac_sparsity_mask.col(j).array() == 0).all()) {
                    continue;
                }
                const double orig = u(j);
                const double h = base_step * std::max(1.0, std::abs(orig));
                u_pert(j) = orig + h;
                Eigen::VectorXd y_plus = eval_fn(u_pert);
                u_pert(j) = orig - h;
                Eigen::VectorXd y_minus = eval_fn(u_pert);
                u_pert(j) = orig;
                J.col(j) = (y_plus - y_minus) / (2.0 * h);
            }
        }
        col_off += sz;
    }
    return J;
}

Eigen::MatrixXd BlackBoxNode::get_or_compute_jacobian(const Eigen::VectorXd& u) const {
    if (!omp_in_parallel()) {
        if (last_u_jac_.size() == u.size() && (last_u_jac_.array() == u.array()).all()) {
            return cached_jac_;
        }
        Eigen::MatrixXd J_new = compute_jacobian_uncached(u);
        cached_jac_ = J_new;
        last_u_jac_ = u;
        return J_new;
    }
    Eigen::MatrixXd J_hit;
    bool hit = false;
    #pragma omp critical(fastopt_bb_jac_cache)
    {
        if (last_u_jac_.size() == u.size() && (last_u_jac_.array() == u.array()).all()) {
            J_hit = cached_jac_;
            hit = true;
        }
    }
    if (hit) {
        return J_hit;
    }
    Eigen::MatrixXd J_new = compute_jacobian_uncached(u);
    #pragma omp critical(fastopt_bb_jac_cache)
    {
        cached_jac_ = J_new;
        last_u_jac_ = u;
    }
    return J_new;
}

Eigen::VectorXd BlackBoxNode::compute_hvp(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& y_bar,
    const Eigen::VectorXd& u_dot) const {
    const Eigen::Index n_in = u.size();
    if (hvp_fn) {
        Eigen::VectorXd res = hvp_fn(u, y_bar, u_dot);
        if (res.size() != n_in) {
            throw std::runtime_error("BlackBoxNode: hvp_fn returned vector with mismatched size.");
        }
        return res;
    }
    if (!fd_hessian) {
        return Eigen::VectorXd::Zero(n_in);
    }

    const double d_norm = u_dot.norm();
    if (d_norm == 0.0 || y_bar.squaredNorm() == 0.0) {
        return Eigen::VectorXd::Zero(n_in);
    }

    const Eigen::VectorXd d_hat = u_dot / d_norm;
    const double scale = std::max(1.0, u.norm());
    if (jac_fn) {
        const double h = fd_step * scale;
        Eigen::MatrixXd J_plus = compute_jacobian_uncached(u + h * d_hat);
        Eigen::MatrixXd J_minus = compute_jacobian_uncached(u - h * d_hat);
        return d_norm * ((J_plus - J_minus).transpose() * y_bar) / (2.0 * h);
    } else {
        const double h_outer = std::pow(fd_step, 0.5) * scale;
        const double h_inner = std::pow(fd_step, 0.65);
        Eigen::MatrixXd J_plus = compute_jacobian_uncached(u + h_outer * d_hat, h_inner);
        Eigen::MatrixXd J_minus = compute_jacobian_uncached(u - h_outer * d_hat, h_inner);
        return d_norm * ((J_plus - J_minus).transpose() * y_bar) / (2.0 * h_outer);
    }
}

void BlackBoxNode::forward(Workspace& ws) const {
    Eigen::VectorXd u = gather_input_val(ws);
    Eigen::VectorXd y = eval_fn(u);
    if (y.size() != size()) {
        throw std::runtime_error("BlackBoxNode::forward: output size mismatch.");
    }
    ws.val.segment(offset, size()) = y;
}

void BlackBoxNode::forward_batch_workspaces(std::vector<Workspace>& batch_ws, Eigen::Index active_pop) const {
    const Eigen::Index n_pop = (active_pop >= 0) ? active_pop : static_cast<Eigen::Index>(batch_ws.size());
    if (n_pop == 0) return;

    if (!batch_eval_fn) {
        for (Eigen::Index p = 0; p < n_pop; ++p) {
            forward(batch_ws[static_cast<std::size_t>(p)]);
        }
        return;
    }

    const Eigen::Index n_in = total_input_size();
    const Eigen::Index m_out = size();
    Eigen::MatrixXd U_batch(n_pop, n_in);
    for (Eigen::Index p = 0; p < n_pop; ++p) {
        const Workspace& ws = batch_ws[static_cast<std::size_t>(p)];
        Eigen::Index col_off = 0;
        for (int in_id : inputs) {
            const Node& in_n = graph->node(in_id);
            const Eigen::Index sz = in_n.size();
            U_batch.row(p).segment(col_off, sz) = ws.val.segment(in_n.offset, sz).transpose();
            col_off += sz;
        }
    }

    Eigen::MatrixXd Y_batch = batch_eval_fn(U_batch);
    if (Y_batch.rows() != n_pop || Y_batch.cols() != m_out) {
        throw std::runtime_error("BlackBoxNode::forward_batch_workspaces: batch output shape mismatch.");
    }

    for (Eigen::Index p = 0; p < n_pop; ++p) {
        batch_ws[static_cast<std::size_t>(p)].val.segment(offset, m_out) = Y_batch.row(p).transpose();
    }
}

void BlackBoxNode::backward(Workspace& ws) const {
    Eigen::VectorXd y_bar = ws.bar.segment(offset, size());
    if (y_bar.squaredNorm() == 0.0) return;

    Eigen::VectorXd u = gather_input_val(ws);
    const Eigen::MatrixXd& J = get_or_compute_jacobian(u);
    Eigen::VectorXd u_bar = J.transpose() * y_bar;

    Eigen::Index col_off = 0;
    for (int in_id : inputs) {
        const Node& in_n = graph->node(in_id);
        const Eigen::Index sz = in_n.size();
        if (in_n.requires_grad) {
            ws.bar.segment(in_n.offset, sz) += u_bar.segment(col_off, sz);
        }
        col_off += sz;
    }
}

void BlackBoxNode::forward_tangent(Workspace& ws) const {
    Eigen::VectorXd u_dot = gather_input_dot(ws);
    if (u_dot.squaredNorm() == 0.0) {
        ws.dot.segment(offset, size()).setZero();
        return;
    }
    Eigen::VectorXd u = gather_input_val(ws);
    const Eigen::MatrixXd& J = get_or_compute_jacobian(u);
    ws.dot.segment(offset, size()) = J * u_dot;
}

void BlackBoxNode::backward_hessian(Workspace& ws) const {
    Eigen::VectorXd y_bar = ws.bar.segment(offset, size());
    Eigen::VectorXd y_dot_bar = ws.dot_bar.segment(offset, size());
    if (y_bar.squaredNorm() == 0.0 && y_dot_bar.squaredNorm() == 0.0) return;

    Eigen::VectorXd u = gather_input_val(ws);
    Eigen::VectorXd u_dot = gather_input_dot(ws);
    const Eigen::MatrixXd& J = get_or_compute_jacobian(u);

    Eigen::VectorXd u_dot_bar = Eigen::VectorXd::Zero(u.size());
    if (y_dot_bar.squaredNorm() > 0.0) {
        u_dot_bar += J.transpose() * y_dot_bar;
    }
    if (y_bar.squaredNorm() > 0.0 && u_dot.squaredNorm() > 0.0) {
        u_dot_bar += compute_hvp(u, y_bar, u_dot);
    }

    Eigen::Index col_off = 0;
    for (int in_id : inputs) {
        const Node& in_n = graph->node(in_id);
        const Eigen::Index sz = in_n.size();
        if (in_n.requires_grad) {
            ws.dot_bar.segment(in_n.offset, sz) += u_dot_bar.segment(col_off, sz);
        }
        col_off += sz;
    }
}

// ============================================================================
// 15. MapNode Implementation (Independent Parallel Loop / fo.vmap)
// ============================================================================
MapNode::MapNode(
    Shape out_shape,
    const std::vector<int>& outer_in_ids,
    std::shared_ptr<Graph> sub_graph,
    int sub_out_id,
    std::vector<int> sub_var_ids,
    std::vector<int> axes_in,
    int axis_out,
    int n_batches,
    std::string name)
    : Node(OpType::Map, out_shape, outer_in_ids, std::move(name)),
      body_graph(std::move(sub_graph)),
      body_out_id(sub_out_id),
      inner_var_ids(std::move(sub_var_ids)),
      in_axes(std::move(axes_in)),
      out_axis(axis_out),
      num_batches(n_batches) {
    if (!body_graph) {
        throw std::invalid_argument("MapNode received null body_graph.");
    }
    if (inputs.size() != inner_var_ids.size() || inputs.size() != in_axes.size()) {
        throw std::invalid_argument("MapNode: inputs, inner_var_ids, and in_axes size mismatch.");
    }
    if (!body_graph->is_compiled()) {
        body_graph->compile({body_out_id});
    }
    body_is_thread_safe = !body_graph->has_unsafe_blackbox();
    const int max_outer = std::max(64, omp_get_max_threads() * 2);
    thread_tapes_.resize(static_cast<std::size_t>(max_outer));
    ensure_tape(0, false, false);
}

std::vector<Workspace>& MapNode::ensure_tape(int outer_tid, bool need_tier1, bool need_tier2) const {
    const std::size_t tid = static_cast<std::size_t>(
        (outer_tid >= 0 && outer_tid < static_cast<int>(thread_tapes_.size())) ? outer_tid : 0);
    auto& tape = thread_tapes_[tid];
    const Eigen::Index mem_sz = body_graph->total_memory_size();
    if (static_cast<int>(tape.size()) != num_batches) {
        tape.resize(static_cast<std::size_t>(num_batches));
        for (int b = 0; b < num_batches; ++b) {
            tape[static_cast<std::size_t>(b)].allocate_tier0(mem_sz);
            body_graph->initialize_static_workspace(tape[static_cast<std::size_t>(b)]);
        }
    }
    if (need_tier1 || need_tier2) {
        for (int b = 0; b < num_batches; ++b) {
            Workspace& sws = tape[static_cast<std::size_t>(b)];
            if (need_tier1) sws.allocate_tier1(mem_sz);
            if (need_tier2) sws.allocate_tier2(mem_sz);
        }
    }
    return tape;
}

void MapNode::extract_slice(
    const Eigen::VectorXd& outer_buf, const Node& outer_n,
    int axis, int b, const Shape& inner_s,
    Eigen::Ref<Eigen::VectorXd> dst) const {
    const Eigen::Index sz = inner_s.size();
    if (axis == -1) {
        dst = outer_buf.segment(outer_n.offset, sz);
    } else if (axis == 1) {
        dst = outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(b) * sz, sz);
    } else if (axis == 0) {
        if (outer_n.shape.cols == 1) {
            dst = outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(b) * sz, sz);
        } else {
            const Eigen::Index r_slice = outer_n.shape.rows / num_batches;
            const Eigen::Index c_slice = outer_n.shape.cols;
            Eigen::Map<const Eigen::MatrixXd> M_out(
                outer_buf.data() + outer_n.offset, outer_n.shape.rows, outer_n.shape.cols);
            Eigen::Map<Eigen::MatrixXd> M_dst(dst.data(), r_slice, c_slice);
            M_dst = M_out.middleRows(static_cast<Eigen::Index>(b) * r_slice, r_slice);
        }
    }
}

void MapNode::accumulate_slice(
    const Eigen::Ref<const Eigen::VectorXd>& src,
    const Shape& inner_s, int axis, int b,
    const Node& outer_n, Eigen::VectorXd& outer_buf) const {
    const Eigen::Index sz = inner_s.size();
    if (axis == -1) {
        outer_buf.segment(outer_n.offset, sz) += src;
    } else if (axis == 1) {
        outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(b) * sz, sz) += src;
    } else if (axis == 0) {
        if (outer_n.shape.cols == 1) {
            outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(b) * sz, sz) += src;
        } else {
            const Eigen::Index r_slice = outer_n.shape.rows / num_batches;
            const Eigen::Index c_slice = outer_n.shape.cols;
            Eigen::Map<Eigen::MatrixXd> M_out(
                outer_buf.data() + outer_n.offset, outer_n.shape.rows, outer_n.shape.cols);
            Eigen::Map<const Eigen::MatrixXd> M_src(src.data(), r_slice, c_slice);
            M_out.middleRows(static_cast<Eigen::Index>(b) * r_slice, r_slice) += M_src;
        }
    }
}

void MapNode::write_output_slice(
    const Eigen::Ref<const Eigen::VectorXd>& src,
    const Shape& slice_s, int b,
    Eigen::VectorXd& outer_buf) const {
    const Eigen::Index sz = slice_s.size();
    if (out_axis == 1 || shape.cols == 1) {
        outer_buf.segment(offset + static_cast<Eigen::Index>(b) * sz, sz) = src;
    } else {
        const Eigen::Index r_slice = shape.rows / num_batches;
        const Eigen::Index c_slice = shape.cols;
        Eigen::Map<Eigen::MatrixXd> M_out(outer_buf.data() + offset, shape.rows, shape.cols);
        Eigen::Map<const Eigen::MatrixXd> M_src(src.data(), r_slice, c_slice);
        M_out.middleRows(static_cast<Eigen::Index>(b) * r_slice, r_slice) = M_src;
    }
}

void MapNode::read_output_slice(
    const Eigen::VectorXd& outer_buf,
    const Shape& slice_s, int b,
    Eigen::Ref<Eigen::VectorXd> dst) const {
    const Eigen::Index sz = slice_s.size();
    if (out_axis == 1 || shape.cols == 1) {
        dst = outer_buf.segment(offset + static_cast<Eigen::Index>(b) * sz, sz);
    } else {
        const Eigen::Index r_slice = shape.rows / num_batches;
        const Eigen::Index c_slice = shape.cols;
        Eigen::Map<const Eigen::MatrixXd> M_out(outer_buf.data() + offset, shape.rows, shape.cols);
        Eigen::Map<Eigen::MatrixXd> M_dst(dst.data(), r_slice, c_slice);
        M_dst = M_out.middleRows(static_cast<Eigen::Index>(b) * r_slice, r_slice);
    }
}

void MapNode::forward(Workspace& ws) const {
    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, false);
    const Node& out_n = body_graph->node(body_out_id);
    const auto& dyn = body_graph->dynamic_exec_order();
    const bool use_omp = body_is_thread_safe && (num_batches >= 4) && !omp_in_parallel();

    #pragma omp parallel for if(use_omp) schedule(static)
    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        for (std::size_t k = 0; k < inputs.size(); ++k) {
            const Node& outer_in = graph->node(inputs[k]);
            const Node& inner_v = body_graph->node(inner_var_ids[k]);
            extract_slice(ws.val, outer_in, in_axes[k], b, inner_v.shape,
                          sws.val.segment(inner_v.offset, inner_v.size()));
        }
        for (int id : dyn) {
            body_graph->node(id).forward(sws);
        }
        write_output_slice(sws.val.segment(out_n.offset, out_n.size()), out_n.shape, b, ws.val);
    }
}

void MapNode::backward(Workspace& ws) const {
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0) {
        const int tid_clr = omp_in_parallel() ? omp_get_thread_num() : 0;
        if (tid_clr < static_cast<int>(thread_tapes_.size())) {
            for (auto& sws : thread_tapes_[static_cast<std::size_t>(tid_clr)]) {
                sws.bar.setZero();
            }
        }
        return;
    }

    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, true, false);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int b = 0; b < num_batches; ++b) {
            tape[static_cast<std::size_t>(b)].val = tape0[static_cast<std::size_t>(b)].val;
        }
    }
    const Node& out_n = body_graph->node(body_out_id);
    const auto& dyn = body_graph->dynamic_exec_order();
    const bool use_omp = body_is_thread_safe && (num_batches >= 4) && !omp_in_parallel();

    #pragma omp parallel for if(use_omp) schedule(static)
    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        read_output_slice(ws.bar, out_n.shape, b, sws.bar.segment(out_n.offset, out_n.size()));
        if (sws.bar.segment(out_n.offset, out_n.size()).squaredNorm() == 0.0) continue;

        for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
            if (body_graph->node(*it).requires_grad) {
                body_graph->node(*it).backward(sws);
            }
        }
    }

    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        if (sws.bar.segment(out_n.offset, out_n.size()).squaredNorm() == 0.0) continue;
        for (std::size_t k = 0; k < inputs.size(); ++k) {
            const Node& outer_in = graph->node(inputs[k]);
            if (outer_in.requires_grad) {
                const Node& inner_v = body_graph->node(inner_var_ids[k]);
                accumulate_slice(sws.bar.segment(inner_v.offset, inner_v.size()),
                                 inner_v.shape, in_axes[k], b, outer_in, ws.bar);
            }
        }
    }
}

void MapNode::forward_tangent(Workspace& ws) const {
    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, true);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int b = 0; b < num_batches; ++b) {
            tape[static_cast<std::size_t>(b)].val = tape0[static_cast<std::size_t>(b)].val;
        }
    }
    const Eigen::Index mem_sz = body_graph->total_memory_size();
    const Node& out_n = body_graph->node(body_out_id);
    const auto& dyn = body_graph->dynamic_exec_order();
    const bool use_omp = body_is_thread_safe && (num_batches >= 4) && !omp_in_parallel();

    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        if (sws.dot.size() != mem_sz) {
            sws.dot.resize(mem_sz);
        }
    }

    #pragma omp parallel for if(use_omp) schedule(static)
    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        sws.dot.setZero();

        for (std::size_t k = 0; k < inputs.size(); ++k) {
            const Node& outer_in = graph->node(inputs[k]);
            if (outer_in.requires_grad) {
                const Node& inner_v = body_graph->node(inner_var_ids[k]);
                extract_slice(ws.dot, outer_in, in_axes[k], b, inner_v.shape,
                              sws.dot.segment(inner_v.offset, inner_v.size()));
            }
        }
        for (int id : dyn) {
            if (body_graph->node(id).requires_grad) {
                body_graph->node(id).forward_tangent(sws);
            }
        }
        write_output_slice(sws.dot.segment(out_n.offset, out_n.size()), out_n.shape, b, ws.dot);
    }
}

void MapNode::backward_hessian(Workspace& ws) const {
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 &&
        ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) {
        return;
    }

    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, false);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int b = 0; b < num_batches; ++b) {
            tape[static_cast<std::size_t>(b)].val = tape0[static_cast<std::size_t>(b)].val;
            tape[static_cast<std::size_t>(b)].bar = tape0[static_cast<std::size_t>(b)].bar;
        }
    }
    const Eigen::Index mem_sz = body_graph->total_memory_size();
    const Node& out_n = body_graph->node(body_out_id);
    const auto& dyn = body_graph->dynamic_exec_order();
    const bool use_omp = body_is_thread_safe && (num_batches >= 4) && !omp_in_parallel();

    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        if (sws.dot_bar.size() != mem_sz) {
            sws.dot_bar.resize(mem_sz);
        }
    }

    #pragma omp parallel for if(use_omp) schedule(static)
    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        sws.dot_bar.setZero();

        read_output_slice(ws.dot_bar, out_n.shape, b, sws.dot_bar.segment(out_n.offset, out_n.size()));

        for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
            if (body_graph->node(*it).requires_grad) {
                body_graph->node(*it).backward_hessian(sws);
            }
        }
    }

    for (int b = 0; b < num_batches; ++b) {
        Workspace& sws = tape[static_cast<std::size_t>(b)];
        for (std::size_t k = 0; k < inputs.size(); ++k) {
            const Node& outer_in = graph->node(inputs[k]);
            if (outer_in.requires_grad) {
                const Node& inner_v = body_graph->node(inner_var_ids[k]);
                accumulate_slice(sws.dot_bar.segment(inner_v.offset, inner_v.size()),
                                 inner_v.shape, in_axes[k], b, outer_in, ws.dot_bar);
            }
        }
    }
}

// ============================================================================
// 16. ScanNode Implementation (Sequential Recurrent Loop / fo.scan)
// ============================================================================
ScanNode::ScanNode(
    Shape out_shape,
    const std::vector<int>& outer_in_ids,
    std::shared_ptr<Graph> sub_graph,
    int sub_carry_id,
    std::vector<int> sub_seq_ids,
    std::vector<int> axes_seq,
    std::vector<int> sub_shared_ids,
    int sub_next_carry_id,
    int sub_y_id,
    int n_steps,
    std::string name)
    : Node(OpType::Scan, out_shape, outer_in_ids, std::move(name)),
      step_graph(std::move(sub_graph)),
      inner_carry_id(sub_carry_id),
      inner_seq_ids(std::move(sub_seq_ids)),
      seq_axes(std::move(axes_seq)),
      inner_shared_ids(std::move(sub_shared_ids)),
      next_carry_out_id(sub_next_carry_id),
      step_y_out_id(sub_y_id),
      num_steps(n_steps) {
    if (!step_graph) {
        throw std::invalid_argument("ScanNode received null step_graph.");
    }
    if (inputs.size() != 1 + inner_seq_ids.size() + inner_shared_ids.size()) {
        throw std::invalid_argument("ScanNode: outer_in_ids count mismatch.");
    }
    if (!step_graph->is_compiled()) {
        if (step_y_out_id >= 0 && step_y_out_id != next_carry_out_id) {
            step_graph->compile({next_carry_out_id, step_y_out_id});
        } else {
            step_graph->compile({next_carry_out_id});
        }
    }
    const int max_outer = std::max(64, omp_get_max_threads() * 2);
    thread_tapes_.resize(static_cast<std::size_t>(max_outer));
    ensure_tape(0, false, false);
}

std::vector<Workspace>& ScanNode::ensure_tape(int outer_tid, bool need_tier1, bool need_tier2) const {
    const std::size_t tid = static_cast<std::size_t>(
        (outer_tid >= 0 && outer_tid < static_cast<int>(thread_tapes_.size())) ? outer_tid : 0);
    auto& tape = thread_tapes_[tid];
    const Eigen::Index mem_sz = step_graph->total_memory_size();
    if (static_cast<int>(tape.size()) != num_steps) {
        tape.resize(static_cast<std::size_t>(num_steps));
        for (int t = 0; t < num_steps; ++t) {
            tape[static_cast<std::size_t>(t)].allocate_tier0(mem_sz);
            step_graph->initialize_static_workspace(tape[static_cast<std::size_t>(t)]);
        }
    }
    if (need_tier1 || need_tier2) {
        for (int t = 0; t < num_steps; ++t) {
            Workspace& sws = tape[static_cast<std::size_t>(t)];
            if (need_tier1) sws.allocate_tier1(mem_sz);
            if (need_tier2) sws.allocate_tier2(mem_sz);
        }
    }
    return tape;
}

void ScanNode::extract_seq_slice(
    const Eigen::VectorXd& outer_buf, const Node& outer_n,
    int axis, int t, const Shape& inner_s,
    Eigen::Ref<Eigen::VectorXd> dst) const {
    const Eigen::Index sz = inner_s.size();
    if (axis == 1 || outer_n.shape.cols == 1) {
        dst = outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(t) * sz, sz);
    } else {
        const Eigen::Index r_slice = outer_n.shape.rows / num_steps;
        const Eigen::Index c_slice = outer_n.shape.cols;
        Eigen::Map<const Eigen::MatrixXd> M_out(
            outer_buf.data() + outer_n.offset, outer_n.shape.rows, outer_n.shape.cols);
        Eigen::Map<Eigen::MatrixXd> M_dst(dst.data(), r_slice, c_slice);
        M_dst = M_out.middleRows(static_cast<Eigen::Index>(t) * r_slice, r_slice);
    }
}

void ScanNode::accumulate_seq_slice(
    const Eigen::Ref<const Eigen::VectorXd>& src,
    const Shape& inner_s, int axis, int t,
    const Node& outer_n, Eigen::VectorXd& outer_buf) const {
    const Eigen::Index sz = inner_s.size();
    if (axis == 1 || outer_n.shape.cols == 1) {
        outer_buf.segment(outer_n.offset + static_cast<Eigen::Index>(t) * sz, sz) += src;
    } else {
        const Eigen::Index r_slice = outer_n.shape.rows / num_steps;
        const Eigen::Index c_slice = outer_n.shape.cols;
        Eigen::Map<Eigen::MatrixXd> M_out(
            outer_buf.data() + outer_n.offset, outer_n.shape.rows, outer_n.shape.cols);
        Eigen::Map<const Eigen::MatrixXd> M_src(src.data(), r_slice, c_slice);
        M_out.middleRows(static_cast<Eigen::Index>(t) * r_slice, r_slice) += M_src;
    }
}

void ScanNode::forward(Workspace& ws) const {
    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, false);
    const Node& outer_c0 = graph->node(inputs[0]);
    const Node& inner_c = step_graph->node(inner_carry_id);
    const Node& next_c = step_graph->node(next_carry_out_id);
    const Eigen::Index n_c = inner_c.size();
    const Eigen::Index n_y = (step_y_out_id >= 0) ? step_graph->node(step_y_out_id).size() : 0;
    const std::size_t n_seq = inner_seq_ids.size();
    const std::size_t n_shared = inner_shared_ids.size();
    const auto& dyn = step_graph->dynamic_exec_order();

    Eigen::VectorXd c_curr = ws.val.segment(outer_c0.offset, n_c);
    for (int t = 0; t < num_steps; ++t) {
        Workspace& sws = tape[static_cast<std::size_t>(t)];
        for (std::size_t j = 0; j < n_shared; ++j) {
            const Node& outer_sh = graph->node(inputs[1 + n_seq + j]);
            const Node& inner_sh = step_graph->node(inner_shared_ids[j]);
            sws.val.segment(inner_sh.offset, inner_sh.size()) =
                ws.val.segment(outer_sh.offset, outer_sh.size());
        }
        sws.val.segment(inner_c.offset, n_c) = c_curr;
        for (std::size_t k = 0; k < n_seq; ++k) {
            const Node& outer_seq = graph->node(inputs[1 + k]);
            const Node& inner_seq = step_graph->node(inner_seq_ids[k]);
            extract_seq_slice(ws.val, outer_seq, seq_axes[k], t, inner_seq.shape,
                              sws.val.segment(inner_seq.offset, inner_seq.size()));
        }
        for (int id : dyn) {
            step_graph->node(id).forward(sws);
        }
        c_curr = sws.val.segment(next_c.offset, n_c);
        if (step_y_out_id >= 0) {
            const Node& y_n = step_graph->node(step_y_out_id);
            ws.val.segment(offset + n_c + static_cast<Eigen::Index>(t) * n_y, n_y) =
                sws.val.segment(y_n.offset, n_y);
        }
    }
    ws.val.segment(offset, n_c) = c_curr;
}

void ScanNode::backward(Workspace& ws) const {
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0) {
        const int tid_clr = omp_in_parallel() ? omp_get_thread_num() : 0;
        if (tid_clr < static_cast<int>(thread_tapes_.size())) {
            for (auto& sws : thread_tapes_[static_cast<std::size_t>(tid_clr)]) {
                sws.bar.setZero();
            }
        }
        return;
    }

    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, true, false);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int t = 0; t < num_steps; ++t) {
            tape[static_cast<std::size_t>(t)].val = tape0[static_cast<std::size_t>(t)].val;
        }
    }
    const Node& outer_c0 = graph->node(inputs[0]);
    const Node& inner_c = step_graph->node(inner_carry_id);
    const Node& next_c = step_graph->node(next_carry_out_id);
    const Eigen::Index n_c = inner_c.size();
    const Eigen::Index n_y = (step_y_out_id >= 0) ? step_graph->node(step_y_out_id).size() : 0;
    const std::size_t n_seq = inner_seq_ids.size();
    const std::size_t n_shared = inner_shared_ids.size();
    const auto& dyn = step_graph->dynamic_exec_order();

    Eigen::VectorXd c_bar = ws.bar.segment(offset, n_c);

    for (int t = num_steps - 1; t >= 0; --t) {
        Workspace& sws = tape[static_cast<std::size_t>(t)];
        sws.bar.segment(next_c.offset, n_c) += c_bar;
        if (step_y_out_id >= 0) {
            const Node& y_n = step_graph->node(step_y_out_id);
            sws.bar.segment(y_n.offset, n_y) +=
                ws.bar.segment(offset + n_c + static_cast<Eigen::Index>(t) * n_y, n_y);
        }

        for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
            if (step_graph->node(*it).requires_grad) {
                step_graph->node(*it).backward(sws);
            }
        }

        c_bar = sws.bar.segment(inner_c.offset, n_c);

        for (std::size_t k = 0; k < n_seq; ++k) {
            const Node& outer_seq = graph->node(inputs[1 + k]);
            if (outer_seq.requires_grad) {
                const Node& inner_seq = step_graph->node(inner_seq_ids[k]);
                accumulate_seq_slice(sws.bar.segment(inner_seq.offset, inner_seq.size()),
                                     inner_seq.shape, seq_axes[k], t, outer_seq, ws.bar);
            }
        }
        for (std::size_t j = 0; j < n_shared; ++j) {
            const Node& outer_sh = graph->node(inputs[1 + n_seq + j]);
            if (outer_sh.requires_grad) {
                const Node& inner_sh = step_graph->node(inner_shared_ids[j]);
                ws.bar.segment(outer_sh.offset, outer_sh.size()) +=
                    sws.bar.segment(inner_sh.offset, inner_sh.size());
            }
        }
    }

    if (outer_c0.requires_grad) {
        ws.bar.segment(outer_c0.offset, n_c) += c_bar;
    }
}

void ScanNode::forward_tangent(Workspace& ws) const {
    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, true);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int t = 0; t < num_steps; ++t) {
            tape[static_cast<std::size_t>(t)].val = tape0[static_cast<std::size_t>(t)].val;
        }
    }
    const Node& outer_c0 = graph->node(inputs[0]);
    const Node& inner_c = step_graph->node(inner_carry_id);
    const Node& next_c = step_graph->node(next_carry_out_id);
    const Eigen::Index n_c = inner_c.size();
    const Eigen::Index n_y = (step_y_out_id >= 0) ? step_graph->node(step_y_out_id).size() : 0;
    const std::size_t n_seq = inner_seq_ids.size();
    const std::size_t n_shared = inner_shared_ids.size();
    const Eigen::Index mem_sz = step_graph->total_memory_size();
    const auto& dyn = step_graph->dynamic_exec_order();

    Eigen::VectorXd c_dot = outer_c0.requires_grad
        ? Eigen::VectorXd(ws.dot.segment(outer_c0.offset, n_c))
        : Eigen::VectorXd::Zero(n_c);

    for (int t = 0; t < num_steps; ++t) {
        Workspace& sws = tape[static_cast<std::size_t>(t)];
        if (sws.dot.size() != mem_sz) {
            sws.dot.resize(mem_sz);
        }
        sws.dot.setZero();

        for (std::size_t j = 0; j < n_shared; ++j) {
            const Node& outer_sh = graph->node(inputs[1 + n_seq + j]);
            if (outer_sh.requires_grad) {
                const Node& inner_sh = step_graph->node(inner_shared_ids[j]);
                sws.dot.segment(inner_sh.offset, inner_sh.size()) =
                    ws.dot.segment(outer_sh.offset, outer_sh.size());
            }
        }

        sws.dot.segment(inner_c.offset, n_c) = c_dot;

        for (std::size_t k = 0; k < n_seq; ++k) {
            const Node& outer_seq = graph->node(inputs[1 + k]);
            if (outer_seq.requires_grad) {
                const Node& inner_seq = step_graph->node(inner_seq_ids[k]);
                extract_seq_slice(ws.dot, outer_seq, seq_axes[k], t, inner_seq.shape,
                                  sws.dot.segment(inner_seq.offset, inner_seq.size()));
            }
        }

        for (int id : dyn) {
            if (step_graph->node(id).requires_grad) {
                step_graph->node(id).forward_tangent(sws);
            }
        }

        c_dot = sws.dot.segment(next_c.offset, n_c);
        if (step_y_out_id >= 0) {
            const Node& y_n = step_graph->node(step_y_out_id);
            ws.dot.segment(offset + n_c + static_cast<Eigen::Index>(t) * n_y, n_y) =
                sws.dot.segment(y_n.offset, n_y);
        }
    }
    ws.dot.segment(offset, n_c) = c_dot;
}

void ScanNode::backward_hessian(Workspace& ws) const {
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 &&
        ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) {
        return;
    }

    const int outer_tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    auto& tape = ensure_tape(outer_tid, false, false);
    if (outer_tid > 0) {
        const auto& tape0 = thread_tapes_[0];
        for (int t = 0; t < num_steps; ++t) {
            tape[static_cast<std::size_t>(t)].val = tape0[static_cast<std::size_t>(t)].val;
            tape[static_cast<std::size_t>(t)].bar = tape0[static_cast<std::size_t>(t)].bar;
        }
    }
    const Node& outer_c0 = graph->node(inputs[0]);
    const Node& inner_c = step_graph->node(inner_carry_id);
    const Node& next_c = step_graph->node(next_carry_out_id);
    const Eigen::Index n_c = inner_c.size();
    const Eigen::Index n_y = (step_y_out_id >= 0) ? step_graph->node(step_y_out_id).size() : 0;
    const std::size_t n_seq = inner_seq_ids.size();
    const std::size_t n_shared = inner_shared_ids.size();
    const Eigen::Index mem_sz = step_graph->total_memory_size();
    const auto& dyn = step_graph->dynamic_exec_order();

    Eigen::VectorXd c_dot_bar = ws.dot_bar.segment(offset, n_c);

    for (int t = num_steps - 1; t >= 0; --t) {
        Workspace& sws = tape[static_cast<std::size_t>(t)];
        if (sws.dot_bar.size() != mem_sz) {
            sws.dot_bar.resize(mem_sz);
        }
        sws.dot_bar.setZero();

        sws.dot_bar.segment(next_c.offset, n_c) += c_dot_bar;
        if (step_y_out_id >= 0) {
            const Node& y_n = step_graph->node(step_y_out_id);
            sws.dot_bar.segment(y_n.offset, n_y) +=
                ws.dot_bar.segment(offset + n_c + static_cast<Eigen::Index>(t) * n_y, n_y);
        }

        for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
            if (step_graph->node(*it).requires_grad) {
                step_graph->node(*it).backward_hessian(sws);
            }
        }

        c_dot_bar = sws.dot_bar.segment(inner_c.offset, n_c);

        for (std::size_t k = 0; k < n_seq; ++k) {
            const Node& outer_seq = graph->node(inputs[1 + k]);
            if (outer_seq.requires_grad) {
                const Node& inner_seq = step_graph->node(inner_seq_ids[k]);
                accumulate_seq_slice(sws.dot_bar.segment(inner_seq.offset, inner_seq.size()),
                                     inner_seq.shape, seq_axes[k], t, outer_seq, ws.dot_bar);
            }
        }
        for (std::size_t j = 0; j < n_shared; ++j) {
            const Node& outer_sh = graph->node(inputs[1 + n_seq + j]);
            if (outer_sh.requires_grad) {
                const Node& inner_sh = step_graph->node(inner_shared_ids[j]);
                ws.dot_bar.segment(outer_sh.offset, outer_sh.size()) +=
                    sws.dot_bar.segment(inner_sh.offset, inner_sh.size());
            }
        }
    }

    if (outer_c0.requires_grad) {
        ws.dot_bar.segment(outer_c0.offset, n_c) += c_dot_bar;
    }
}

// ============================================================================
// 17. Operator Builder Functions (with CSE Hash-Consing & Constant Folding)
// ============================================================================
// ============================================================================
// 19. Fused CPU Neural Network Surrogate Implementation (MLP / ResNet / LN)
// ============================================================================
namespace {

template <typename Scalar>
inline void nn_act_val_deriv(NNActivation act, Scalar u, Scalar& y, Scalar& dy) {
    constexpr Scalar kZero = Scalar(0);
    constexpr Scalar kOne = Scalar(1);
    constexpr Scalar kHalf = Scalar(0.5);
    constexpr Scalar kInvSqrt2 = Scalar(0.70710678118654752440);
    constexpr Scalar kInvSqrt2Pi = Scalar(0.39894228040143267794);

    switch (act) {
        case NNActivation::Identity:
            y = u;
            dy = kOne;
            return;
        case NNActivation::Tanh: {
            const Scalar t = std::tanh(u);
            y = t;
            dy = kOne - t * t;
            return;
        }
        case NNActivation::SiLU: {
            // Numerically stable sigmoid
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            y = u * s;
            dy = s * (kOne + u * (kOne - s));
            return;
        }
        case NNActivation::GELU: {
            const Scalar cdf = kHalf * (kOne + std::erf(u * kInvSqrt2));
            const Scalar pdf = kInvSqrt2Pi * std::exp(-kHalf * u * u);
            y = u * cdf;
            dy = cdf + u * pdf;
            return;
        }
        case NNActivation::ReLU:
            if (u > kZero) {
                y = u;
                dy = kOne;
            } else {
                y = kZero;
                dy = kZero;
            }
            return;
        case NNActivation::Softplus: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            y = (u > kZero) ? (u + std::log1p(std::exp(-u))) : std::log1p(std::exp(u));
            dy = s;
            return;
        }
        case NNActivation::ELU:
            if (u >= kZero) {
                y = u;
                dy = kOne;
            } else {
                const Scalar e = std::exp(u);
                y = e - kOne;
                dy = e;
            }
            return;
        case NNActivation::Sin:
            y = std::sin(u);
            dy = std::cos(u);
            return;
        case NNActivation::Sigmoid: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            y = s;
            dy = s * (kOne - s);
            return;
        }
        case NNActivation::LeakyReLU: {
            constexpr Scalar kSlope = Scalar(0.01);
            if (u >= kZero) { y = u; dy = kOne; }
            else { y = kSlope * u; dy = kSlope; }
            return;
        }
        case NNActivation::Mish: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            const Scalar sp = (u > kZero) ? (u + std::log1p(std::exp(-u))) : std::log1p(std::exp(u));
            const Scalar t = std::tanh(sp);
            y = u * t;
            dy = t + u * (kOne - t * t) * s;
            return;
        }
        case NNActivation::SELU: {
            constexpr Scalar kLam = Scalar(1.0507009873554804934193349852946);
            constexpr Scalar kAlp = Scalar(1.6732632423543772848170429916717);
            if (u >= kZero) { y = kLam * u; dy = kLam; }
            else { const Scalar e = std::exp(u); y = kLam * kAlp * (e - kOne); dy = kLam * kAlp * e; }
            return;
        }
        case NNActivation::HardSwish: {
            if (u <= Scalar(-3)) { y = kZero; dy = kZero; }
            else if (u >= Scalar(3)) { y = u; dy = kOne; }
            else { y = u * (u + Scalar(3)) / Scalar(6); dy = (Scalar(2) * u + Scalar(3)) / Scalar(6); }
            return;
        }
    }
    y = u;
    dy = kOne;
}

template <typename Scalar>
inline void nn_act_jet2(NNActivation act, Scalar u, Scalar& dy, Scalar& d2y) {
    constexpr Scalar kZero = Scalar(0);
    constexpr Scalar kOne = Scalar(1);
    constexpr Scalar kTwo = Scalar(2);
    constexpr Scalar kHalf = Scalar(0.5);
    constexpr Scalar kInvSqrt2 = Scalar(0.70710678118654752440);
    constexpr Scalar kInvSqrt2Pi = Scalar(0.39894228040143267794);

    switch (act) {
        case NNActivation::Identity:
            dy = kOne;
            d2y = kZero;
            return;
        case NNActivation::Tanh: {
            const Scalar t = std::tanh(u);
            const Scalar dt = kOne - t * t;
            dy = dt;
            d2y = -kTwo * t * dt;
            return;
        }
        case NNActivation::SiLU: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            const Scalar one_m_s = kOne - s;
            dy = s * (kOne + u * one_m_s);
            d2y = s * one_m_s * (kTwo + u * (kOne - kTwo * s));
            return;
        }
        case NNActivation::GELU: {
            const Scalar cdf = kHalf * (kOne + std::erf(u * kInvSqrt2));
            const Scalar pdf = kInvSqrt2Pi * std::exp(-kHalf * u * u);
            dy = cdf + u * pdf;
            d2y = pdf * (kTwo - u * u);
            return;
        }
        case NNActivation::ReLU:
            dy = (u > kZero) ? kOne : kZero;
            d2y = kZero;
            return;
        case NNActivation::Softplus: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            dy = s;
            d2y = s * (kOne - s);
            return;
        }
        case NNActivation::ELU:
            if (u >= kZero) {
                dy = kOne;
                d2y = kZero;
            } else {
                const Scalar e = std::exp(u);
                dy = e;
                d2y = e;
            }
            return;
        case NNActivation::Sin:
            dy = std::cos(u);
            d2y = -std::sin(u);
            return;
        case NNActivation::Sigmoid: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            const Scalar ds = s * (kOne - s);
            dy = ds;
            d2y = ds * (kOne - kTwo * s);
            return;
        }
        case NNActivation::LeakyReLU: {
            constexpr Scalar kSlope = Scalar(0.01);
            dy = (u >= kZero) ? kOne : kSlope;
            d2y = kZero;
            return;
        }
        case NNActivation::Mish: {
            const Scalar s = (u >= kZero)
                ? (kOne / (kOne + std::exp(-u)))
                : (std::exp(u) / (kOne + std::exp(u)));
            const Scalar sp = (u > kZero) ? (u + std::log1p(std::exp(-u))) : std::log1p(std::exp(u));
            const Scalar t = std::tanh(sp);
            const Scalar sech2 = kOne - t * t;
            dy = t + u * sech2 * s;
            d2y = sech2 * s * (kTwo + u * (kOne - s) - kTwo * u * t * s);
            return;
        }
        case NNActivation::SELU: {
            constexpr Scalar kLam = Scalar(1.0507009873554804934193349852946);
            constexpr Scalar kAlp = Scalar(1.6732632423543772848170429916717);
            if (u >= kZero) { dy = kLam; d2y = kZero; }
            else { const Scalar e = std::exp(u); dy = kLam * kAlp * e; d2y = dy; }
            return;
        }
        case NNActivation::HardSwish: {
            if (u <= Scalar(-3)) { dy = kZero; d2y = kZero; }
            else if (u >= Scalar(3)) { dy = kOne; d2y = kZero; }
            else { dy = (kTwo * u + Scalar(3)) / Scalar(6); d2y = Scalar(1) / Scalar(3); }
            return;
        }
    }
    dy = kOne;
    d2y = kZero;
}

template <typename Scalar>
struct MLPLayerTraits;

template <>
struct MLPLayerTraits<double> {
    using Layer = MLPEngineData::LayerD;
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;
    static const std::vector<Layer>& layers(const MLPEngineData& d) { return d.layers_d; }
    static const Vec& in_mean(const MLPEngineData& d) { return d.in_mean_d; }
    static const Vec& in_inv_std(const MLPEngineData& d) { return d.in_inv_std_d; }
    static const Vec& out_mean(const MLPEngineData& d) { return d.out_mean_d; }
    static const Vec& out_std(const MLPEngineData& d) { return d.out_std_d; }
};

template <>
struct MLPLayerTraits<float> {
    using Layer = MLPEngineData::LayerF;
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    static const std::vector<Layer>& layers(const MLPEngineData& d) { return d.layers_f; }
    static const Vec& in_mean(const MLPEngineData& d) { return d.in_mean_f; }
    static const Vec& in_inv_std(const MLPEngineData& d) { return d.in_inv_std_f; }
    static const Vec& out_mean(const MLPEngineData& d) { return d.out_mean_f; }
    static const Vec& out_std(const MLPEngineData& d) { return d.out_std_f; }
};

template <typename Scalar>
struct MLPScratch {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    bool has_primal = false;
    bool has_tangent = false;
    Eigen::VectorXd cached_x;
    Eigen::VectorXd cached_xdot;

    // Primal buffers: H[0] is normalized input, H[l+1] is output of layer l
    std::vector<Mat> H;        // size L + 1
    std::vector<Mat> Z;        // size L (post-linear)
    std::vector<Mat> U;        // size L (pre-activation, post-LN)
    std::vector<Mat> LN_zhat;  // size L (only resized if layer_norm)
    std::vector<Vec> LN_s;     // size L (length B, only resized if layer_norm)

    // Tangent buffers
    std::vector<Mat> dH;       // size L + 1
    std::vector<Mat> dZ;       // size L
    std::vector<Mat> dU;       // size L
    std::vector<Mat> LN_dzhat; // size L
    std::vector<Vec> LN_p;     // size L (length B: (1/D) * sum(zhat * dZ))

    // Reverse scratch buffers (reused across layers to avoid heap allocations)
    Mat bar_cur;
    Mat bar_next;
    Mat bar_u;
    Mat bar_z;
    Mat dbar_cur;
    Mat dbar_next;
    Mat dbar_u;
    Mat dbar_z;

    void ensure_primal(const MLPEngineData& data, Eigen::Index B) {
        const auto& layers = MLPLayerTraits<Scalar>::layers(data);
        const std::size_t L = layers.size();
        if (H.size() == L + 1 && H[0].rows() == data.d_in && H[0].cols() == B) {
            bool ok = true;
            for (std::size_t l = 0; l < L; ++l) {
                if (U[l].rows() != layers[l].d_out || U[l].cols() != B) {
                    ok = false;
                    break;
                }
            }
            if (ok) return;
        }
        H.resize(L + 1);
        Z.resize(L);
        U.resize(L);
        LN_zhat.resize(L);
        LN_s.resize(L);
        H[0].resize(data.d_in, B);
        for (std::size_t l = 0; l < L; ++l) {
            const Eigen::Index dout = layers[l].d_out;
            U[l].resize(dout, B);
            H[l + 1].resize(dout, B);
            if (layers[l].layer_norm) {
                Z[l].resize(dout, B);
                LN_zhat[l].resize(dout, B);
                LN_s[l].resize(B);
            }
        }
    }

    void ensure_tangent(const MLPEngineData& data, Eigen::Index B) {
        const auto& layers = MLPLayerTraits<Scalar>::layers(data);
        const std::size_t L = layers.size();
        if (dH.size() == L + 1 && dH[0].cols() == B) return;
        dH.resize(L + 1);
        dZ.resize(L);
        dU.resize(L);
        LN_dzhat.resize(L);
        LN_p.resize(L);
        dH[0].resize(data.d_in, B);
        for (std::size_t l = 0; l < L; ++l) {
            const Eigen::Index dout = layers[l].d_out;
            dZ[l].resize(dout, B);
            dU[l].resize(dout, B);
            dH[l + 1].resize(dout, B);
            if (layers[l].layer_norm) {
                LN_dzhat[l].resize(dout, B);
                LN_p[l].resize(B);
            }
        }
    }
};

// Fast contiguous column-major GEMV helpers for B == 1 small/medium layers.
// Bypassing external BLAS (OpenBLAS/MKL) for tiny (<= 360x360) single-vector products
// eliminates 10-15 us/layer thread-pool barrier overhead while letting GCC/Clang/MSVC
// emit pure AVX2+FMA instructions on contiguous W.col(j).
template <typename Scalar>
inline void mlp_gemv_add_bias(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H,
                              Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Z) {
    const Eigen::Index d_out = W.rows();
    const Eigen::Index d_in = W.cols();
    const Eigen::Index B = H.cols();
    if (B == 1 && d_out * d_in <= 131072) {
        Scalar* z_ptr = Z.data();
        const Scalar* b_ptr = b.data();
        const Scalar* h_ptr = H.data();
        std::memcpy(z_ptr, b_ptr, static_cast<std::size_t>(d_out) * sizeof(Scalar));
        for (Eigen::Index j = 0; j < d_in; ++j) {
            const Scalar hj = h_ptr[j];
            const Scalar* w_col = W.data() + j * d_out;
            for (Eigen::Index i = 0; i < d_out; ++i) {
                z_ptr[i] += w_col[i] * hj;
            }
        }
    } else {
        Z.noalias() = W * H;
        Z.colwise() += b;
    }
}

template <typename Scalar>
inline void mlp_gemv_nobias(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W,
                            const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H,
                            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Z) {
    const Eigen::Index d_out = W.rows();
    const Eigen::Index d_in = W.cols();
    const Eigen::Index B = H.cols();
    if (B == 1 && d_out * d_in <= 131072) {
        Scalar* z_ptr = Z.data();
        const Scalar* h_ptr = H.data();
        std::memset(z_ptr, 0, static_cast<std::size_t>(d_out) * sizeof(Scalar));
        for (Eigen::Index j = 0; j < d_in; ++j) {
            const Scalar hj = h_ptr[j];
            const Scalar* w_col = W.data() + j * d_out;
            for (Eigen::Index i = 0; i < d_out; ++i) {
                z_ptr[i] += w_col[i] * hj;
            }
        }
    } else {
        Z.noalias() = W * H;
    }
}

template <typename Scalar>
inline void mlp_gemv_trans(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W,
                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Y,
                           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
    const Eigen::Index d_out = W.rows();
    const Eigen::Index d_in = W.cols();
    const Eigen::Index B = Y.cols();
    if (B == 1 && d_out * d_in <= 131072) {
        Scalar* x_ptr = X.data();
        const Scalar* y_ptr = Y.data();
        for (Eigen::Index j = 0; j < d_in; ++j) {
            const Scalar* w_col = W.data() + j * d_out;
            Scalar acc = Scalar(0);
            for (Eigen::Index i = 0; i < d_out; ++i) {
                acc += w_col[i] * y_ptr[i];
            }
            x_ptr[j] = acc;
        }
    } else {
        X.noalias() = W.transpose() * Y;
    }
}

// Fused single-pass (bias_add -> U, activation -> Hnext, optional residual += Hprev)
// Eliminates redundant Z->U copies and separate bias/residual RAM sweeps.
template <typename Scalar>
inline void mlp_fused_bias_act_res_fwd(
    NNActivation act,
    bool residual,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& bias,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Hprev,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Hnext) {
    const Eigen::Index D = U.rows();
    const Eigen::Index B = U.cols();
    const Eigen::Index n_tot = D * B;
    const bool use_omp = (!omp_in_parallel()) && (n_tot >= 16384) && (B >= 8) && (omp_get_max_threads() > 1);

    auto process_cols = [&](Eigen::Index b0, Eigen::Index cols) {
        const Eigen::Index len = D * cols;
        Scalar* u_ptr = U.data() + b0 * D;
        Scalar* h_ptr = Hnext.data() + b0 * D;
        const Scalar* prev_ptr = residual ? (Hprev.data() + b0 * D) : nullptr;
        const Scalar* b_ptr = bias.data();

        // Add bias per column into U (contiguous column-major, L1 cache resident)
        for (Eigen::Index c = 0; c < cols; ++c) {
            Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> uc(u_ptr + c * D, D);
            Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> bv(b_ptr, D);
            uc += bv;
        }

        Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> u_seg(u_ptr, len);
        Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> h_seg(h_ptr, len);
        if (act == NNActivation::Identity) {
            h_seg = u_seg;
        } else if (act == NNActivation::Tanh) {
            h_seg = u_seg.tanh();
        } else if (act == NNActivation::ReLU) {
            h_seg = u_seg.max(Scalar(0));
        } else if (act == NNActivation::Sin) {
            h_seg = u_seg.sin();
        } else if (act == NNActivation::SiLU) {
            h_seg = u_seg / (Scalar(1) + (-u_seg).min(Scalar(80)).exp());
        } else if (act == NNActivation::Softplus) {
            h_seg = u_seg.max(Scalar(0)) + (-u_seg.abs()).exp().log1p();
        } else {
            for (Eigen::Index k = 0; k < len; ++k) {
                Scalar y_val, dy_val;
                nn_act_val_deriv(act, u_ptr[k], y_val, dy_val);
                h_ptr[k] = y_val;
            }
        }
        if (residual) {
            Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> p_seg(prev_ptr, len);
            h_seg += p_seg;
        }
    };

    if (!use_omp) {
        process_cols(0, B);
        return;
    }
    const int n_threads = std::min<int>(omp_get_max_threads(), static_cast<int>(B));
    const Eigen::Index chunk = (B + n_threads - 1) / n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int tid = 0; tid < n_threads; ++tid) {
        const Eigen::Index b0 = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index cols = std::min(chunk, B - b0);
        if (cols > 0) process_cols(b0, cols);
    }
}

// Vectorized & OpenMP-accelerated activation backward VJP: bar_u = bar_cur * phi'(U)
template <typename Scalar>
inline void mlp_apply_activation_bwd(
    NNActivation act,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& bar_cur,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& bar_u) {
    const Eigen::Index n_tot = U.size();
    if (act == NNActivation::Identity) {
        bar_u = bar_cur;
        return;
    }
    const bool use_omp = (!omp_in_parallel()) && (n_tot >= 16384) && (omp_get_max_threads() > 1);
    const Scalar* u_base = U.data();
    const Scalar* bc_base = bar_cur.data();
    Scalar* bu_base = bar_u.data();

    auto process_range = [&](Eigen::Index i0, Eigen::Index len) {
        Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> u_seg(u_base + i0, len);
        Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> bc_seg(bc_base + i0, len);
        Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> bu_seg(bu_base + i0, len);
        if (act == NNActivation::SiLU) {
            // s = 1 / (1 + exp(-u)), dy = s * (1 + u * (1 - s))
            auto s = (Scalar(1) + (-u_seg).min(Scalar(80)).exp()).inverse();
            bu_seg = bc_seg * s * (Scalar(1) + u_seg * (Scalar(1) - s));
        } else if (act == NNActivation::Tanh) {
            auto t_val = u_seg.tanh();
            bu_seg = bc_seg * (Scalar(1) - t_val * t_val);
        } else if (act == NNActivation::ReLU) {
            bu_seg = (u_seg > Scalar(0)).select(bc_seg, Scalar(0));
        } else if (act == NNActivation::Sin) {
            bu_seg = bc_seg * u_seg.cos();
        } else if (act == NNActivation::Softplus) {
            auto s = (Scalar(1) + (-u_seg).min(Scalar(80)).exp()).inverse();
            bu_seg = bc_seg * s;
        } else {
            for (Eigen::Index k = 0; k < len; ++k) {
                Scalar y_val, dy_val;
                nn_act_val_deriv(act, u_base[i0 + k], y_val, dy_val);
                bu_base[i0 + k] = bc_base[i0 + k] * dy_val;
            }
        }
    };

    if (!use_omp) {
        process_range(0, n_tot);
        return;
    }
    const int n_threads = omp_get_max_threads();
    const Eigen::Index chunk = (n_tot + n_threads - 1) / n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int tid = 0; tid < n_threads; ++tid) {
        const Eigen::Index i0 = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index len = std::min(chunk, n_tot - i0);
        if (len > 0) process_range(i0, len);
    }
}

// Forward-only L3-cache ping-pong scratchpad for CompiledMLP::eval (zero malloc/free per call,
// only 2 matrices allocated instead of 3*(L+1), keeping large batches inside L3 cache).
template <typename Scalar>
struct MLPEvalPingPong {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Mat buf_a;
    Mat buf_b;
    Mat buf_ln;
};

template <typename Scalar>
inline void mlp_apply_activation_fwd(NNActivation act,
                                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U,
                                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Hnext);

template <typename Scalar>
void mlp_run_eval_only(const MLPEngineData& data,
                       const Eigen::Map<const Eigen::MatrixXd>& X_in,
                       Eigen::Map<Eigen::MatrixXd>& Y_out,
                       MLPEvalPingPong<Scalar>& pp) {
    const auto& layers = MLPLayerTraits<Scalar>::layers(data);
    const std::size_t L = layers.size();
    const Eigen::Index B = X_in.cols();

    pp.buf_a.resize(data.d_in, B);
    pp.buf_a = X_in.cast<Scalar>();
    if (data.has_in_norm) {
        const auto& mu = MLPLayerTraits<Scalar>::in_mean(data);
        const auto& inv_s = MLPLayerTraits<Scalar>::in_inv_std(data);
        pp.buf_a.colwise() -= mu;
        pp.buf_a.array().colwise() *= inv_s.array();
    }

    for (std::size_t l = 0; l < L; ++l) {
        const auto& ly = layers[l];
        pp.buf_b.resize(ly.d_out, B);

        if (!ly.layer_norm) {
            // Fast path: GEMM into buf_b, then single-pass fused (bias + activation + residual)
            mlp_gemv_nobias<Scalar>(ly.W, pp.buf_a, pp.buf_b);
            const Eigen::Index D = ly.d_out;
            const Eigen::Index n_tot = D * B;
            const bool use_omp = (!omp_in_parallel()) && (n_tot >= 16384) && (B >= 8) && (omp_get_max_threads() > 1);

            auto fuse_cols = [&](Eigen::Index b0, Eigen::Index cols) {
                const Eigen::Index len = D * cols;
                Scalar* z_ptr = pp.buf_b.data() + b0 * D;
                const Scalar* prev_ptr = ly.residual ? (pp.buf_a.data() + b0 * D) : nullptr;
                const Scalar* b_ptr = ly.b.data();
                for (Eigen::Index c = 0; c < cols; ++c) {
                    Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> zc(z_ptr + c * D, D);
                    Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> bv(b_ptr, D);
                    zc += bv;
                }
                Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> z_seg(z_ptr, len);
                if (ly.activation == NNActivation::Tanh) {
                    z_seg = z_seg.tanh();
                } else if (ly.activation == NNActivation::ReLU) {
                    z_seg = z_seg.max(Scalar(0));
                } else if (ly.activation == NNActivation::Sin) {
                    z_seg = z_seg.sin();
                } else if (ly.activation == NNActivation::SiLU) {
                    z_seg = z_seg / (Scalar(1) + (-z_seg).min(Scalar(80)).exp());
                } else if (ly.activation == NNActivation::Softplus) {
                    z_seg = z_seg.max(Scalar(0)) + (-z_seg.abs()).exp().log1p();
                } else if (ly.activation != NNActivation::Identity) {
                    for (Eigen::Index k = 0; k < len; ++k) {
                        Scalar y_val, dy_val;
                        nn_act_val_deriv(ly.activation, z_ptr[k], y_val, dy_val);
                        z_ptr[k] = y_val;
                    }
                }
                if (ly.residual) {
                    Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> p_seg(prev_ptr, len);
                    z_seg += p_seg;
                }
            };

            if (!use_omp) {
                fuse_cols(0, B);
            } else {
                const int n_threads = std::min<int>(omp_get_max_threads(), static_cast<int>(B));
                const Eigen::Index chunk = (B + n_threads - 1) / n_threads;
                #pragma omp parallel for num_threads(n_threads) schedule(static)
                for (int tid = 0; tid < n_threads; ++tid) {
                    const Eigen::Index b0 = static_cast<Eigen::Index>(tid) * chunk;
                    const Eigen::Index cols = std::min(chunk, B - b0);
                    if (cols > 0) fuse_cols(b0, cols);
                }
            }
        } else {
            // LayerNorm path
            mlp_gemv_add_bias<Scalar>(ly.W, ly.b, pp.buf_a, pp.buf_b);
            const Eigen::Index D = ly.d_out;
            const Scalar inv_D = Scalar(1) / static_cast<Scalar>(D);
            const bool has_gamma = (ly.ln_gamma.size() == D);
            const bool has_beta = (ly.ln_beta.size() == D);
            pp.buf_ln.resize(D, B);
            for (Eigen::Index b = 0; b < B; ++b) {
                const Scalar mean = pp.buf_b.col(b).sum() * inv_D;
                Scalar var = Scalar(0);
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar c = pp.buf_b(i, b) - mean;
                    var += c * c;
                }
                var *= inv_D;
                const Scalar s = Scalar(1) / std::sqrt(var + ly.ln_eps);
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar zh = (pp.buf_b(i, b) - mean) * s;
                    const Scalar g = has_gamma ? ly.ln_gamma(i) : Scalar(1);
                    const Scalar bt = has_beta ? ly.ln_beta(i) : Scalar(0);
                    pp.buf_ln(i, b) = g * zh + bt;
                }
            }
            mlp_apply_activation_fwd<Scalar>(ly.activation, pp.buf_ln, pp.buf_b);
            if (ly.residual) {
                pp.buf_b.noalias() += pp.buf_a;
            }
        }
        pp.buf_a.swap(pp.buf_b);
    }

    if (data.has_out_norm) {
        const auto& out_mu = MLPLayerTraits<Scalar>::out_mean(data);
        const auto& out_sd = MLPLayerTraits<Scalar>::out_std(data);
        for (Eigen::Index b = 0; b < B; ++b) {
            for (Eigen::Index i = 0; i < data.d_out; ++i) {
                Y_out(i, b) = static_cast<double>(out_mu(i) + out_sd(i) * pp.buf_a(i, b));
            }
        }
    } else {
        Y_out = pp.buf_a.template cast<double>();
    }
}

// Vectorized & OpenMP-accelerated activation forward pass
template <typename Scalar>
inline void mlp_apply_activation_fwd(NNActivation act,
                                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U,
                                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Hnext) {
    const Eigen::Index n_tot = U.size();
    const bool use_omp = (!omp_in_parallel()) && (n_tot >= 16384) && (omp_get_max_threads() > 1);
    if (act == NNActivation::Identity) {
        Hnext = U;
        return;
    }
    if (!use_omp) {
        if (act == NNActivation::Tanh) {
            Hnext.array() = U.array().tanh();
            return;
        }
        if (act == NNActivation::ReLU) {
            Hnext.array() = U.array().max(Scalar(0));
            return;
        }
        if (act == NNActivation::Sin) {
            Hnext.array() = U.array().sin();
            return;
        }
        if (act == NNActivation::SiLU) {
            // Clamp exponent argument for overflow-free SIMD evaluation
            auto u = U.array();
            Hnext.array() = u / (Scalar(1) + (-u).min(Scalar(80)).exp());
            return;
        }
        const Scalar* u_ptr = U.data();
        Scalar* h_ptr = Hnext.data();
        for (Eigen::Index i = 0; i < n_tot; ++i) {
            Scalar y_val, dy_val;
            nn_act_val_deriv(act, u_ptr[i], y_val, dy_val);
            h_ptr[i] = y_val;
        }
        return;
    }

    // Parallel SIMD chunks across OpenMP threads for large batches (e.g. 512 x 1024)
    const int n_threads = omp_get_max_threads();
    const Eigen::Index chunk = (n_tot + n_threads - 1) / n_threads;
    const Scalar* u_base = U.data();
    Scalar* h_base = Hnext.data();
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int tid = 0; tid < n_threads; ++tid) {
        const Eigen::Index i0 = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index len = std::min(chunk, n_tot - i0);
        if (len <= 0) continue;
        Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>> u_seg(u_base + i0, len);
        Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>> h_seg(h_base + i0, len);
        if (act == NNActivation::Tanh) {
            h_seg = u_seg.tanh();
        } else if (act == NNActivation::ReLU) {
            h_seg = u_seg.max(Scalar(0));
        } else if (act == NNActivation::Sin) {
            h_seg = u_seg.sin();
        } else if (act == NNActivation::SiLU) {
            h_seg = u_seg / (Scalar(1) + (-u_seg).min(Scalar(80)).exp());
        } else {
            for (Eigen::Index k = 0; k < len; ++k) {
                Scalar y_val, dy_val;
                nn_act_val_deriv(act, u_base[i0 + k], y_val, dy_val);
                h_base[i0 + k] = y_val;
            }
        }
    }
}

template <typename Scalar>
void mlp_run_primal(const MLPEngineData& data,
                    const Eigen::Map<const Eigen::MatrixXd>& X_in,
                    Eigen::Map<Eigen::MatrixXd>& Y_out,
                    MLPScratch<Scalar>& sc) {
    const auto& layers = MLPLayerTraits<Scalar>::layers(data);
    const std::size_t L = layers.size();
    const Eigen::Index B = X_in.cols();
    sc.ensure_primal(data, B);

    // Input conversion + optional affine normalization
    sc.H[0] = X_in.cast<Scalar>();
    if (data.has_in_norm) {
        const auto& mu = MLPLayerTraits<Scalar>::in_mean(data);
        const auto& inv_s = MLPLayerTraits<Scalar>::in_inv_std(data);
        sc.H[0].colwise() -= mu;
        sc.H[0].array().colwise() *= inv_s.array();
    }

    for (std::size_t l = 0; l < L; ++l) {
        const auto& ly = layers[l];
        auto& Ul = sc.U[l];
        auto& Hnext = sc.H[l + 1];

        if (!ly.layer_norm) {
            // Fast fused path: GEMM directly into Ul, then single-pass (bias + act + residual)
            mlp_gemv_nobias<Scalar>(ly.W, sc.H[l], Ul);
            mlp_fused_bias_act_res_fwd<Scalar>(ly.activation, ly.residual, ly.b, Ul, sc.H[l], Hnext);
        } else {
            auto& Zl = sc.Z[l];
            mlp_gemv_add_bias<Scalar>(ly.W, ly.b, sc.H[l], Zl);
            const Eigen::Index D = ly.d_out;
            const Scalar inv_D = Scalar(1) / static_cast<Scalar>(D);
            const bool has_gamma = (ly.ln_gamma.size() == D);
            const bool has_beta = (ly.ln_beta.size() == D);
            for (Eigen::Index b = 0; b < B; ++b) {
                const Scalar mean = Zl.col(b).sum() * inv_D;
                Scalar var = Scalar(0);
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar c = Zl(i, b) - mean;
                    var += c * c;
                }
                var *= inv_D;
                const Scalar s = Scalar(1) / std::sqrt(var + ly.ln_eps);
                sc.LN_s[l](b) = s;
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar zh = (Zl(i, b) - mean) * s;
                    sc.LN_zhat[l](i, b) = zh;
                    const Scalar g = has_gamma ? ly.ln_gamma(i) : Scalar(1);
                    const Scalar bt = has_beta ? ly.ln_beta(i) : Scalar(0);
                    Ul(i, b) = g * zh + bt;
                }
            }
            mlp_apply_activation_fwd<Scalar>(ly.activation, Ul, Hnext);
            if (ly.residual) {
                Hnext.noalias() += sc.H[l];
            }
        }
    }

    // Output denormalization + write to Y_out
    if (data.has_out_norm) {
        const auto& out_mu = MLPLayerTraits<Scalar>::out_mean(data);
        const auto& out_sd = MLPLayerTraits<Scalar>::out_std(data);
        for (Eigen::Index b = 0; b < B; ++b) {
            for (Eigen::Index i = 0; i < data.d_out; ++i) {
                Y_out(i, b) = static_cast<double>(out_mu(i) + out_sd(i) * sc.H[L](i, b));
            }
        }
    } else {
        Y_out = sc.H[L].template cast<double>();
    }
}

template <typename Scalar>
void mlp_run_backward(const MLPEngineData& data,
                      const MLPScratch<Scalar>& primal,
                      const Eigen::Map<const Eigen::MatrixXd>& Y_bar,
                      Eigen::Map<Eigen::MatrixXd>& X_bar,
                      MLPScratch<Scalar>& work) {
    const auto& layers = MLPLayerTraits<Scalar>::layers(data);
    const std::size_t L = layers.size();
    const Eigen::Index B = Y_bar.cols();

    work.bar_cur = Y_bar.cast<Scalar>();
    if (data.has_out_norm) {
        const auto& out_sd = MLPLayerTraits<Scalar>::out_std(data);
        work.bar_cur.array().colwise() *= out_sd.array();
    }

    for (std::size_t rev = 0; rev < L; ++rev) {
        const std::size_t l = L - 1 - rev;
        const auto& ly = layers[l];
        const auto& Ul = primal.U[l];

        // Vectorized + OpenMP-accelerated Activation VJP: bar_u = bar_cur * phi'(U_l)
        work.bar_u.resize(ly.d_out, B);
        mlp_apply_activation_bwd<Scalar>(ly.activation, Ul, work.bar_cur, work.bar_u);

        // LayerNorm VJP: bar_u -> bar_z
        const auto* bz_mat = &work.bar_u;
        if (ly.layer_norm) {
            const Eigen::Index D = ly.d_out;
            const Scalar inv_D = Scalar(1) / static_cast<Scalar>(D);
            const bool has_gamma = (ly.ln_gamma.size() == D);
            work.bar_z.resize(D, B);
            for (Eigen::Index b = 0; b < B; ++b) {
                const Scalar s = primal.LN_s[l](b);
                Scalar mg = Scalar(0);
                Scalar mgz = Scalar(0);
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar g = (has_gamma ? ly.ln_gamma(i) : Scalar(1)) * work.bar_u(i, b);
                    mg += g;
                    mgz += g * primal.LN_zhat[l](i, b);
                }
                mg *= inv_D;
                mgz *= inv_D;
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar g = (has_gamma ? ly.ln_gamma(i) : Scalar(1)) * work.bar_u(i, b);
                    work.bar_z(i, b) = s * (g - mg - primal.LN_zhat[l](i, b) * mgz);
                }
            }
            bz_mat = &work.bar_z;
        }

        // Linear VJP: bar_next = W_l^T * bar_z (+ bar_cur if residual)
        work.bar_next.resize(ly.d_in, B);
        mlp_gemv_trans<Scalar>(ly.W, *bz_mat, work.bar_next);
        if (ly.residual) {
            work.bar_next.noalias() += work.bar_cur;
        }
        work.bar_cur.swap(work.bar_next);
    }

    if (data.has_in_norm) {
        const auto& inv_s = MLPLayerTraits<Scalar>::in_inv_std(data);
        work.bar_cur.array().colwise() *= inv_s.array();
    }
    X_bar.noalias() += work.bar_cur.template cast<double>();
}

template <typename Scalar>
void mlp_run_tangent(const MLPEngineData& data,
                     const MLPScratch<Scalar>& primal,
                     const Eigen::Map<const Eigen::MatrixXd>& X_dot,
                     Eigen::Map<Eigen::MatrixXd>& Y_dot,
                     MLPScratch<Scalar>& tang) {
    const auto& layers = MLPLayerTraits<Scalar>::layers(data);
    const std::size_t L = layers.size();
    const Eigen::Index B = X_dot.cols();
    tang.ensure_tangent(data, B);

    tang.dH[0] = X_dot.cast<Scalar>();
    if (data.has_in_norm) {
        const auto& inv_s = MLPLayerTraits<Scalar>::in_inv_std(data);
        tang.dH[0].array().colwise() *= inv_s.array();
    }

    for (std::size_t l = 0; l < L; ++l) {
        const auto& ly = layers[l];
        auto& dZl = tang.dZ[l];
        auto& dUl = tang.dU[l];
        auto& dHnext = tang.dH[l + 1];

        mlp_gemv_nobias<Scalar>(ly.W, tang.dH[l], dZl);

        if (ly.layer_norm) {
            const Eigen::Index D = ly.d_out;
            const Scalar inv_D = Scalar(1) / static_cast<Scalar>(D);
            const bool has_gamma = (ly.ln_gamma.size() == D);
            for (Eigen::Index b = 0; b < B; ++b) {
                const Scalar s = primal.LN_s[l](b);
                const Scalar dmean = dZl.col(b).sum() * inv_D;
                Scalar p = Scalar(0);
                for (Eigen::Index i = 0; i < D; ++i) {
                    p += primal.LN_zhat[l](i, b) * dZl(i, b);
                }
                p *= inv_D;
                tang.LN_p[l](b) = p;
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar dzh = s * (dZl(i, b) - dmean - primal.LN_zhat[l](i, b) * p);
                    tang.LN_dzhat[l](i, b) = dzh;
                    const Scalar g = has_gamma ? ly.ln_gamma(i) : Scalar(1);
                    dUl(i, b) = g * dzh;
                }
            }
        } else {
            dUl = dZl;
        }

        if (ly.activation == NNActivation::Identity) {
            dHnext = dUl;
        } else {
            const Eigen::Index n_tot = primal.U[l].size();
            const Scalar* u_ptr = primal.U[l].data();
            const Scalar* du_ptr = dUl.data();
            Scalar* dh_ptr = dHnext.data();
            for (Eigen::Index i = 0; i < n_tot; ++i) {
                Scalar y_val, dy_val;
                nn_act_val_deriv(ly.activation, u_ptr[i], y_val, dy_val);
                dh_ptr[i] = dy_val * du_ptr[i];
            }
        }

        if (ly.residual) {
            dHnext.noalias() += tang.dH[l];
        }
    }

    if (data.has_out_norm) {
        const auto& out_sd = MLPLayerTraits<Scalar>::out_std(data);
        for (Eigen::Index b = 0; b < B; ++b) {
            for (Eigen::Index i = 0; i < data.d_out; ++i) {
                Y_dot(i, b) = static_cast<double>(out_sd(i) * tang.dH[L](i, b));
            }
        }
    } else {
        Y_dot = tang.dH[L].template cast<double>();
    }
}

template <typename Scalar>
void mlp_run_hessian(const MLPEngineData& data,
                     const MLPScratch<Scalar>& primal,
                     const MLPScratch<Scalar>& tang,
                     const Eigen::Map<const Eigen::MatrixXd>& Y_bar,
                     const Eigen::Map<const Eigen::MatrixXd>& Y_dbar,
                     Eigen::Map<Eigen::MatrixXd>& X_dbar,
                     MLPScratch<Scalar>& work) {
    const auto& layers = MLPLayerTraits<Scalar>::layers(data);
    const std::size_t L = layers.size();
    const Eigen::Index B = Y_bar.cols();

    work.bar_cur = Y_bar.cast<Scalar>();
    work.dbar_cur = Y_dbar.cast<Scalar>();
    if (data.has_out_norm) {
        const auto& out_sd = MLPLayerTraits<Scalar>::out_std(data);
        work.bar_cur.array().colwise() *= out_sd.array();
        work.dbar_cur.array().colwise() *= out_sd.array();
    }

    for (std::size_t rev = 0; rev < L; ++rev) {
        const std::size_t l = L - 1 - rev;
        const auto& ly = layers[l];
        const auto& Ul = primal.U[l];
        const auto& dUl = tang.dU[l];

        // Activation Forward-over-Reverse:
        //   bar_u  = bar_cur * phi'(U)
        //   dbar_u = dbar_cur * phi'(U) + bar_cur * phi''(U) * dU
        work.bar_u.resize(ly.d_out, B);
        work.dbar_u.resize(ly.d_out, B);
        if (ly.activation == NNActivation::Identity) {
            work.bar_u = work.bar_cur;
            work.dbar_u = work.dbar_cur;
        } else {
            const Eigen::Index n_tot = Ul.size();
            const Scalar* u_ptr = Ul.data();
            const Scalar* du_ptr = dUl.data();
            const Scalar* bc_ptr = work.bar_cur.data();
            const Scalar* dbc_ptr = work.dbar_cur.data();
            Scalar* bu_ptr = work.bar_u.data();
            Scalar* dbu_ptr = work.dbar_u.data();
            for (Eigen::Index i = 0; i < n_tot; ++i) {
                Scalar dy_val, d2y_val;
                nn_act_jet2(ly.activation, u_ptr[i], dy_val, d2y_val);
                bu_ptr[i] = bc_ptr[i] * dy_val;
                dbu_ptr[i] = dbc_ptr[i] * dy_val + bc_ptr[i] * d2y_val * du_ptr[i];
            }
        }

        // LayerNorm Forward-over-Reverse: (bar_u, dbar_u) -> (bar_z, dbar_z)
        const auto* bz_mat = &work.bar_u;
        const auto* dbz_mat = &work.dbar_u;
        if (ly.layer_norm) {
            const Eigen::Index D = ly.d_out;
            const Scalar inv_D = Scalar(1) / static_cast<Scalar>(D);
            const bool has_gamma = (ly.ln_gamma.size() == D);
            work.bar_z.resize(D, B);
            work.dbar_z.resize(D, B);
            for (Eigen::Index b = 0; b < B; ++b) {
                const Scalar s = primal.LN_s[l](b);
                const Scalar p = tang.LN_p[l](b);
                Scalar mg = Scalar(0), mgz = Scalar(0);
                Scalar dmg = Scalar(0), dmgz = Scalar(0);
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar gam = has_gamma ? ly.ln_gamma(i) : Scalar(1);
                    const Scalar g = gam * work.bar_u(i, b);
                    const Scalar dg = gam * work.dbar_u(i, b);
                    const Scalar zh = primal.LN_zhat[l](i, b);
                    const Scalar dzh = tang.LN_dzhat[l](i, b);
                    mg += g;
                    mgz += g * zh;
                    dmg += dg;
                    dmgz += dg * zh + g * dzh;
                }
                mg *= inv_D;
                mgz *= inv_D;
                dmg *= inv_D;
                dmgz *= inv_D;
                for (Eigen::Index i = 0; i < D; ++i) {
                    const Scalar gam = has_gamma ? ly.ln_gamma(i) : Scalar(1);
                    const Scalar g = gam * work.bar_u(i, b);
                    const Scalar dg = gam * work.dbar_u(i, b);
                    const Scalar zh = primal.LN_zhat[l](i, b);
                    const Scalar dzh = tang.LN_dzhat[l](i, b);
                    const Scalar bz = s * (g - mg - zh * mgz);
                    work.bar_z(i, b) = bz;
                    work.dbar_z(i, b) = -s * p * bz + s * (dg - dmg - dzh * mgz - zh * dmgz);
                }
            }
            bz_mat = &work.bar_z;
            dbz_mat = &work.dbar_z;
        }

        // Linear Forward-over-Reverse
        work.bar_next.resize(ly.d_in, B);
        work.dbar_next.resize(ly.d_in, B);
        mlp_gemv_trans<Scalar>(ly.W, *bz_mat, work.bar_next);
        mlp_gemv_trans<Scalar>(ly.W, *dbz_mat, work.dbar_next);
        if (ly.residual) {
            work.bar_next.noalias() += work.bar_cur;
            work.dbar_next.noalias() += work.dbar_cur;
        }
        work.bar_cur.swap(work.bar_next);
        work.dbar_cur.swap(work.dbar_next);
    }

    if (data.has_in_norm) {
        const auto& inv_s = MLPLayerTraits<Scalar>::in_inv_std(data);
        work.dbar_cur.array().colwise() *= inv_s.array();
    }
    X_dbar.noalias() += work.dbar_cur.template cast<double>();
}

} // namespace

std::shared_ptr<const MLPEngineData> MLPEngineData::create(
    const std::vector<MLPLayerSpec>& layers,
    bool use_float32,
    const Eigen::VectorXd& in_mean,
    const Eigen::VectorXd& in_inv_std,
    const Eigen::VectorXd& out_mean,
    const Eigen::VectorXd& out_std) {
    if (layers.empty()) {
        throw std::invalid_argument("MLP requires at least one layer.");
    }
    auto d = std::make_shared<MLPEngineData>();
    d->use_float32 = use_float32;
    d->d_in = layers.front().W.cols();
    d->d_out = layers.back().W.rows();

    Eigen::Index prev_dim = d->d_in;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& spec = layers[l];
        if (spec.W.rows() <= 0 || spec.W.cols() <= 0) {
            throw std::invalid_argument("MLP layer weight matrix W must be non-empty.");
        }
        if (spec.W.cols() != prev_dim) {
            throw std::invalid_argument("MLP layer dimension mismatch between consecutive layers.");
        }
        if (spec.b.size() != spec.W.rows()) {
            throw std::invalid_argument("MLP layer bias b size must match W.rows().");
        }
        if (spec.residual && spec.W.rows() != spec.W.cols()) {
            throw std::invalid_argument("MLP residual skip connection requires W.rows() == W.cols().");
        }
        if (spec.layer_norm) {
            if (spec.ln_gamma.size() > 0 && spec.ln_gamma.size() != spec.W.rows()) {
                throw std::invalid_argument("MLP LayerNorm gamma size must match W.rows().");
            }
            if (spec.ln_beta.size() > 0 && spec.ln_beta.size() != spec.W.rows()) {
                throw std::invalid_argument("MLP LayerNorm beta size must match W.rows().");
            }
        }
        prev_dim = spec.W.rows();
    }

    if (in_mean.size() > 0 || in_inv_std.size() > 0) {
        if (in_mean.size() != d->d_in || in_inv_std.size() != d->d_in) {
            throw std::invalid_argument("MLP in_mean and in_inv_std must both have size d_in.");
        }
        d->has_in_norm = true;
        d->in_mean_d = in_mean;
        d->in_inv_std_d = in_inv_std;
        if (use_float32) {
            d->in_mean_f = in_mean.cast<float>();
            d->in_inv_std_f = in_inv_std.cast<float>();
        }
    }
    if (out_mean.size() > 0 || out_std.size() > 0) {
        if (out_mean.size() != d->d_out || out_std.size() != d->d_out) {
            throw std::invalid_argument("MLP out_mean and out_std must both have size d_out.");
        }
        d->has_out_norm = true;
        d->out_mean_d = out_mean;
        d->out_std_d = out_std;
        if (use_float32) {
            d->out_mean_f = out_mean.cast<float>();
            d->out_std_f = out_std.cast<float>();
        }
    }

    if (use_float32) {
        d->layers_f.reserve(layers.size());
        for (const auto& spec : layers) {
            LayerF lf;
            lf.W = spec.W.cast<float>();
            lf.b = spec.b.cast<float>();
            lf.activation = spec.activation;
            lf.residual = spec.residual;
            lf.layer_norm = spec.layer_norm;
            if (spec.ln_gamma.size() > 0) lf.ln_gamma = spec.ln_gamma.cast<float>();
            if (spec.ln_beta.size() > 0) lf.ln_beta = spec.ln_beta.cast<float>();
            lf.ln_eps = static_cast<float>(spec.ln_eps);
            lf.d_in = spec.W.cols();
            lf.d_out = spec.W.rows();
            d->layers_f.push_back(std::move(lf));
        }
    } else {
        d->layers_d.reserve(layers.size());
        for (const auto& spec : layers) {
            LayerD ld;
            ld.W = spec.W;
            ld.b = spec.b;
            ld.activation = spec.activation;
            ld.residual = spec.residual;
            ld.layer_norm = spec.layer_norm;
            ld.ln_gamma = spec.ln_gamma;
            ld.ln_beta = spec.ln_beta;
            ld.ln_eps = spec.ln_eps;
            ld.d_in = spec.W.cols();
            ld.d_out = spec.W.rows();
            d->layers_d.push_back(std::move(ld));
        }
    }
    return d;
}

CompiledMLP::CompiledMLP(const std::vector<MLPLayerSpec>& layers,
                         bool use_float32,
                         const Eigen::VectorXd& in_mean,
                         const Eigen::VectorXd& in_inv_std,
                         const Eigen::VectorXd& out_mean,
                         const Eigen::VectorXd& out_std,
                         std::string name_)
    : data(MLPEngineData::create(layers, use_float32, in_mean, in_inv_std, out_mean, out_std)),
      name(std::move(name_)) {}

Eigen::MatrixXd CompiledMLP::eval(const Eigen::MatrixXd& X, int num_threads) const {
    if (X.rows() != data->d_in) {
        throw std::invalid_argument("CompiledMLP::eval: X.rows() does not match d_in.");
    }
    const Eigen::Index B = X.cols();
    Eigen::MatrixXd Y(data->d_out, B);
    if (B == 0) return Y;

#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_MKL_ALL)
    // External BLAS (OpenBLAS / Intel MKL) parallelizes large GEMMs internally.
    // Skip outer OpenMP column chunking to prevent nested thread oversubscription.
    (void)num_threads;
    const int max_t = 1;
    const bool use_omp = false;
#else
    const int max_t = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const bool use_omp = (!omp_in_parallel()) && (max_t > 1) && (B >= 64);
#endif

    if (!use_omp) {
        Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), data->d_in, B);
        Eigen::Map<Eigen::MatrixXd> Ym(Y.data(), data->d_out, B);
        if (data->use_float32) {
            static thread_local MLPEvalPingPong<float> pp_f;
            mlp_run_eval_only<float>(*data, Xm, Ym, pp_f);
        } else {
            static thread_local MLPEvalPingPong<double> pp_d;
            mlp_run_eval_only<double>(*data, Xm, Ym, pp_d);
        }
        return Y;
    }

    const int n_chunks = std::min<int>(max_t, static_cast<int>((B + 31) / 32));
    const Eigen::Index chunk_sz = (B + n_chunks - 1) / n_chunks;
    #pragma omp parallel for num_threads(n_chunks) schedule(static)
    for (int c = 0; c < n_chunks; ++c) {
        const Eigen::Index col0 = static_cast<Eigen::Index>(c) * chunk_sz;
        const Eigen::Index cols = std::min(chunk_sz, B - col0);
        if (cols <= 0) continue;
        Eigen::Map<const Eigen::MatrixXd> Xm(X.data() + col0 * data->d_in, data->d_in, cols);
        Eigen::Map<Eigen::MatrixXd> Ym(Y.data() + col0 * data->d_out, data->d_out, cols);
        if (data->use_float32) {
            static thread_local MLPEvalPingPong<float> pp_f;
            mlp_run_eval_only<float>(*data, Xm, Ym, pp_f);
        } else {
            static thread_local MLPEvalPingPong<double> pp_d;
            mlp_run_eval_only<double>(*data, Xm, Ym, pp_d);
        }
    }
    return Y;
}

struct MLPNode::ScratchD : public MLPScratch<double> {};
struct MLPNode::ScratchF : public MLPScratch<float> {};

MLPNode::MLPNode(Shape out_shape, int x_id,
                 std::shared_ptr<const MLPEngineData> engine_data,
                 std::string name_)
    : Node(OpType::MLP, out_shape, {x_id}, std::move(name_)),
      data(std::move(engine_data)),
      batch_cols(out_shape.cols) {
    const int n_threads = std::max(1, omp_get_max_threads());
    if (data->use_float32) {
        thread_scratch_f_.resize(static_cast<std::size_t>(n_threads));
    } else {
        thread_scratch_d_.resize(static_cast<std::size_t>(n_threads));
    }
}

MLPNode::ScratchD& MLPNode::ensure_scratch_d(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_d_.size()) {
        #pragma omp critical(fastopt_mlp_scratch)
        {
            if (idx >= thread_scratch_d_.size()) {
                thread_scratch_d_.resize(idx + 1);
            }
        }
    }
    auto& ptr = thread_scratch_d_[idx];
    if (!ptr) {
        ptr = std::make_shared<ScratchD>();
    }
    return *ptr;
}

MLPNode::ScratchF& MLPNode::ensure_scratch_f(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_f_.size()) {
        #pragma omp critical(fastopt_mlp_scratch)
        {
            if (idx >= thread_scratch_f_.size()) {
                thread_scratch_f_.resize(idx + 1);
            }
        }
    }
    auto& ptr = thread_scratch_f_[idx];
    if (!ptr) {
        ptr = std::make_shared<ScratchF>();
    }
    return *ptr;
}

void MLPNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<Eigen::MatrixXd> Ym(ws.val.data() + offset, data->d_out, batch_cols);

    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        mlp_run_primal<float>(*data, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true;
        sc.has_tangent = false;
    } else {
        auto& sc = ensure_scratch_d(tid);
        mlp_run_primal<double>(*data, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true;
        sc.has_tangent = false;
    }
}

void MLPNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;

    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->d_out, batch_cols);
    Eigen::Map<Eigen::MatrixXd> Xb(ws.bar.data() + in.offset, data->d_in, batch_cols);

    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal &&
                thread_scratch_f_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_f_[0]->cached_x == x_seg) {
                primal = thread_scratch_f_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<float>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        mlp_run_backward<float>(*data, *primal, Yb, Xb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal &&
                thread_scratch_d_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_d_[0]->cached_x == x_seg) {
                primal = thread_scratch_d_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<double>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        mlp_run_backward<double>(*data, *primal, Yb, Xb, sc);
    }
}

void MLPNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) {
        ws.dot.segment(offset, size()).setZero();
        return;
    }

    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<Eigen::MatrixXd> Yd(ws.dot.data() + offset, data->d_out, batch_cols);

    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal &&
                thread_scratch_f_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_f_[0]->cached_x == x_seg) {
                primal = thread_scratch_f_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<float>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        mlp_run_tangent<float>(*data, *primal, Xd, Yd, sc);
        sc.cached_xdot = xd_seg;
        sc.has_tangent = true;
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal &&
                thread_scratch_d_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_d_[0]->cached_x == x_seg) {
                primal = thread_scratch_d_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<double>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        mlp_run_tangent<double>(*data, *primal, Xd, Yd, sc);
        sc.cached_xdot = xd_seg;
        sc.has_tangent = true;
    }
}

void MLPNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 &&
        ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) {
        return;
    }

    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->d_in, batch_cols);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->d_out, batch_cols);
    Eigen::Map<const Eigen::MatrixXd> Ydb(ws.dot_bar.data() + offset, data->d_out, batch_cols);
    Eigen::Map<Eigen::MatrixXd> Xdb(ws.dot_bar.data() + in.offset, data->d_in, batch_cols);

    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal &&
                thread_scratch_f_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_f_[0]->cached_x == x_seg) {
                primal = thread_scratch_f_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<float>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        if (!(sc.has_tangent && sc.cached_xdot.size() == xd_seg.size() && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tmp_yd(data->d_out, batch_cols);
            Eigen::Map<Eigen::MatrixXd> Ydm(tmp_yd.data(), data->d_out, batch_cols);
            mlp_run_tangent<float>(*data, *primal, Xd, Ydm, sc);
            sc.cached_xdot = xd_seg;
            sc.has_tangent = true;
        }
        mlp_run_hessian<float>(*data, *primal, sc, Yb, Ydb, Xdb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* primal = &sc;
        if (!(sc.has_primal && sc.cached_x.size() == x_seg.size() && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal &&
                thread_scratch_d_[0]->cached_x.size() == x_seg.size() &&
                thread_scratch_d_[0]->cached_x == x_seg) {
                primal = thread_scratch_d_[0].get();
            } else {
                Eigen::MatrixXd tmp_y(data->d_out, batch_cols);
                Eigen::Map<Eigen::MatrixXd> Ym(tmp_y.data(), data->d_out, batch_cols);
                mlp_run_primal<double>(*data, Xm, Ym, sc);
                sc.cached_x = x_seg;
                sc.has_primal = true;
                primal = &sc;
            }
        }
        if (!(sc.has_tangent && sc.cached_xdot.size() == xd_seg.size() && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tmp_yd(data->d_out, batch_cols);
            Eigen::Map<Eigen::MatrixXd> Ydm(tmp_yd.data(), data->d_out, batch_cols);
            mlp_run_tangent<double>(*data, *primal, Xd, Ydm, sc);
            sc.cached_xdot = xd_seg;
            sc.has_tangent = true;
        }
        mlp_run_hessian<double>(*data, *primal, sc, Yb, Ydb, Xdb, sc);
    }
}

// ============================================================================
// 19B. Fused Conv1D Implementation (im2col + BLAS3 GEMM + 0/1/2 AD, fp32/fp64)
// ============================================================================
namespace {

template <typename Scalar>
struct Conv1DTraits;

template <>
struct Conv1DTraits<double> {
    static const Eigen::MatrixXd& W(const Conv1DEngineData& d) { return d.W_d; }
    static const Eigen::VectorXd& b(const Conv1DEngineData& d) { return d.b_d; }
};

template <>
struct Conv1DTraits<float> {
    static const Eigen::MatrixXf& W(const Conv1DEngineData& d) { return d.W_f; }
    static const Eigen::VectorXf& b(const Conv1DEngineData& d) { return d.b_f; }
};

template <typename Scalar>
struct Conv1DScratch {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    bool has_primal = false;
    bool has_tangent = false;
    Eigen::VectorXd cached_x;
    Eigen::VectorXd cached_xdot;

    Mat X_buf;
    Mat Col;
    Mat Z;
    Mat dX_buf;
    Mat dCol;
    Mat dZ;
    Mat bar_z;
    Mat bar_col;
    Mat bar_x;
    Mat dbar_z;
    Mat dbar_col;
    Mat dbar_x;
};

template <typename Scalar>
void im2col_1d(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
               Eigen::Index c_in, Eigen::Index l_in,
               int K, int S, int P, int D, Eigen::Index l_out,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Col) {
    Col.resize(c_in * K, l_out);
    Col.setZero();
    for (Eigen::Index lo = 0; lo < l_out; ++lo) {
        const Eigen::Index base = lo * S - P;
        for (int k = 0; k < K; ++k) {
            const Eigen::Index li = base + static_cast<Eigen::Index>(k) * D;
            if (li >= 0 && li < l_in) {
                Col.block(static_cast<Eigen::Index>(k) * c_in, lo, c_in, 1) = X.col(li);
            }
        }
    }
}

template <typename Scalar>
void col2im_1d(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& dCol,
               Eigen::Index c_in, Eigen::Index l_in,
               int K, int S, int P, int D, Eigen::Index l_out,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& dX) {
    dX.resize(c_in, l_in);
    dX.setZero();
    for (Eigen::Index lo = 0; lo < l_out; ++lo) {
        const Eigen::Index base = lo * S - P;
        for (int k = 0; k < K; ++k) {
            const Eigen::Index li = base + static_cast<Eigen::Index>(k) * D;
            if (li >= 0 && li < l_in) {
                dX.col(li).noalias() += dCol.block(static_cast<Eigen::Index>(k) * c_in, lo, c_in, 1);
            }
        }
    }
}

template <typename Scalar>
void conv1d_run_primal(const Conv1DEngineData& d, Eigen::Index l_in, Eigen::Index l_out,
                       const Eigen::Map<const Eigen::MatrixXd>& Xm,
                       Eigen::Map<Eigen::MatrixXd>& Ym,
                       Conv1DScratch<Scalar>& sc) {
    sc.X_buf = Xm.cast<Scalar>();
    im2col_1d<Scalar>(sc.X_buf, d.c_in, l_in, d.kernel_size, d.stride, d.padding, d.dilation, l_out, sc.Col);
    sc.Z.resize(d.c_out, l_out);
    sc.Z.noalias() = Conv1DTraits<Scalar>::W(d) * sc.Col;
    sc.Z.colwise() += Conv1DTraits<Scalar>::b(d);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y_s(d.c_out, l_out);
    mlp_apply_activation_fwd<Scalar>(d.activation, sc.Z, Y_s);
    Ym = Y_s.template cast<double>();
}

template <typename Scalar>
void conv1d_run_backward(const Conv1DEngineData& d, Eigen::Index l_in, Eigen::Index l_out,
                         const Conv1DScratch<Scalar>& primal,
                         const Eigen::Map<const Eigen::MatrixXd>& Yb,
                         Eigen::Map<Eigen::MatrixXd>& Xb,
                         Conv1DScratch<Scalar>& work) {
    work.bar_z.resize(d.c_out, l_out);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yb_s = Yb.cast<Scalar>();
    mlp_apply_activation_bwd<Scalar>(d.activation, primal.Z, yb_s, work.bar_z);
    work.bar_col.resize(d.c_in * d.kernel_size, l_out);
    work.bar_col.noalias() = Conv1DTraits<Scalar>::W(d).transpose() * work.bar_z;
    col2im_1d<Scalar>(work.bar_col, d.c_in, l_in, d.kernel_size, d.stride, d.padding, d.dilation, l_out, work.bar_x);
    Xb.noalias() += work.bar_x.template cast<double>();
}

template <typename Scalar>
void conv1d_run_tangent(const Conv1DEngineData& d, Eigen::Index l_in, Eigen::Index l_out,
                        const Conv1DScratch<Scalar>& primal,
                        const Eigen::Map<const Eigen::MatrixXd>& Xd,
                        Eigen::Map<Eigen::MatrixXd>& Yd,
                        Conv1DScratch<Scalar>& tang) {
    tang.dX_buf = Xd.cast<Scalar>();
    im2col_1d<Scalar>(tang.dX_buf, d.c_in, l_in, d.kernel_size, d.stride, d.padding, d.dilation, l_out, tang.dCol);
    tang.dZ.resize(d.c_out, l_out);
    tang.dZ.noalias() = Conv1DTraits<Scalar>::W(d) * tang.dCol;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yd_s(d.c_out, l_out);
    mlp_apply_activation_bwd<Scalar>(d.activation, primal.Z, tang.dZ, yd_s);
    Yd = yd_s.template cast<double>();
}

template <typename Scalar>
void conv1d_run_hessian(const Conv1DEngineData& d, Eigen::Index l_in, Eigen::Index l_out,
                        const Conv1DScratch<Scalar>& primal,
                        const Conv1DScratch<Scalar>& tang,
                        const Eigen::Map<const Eigen::MatrixXd>& Yb,
                        const Eigen::Map<const Eigen::MatrixXd>& Ydb,
                        Eigen::Map<Eigen::MatrixXd>& Xdb,
                        Conv1DScratch<Scalar>& work) {
    const Eigen::Index n_tot = d.c_out * l_out;
    work.dbar_z.resize(d.c_out, l_out);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yb_s = Yb.cast<Scalar>();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ydb_s = Ydb.cast<Scalar>();
    const Scalar* z_ptr = primal.Z.data();
    const Scalar* dz_ptr = tang.dZ.data();
    const Scalar* yb_ptr = yb_s.data();
    const Scalar* ydb_ptr = ydb_s.data();
    Scalar* dbz_ptr = work.dbar_z.data();
    for (Eigen::Index i = 0; i < n_tot; ++i) {
        Scalar dy_v, d2y_v;
        nn_act_jet2(d.activation, z_ptr[i], dy_v, d2y_v);
        dbz_ptr[i] = ydb_ptr[i] * dy_v + yb_ptr[i] * d2y_v * dz_ptr[i];
    }
    work.dbar_col.resize(d.c_in * d.kernel_size, l_out);
    work.dbar_col.noalias() = Conv1DTraits<Scalar>::W(d).transpose() * work.dbar_z;
    col2im_1d<Scalar>(work.dbar_col, d.c_in, l_in, d.kernel_size, d.stride, d.padding, d.dilation, l_out, work.dbar_x);
    Xdb.noalias() += work.dbar_x.template cast<double>();
}

// ============================================================================
// 19C. Fused Multi-Head Self-Attention (Exact 0/1/2 AD, fp32/fp64)
// ============================================================================
template <typename Scalar>
struct AttnTraits;

template <>
struct AttnTraits<double> {
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;
    static const Mat& Wq(const AttentionEngineData& d) { return d.Wq_d; }
    static const Mat& Wk(const AttentionEngineData& d) { return d.Wk_d; }
    static const Mat& Wv(const AttentionEngineData& d) { return d.Wv_d; }
    static const Mat& Wo(const AttentionEngineData& d) { return d.Wo_d; }
    static const Vec& bq(const AttentionEngineData& d) { return d.bq_d; }
    static const Vec& bk(const AttentionEngineData& d) { return d.bk_d; }
    static const Vec& bv(const AttentionEngineData& d) { return d.bv_d; }
    static const Vec& bo(const AttentionEngineData& d) { return d.bo_d; }
};

template <>
struct AttnTraits<float> {
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    static const Mat& Wq(const AttentionEngineData& d) { return d.Wq_f; }
    static const Mat& Wk(const AttentionEngineData& d) { return d.Wk_f; }
    static const Mat& Wv(const AttentionEngineData& d) { return d.Wv_f; }
    static const Mat& Wo(const AttentionEngineData& d) { return d.Wo_f; }
    static const Vec& bq(const AttentionEngineData& d) { return d.bq_f; }
    static const Vec& bk(const AttentionEngineData& d) { return d.bk_f; }
    static const Vec& bv(const AttentionEngineData& d) { return d.bv_f; }
    static const Vec& bo(const AttentionEngineData& d) { return d.bo_f; }
};

template <typename Scalar>
struct AttnScratch {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    bool has_primal = false;
    bool has_tangent = false;
    Eigen::VectorXd cached_x;
    Eigen::VectorXd cached_xdot;

    Mat Q, K, V, O_cat;
    std::vector<Mat> P;   // per-head softmax weights (L x L): col j = attention weights for query j
    Mat dQ, dK, dV;
    std::vector<Mat> dP;  // per-head tangent softmax weights (L x L)
};

template <typename Scalar>
void attn_run_primal(const AttentionEngineData& d, Eigen::Index L,
                     const Eigen::Map<const Eigen::MatrixXd>& Xm,
                     Eigen::Map<Eigen::MatrixXd>& Ym,
                     AttnScratch<Scalar>& sc) {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Mat X_s = Xm.cast<Scalar>();
    const Eigen::Index dm = d.d_model;
    const Eigen::Index dh = d.d_head;
    const int H = d.num_heads;
    const Scalar sc_val = static_cast<Scalar>(d.scale);

    sc.Q.resize(dm, L); sc.Q.noalias() = AttnTraits<Scalar>::Wq(d) * X_s; sc.Q.colwise() += AttnTraits<Scalar>::bq(d);
    sc.K.resize(dm, L); sc.K.noalias() = AttnTraits<Scalar>::Wk(d) * X_s; sc.K.colwise() += AttnTraits<Scalar>::bk(d);
    sc.V.resize(dm, L); sc.V.noalias() = AttnTraits<Scalar>::Wv(d) * X_s; sc.V.colwise() += AttnTraits<Scalar>::bv(d);
    sc.O_cat.resize(dm, L);
    sc.P.resize(static_cast<std::size_t>(H));

    Mat S(L, L);
    for (int h = 0; h < H; ++h) {
        const Eigen::Index r0 = static_cast<Eigen::Index>(h) * dh;
        auto Qh = sc.Q.middleRows(r0, dh);
        auto Kh = sc.K.middleRows(r0, dh);
        auto Vh = sc.V.middleRows(r0, dh);
        S.noalias() = (Kh.transpose() * Qh) * sc_val; // S(i, j) = <K_i, Q_j> * scale
        auto& Ph = sc.P[static_cast<std::size_t>(h)];
        Ph.resize(L, L);
        for (Eigen::Index j = 0; j < L; ++j) {
            const Eigen::Index i_lim = d.causal ? (j + 1) : L;
            Scalar m_val = S(0, j);
            for (Eigen::Index i = 1; i < i_lim; ++i) m_val = std::max(m_val, S(i, j));
            Scalar sum_e = Scalar(0);
            for (Eigen::Index i = 0; i < i_lim; ++i) {
                const Scalar e = std::exp(S(i, j) - m_val);
                Ph(i, j) = e;
                sum_e += e;
            }
            const Scalar inv_s = Scalar(1) / sum_e;
            for (Eigen::Index i = 0; i < i_lim; ++i) Ph(i, j) *= inv_s;
            for (Eigen::Index i = i_lim; i < L; ++i) Ph(i, j) = Scalar(0);
        }
        sc.O_cat.middleRows(r0, dh).noalias() = Vh * Ph;
    }

    Mat Y_s(dm, L);
    Y_s.noalias() = AttnTraits<Scalar>::Wo(d) * sc.O_cat;
    Y_s.colwise() += AttnTraits<Scalar>::bo(d);
    if (d.residual) Y_s.noalias() += X_s;
    Ym = Y_s.template cast<double>();
}

template <typename Scalar>
void attn_run_backward(const AttentionEngineData& d, Eigen::Index L,
                       const AttnScratch<Scalar>& primal,
                       const Eigen::Map<const Eigen::MatrixXd>& Yb,
                       Eigen::Map<Eigen::MatrixXd>& Xb) {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Mat Yb_s = Yb.cast<Scalar>();
    const Eigen::Index dm = d.d_model;
    const Eigen::Index dh = d.d_head;
    const int H = d.num_heads;
    const Scalar sc_val = static_cast<Scalar>(d.scale);

    Mat O_bar = AttnTraits<Scalar>::Wo(d).transpose() * Yb_s;
    Mat Q_bar = Mat::Zero(dm, L);
    Mat K_bar = Mat::Zero(dm, L);
    Mat V_bar = Mat::Zero(dm, L);

    Mat P_bar(L, L), S_bar(L, L);
    for (int h = 0; h < H; ++h) {
        const Eigen::Index r0 = static_cast<Eigen::Index>(h) * dh;
        const auto Qh = primal.Q.middleRows(r0, dh);
        const auto Kh = primal.K.middleRows(r0, dh);
        const auto Vh = primal.V.middleRows(r0, dh);
        const auto& Ph = primal.P[static_cast<std::size_t>(h)];
        const auto Oh_bar = O_bar.middleRows(r0, dh);

        V_bar.middleRows(r0, dh).noalias() = Oh_bar * Ph.transpose();
        P_bar.noalias() = Vh.transpose() * Oh_bar;

        // Softmax VJP per column j: S_bar(i,j) = P(i,j) * (P_bar(i,j) - sum_k P(k,j)*P_bar(k,j))
        for (Eigen::Index j = 0; j < L; ++j) {
            const Eigen::Index i_lim = d.causal ? (j + 1) : L;
            Scalar dot_pb = Scalar(0);
            for (Eigen::Index i = 0; i < i_lim; ++i) dot_pb += Ph(i, j) * P_bar(i, j);
            for (Eigen::Index i = 0; i < i_lim; ++i) S_bar(i, j) = Ph(i, j) * (P_bar(i, j) - dot_pb);
            for (Eigen::Index i = i_lim; i < L; ++i) S_bar(i, j) = Scalar(0);
        }
        Q_bar.middleRows(r0, dh).noalias() = (Kh * S_bar) * sc_val;
        K_bar.middleRows(r0, dh).noalias() = (Qh * S_bar.transpose()) * sc_val;
    }

    Mat X_bar = AttnTraits<Scalar>::Wq(d).transpose() * Q_bar
              + AttnTraits<Scalar>::Wk(d).transpose() * K_bar
              + AttnTraits<Scalar>::Wv(d).transpose() * V_bar;
    if (d.residual) X_bar.noalias() += Yb_s;
    Xb.noalias() += X_bar.template cast<double>();
}

template <typename Scalar>
void attn_run_tangent(const AttentionEngineData& d, Eigen::Index L,
                      const AttnScratch<Scalar>& primal,
                      const Eigen::Map<const Eigen::MatrixXd>& Xd,
                      Eigen::Map<Eigen::MatrixXd>& Yd,
                      AttnScratch<Scalar>& tang) {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Mat Xd_s = Xd.cast<Scalar>();
    const Eigen::Index dm = d.d_model;
    const Eigen::Index dh = d.d_head;
    const int H = d.num_heads;
    const Scalar sc_val = static_cast<Scalar>(d.scale);

    tang.dQ.resize(dm, L); tang.dQ.noalias() = AttnTraits<Scalar>::Wq(d) * Xd_s;
    tang.dK.resize(dm, L); tang.dK.noalias() = AttnTraits<Scalar>::Wk(d) * Xd_s;
    tang.dV.resize(dm, L); tang.dV.noalias() = AttnTraits<Scalar>::Wv(d) * Xd_s;
    tang.dP.resize(static_cast<std::size_t>(H));

    Mat dO_cat(dm, L);
    Mat dS(L, L);
    for (int h = 0; h < H; ++h) {
        const Eigen::Index r0 = static_cast<Eigen::Index>(h) * dh;
        const auto Qh = primal.Q.middleRows(r0, dh);
        const auto Kh = primal.K.middleRows(r0, dh);
        const auto Vh = primal.V.middleRows(r0, dh);
        const auto dQh = tang.dQ.middleRows(r0, dh);
        const auto dKh = tang.dK.middleRows(r0, dh);
        const auto dVh = tang.dV.middleRows(r0, dh);
        const auto& Ph = primal.P[static_cast<std::size_t>(h)];

        dS.noalias() = (dKh.transpose() * Qh + Kh.transpose() * dQh) * sc_val;
        auto& dPh = tang.dP[static_cast<std::size_t>(h)];
        dPh.resize(L, L);
        for (Eigen::Index j = 0; j < L; ++j) {
            const Eigen::Index i_lim = d.causal ? (j + 1) : L;
            Scalar dot_ds = Scalar(0);
            for (Eigen::Index i = 0; i < i_lim; ++i) dot_ds += Ph(i, j) * dS(i, j);
            for (Eigen::Index i = 0; i < i_lim; ++i) dPh(i, j) = Ph(i, j) * (dS(i, j) - dot_ds);
            for (Eigen::Index i = i_lim; i < L; ++i) dPh(i, j) = Scalar(0);
        }
        dO_cat.middleRows(r0, dh).noalias() = dVh * Ph + Vh * dPh;
    }

    Mat Yd_s = AttnTraits<Scalar>::Wo(d) * dO_cat;
    if (d.residual) Yd_s.noalias() += Xd_s;
    Yd = Yd_s.template cast<double>();
}

template <typename Scalar>
void attn_run_hessian(const AttentionEngineData& d, Eigen::Index L,
                      const AttnScratch<Scalar>& primal,
                      const AttnScratch<Scalar>& tang,
                      const Eigen::Map<const Eigen::MatrixXd>& Yb,
                      const Eigen::Map<const Eigen::MatrixXd>& Ydb,
                      Eigen::Map<Eigen::MatrixXd>& Xdb) {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Mat Yb_s = Yb.cast<Scalar>();
    const Mat Ydb_s = Ydb.cast<Scalar>();
    const Eigen::Index dm = d.d_model;
    const Eigen::Index dh = d.d_head;
    const int H = d.num_heads;
    const Scalar sc_val = static_cast<Scalar>(d.scale);

    const Mat O_bar = AttnTraits<Scalar>::Wo(d).transpose() * Yb_s;
    const Mat O_dbar = AttnTraits<Scalar>::Wo(d).transpose() * Ydb_s;

    Mat Q_dbar = Mat::Zero(dm, L);
    Mat K_dbar = Mat::Zero(dm, L);
    Mat V_dbar = Mat::Zero(dm, L);

    Mat P_bar(L, L), P_dbar(L, L), S_bar(L, L), S_dbar(L, L);
    for (int h = 0; h < H; ++h) {
        const Eigen::Index r0 = static_cast<Eigen::Index>(h) * dh;
        const auto Qh = primal.Q.middleRows(r0, dh);
        const auto Kh = primal.K.middleRows(r0, dh);
        const auto Vh = primal.V.middleRows(r0, dh);
        const auto dQh = tang.dQ.middleRows(r0, dh);
        const auto dKh = tang.dK.middleRows(r0, dh);
        const auto dVh = tang.dV.middleRows(r0, dh);
        const auto& Ph = primal.P[static_cast<std::size_t>(h)];
        const auto& dPh = tang.dP[static_cast<std::size_t>(h)];
        const auto Oh_bar = O_bar.middleRows(r0, dh);
        const auto Oh_dbar = O_dbar.middleRows(r0, dh);

        V_dbar.middleRows(r0, dh).noalias() = Oh_dbar * Ph.transpose() + Oh_bar * dPh.transpose();
        P_bar.noalias() = Vh.transpose() * Oh_bar;
        P_dbar.noalias() = dVh.transpose() * Oh_bar + Vh.transpose() * Oh_dbar;

        // Forward-over-Reverse exact Softmax Hessian per column j:
        //   S_bar(i,j)  = P(i,j) * (P_bar(i,j) - mu)
        //   S_dbar(i,j) = dP(i,j) * (P_bar(i,j) - mu) + P(i,j) * (P_dbar(i,j) - dmu)
        for (Eigen::Index j = 0; j < L; ++j) {
            const Eigen::Index i_lim = d.causal ? (j + 1) : L;
            Scalar mu = Scalar(0), dmu = Scalar(0);
            for (Eigen::Index i = 0; i < i_lim; ++i) {
                mu += Ph(i, j) * P_bar(i, j);
                dmu += dPh(i, j) * P_bar(i, j) + Ph(i, j) * P_dbar(i, j);
            }
            for (Eigen::Index i = 0; i < i_lim; ++i) {
                const Scalar diff = P_bar(i, j) - mu;
                S_bar(i, j) = Ph(i, j) * diff;
                S_dbar(i, j) = dPh(i, j) * diff + Ph(i, j) * (P_dbar(i, j) - dmu);
            }
            for (Eigen::Index i = i_lim; i < L; ++i) {
                S_bar(i, j) = Scalar(0);
                S_dbar(i, j) = Scalar(0);
            }
        }
        Q_dbar.middleRows(r0, dh).noalias() = (dKh * S_bar + Kh * S_dbar) * sc_val;
        K_dbar.middleRows(r0, dh).noalias() = (dQh * S_bar.transpose() + Qh * S_dbar.transpose()) * sc_val;
    }

    Mat X_dbar = AttnTraits<Scalar>::Wq(d).transpose() * Q_dbar
               + AttnTraits<Scalar>::Wk(d).transpose() * K_dbar
               + AttnTraits<Scalar>::Wv(d).transpose() * V_dbar;
    if (d.residual) X_dbar.noalias() += Ydb_s;
    Xdb.noalias() += X_dbar.template cast<double>();
}

} // namespace

std::shared_ptr<const Conv1DEngineData> Conv1DEngineData::create(
    const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
    int c_in_, int kernel_size_, int stride_, int padding_, int dilation_,
    NNActivation activation_, bool use_float32_) {
    if (c_in_ <= 0 || kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0 || dilation_ <= 0) {
        throw std::invalid_argument("Conv1D: invalid convolution hyperparameters.");
    }
    if (W.cols() != static_cast<Eigen::Index>(c_in_) * kernel_size_) {
        throw std::invalid_argument("Conv1D: W.cols() must equal c_in * kernel_size.");
    }
    if (b.size() != W.rows()) {
        throw std::invalid_argument("Conv1D: bias size must equal W.rows() (c_out).");
    }
    auto d = std::make_shared<Conv1DEngineData>();
    d->use_float32 = use_float32_;
    d->c_in = c_in_;
    d->c_out = W.rows();
    d->kernel_size = kernel_size_;
    d->stride = stride_;
    d->padding = padding_;
    d->dilation = dilation_;
    d->activation = activation_;
    if (use_float32_) {
        d->W_f = W.cast<float>();
        d->b_f = b.cast<float>();
    } else {
        d->W_d = W;
        d->b_d = b;
    }
    return d;
}

CompiledConv1D::CompiledConv1D(const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
                               int c_in_, int kernel_size_, int stride_, int padding_, int dilation_,
                               NNActivation activation_, bool use_float32_, std::string name_)
    : data(Conv1DEngineData::create(W, b, c_in_, kernel_size_, stride_, padding_, dilation_, activation_, use_float32_)),
      name(std::move(name_)) {}

Eigen::MatrixXd CompiledConv1D::eval(const Eigen::MatrixXd& X) const {
    if (X.rows() != data->c_in) {
        throw std::invalid_argument("CompiledConv1D::eval: X.rows() must match c_in.");
    }
    const Eigen::Index l_in = X.cols();
    const Eigen::Index l_out = data->out_length(l_in);
    Eigen::MatrixXd Y(data->c_out, l_out);
    Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), data->c_in, l_in);
    Eigen::Map<Eigen::MatrixXd> Ym(Y.data(), data->c_out, l_out);
    if (data->use_float32) {
        static thread_local Conv1DScratch<float> sc;
        conv1d_run_primal<float>(*data, l_in, l_out, Xm, Ym, sc);
    } else {
        static thread_local Conv1DScratch<double> sc;
        conv1d_run_primal<double>(*data, l_in, l_out, Xm, Ym, sc);
    }
    return Y;
}

struct Conv1DNode::ScratchD : public Conv1DScratch<double> {};
struct Conv1DNode::ScratchF : public Conv1DScratch<float> {};

Conv1DNode::Conv1DNode(Shape out_shape, int x_id, Eigen::Index l_in_,
                       std::shared_ptr<const Conv1DEngineData> engine_data,
                       std::string name_)
    : Node(OpType::Conv1D, out_shape, {x_id}, std::move(name_)),
      data(std::move(engine_data)), l_in(l_in_), l_out(out_shape.cols) {
    const int nt = std::max(1, omp_get_max_threads());
    if (data->use_float32) thread_scratch_f_.resize(static_cast<std::size_t>(nt));
    else thread_scratch_d_.resize(static_cast<std::size_t>(nt));
}

Conv1DNode::ScratchD& Conv1DNode::ensure_scratch_d(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_d_.size()) {
        #pragma omp critical(fastopt_conv1d_scratch)
        { if (idx >= thread_scratch_d_.size()) thread_scratch_d_.resize(idx + 1); }
    }
    if (!thread_scratch_d_[idx]) thread_scratch_d_[idx] = std::make_shared<ScratchD>();
    return *thread_scratch_d_[idx];
}

Conv1DNode::ScratchF& Conv1DNode::ensure_scratch_f(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_f_.size()) {
        #pragma omp critical(fastopt_conv1d_scratch)
        { if (idx >= thread_scratch_f_.size()) thread_scratch_f_.resize(idx + 1); }
    }
    if (!thread_scratch_f_[idx]) thread_scratch_f_[idx] = std::make_shared<ScratchF>();
    return *thread_scratch_f_[idx];
}

void Conv1DNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, l_in);
    Eigen::Map<Eigen::MatrixXd> Ym(ws.val.data() + offset, data->c_out, l_out);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        conv1d_run_primal<float>(*data, l_in, l_out, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true; sc.has_tangent = false;
    } else {
        auto& sc = ensure_scratch_d(tid);
        conv1d_run_primal<double>(*data, l_in, l_out, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true; sc.has_tangent = false;
    }
}

void Conv1DNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad || ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, l_in);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->c_out, l_out);
    Eigen::Map<Eigen::MatrixXd> Xb(ws.bar.data() + in.offset, data->c_in, l_in);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<float>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv1d_run_backward<float>(*data, l_in, l_out, *pr, Yb, Xb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<double>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv1d_run_backward<double>(*data, l_in, l_out, *pr, Yb, Xb, sc);
    }
}

void Conv1DNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) { ws.dot.segment(offset, size()).setZero(); return; }
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, l_in);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->c_in, l_in);
    Eigen::Map<Eigen::MatrixXd> Yd(ws.dot.data() + offset, data->c_out, l_out);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<float>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv1d_run_tangent<float>(*data, l_in, l_out, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<double>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv1d_run_tangent<double>(*data, l_in, l_out, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    }
}

void Conv1DNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 && ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, l_in);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->c_in, l_in);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->c_out, l_out);
    Eigen::Map<const Eigen::MatrixXd> Ydb(ws.dot_bar.data() + offset, data->c_out, l_out);
    Eigen::Map<Eigen::MatrixXd> Xdb(ws.dot_bar.data() + in.offset, data->c_in, l_in);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<float>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->c_out, l_out);
            conv1d_run_tangent<float>(*data, l_in, l_out, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        conv1d_run_hessian<float>(*data, l_in, l_out, *pr, sc, Yb, Ydb, Xdb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, l_out); conv1d_run_primal<double>(*data, l_in, l_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->c_out, l_out); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->c_out, l_out);
            conv1d_run_tangent<double>(*data, l_in, l_out, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        conv1d_run_hessian<double>(*data, l_in, l_out, *pr, sc, Yb, Ydb, Xdb, sc);
    }
}

std::shared_ptr<const AttentionEngineData> AttentionEngineData::create(
    const Eigen::MatrixXd& Wq, const Eigen::VectorXd& bq,
    const Eigen::MatrixXd& Wk, const Eigen::VectorXd& bk,
    const Eigen::MatrixXd& Wv, const Eigen::VectorXd& bv,
    const Eigen::MatrixXd& Wo, const Eigen::VectorXd& bo,
    int num_heads_, bool causal_, bool residual_, bool use_float32_) {
    const Eigen::Index dm = Wq.rows();
    if (dm <= 0 || num_heads_ <= 0 || (dm % num_heads_) != 0) {
        throw std::invalid_argument("Attention: d_model must be positive and divisible by num_heads.");
    }
    auto check_mat = [&](const Eigen::MatrixXd& M, const Eigen::VectorXd& v) {
        if (M.rows() != dm || M.cols() != dm || v.size() != dm) {
            throw std::invalid_argument("Attention: Wq/Wk/Wv/Wo must be (d_model x d_model) and biases (d_model).");
        }
    };
    check_mat(Wq, bq); check_mat(Wk, bk); check_mat(Wv, bv); check_mat(Wo, bo);

    auto d = std::make_shared<AttentionEngineData>();
    d->use_float32 = use_float32_;
    d->d_model = dm;
    d->num_heads = num_heads_;
    d->d_head = dm / num_heads_;
    d->scale = 1.0 / std::sqrt(static_cast<double>(d->d_head));
    d->causal = causal_;
    d->residual = residual_;
    if (use_float32_) {
        d->Wq_f = Wq.cast<float>(); d->bq_f = bq.cast<float>();
        d->Wk_f = Wk.cast<float>(); d->bk_f = bk.cast<float>();
        d->Wv_f = Wv.cast<float>(); d->bv_f = bv.cast<float>();
        d->Wo_f = Wo.cast<float>(); d->bo_f = bo.cast<float>();
    } else {
        d->Wq_d = Wq; d->bq_d = bq;
        d->Wk_d = Wk; d->bk_d = bk;
        d->Wv_d = Wv; d->bv_d = bv;
        d->Wo_d = Wo; d->bo_d = bo;
    }
    return d;
}

CompiledAttention::CompiledAttention(
    const Eigen::MatrixXd& Wq, const Eigen::VectorXd& bq,
    const Eigen::MatrixXd& Wk, const Eigen::VectorXd& bk,
    const Eigen::MatrixXd& Wv, const Eigen::VectorXd& bv,
    const Eigen::MatrixXd& Wo, const Eigen::VectorXd& bo,
    int num_heads_, bool causal_, bool residual_, bool use_float32_, std::string name_)
    : data(AttentionEngineData::create(Wq, bq, Wk, bk, Wv, bv, Wo, bo, num_heads_, causal_, residual_, use_float32_)),
      name(std::move(name_)) {}

Eigen::MatrixXd CompiledAttention::eval(const Eigen::MatrixXd& X) const {
    if (X.rows() != data->d_model) {
        throw std::invalid_argument("CompiledAttention::eval: X.rows() must match d_model.");
    }
    const Eigen::Index L = X.cols();
    Eigen::MatrixXd Y(data->d_model, L);
    Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), data->d_model, L);
    Eigen::Map<Eigen::MatrixXd> Ym(Y.data(), data->d_model, L);
    if (data->use_float32) {
        static thread_local AttnScratch<float> sc;
        attn_run_primal<float>(*data, L, Xm, Ym, sc);
    } else {
        static thread_local AttnScratch<double> sc;
        attn_run_primal<double>(*data, L, Xm, Ym, sc);
    }
    return Y;
}

struct AttentionNode::ScratchD : public AttnScratch<double> {};
struct AttentionNode::ScratchF : public AttnScratch<float> {};

AttentionNode::AttentionNode(Shape out_shape, int x_id,
                             std::shared_ptr<const AttentionEngineData> engine_data,
                             std::string name_)
    : Node(OpType::Attention, out_shape, {x_id}, std::move(name_)),
      data(std::move(engine_data)), seq_len(out_shape.cols) {
    const int nt = std::max(1, omp_get_max_threads());
    if (data->use_float32) thread_scratch_f_.resize(static_cast<std::size_t>(nt));
    else thread_scratch_d_.resize(static_cast<std::size_t>(nt));
}

AttentionNode::ScratchD& AttentionNode::ensure_scratch_d(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_d_.size()) {
        #pragma omp critical(fastopt_attn_scratch)
        { if (idx >= thread_scratch_d_.size()) thread_scratch_d_.resize(idx + 1); }
    }
    if (!thread_scratch_d_[idx]) thread_scratch_d_[idx] = std::make_shared<ScratchD>();
    return *thread_scratch_d_[idx];
}

AttentionNode::ScratchF& AttentionNode::ensure_scratch_f(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_f_.size()) {
        #pragma omp critical(fastopt_attn_scratch)
        { if (idx >= thread_scratch_f_.size()) thread_scratch_f_.resize(idx + 1); }
    }
    if (!thread_scratch_f_[idx]) thread_scratch_f_[idx] = std::make_shared<ScratchF>();
    return *thread_scratch_f_[idx];
}

void AttentionNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<Eigen::MatrixXd> Ym(ws.val.data() + offset, data->d_model, seq_len);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        attn_run_primal<float>(*data, seq_len, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true; sc.has_tangent = false;
    } else {
        auto& sc = ensure_scratch_d(tid);
        attn_run_primal<double>(*data, seq_len, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size());
        sc.has_primal = true; sc.has_tangent = false;
    }
}

void AttentionNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad || ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->d_model, seq_len);
    Eigen::Map<Eigen::MatrixXd> Xb(ws.bar.data() + in.offset, data->d_model, seq_len);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<float>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        attn_run_backward<float>(*data, seq_len, *pr, Yb, Xb);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<double>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        attn_run_backward<double>(*data, seq_len, *pr, Yb, Xb);
    }
}

void AttentionNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) { ws.dot.segment(offset, size()).setZero(); return; }
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<Eigen::MatrixXd> Yd(ws.dot.data() + offset, data->d_model, seq_len);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<float>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        attn_run_tangent<float>(*data, seq_len, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<double>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        attn_run_tangent<double>(*data, seq_len, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    }
}

void AttentionNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 && ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->d_model, seq_len);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->d_model, seq_len);
    Eigen::Map<const Eigen::MatrixXd> Ydb(ws.dot_bar.data() + offset, data->d_model, seq_len);
    Eigen::Map<Eigen::MatrixXd> Xdb(ws.dot_bar.data() + in.offset, data->d_model, seq_len);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg)
                pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<float>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->d_model, seq_len);
            attn_run_tangent<float>(*data, seq_len, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        attn_run_hessian<float>(*data, seq_len, *pr, sc, Yb, Ydb, Xdb);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg)
                pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->d_model, seq_len); attn_run_primal<double>(*data, seq_len, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->d_model, seq_len); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->d_model, seq_len);
            attn_run_tangent<double>(*data, seq_len, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        attn_run_hessian<double>(*data, seq_len, *pr, sc, Yb, Ydb, Xdb);
    }
}

// ============================================================================
// 19D. Fused Conv2D, Pool2D, and RMSNorm Implementations (0/1/2 AD)
// ============================================================================
namespace {

template <typename Scalar>
struct Conv2DTraits;
template <>
struct Conv2DTraits<double> {
    static const Eigen::MatrixXd& W(const Conv2DEngineData& d) { return d.W_d; }
    static const Eigen::VectorXd& b(const Conv2DEngineData& d) { return d.b_d; }
};
template <>
struct Conv2DTraits<float> {
    static const Eigen::MatrixXf& W(const Conv2DEngineData& d) { return d.W_f; }
    static const Eigen::VectorXf& b(const Conv2DEngineData& d) { return d.b_f; }
};

template <typename Scalar>
struct Conv2DScratch {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    bool has_primal = false;
    bool has_tangent = false;
    Eigen::VectorXd cached_x;
    Eigen::VectorXd cached_xdot;
    Mat X_buf, Col, Z, dX_buf, dCol, dZ, bar_z, bar_col, bar_x, dbar_z, dbar_col, dbar_x;
};

template <typename Scalar>
void im2col_2d(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
               Eigen::Index c_in, int h_in, int w_in,
               int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
               int h_out, int w_out,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Col) {
    Col.resize(c_in * kh * kw, static_cast<Eigen::Index>(h_out) * w_out);
    Col.setZero();
    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index col_idx = static_cast<Eigen::Index>(ho) * w_out + wo;
            int k_idx = 0;
            for (int r = 0; r < kh; ++r) {
                const int hi = ho * sh - ph + r * dh;
                for (int c = 0; c < kw; ++c) {
                    const int wi = wo * sw - pw + c * dw;
                    if (hi >= 0 && hi < h_in && wi >= 0 && wi < w_in) {
                        const Eigen::Index in_col = static_cast<Eigen::Index>(hi) * w_in + wi;
                        Col.block(static_cast<Eigen::Index>(k_idx) * c_in, col_idx, c_in, 1) = X.col(in_col);
                    }
                    ++k_idx;
                }
            }
        }
    }
}

template <typename Scalar>
void col2im_2d(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& dCol,
               Eigen::Index c_in, int h_in, int w_in,
               int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
               int h_out, int w_out,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& dX) {
    dX.resize(c_in, static_cast<Eigen::Index>(h_in) * w_in);
    dX.setZero();
    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index col_idx = static_cast<Eigen::Index>(ho) * w_out + wo;
            int k_idx = 0;
            for (int r = 0; r < kh; ++r) {
                const int hi = ho * sh - ph + r * dh;
                for (int c = 0; c < kw; ++c) {
                    const int wi = wo * sw - pw + c * dw;
                    if (hi >= 0 && hi < h_in && wi >= 0 && wi < w_in) {
                        const Eigen::Index in_col = static_cast<Eigen::Index>(hi) * w_in + wi;
                        dX.col(in_col).noalias() += dCol.block(static_cast<Eigen::Index>(k_idx) * c_in, col_idx, c_in, 1);
                    }
                    ++k_idx;
                }
            }
        }
    }
}

template <typename Scalar>
void conv2d_run_primal(const Conv2DEngineData& d, int h_in, int w_in, int h_out, int w_out,
                       const Eigen::Map<const Eigen::MatrixXd>& Xm,
                       Eigen::Map<Eigen::MatrixXd>& Ym, Conv2DScratch<Scalar>& sc) {
    sc.X_buf = Xm.cast<Scalar>();
    im2col_2d<Scalar>(sc.X_buf, d.c_in, h_in, w_in, d.kh, d.kw, d.sh, d.sw, d.ph, d.pw, d.dh, d.dw, h_out, w_out, sc.Col);
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    sc.Z.resize(d.c_out, s_out);
    sc.Z.noalias() = Conv2DTraits<Scalar>::W(d) * sc.Col;
    sc.Z.colwise() += Conv2DTraits<Scalar>::b(d);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y_s(d.c_out, s_out);
    mlp_apply_activation_fwd<Scalar>(d.activation, sc.Z, Y_s);
    Ym = Y_s.template cast<double>();
}

template <typename Scalar>
void conv2d_run_backward(const Conv2DEngineData& d, int h_in, int w_in, int h_out, int w_out,
                         const Conv2DScratch<Scalar>& primal,
                         const Eigen::Map<const Eigen::MatrixXd>& Yb,
                         Eigen::Map<Eigen::MatrixXd>& Xb, Conv2DScratch<Scalar>& work) {
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    work.bar_z.resize(d.c_out, s_out);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yb_s = Yb.cast<Scalar>();
    mlp_apply_activation_bwd<Scalar>(d.activation, primal.Z, yb_s, work.bar_z);
    work.bar_col.resize(d.c_in * d.kh * d.kw, s_out);
    work.bar_col.noalias() = Conv2DTraits<Scalar>::W(d).transpose() * work.bar_z;
    col2im_2d<Scalar>(work.bar_col, d.c_in, h_in, w_in, d.kh, d.kw, d.sh, d.sw, d.ph, d.pw, d.dh, d.dw, h_out, w_out, work.bar_x);
    Xb.noalias() += work.bar_x.template cast<double>();
}

template <typename Scalar>
void conv2d_run_tangent(const Conv2DEngineData& d, int h_in, int w_in, int h_out, int w_out,
                        const Conv2DScratch<Scalar>& primal,
                        const Eigen::Map<const Eigen::MatrixXd>& Xd,
                        Eigen::Map<Eigen::MatrixXd>& Yd, Conv2DScratch<Scalar>& tang) {
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    tang.dX_buf = Xd.cast<Scalar>();
    im2col_2d<Scalar>(tang.dX_buf, d.c_in, h_in, w_in, d.kh, d.kw, d.sh, d.sw, d.ph, d.pw, d.dh, d.dw, h_out, w_out, tang.dCol);
    tang.dZ.resize(d.c_out, s_out);
    tang.dZ.noalias() = Conv2DTraits<Scalar>::W(d) * tang.dCol;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yd_s(d.c_out, s_out);
    mlp_apply_activation_bwd<Scalar>(d.activation, primal.Z, tang.dZ, yd_s);
    Yd = yd_s.template cast<double>();
}

template <typename Scalar>
void conv2d_run_hessian(const Conv2DEngineData& d, int h_in, int w_in, int h_out, int w_out,
                        const Conv2DScratch<Scalar>& primal, const Conv2DScratch<Scalar>& tang,
                        const Eigen::Map<const Eigen::MatrixXd>& Yb,
                        const Eigen::Map<const Eigen::MatrixXd>& Ydb,
                        Eigen::Map<Eigen::MatrixXd>& Xdb, Conv2DScratch<Scalar>& work) {
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    const Eigen::Index n_tot = d.c_out * s_out;
    work.dbar_z.resize(d.c_out, s_out);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> yb_s = Yb.cast<Scalar>();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ydb_s = Ydb.cast<Scalar>();
    const Scalar* zp = primal.Z.data();
    const Scalar* dzp = tang.dZ.data();
    const Scalar* ybp = yb_s.data();
    const Scalar* ydbp = ydb_s.data();
    Scalar* dbzp = work.dbar_z.data();
    for (Eigen::Index i = 0; i < n_tot; ++i) {
        Scalar dy_v, d2y_v;
        nn_act_jet2(d.activation, zp[i], dy_v, d2y_v);
        dbzp[i] = ydbp[i] * dy_v + ybp[i] * d2y_v * dzp[i];
    }
    work.dbar_col.resize(d.c_in * d.kh * d.kw, s_out);
    work.dbar_col.noalias() = Conv2DTraits<Scalar>::W(d).transpose() * work.dbar_z;
    col2im_2d<Scalar>(work.dbar_col, d.c_in, h_in, w_in, d.kh, d.kw, d.sh, d.sw, d.ph, d.pw, d.dh, d.dw, h_out, w_out, work.dbar_x);
    Xdb.noalias() += work.dbar_x.template cast<double>();
}

} // namespace

std::shared_ptr<const Conv2DEngineData> Conv2DEngineData::create(
    const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
    int c_in_, int kh_, int kw_, int sh_, int sw_, int ph_, int pw_, int dh_, int dw_,
    NNActivation activation_, bool use_float32_) {
    if (c_in_ <= 0 || kh_ <= 0 || kw_ <= 0 || sh_ <= 0 || sw_ <= 0 || ph_ < 0 || pw_ < 0 || dh_ <= 0 || dw_ <= 0)
        throw std::invalid_argument("Conv2D: invalid hyperparameters.");
    if (W.cols() != static_cast<Eigen::Index>(c_in_) * kh_ * kw_)
        throw std::invalid_argument("Conv2D: W.cols() must equal c_in * kh * kw.");
    if (b.size() != W.rows())
        throw std::invalid_argument("Conv2D: bias size must equal W.rows() (c_out).");
    auto d = std::make_shared<Conv2DEngineData>();
    d->use_float32 = use_float32_;
    d->c_in = c_in_; d->c_out = W.rows();
    d->kh = kh_; d->kw = kw_; d->sh = sh_; d->sw = sw_;
    d->ph = ph_; d->pw = pw_; d->dh = dh_; d->dw = dw_;
    d->activation = activation_;
    if (use_float32_) { d->W_f = W.cast<float>(); d->b_f = b.cast<float>(); }
    else { d->W_d = W; d->b_d = b; }
    return d;
}

CompiledConv2D::CompiledConv2D(const Eigen::MatrixXd& W, const Eigen::VectorXd& b,
                               int c_in_, int kh_, int kw_, int sh_, int sw_, int ph_, int pw_, int dh_, int dw_,
                               NNActivation activation_, bool use_float32_, std::string name_)
    : data(Conv2DEngineData::create(W, b, c_in_, kh_, kw_, sh_, sw_, ph_, pw_, dh_, dw_, activation_, use_float32_)),
      name(std::move(name_)) {}

Eigen::MatrixXd CompiledConv2D::eval(const Eigen::MatrixXd& X, int h_in_, int w_in_) const {
    if (X.rows() != data->c_in || X.cols() != static_cast<Eigen::Index>(h_in_) * w_in_)
        throw std::invalid_argument("CompiledConv2D::eval: X shape must be (c_in, h_in * w_in).");
    auto [ho, wo] = data->out_hw(h_in_, w_in_);
    Eigen::MatrixXd Y(data->c_out, static_cast<Eigen::Index>(ho) * wo);
    Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), data->c_in, X.cols());
    Eigen::Map<Eigen::MatrixXd> Ym(Y.data(), data->c_out, Y.cols());
    if (data->use_float32) {
        static thread_local Conv2DScratch<float> sc;
        conv2d_run_primal<float>(*data, h_in_, w_in_, ho, wo, Xm, Ym, sc);
    } else {
        static thread_local Conv2DScratch<double> sc;
        conv2d_run_primal<double>(*data, h_in_, w_in_, ho, wo, Xm, Ym, sc);
    }
    return Y;
}

struct Conv2DNode::ScratchD : public Conv2DScratch<double> {};
struct Conv2DNode::ScratchF : public Conv2DScratch<float> {};

Conv2DNode::Conv2DNode(Shape out_shape, int x_id, int h_in_, int w_in_, int h_out_, int w_out_,
                       std::shared_ptr<const Conv2DEngineData> engine_data, std::string name_)
    : Node(OpType::Conv2D, out_shape, {x_id}, std::move(name_)),
      data(std::move(engine_data)), h_in(h_in_), w_in(w_in_), h_out(h_out_), w_out(w_out_) {
    const int nt = std::max(1, omp_get_max_threads());
    if (data->use_float32) thread_scratch_f_.resize(static_cast<std::size_t>(nt));
    else thread_scratch_d_.resize(static_cast<std::size_t>(nt));
}

Conv2DNode::ScratchD& Conv2DNode::ensure_scratch_d(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_d_.size()) {
        #pragma omp critical(fastopt_conv2d_scratch)
        { if (idx >= thread_scratch_d_.size()) thread_scratch_d_.resize(idx + 1); }
    }
    if (!thread_scratch_d_[idx]) thread_scratch_d_[idx] = std::make_shared<ScratchD>();
    return *thread_scratch_d_[idx];
}

Conv2DNode::ScratchF& Conv2DNode::ensure_scratch_f(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_scratch_f_.size()) {
        #pragma omp critical(fastopt_conv2d_scratch)
        { if (idx >= thread_scratch_f_.size()) thread_scratch_f_.resize(idx + 1); }
    }
    if (!thread_scratch_f_[idx]) thread_scratch_f_[idx] = std::make_shared<ScratchF>();
    return *thread_scratch_f_[idx];
}

void Conv2DNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, static_cast<Eigen::Index>(h_in) * w_in);
    Eigen::Map<Eigen::MatrixXd> Ym(ws.val.data() + offset, data->c_out, static_cast<Eigen::Index>(h_out) * w_out);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        conv2d_run_primal<float>(*data, h_in, w_in, h_out, w_out, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size()); sc.has_primal = true; sc.has_tangent = false;
    } else {
        auto& sc = ensure_scratch_d(tid);
        conv2d_run_primal<double>(*data, h_in, w_in, h_out, w_out, Xm, Ym, sc);
        sc.cached_x = ws.val.segment(in.offset, in.size()); sc.has_primal = true; sc.has_tangent = false;
    }
}

void Conv2DNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad || ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const Eigen::Index s_in = static_cast<Eigen::Index>(h_in) * w_in;
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, s_in);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->c_out, s_out);
    Eigen::Map<Eigen::MatrixXd> Xb(ws.bar.data() + in.offset, data->c_in, s_in);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg) pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<float>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv2d_run_backward<float>(*data, h_in, w_in, h_out, w_out, *pr, Yb, Xb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg) pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<double>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv2d_run_backward<double>(*data, h_in, w_in, h_out, w_out, *pr, Yb, Xb, sc);
    }
}

void Conv2DNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) { ws.dot.segment(offset, size()).setZero(); return; }
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    const Eigen::Index s_in = static_cast<Eigen::Index>(h_in) * w_in;
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, s_in);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->c_in, s_in);
    Eigen::Map<Eigen::MatrixXd> Yd(ws.dot.data() + offset, data->c_out, s_out);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg) pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<float>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv2d_run_tangent<float>(*data, h_in, w_in, h_out, w_out, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg) pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<double>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        conv2d_run_tangent<double>(*data, h_in, w_in, h_out, w_out, *pr, Xd, Yd, sc);
        sc.cached_xdot = xd_seg; sc.has_tangent = true;
    }
}

void Conv2DNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0 && ws.dot_bar.segment(offset, size()).squaredNorm() == 0.0) return;
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const auto x_seg = ws.val.segment(in.offset, in.size());
    const auto xd_seg = ws.dot.segment(in.offset, in.size());
    const Eigen::Index s_in = static_cast<Eigen::Index>(h_in) * w_in;
    const Eigen::Index s_out = static_cast<Eigen::Index>(h_out) * w_out;
    Eigen::Map<const Eigen::MatrixXd> Xm(ws.val.data() + in.offset, data->c_in, s_in);
    Eigen::Map<const Eigen::MatrixXd> Xd(ws.dot.data() + in.offset, data->c_in, s_in);
    Eigen::Map<const Eigen::MatrixXd> Yb(ws.bar.data() + offset, data->c_out, s_out);
    Eigen::Map<const Eigen::MatrixXd> Ydb(ws.dot_bar.data() + offset, data->c_out, s_out);
    Eigen::Map<Eigen::MatrixXd> Xdb(ws.dot_bar.data() + in.offset, data->c_in, s_in);
    if (data->use_float32) {
        auto& sc = ensure_scratch_f(tid);
        const ScratchF* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_f_[0] && thread_scratch_f_[0]->has_primal && thread_scratch_f_[0]->cached_x == x_seg) pr = thread_scratch_f_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<float>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->c_out, s_out);
            conv2d_run_tangent<float>(*data, h_in, w_in, h_out, w_out, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        conv2d_run_hessian<float>(*data, h_in, w_in, h_out, w_out, *pr, sc, Yb, Ydb, Xdb, sc);
    } else {
        auto& sc = ensure_scratch_d(tid);
        const ScratchD* pr = &sc;
        if (!(sc.has_primal && sc.cached_x == x_seg)) {
            if (tid > 0 && thread_scratch_d_[0] && thread_scratch_d_[0]->has_primal && thread_scratch_d_[0]->cached_x == x_seg) pr = thread_scratch_d_[0].get();
            else { Eigen::MatrixXd ty(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tym(ty.data(), data->c_out, s_out); conv2d_run_primal<double>(*data, h_in, w_in, h_out, w_out, Xm, tym, sc); sc.cached_x = x_seg; sc.has_primal = true; }
        }
        if (!(sc.has_tangent && sc.cached_xdot == xd_seg)) {
            Eigen::MatrixXd tyd(data->c_out, s_out); Eigen::Map<Eigen::MatrixXd> tydm(tyd.data(), data->c_out, s_out);
            conv2d_run_tangent<double>(*data, h_in, w_in, h_out, w_out, *pr, Xd, tydm, sc); sc.cached_xdot = xd_seg; sc.has_tangent = true;
        }
        conv2d_run_hessian<double>(*data, h_in, w_in, h_out, w_out, *pr, sc, Yb, Ydb, Xdb, sc);
    }
}

// Pool2DNode implementation
Pool2DNode::Pool2DNode(Shape out_shape, int x_id, Pool2DMode mode_,
                       Eigen::Index channels_, int h_in_, int w_in_, int h_out_, int w_out_,
                       int kh_, int kw_, int sh_, int sw_, int ph_, int pw_, std::string name_)
    : Node(OpType::Pool2D, out_shape, {x_id}, std::move(name_)),
      mode(mode_), channels(channels_), h_in(h_in_), w_in(w_in_), h_out(h_out_), w_out(w_out_),
      kh(kh_), kw(kw_), sh(sh_), sw(sw_), ph(ph_), pw(pw_) {
    thread_argmax_.resize(static_cast<std::size_t>(std::max(1, omp_get_max_threads())));
}

std::vector<Eigen::Index>& Pool2DNode::ensure_argmax(int tid) const {
    const std::size_t idx = static_cast<std::size_t>(tid);
    if (idx >= thread_argmax_.size()) {
        #pragma omp critical(fastopt_pool2d_scratch)
        { if (idx >= thread_argmax_.size()) thread_argmax_.resize(idx + 1); }
    }
    return thread_argmax_[idx];
}

void Pool2DNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    const double* xp = ws.val.data() + in.offset;
    double* yp = ws.val.data() + offset;
    auto& am = ensure_argmax(tid);
    if (mode == Pool2DMode::Max) am.resize(static_cast<std::size_t>(size()));

    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index out_col = static_cast<Eigen::Index>(ho) * w_out + wo;
            for (Eigen::Index ch = 0; ch < channels; ++ch) {
                const Eigen::Index out_idx = out_col * channels + ch;
                if (mode == Pool2DMode::Avg) {
                    double acc = 0.0;
                    int cnt = 0;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi < 0 || wi >= w_in) continue;
                            acc += xp[(static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch];
                            ++cnt;
                        }
                    }
                    yp[out_idx] = (cnt > 0) ? (acc / static_cast<double>(cnt)) : 0.0;
                } else {
                    double best = -std::numeric_limits<double>::infinity();
                    Eigen::Index best_idx = -1;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi < 0 || wi >= w_in) continue;
                            const Eigen::Index in_idx = (static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch;
                            if (best_idx < 0 || xp[in_idx] > best) { best = xp[in_idx]; best_idx = in_idx; }
                        }
                    }
                    yp[out_idx] = (best_idx >= 0) ? best : 0.0;
                    am[static_cast<std::size_t>(out_idx)] = best_idx;
                }
            }
        }
    }
}

void Pool2DNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double* xp = ws.val.data() + in.offset;
    const double* yb = ws.bar.data() + offset;
    double* xb = ws.bar.data() + in.offset;
    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index out_col = static_cast<Eigen::Index>(ho) * w_out + wo;
            for (Eigen::Index ch = 0; ch < channels; ++ch) {
                const Eigen::Index out_idx = out_col * channels + ch;
                const double g = yb[out_idx];
                if (g == 0.0) continue;
                if (mode == Pool2DMode::Avg) {
                    int cnt = 0;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi >= 0 && wi < w_in) ++cnt;
                        }
                    }
                    if (cnt == 0) continue;
                    const double gv = g / static_cast<double>(cnt);
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi >= 0 && wi < w_in)
                                xb[(static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch] += gv;
                        }
                    }
                } else {
                    double best = -std::numeric_limits<double>::infinity();
                    Eigen::Index best_idx = -1;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi < 0 || wi >= w_in) continue;
                            const Eigen::Index in_idx = (static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch;
                            if (best_idx < 0 || xp[in_idx] > best) { best = xp[in_idx]; best_idx = in_idx; }
                        }
                    }
                    if (best_idx >= 0) xb[best_idx] += g;
                }
            }
        }
    }
}

void Pool2DNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) { ws.dot.segment(offset, size()).setZero(); return; }
    const double* xp = ws.val.data() + in.offset;
    const double* xd = ws.dot.data() + in.offset;
    double* yd = ws.dot.data() + offset;
    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index out_col = static_cast<Eigen::Index>(ho) * w_out + wo;
            for (Eigen::Index ch = 0; ch < channels; ++ch) {
                const Eigen::Index out_idx = out_col * channels + ch;
                if (mode == Pool2DMode::Avg) {
                    double acc = 0.0;
                    int cnt = 0;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi >= 0 && wi < w_in) {
                                acc += xd[(static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch];
                                ++cnt;
                            }
                        }
                    }
                    yd[out_idx] = (cnt > 0) ? (acc / static_cast<double>(cnt)) : 0.0;
                } else {
                    double best = -std::numeric_limits<double>::infinity();
                    Eigen::Index best_idx = -1;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi < 0 || wi >= w_in) continue;
                            const Eigen::Index in_idx = (static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch;
                            if (best_idx < 0 || xp[in_idx] > best) { best = xp[in_idx]; best_idx = in_idx; }
                        }
                    }
                    yd[out_idx] = (best_idx >= 0) ? xd[best_idx] : 0.0;
                }
            }
        }
    }
}

void Pool2DNode::backward_hessian(Workspace& ws) const {
    // Pool2D (Avg & piecewise-linear Max) has zero 2nd-order local curvature; propagate dot_bar identically to bar.
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double* xp = ws.val.data() + in.offset;
    const double* ydb = ws.dot_bar.data() + offset;
    double* xdb = ws.dot_bar.data() + in.offset;
    for (int ho = 0; ho < h_out; ++ho) {
        for (int wo = 0; wo < w_out; ++wo) {
            const Eigen::Index out_col = static_cast<Eigen::Index>(ho) * w_out + wo;
            for (Eigen::Index ch = 0; ch < channels; ++ch) {
                const Eigen::Index out_idx = out_col * channels + ch;
                const double g = ydb[out_idx];
                if (g == 0.0) continue;
                if (mode == Pool2DMode::Avg) {
                    int cnt = 0;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi >= 0 && wi < w_in) ++cnt;
                        }
                    }
                    if (cnt == 0) continue;
                    const double gv = g / static_cast<double>(cnt);
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi >= 0 && wi < w_in)
                                xdb[(static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch] += gv;
                        }
                    }
                } else {
                    double best = -std::numeric_limits<double>::infinity();
                    Eigen::Index best_idx = -1;
                    for (int r = 0; r < kh; ++r) {
                        const int hi = ho * sh - ph + r;
                        if (hi < 0 || hi >= h_in) continue;
                        for (int c = 0; c < kw; ++c) {
                            const int wi = wo * sw - pw + c;
                            if (wi < 0 || wi >= w_in) continue;
                            const Eigen::Index in_idx = (static_cast<Eigen::Index>(hi) * w_in + wi) * channels + ch;
                            if (best_idx < 0 || xp[in_idx] > best) { best = xp[in_idx]; best_idx = in_idx; }
                        }
                    }
                    if (best_idx >= 0) xdb[best_idx] += g;
                }
            }
        }
    }
}

// RMSNormNode implementation (exact 0th, 1st, and 2nd-order Forward-over-Reverse HVP)
RMSNormNode::RMSNormNode(Shape out_shape, int x_id, Eigen::VectorXd gamma_, double eps_, std::string name_)
    : Node(OpType::RMSNorm, out_shape, {x_id}, std::move(name_)),
      gamma(std::move(gamma_)), eps(eps_), dim(out_shape.rows), cols(out_shape.cols) {}

void RMSNormNode::forward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    const double inv_D = 1.0 / static_cast<double>(dim);
    const bool has_g = (gamma.size() == dim);
    for (Eigen::Index b = 0; b < cols; ++b) {
        const double* x = ws.val.data() + in.offset + b * dim;
        double* y = ws.val.data() + offset + b * dim;
        double ms = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) ms += x[i] * x[i];
        const double s = 1.0 / std::sqrt(ms * inv_D + eps);
        for (Eigen::Index i = 0; i < dim; ++i) {
            y[i] = (has_g ? gamma[i] : 1.0) * (x[i] * s);
        }
    }
}

void RMSNormNode::backward(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double inv_D = 1.0 / static_cast<double>(dim);
    const bool has_g = (gamma.size() == dim);
    for (Eigen::Index b = 0; b < cols; ++b) {
        const double* x = ws.val.data() + in.offset + b * dim;
        const double* yb = ws.bar.data() + offset + b * dim;
        double* xb = ws.bar.data() + in.offset + b * dim;
        double ms = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) ms += x[i] * x[i];
        const double s = 1.0 / std::sqrt(ms * inv_D + eps);
        double m = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const double g = (has_g ? gamma[i] : 1.0) * yb[i];
            m += g * (x[i] * s);
        }
        m *= inv_D;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const double g = (has_g ? gamma[i] : 1.0) * yb[i];
            xb[i] += s * (g - (x[i] * s) * m);
        }
    }
}

void RMSNormNode::forward_tangent(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) { ws.dot.segment(offset, size()).setZero(); return; }
    const double inv_D = 1.0 / static_cast<double>(dim);
    const bool has_g = (gamma.size() == dim);
    for (Eigen::Index b = 0; b < cols; ++b) {
        const double* x = ws.val.data() + in.offset + b * dim;
        const double* xd = ws.dot.data() + in.offset + b * dim;
        double* yd = ws.dot.data() + offset + b * dim;
        double ms = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) ms += x[i] * x[i];
        const double s = 1.0 / std::sqrt(ms * inv_D + eps);
        double p = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) p += (x[i] * s) * xd[i];
        p *= inv_D;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const double dxhat = s * (xd[i] - (x[i] * s) * p);
            yd[i] = (has_g ? gamma[i] : 1.0) * dxhat;
        }
    }
}

void RMSNormNode::backward_hessian(Workspace& ws) const {
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const double inv_D = 1.0 / static_cast<double>(dim);
    const bool has_g = (gamma.size() == dim);
    for (Eigen::Index b = 0; b < cols; ++b) {
        const double* x = ws.val.data() + in.offset + b * dim;
        const double* xd = ws.dot.data() + in.offset + b * dim;
        const double* yb = ws.bar.data() + offset + b * dim;
        const double* ydb = ws.dot_bar.data() + offset + b * dim;
        double* xdb = ws.dot_bar.data() + in.offset + b * dim;
        double ms = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) ms += x[i] * x[i];
        const double s = 1.0 / std::sqrt(ms * inv_D + eps);
        double p = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) p += (x[i] * s) * xd[i];
        p *= inv_D;
        double m = 0.0, dm = 0.0;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const double gam = has_g ? gamma[i] : 1.0;
            const double g = gam * yb[i];
            const double dg = gam * ydb[i];
            const double xh = x[i] * s;
            const double dxh = s * (xd[i] - xh * p);
            m += g * xh;
            dm += dg * xh + g * dxh;
        }
        m *= inv_D;
        dm *= inv_D;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const double gam = has_g ? gamma[i] : 1.0;
            const double g = gam * yb[i];
            const double dg = gam * ydb[i];
            const double xh = x[i] * s;
            const double dxh = s * (xd[i] - xh * p);
            const double xb_i = s * (g - xh * m);
            xdb[i] += -s * p * xb_i + s * (dg - dxh * m - xh * dm);
        }
    }
}


// ============================================================================
// Phase 19: Native C++17 Kernel Surrogate Engine (RBF, Kriging, GEK, KPLS, CoKriging)
// ============================================================================
namespace {

inline void surrogate_kernel_jet3(SurrogateKernel kern, double q, double r_eps,
                                  double& p0, double& p1, double& p2, double& p3) {
    switch (kern) {
        case SurrogateKernel::Gaussian: {
            const double e = std::exp(-q);
            p0 = e; p1 = -e; p2 = e; p3 = -e;
            return;
        }
        case SurrogateKernel::Multiquadric: {
            const double qp1 = q + 1.0;
            const double sq = std::sqrt(qp1);
            const double inv1 = 1.0 / sq;
            const double inv3 = inv1 / qp1;
            const double inv5 = inv3 / qp1;
            p0 = sq;
            p1 = 0.5 * inv1;
            p2 = -0.25 * inv3;
            p3 = 0.375 * inv5;
            return;
        }
        case SurrogateKernel::InvMultiquadric: {
            const double qp1 = q + 1.0;
            const double sq = std::sqrt(qp1);
            const double inv1 = 1.0 / sq;
            const double inv3 = inv1 / qp1;
            const double inv5 = inv3 / qp1;
            const double inv7 = inv5 / qp1;
            p0 = inv1;
            p1 = -0.5 * inv3;
            p2 = 0.75 * inv5;
            p3 = -1.875 * inv7;
            return;
        }
        case SurrogateKernel::Cubic: {
            const double qs = q + r_eps;
            const double r = std::sqrt(qs);
            const double inv_r = 1.0 / r;
            p0 = qs * r;
            p1 = 1.5 * r;
            p2 = 0.75 * inv_r;
            p3 = -0.375 * inv_r / qs;
            return;
        }
        case SurrogateKernel::ThinPlateSpline: {
            const double qs = q + r_eps;
            const double inv1 = 1.0 / qs;
            const double inv2 = inv1 * inv1;
            const double inv3 = inv2 * inv1;
            const double lq = std::log(qs);
            p0 = 0.5 * q * lq;
            p1 = 0.5 * (lq + q * inv1);
            p2 = 0.5 * (q + 2.0 * r_eps) * inv2;
            p3 = -0.5 * (q + 3.0 * r_eps) * inv3;
            return;
        }
        case SurrogateKernel::Matern32: {
            const double s2 = 3.0 * q + r_eps;
            const double s = std::sqrt(s2);
            const double e = std::exp(-s);
            p0 = (1.0 + s) * e;
            p1 = -1.5 * e;
            p2 = 2.25 * e / s;
            p3 = -3.375 * e * (s + 1.0) / (s2 * s);
            return;
        }
        case SurrogateKernel::Matern52: {
            const double s2 = 5.0 * q + r_eps;
            const double s = std::sqrt(s2);
            const double e = std::exp(-s);
            p0 = (1.0 + s + (1.0 / 3.0) * s2) * e;
            p1 = -(5.0 / 6.0) * (1.0 + s) * e;
            p2 = (25.0 / 12.0) * e;
            p3 = -(125.0 / 24.0) * e / s;
            return;
        }
    }
    p0 = 1.0; p1 = 0.0; p2 = 0.0; p3 = 0.0;
}

inline void eval_poly_basis(const Eigen::VectorXd& x, int deg, Eigen::VectorXd& p) {
    const Eigen::Index d = x.size();
    const Eigen::Index np = SurrogateEngineData::poly_dim(d, deg);
    p.resize(np);
    if (deg < 0) return;
    Eigen::Index idx = 0;
    p(idx++) = 1.0;
    if (deg >= 1) {
        for (Eigen::Index j = 0; j < d; ++j) p(idx++) = x(j);
    }
    if (deg >= 2) {
        for (Eigen::Index a = 0; a < d; ++a) {
            const double xa = x(a);
            for (Eigen::Index b = a; b < d; ++b) {
                p(idx++) = xa * x(b);
            }
        }
    }
}

inline void eval_poly_tangent(const Eigen::VectorXd& x, const Eigen::VectorXd& v, int deg, Eigen::VectorXd& dp) {
    const Eigen::Index d = x.size();
    const Eigen::Index np = SurrogateEngineData::poly_dim(d, deg);
    dp.resize(np);
    if (deg < 0) return;
    Eigen::Index idx = 0;
    dp(idx++) = 0.0;
    if (deg >= 1) {
        for (Eigen::Index j = 0; j < d; ++j) dp(idx++) = v(j);
    }
    if (deg >= 2) {
        for (Eigen::Index a = 0; a < d; ++a) {
            for (Eigen::Index b = a; b < d; ++b) {
                dp(idx++) = (a == b) ? (2.0 * x(a) * v(a)) : (x(a) * v(b) + x(b) * v(a));
            }
        }
    }
}

inline void accum_poly_grad(const Eigen::VectorXd& x, const Eigen::VectorXd& w, int deg, Eigen::VectorXd& gx) {
    if (deg < 1 || w.size() == 0) return;
    const Eigen::Index d = x.size();
    Eigen::Index idx = 1;
    for (Eigen::Index j = 0; j < d; ++j) gx(j) += w(idx++);
    if (deg >= 2) {
        for (Eigen::Index a = 0; a < d; ++a) {
            for (Eigen::Index b = a; b < d; ++b) {
                const double cw = w(idx++);
                if (a == b) {
                    gx(a) += 2.0 * cw * x(a);
                } else {
                    gx(a) += cw * x(b);
                    gx(b) += cw * x(a);
                }
            }
        }
    }
}

inline void accum_poly_hvp(const Eigen::VectorXd& v, const Eigen::VectorXd& w, int deg, Eigen::VectorXd& hv) {
    if (deg < 2 || w.size() == 0) return;
    const Eigen::Index d = v.size();
    Eigen::Index idx = 1 + d;
    for (Eigen::Index a = 0; a < d; ++a) {
        for (Eigen::Index b = a; b < d; ++b) {
            const double cw = w(idx++);
            if (a == b) {
                hv(a) += 2.0 * cw * v(a);
            } else {
                hv(a) += cw * v(b);
                hv(b) += cw * v(a);
            }
        }
    }
}

inline void accum_kernel_lincomb_grad_and_hvp(
    const SurrogateEngineData& D,
    const Eigen::VectorXd& xt,
    const Eigen::VectorXd* vt,
    const Eigen::VectorXd& w_val,
    const Eigen::VectorXd* w_grad,
    bool w_grad_prescaled,
    Eigen::VectorXd* gx_out,
    Eigen::VectorXd* hv_out,
    double scale = 1.0) {
    const Eigen::Index d = D.d_in;
    const Eigen::Index N = D.n_centers;
    Eigen::VectorXd u(d), tu(d), tv;
    if (vt) tv = D.theta.cwiseProduct(*vt);

    for (Eigen::Index i = 0; i < N; ++i) {
        u = xt - D.X_centers.col(i);
        tu = D.theta.cwiseProduct(u);
        const double q = u.dot(tu);
        double p0, p1, p2, p3;
        surrogate_kernel_jet3(D.kernel, q, D.r_eps, p0, p1, p2, p3);

        const double a_i = w_val(i);
        double s_i = 0.0;
        double ds_i = 0.0;
        const bool use_g = (w_grad != nullptr && w_grad->size() == N * d);
        if (use_g) {
            const auto gi = w_grad->segment(i * d, d);
            if (w_grad_prescaled) {
                s_i = gi.dot(u);
                if (vt) ds_i = gi.dot(*vt);
            } else {
                s_i = gi.dot(tu);
                if (vt) ds_i = gi.dot(tv);
            }
        }

        const double c1 = 2.0 * a_i * p1 - 4.0 * p2 * s_i;
        if (gx_out) {
            gx_out->noalias() += (scale * c1) * tu;
            if (use_g) {
                const auto gi = w_grad->segment(i * d, d);
                if (w_grad_prescaled) {
                    gx_out->noalias() -= (scale * 2.0 * p1) * gi;
                } else {
                    gx_out->noalias() -= (scale * 2.0 * p1) * D.theta.cwiseProduct(gi);
                }
            }
        }
        if (hv_out && vt) {
            const double dq = 2.0 * tu.dot(*vt);
            const double dc1 = (2.0 * a_i * p2 - 4.0 * p3 * s_i) * dq - 4.0 * p2 * ds_i;
            hv_out->noalias() += (scale * dc1) * tu + (scale * c1) * tv;
            if (use_g) {
                const auto gi = w_grad->segment(i * d, d);
                if (w_grad_prescaled) {
                    hv_out->noalias() -= (scale * 2.0 * p2 * dq) * gi;
                } else {
                    hv_out->noalias() -= (scale * 2.0 * p2 * dq) * D.theta.cwiseProduct(gi);
                }
            }
        }
    }
}

inline void eval_kernel_basis_and_tangent(
    const SurrogateEngineData& D,
    const Eigen::VectorXd& xt,
    const Eigen::VectorXd* vt,
    Eigen::VectorXd& k_out,
    Eigen::VectorXd* dk_out) {
    const Eigen::Index d = D.d_in;
    const Eigen::Index N = D.n_centers;
    const Eigen::Index M = D.has_grad_basis ? N * (1 + d) : N;
    k_out.resize(M);
    if (dk_out) dk_out->resize(M);

    Eigen::VectorXd u(d), tu(d), tv;
    if (vt) tv = D.theta.cwiseProduct(*vt);

    for (Eigen::Index i = 0; i < N; ++i) {
        u = xt - D.X_centers.col(i);
        tu = D.theta.cwiseProduct(u);
        const double q = u.dot(tu);
        double p0, p1, p2, p3;
        surrogate_kernel_jet3(D.kernel, q, D.r_eps, p0, p1, p2, p3);

        k_out(i) = p0;
        double dq = 0.0;
        if (vt && dk_out) {
            dq = 2.0 * tu.dot(*vt);
            (*dk_out)(i) = p1 * dq;
        }
        if (D.has_grad_basis) {
            const Eigen::Index base = N + i * d;
            k_out.segment(base, d) = (-2.0 * p1) * tu;
            if (vt && dk_out) {
                dk_out->segment(base, d) = (-2.0 * p2 * dq) * tu - (2.0 * p1) * tv;
            }
        }
    }
}


// Numerically rock-solid Cholesky evaluation of Universal Kriging / GEK variance:
// D.M_kk stores lower-triangular L_R (R = L_R L_R^T)
// D.M_kp stores W_F = L_R^{-1} F (shape M x n_poly)
// D.M_pp stores lower-triangular L_A (W_F^T W_F = L_A L_A^T)
// V(x) = 1 - ||z1||^2 + ||z2||^2, where z1 = L_R^{-1} k, u = W_F^T z1 - p, z2 = L_A^{-1} u.
inline void eval_cholesky_variance_state(
    const SurrogateEngineData& D,
    const Eigen::VectorXd& k_vec,
    const Eigen::VectorXd& p_vec,
    double& v_raw,
    Eigen::VectorXd* qk_out,
    Eigen::VectorXd* qp_out) {
    Eigen::VectorXd z1 = D.M_kk.triangularView<Eigen::Lower>().solve(k_vec);
    double v = 1.0 - z1.squaredNorm();
    Eigen::VectorXd a_vec;
    if (D.n_poly > 0) {
        Eigen::VectorXd u = D.M_kp.transpose() * z1 - p_vec;
        Eigen::VectorXd z2 = D.M_pp.triangularView<Eigen::Lower>().solve(u);
        v += z2.squaredNorm();
        if (qk_out || qp_out) {
            a_vec = D.M_pp.triangularView<Eigen::Lower>().transpose().solve(z2);
        }
    }
    v_raw = v;
    if (qk_out) {
        Eigen::VectorXd rhs = z1;
        if (D.n_poly > 0) rhs.noalias() -= D.M_kp * a_vec;
        *qk_out = D.M_kk.triangularView<Eigen::Lower>().transpose().solve(rhs);
    }
    if (qp_out && D.n_poly > 0) {
        *qp_out = a_vec;
    }
}

inline void eval_cholesky_variance_tangent(
    const SurrogateEngineData& D,
    const Eigen::VectorXd& dk_vec,
    const Eigen::VectorXd& dp_vec,
    Eigen::VectorXd& dqk_out,
    Eigen::VectorXd& dqp_out) {
    Eigen::VectorXd dz1 = D.M_kk.triangularView<Eigen::Lower>().solve(dk_vec);
    Eigen::VectorXd da_vec;
    if (D.n_poly > 0) {
        Eigen::VectorXd du = D.M_kp.transpose() * dz1 - dp_vec;
        Eigen::VectorXd dz2 = D.M_pp.triangularView<Eigen::Lower>().solve(du);
        da_vec = D.M_pp.triangularView<Eigen::Lower>().transpose().solve(dz2);
        dqp_out = da_vec;
    }
    Eigen::VectorXd rhs = dz1;
    if (D.n_poly > 0) rhs.noalias() -= D.M_kp * da_vec;
    dqk_out = D.M_kk.triangularView<Eigen::Lower>().transpose().solve(rhs);
}

} // anonymous namespace

std::shared_ptr<const SurrogateEngineData> SurrogateEngineData::create(
    SurrogateKernel kernel,
    int poly_degree,
    const Eigen::MatrixXd& X_centers,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& alpha,
    const Eigen::MatrixXd& beta_grad,
    const Eigen::MatrixXd& poly_coef,
    const Eigen::VectorXd& in_mean,
    const Eigen::VectorXd& in_inv_std,
    const Eigen::VectorXd& out_mean,
    const Eigen::VectorXd& out_std,
    const Eigen::VectorXd& sigma2,
    const Eigen::MatrixXd& M_kk,
    const Eigen::MatrixXd& M_kp,
    const Eigen::MatrixXd& M_pp,
    double r_eps) {
    if (X_centers.rows() <= 0 || X_centers.cols() <= 0) {
        throw std::invalid_argument("Surrogate: X_centers must be non-empty (d_in, n_centers).");
    }
    const Eigen::Index d_in = X_centers.rows();
    const Eigen::Index N = X_centers.cols();
    if (theta.size() != d_in) {
        throw std::invalid_argument("Surrogate: theta length must equal d_in.");
    }
    if (alpha.rows() != N || alpha.cols() <= 0) {
        throw std::invalid_argument("Surrogate: alpha must have shape (n_centers, d_out).");
    }
    const Eigen::Index d_out = alpha.cols();
    if (poly_degree < -1 || poly_degree > 2) {
        throw std::invalid_argument("Surrogate: poly_degree must be -1, 0, 1, or 2.");
    }
    const Eigen::Index np = poly_dim(d_in, poly_degree);

    auto data = std::make_shared<SurrogateEngineData>();
    data->kernel = kernel;
    data->poly_degree = poly_degree;
    data->d_in = d_in;
    data->d_out = d_out;
    data->n_centers = N;
    data->n_poly = np;
    data->r_eps = r_eps;
    data->X_centers = X_centers;
    data->theta = theta;
    data->alpha = alpha;

    data->in_mean = (in_mean.size() == d_in) ? in_mean : Eigen::VectorXd::Zero(d_in);
    data->in_inv_std = (in_inv_std.size() == d_in) ? in_inv_std : Eigen::VectorXd::Ones(d_in);
    data->out_mean = (out_mean.size() == d_out) ? out_mean : Eigen::VectorXd::Zero(d_out);
    data->out_std = (out_std.size() == d_out) ? out_std : Eigen::VectorXd::Ones(d_out);

    if (beta_grad.size() > 0) {
        if (beta_grad.rows() != N * d_in || beta_grad.cols() != d_out) {
            throw std::invalid_argument("Surrogate: beta_grad must have shape (n_centers * d_in, d_out).");
        }
        data->has_grad_basis = true;
        data->beta_scaled.resize(N * d_in, d_out);
        for (Eigen::Index i = 0; i < N; ++i) {
            data->beta_scaled.middleRows(i * d_in, d_in) =
                beta_grad.middleRows(i * d_in, d_in).array().colwise() * theta.array();
        }
    }

    if (np > 0) {
        if (poly_coef.size() == 0) {
            data->poly_coef = Eigen::MatrixXd::Zero(np, d_out);
        } else {
            if (poly_coef.rows() != np || poly_coef.cols() != d_out) {
                throw std::invalid_argument("Surrogate: poly_coef shape mismatch.");
            }
            data->poly_coef = poly_coef;
        }
    }

    if (sigma2.size() > 0 && M_kk.size() > 0) {
        const Eigen::Index M = data->has_grad_basis ? N * (1 + d_in) : N;
        if (sigma2.size() != d_out || M_kk.rows() != M || M_kk.cols() != M) {
            throw std::invalid_argument("Surrogate: sigma2 or M_kk shape mismatch.");
        }
        data->has_variance = true;
        data->sigma2 = sigma2;
        data->M_kk = M_kk;
        if (np > 0) {
            if (M_kp.rows() != M || M_kp.cols() != np || M_pp.rows() != np || M_pp.cols() != np) {
                throw std::invalid_argument("Surrogate: M_kp or M_pp shape mismatch.");
            }
            data->M_kp = M_kp;
            data->M_pp = M_pp;
        }
    }
    return data;
}

CompiledSurrogate::CompiledSurrogate(
    SurrogateKernel kernel,
    int poly_degree,
    const Eigen::MatrixXd& X_centers,
    const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& alpha,
    const Eigen::MatrixXd& beta_grad,
    const Eigen::MatrixXd& poly_coef,
    const Eigen::VectorXd& in_mean,
    const Eigen::VectorXd& in_inv_std,
    const Eigen::VectorXd& out_mean,
    const Eigen::VectorXd& out_std,
    const Eigen::VectorXd& sigma2,
    const Eigen::MatrixXd& M_kk,
    const Eigen::MatrixXd& M_kp,
    const Eigen::MatrixXd& M_pp,
    double r_eps,
    std::string name_)
    : data(SurrogateEngineData::create(
          kernel, poly_degree, X_centers, theta, alpha, beta_grad, poly_coef,
          in_mean, in_inv_std, out_mean, out_std, sigma2, M_kk, M_kp, M_pp, r_eps)),
      name(std::move(name_)) {}

Eigen::MatrixXd CompiledSurrogate::eval(const Eigen::MatrixXd& X, int num_threads) const {
    const SurrogateEngineData& D = *data;
    if (X.rows() != D.d_in) {
        throw std::invalid_argument("CompiledSurrogate::eval: X rows must equal d_in.");
    }
    const Eigen::Index B = X.cols();
    Eigen::MatrixXd Y(D.d_out, B);
#ifdef _OPENMP
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(threads) if (B >= 32 && threads > 1)
#endif
    for (Eigen::Index col = 0; col < B; ++col) {
        Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
        Eigen::VectorXd y_col = Eigen::VectorXd::Zero(D.d_out);
        Eigen::VectorXd u(D.d_in);
        for (Eigen::Index i = 0; i < D.n_centers; ++i) {
            u = xt - D.X_centers.col(i);
            const double q = u.dot(D.theta.cwiseProduct(u));
            double p0, p1, p2, p3;
            surrogate_kernel_jet3(D.kernel, q, D.r_eps, p0, p1, p2, p3);
            y_col.noalias() += p0 * D.alpha.row(i).transpose();
            if (D.has_grad_basis) {
                y_col.noalias() -= (2.0 * p1) * (D.beta_scaled.middleRows(i * D.d_in, D.d_in).transpose() * u);
            }
        }
        if (D.n_poly > 0) {
            Eigen::VectorXd p;
            eval_poly_basis(xt, D.poly_degree, p);
            y_col.noalias() += D.poly_coef.transpose() * p;
        }
        Y.col(col) = D.out_mean + D.out_std.cwiseProduct(y_col);
    }
    return Y;
}

Eigen::MatrixXd CompiledSurrogate::eval_variance(const Eigen::MatrixXd& X, int num_threads) const {
    const SurrogateEngineData& D = *data;
    if (!D.has_variance) {
        throw std::runtime_error("CompiledSurrogate::eval_variance: variance matrices not provided for this surrogate.");
    }
    if (X.rows() != D.d_in) {
        throw std::invalid_argument("CompiledSurrogate::eval_variance: X rows must equal d_in.");
    }
    const Eigen::Index B = X.cols();
    Eigen::MatrixXd V(D.d_out, B);
#ifdef _OPENMP
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(threads) if (B >= 32 && threads > 1)
#endif
    for (Eigen::Index col = 0; col < B; ++col) {
        Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
        Eigen::VectorXd k_vec, p_vec;
        eval_kernel_basis_and_tangent(D, xt, nullptr, k_vec, nullptr);
        if (D.n_poly > 0) eval_poly_basis(xt, D.poly_degree, p_vec);
        double v_unclamped = 0.0;
        eval_cholesky_variance_state(D, k_vec, p_vec, v_unclamped, nullptr, nullptr);
        const double v_raw = std::max(0.0, v_unclamped);
        V.col(col) = (D.out_std.array().square() * D.sigma2.array() * v_raw).matrix();
    }
    return V;
}

SurrogateNode::SurrogateNode(Shape out_shape, int x_id,
                             std::shared_ptr<const SurrogateEngineData> engine_data,
                             SurrogateOutputMode mode_,
                             std::string name)
    : Node(OpType::Surrogate, out_shape, {x_id}, std::move(name)),
      data(std::move(engine_data)),
      mode(mode_) {
    if (mode == SurrogateOutputMode::Variance && !data->has_variance) {
        throw std::invalid_argument("SurrogateNode: Variance mode requested but surrogate has no variance matrices.");
    }
}

void SurrogateNode::forward(Workspace& ws) const {
    const SurrogateEngineData& D = *data;
    const Node& in = graph->node(inputs[0]);
    const auto X = ws.val_mat(in.offset, in.shape);
    auto Y = ws.val_mat(offset, shape);
    const Eigen::Index B = shape.cols;

    if (mode == SurrogateOutputMode::Mean) {
        Eigen::VectorXd u(D.d_in), p;
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd y_col = Eigen::VectorXd::Zero(D.d_out);
            for (Eigen::Index i = 0; i < D.n_centers; ++i) {
                u = xt - D.X_centers.col(i);
                const double q = u.dot(D.theta.cwiseProduct(u));
                double p0, p1, p2, p3;
                surrogate_kernel_jet3(D.kernel, q, D.r_eps, p0, p1, p2, p3);
                y_col.noalias() += p0 * D.alpha.row(i).transpose();
                if (D.has_grad_basis) {
                    y_col.noalias() -= (2.0 * p1) * (D.beta_scaled.middleRows(i * D.d_in, D.d_in).transpose() * u);
                }
            }
            if (D.n_poly > 0) {
                eval_poly_basis(xt, D.poly_degree, p);
                y_col.noalias() += D.poly_coef.transpose() * p;
            }
            Y.col(col) = D.out_mean + D.out_std.cwiseProduct(y_col);
        }
    } else {
        Eigen::VectorXd k_vec, p_vec;
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            eval_kernel_basis_and_tangent(D, xt, nullptr, k_vec, nullptr);
            if (D.n_poly > 0) eval_poly_basis(xt, D.poly_degree, p_vec);
            double v_unclamped = 0.0;
            eval_cholesky_variance_state(D, k_vec, p_vec, v_unclamped, nullptr, nullptr);
            const double v_raw = std::max(0.0, v_unclamped);
            Y.col(col) = (D.out_std.array().square() * D.sigma2.array() * v_raw).matrix();
        }
    }
}

void SurrogateNode::backward(Workspace& ws) const {
    const SurrogateEngineData& D = *data;
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const auto X = ws.val_mat(in.offset, in.shape);
    const auto G_out = ws.bar_mat(offset, shape);
    auto G_in = ws.bar_mat(in.offset, in.shape);
    const Eigen::Index B = shape.cols;

    if (mode == SurrogateOutputMode::Mean) {
        Eigen::VectorXd gx(D.d_in);
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd wy = G_out.col(col).cwiseProduct(D.out_std);
            Eigen::VectorXd w_val = D.alpha * wy;
            Eigen::VectorXd w_grad;
            const Eigen::VectorXd* w_grad_ptr = nullptr;
            if (D.has_grad_basis) {
                w_grad = D.beta_scaled * wy;
                w_grad_ptr = &w_grad;
            }
            gx.setZero();
            accum_kernel_lincomb_grad_and_hvp(D, xt, nullptr, w_val, w_grad_ptr, true, &gx, nullptr, 1.0);
            if (D.n_poly > 0) {
                Eigen::VectorXd wp = D.poly_coef * wy;
                accum_poly_grad(xt, wp, D.poly_degree, gx);
            }
            G_in.col(col).noalias() += gx.cwiseProduct(D.in_inv_std);
        }
    } else {
        Eigen::VectorXd k_vec, p_vec, gx(D.d_in);
        for (Eigen::Index col = 0; col < B; ++col) {
            const double w_var = (G_out.col(col).array() * D.out_std.array().square() * D.sigma2.array()).sum();
            if (w_var == 0.0) continue;
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            eval_kernel_basis_and_tangent(D, xt, nullptr, k_vec, nullptr);
            if (D.n_poly > 0) eval_poly_basis(xt, D.poly_degree, p_vec);
            double v_unclamped = 0.0;
            Eigen::VectorXd qk, qp;
            eval_cholesky_variance_state(D, k_vec, p_vec, v_unclamped, &qk, &qp);
            if (v_unclamped <= 0.0) continue;
            Eigen::VectorXd qk_val = qk.head(D.n_centers);
            Eigen::VectorXd qk_grad;
            const Eigen::VectorXd* qk_grad_ptr = nullptr;
            if (D.has_grad_basis) {
                qk_grad = qk.tail(D.n_centers * D.d_in);
                qk_grad_ptr = &qk_grad;
            }
            gx.setZero();
            accum_kernel_lincomb_grad_and_hvp(D, xt, nullptr, qk_val, qk_grad_ptr, false, &gx, nullptr, -2.0 * w_var);
            if (D.n_poly > 0) {
                Eigen::VectorXd wp = (-2.0 * w_var) * qp;
                accum_poly_grad(xt, wp, D.poly_degree, gx);
            }
            G_in.col(col).noalias() += gx.cwiseProduct(D.in_inv_std);
        }
    }
}

void SurrogateNode::forward_tangent(Workspace& ws) const {
    const SurrogateEngineData& D = *data;
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) {
        ws.dot_mat(offset, shape).setZero();
        return;
    }
    const auto X = ws.val_mat(in.offset, in.shape);
    const auto dX = ws.dot_mat(in.offset, in.shape);
    auto dY = ws.dot_mat(offset, shape);
    const Eigen::Index B = shape.cols;

    if (mode == SurrogateOutputMode::Mean) {
        Eigen::VectorXd u(D.d_in), tu(D.d_in), tv(D.d_in), dp;
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd vt = dX.col(col).cwiseProduct(D.in_inv_std);
            tv = D.theta.cwiseProduct(vt);
            Eigen::VectorXd dy_col = Eigen::VectorXd::Zero(D.d_out);
            for (Eigen::Index i = 0; i < D.n_centers; ++i) {
                u = xt - D.X_centers.col(i);
                tu = D.theta.cwiseProduct(u);
                const double q = u.dot(tu);
                const double dq = 2.0 * tu.dot(vt);
                double p0, p1, p2, p3;
                surrogate_kernel_jet3(D.kernel, q, D.r_eps, p0, p1, p2, p3);
                dy_col.noalias() += (p1 * dq) * D.alpha.row(i).transpose();
                if (D.has_grad_basis) {
                    const auto Bi = D.beta_scaled.middleRows(i * D.d_in, D.d_in);
                    dy_col.noalias() -= (2.0 * p2 * dq) * (Bi.transpose() * u) + (2.0 * p1) * (Bi.transpose() * vt);
                }
            }
            if (D.n_poly > 0) {
                eval_poly_tangent(xt, vt, D.poly_degree, dp);
                dy_col.noalias() += D.poly_coef.transpose() * dp;
            }
            dY.col(col) = D.out_std.cwiseProduct(dy_col);
        }
    } else {
        Eigen::VectorXd k_vec, dk_vec, p_vec, dp_vec;
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd vt = dX.col(col).cwiseProduct(D.in_inv_std);
            eval_kernel_basis_and_tangent(D, xt, &vt, k_vec, &dk_vec);
            if (D.n_poly > 0) {
                eval_poly_basis(xt, D.poly_degree, p_vec);
                eval_poly_tangent(xt, vt, D.poly_degree, dp_vec);
            }
            double v_unclamped = 0.0;
            Eigen::VectorXd qk, qp;
            eval_cholesky_variance_state(D, k_vec, p_vec, v_unclamped, &qk, &qp);
            if (v_unclamped <= 0.0) {
                dY.col(col).setZero();
                continue;
            }
            double dv = -2.0 * dk_vec.dot(qk);
            if (D.n_poly > 0) dv -= 2.0 * dp_vec.dot(qp);
            dY.col(col) = (D.out_std.array().square() * D.sigma2.array() * dv).matrix();
        }
    }
}

void SurrogateNode::backward_hessian(Workspace& ws) const {
    const SurrogateEngineData& D = *data;
    const Node& in = graph->node(inputs[0]);
    if (!in.requires_grad) return;
    const auto X = ws.val_mat(in.offset, in.shape);
    const auto dX = ws.dot_mat(in.offset, in.shape);
    const auto G_out = ws.bar_mat(offset, shape);
    const auto dG_out = ws.dot_bar_mat(offset, shape);
    auto dG_in = ws.dot_bar_mat(in.offset, in.shape);
    const Eigen::Index B = shape.cols;

    if (mode == SurrogateOutputMode::Mean) {
        Eigen::VectorXd hv(D.d_in);
        for (Eigen::Index col = 0; col < B; ++col) {
            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd vt = dX.col(col).cwiseProduct(D.in_inv_std);
            hv.setZero();

            Eigen::VectorXd dwy = dG_out.col(col).cwiseProduct(D.out_std);
            Eigen::VectorXd dw_val = D.alpha * dwy;
            Eigen::VectorXd dw_grad;
            const Eigen::VectorXd* dw_grad_ptr = nullptr;
            if (D.has_grad_basis) {
                dw_grad = D.beta_scaled * dwy;
                dw_grad_ptr = &dw_grad;
            }
            accum_kernel_lincomb_grad_and_hvp(D, xt, nullptr, dw_val, dw_grad_ptr, true, &hv, nullptr, 1.0);
            if (D.n_poly > 0) {
                Eigen::VectorXd dwp = D.poly_coef * dwy;
                accum_poly_grad(xt, dwp, D.poly_degree, hv);
            }

            Eigen::VectorXd wy = G_out.col(col).cwiseProduct(D.out_std);
            Eigen::VectorXd w_val = D.alpha * wy;
            Eigen::VectorXd w_grad;
            const Eigen::VectorXd* w_grad_ptr = nullptr;
            if (D.has_grad_basis) {
                w_grad = D.beta_scaled * wy;
                w_grad_ptr = &w_grad;
            }
            accum_kernel_lincomb_grad_and_hvp(D, xt, &vt, w_val, w_grad_ptr, true, nullptr, &hv, 1.0);
            if (D.n_poly > 0) {
                Eigen::VectorXd wp = D.poly_coef * wy;
                accum_poly_hvp(vt, wp, D.poly_degree, hv);
            }

            dG_in.col(col).noalias() += hv.cwiseProduct(D.in_inv_std);
        }
    } else {
        Eigen::VectorXd k_vec, dk_vec, p_vec, dp_vec, hv(D.d_in);
        for (Eigen::Index col = 0; col < B; ++col) {
            const double w_var = (G_out.col(col).array() * D.out_std.array().square() * D.sigma2.array()).sum();
            const double dw_var = (dG_out.col(col).array() * D.out_std.array().square() * D.sigma2.array()).sum();
            if (w_var == 0.0 && dw_var == 0.0) continue;

            Eigen::VectorXd xt = (X.col(col) - D.in_mean).cwiseProduct(D.in_inv_std);
            Eigen::VectorXd vt = dX.col(col).cwiseProduct(D.in_inv_std);
            eval_kernel_basis_and_tangent(D, xt, &vt, k_vec, &dk_vec);
            if (D.n_poly > 0) {
                eval_poly_basis(xt, D.poly_degree, p_vec);
                eval_poly_tangent(xt, vt, D.poly_degree, dp_vec);
            }
            double v_unclamped = 0.0;
            Eigen::VectorXd qk, qp, dqk, dqp;
            eval_cholesky_variance_state(D, k_vec, p_vec, v_unclamped, &qk, &qp);
            if (v_unclamped <= 0.0) continue;
            eval_cholesky_variance_tangent(D, dk_vec, dp_vec, dqk, dqp);

            hv.setZero();
            Eigen::VectorXd w_lin_k = (-2.0 * dw_var) * qk - (2.0 * w_var) * dqk;
            Eigen::VectorXd wl_val = w_lin_k.head(D.n_centers);
            Eigen::VectorXd wl_grad;
            const Eigen::VectorXd* wl_grad_ptr = nullptr;
            if (D.has_grad_basis) {
                wl_grad = w_lin_k.tail(D.n_centers * D.d_in);
                wl_grad_ptr = &wl_grad;
            }
            accum_kernel_lincomb_grad_and_hvp(D, xt, nullptr, wl_val, wl_grad_ptr, false, &hv, nullptr, 1.0);
            if (D.n_poly > 0) {
                Eigen::VectorXd w_lin_p = (-2.0 * dw_var) * qp - (2.0 * w_var) * dqp;
                accum_poly_grad(xt, w_lin_p, D.poly_degree, hv);
            }

            if (w_var != 0.0) {
                Eigen::VectorXd qk_val = qk.head(D.n_centers);
                Eigen::VectorXd qk_grad;
                const Eigen::VectorXd* qk_grad_ptr = nullptr;
                if (D.has_grad_basis) {
                    qk_grad = qk.tail(D.n_centers * D.d_in);
                    qk_grad_ptr = &qk_grad;
                }
                accum_kernel_lincomb_grad_and_hvp(D, xt, &vt, qk_val, qk_grad_ptr, false, nullptr, &hv, -2.0 * w_var);
                if (D.n_poly > 0) {
                    Eigen::VectorXd wp = (-2.0 * w_var) * qp;
                    accum_poly_hvp(vt, wp, D.poly_degree, hv);
                }
            }

            dG_in.col(col).noalias() += hv.cwiseProduct(D.in_inv_std);
        }
    }
}

namespace ops {

namespace {

inline double eval_unary_scalar(OpType op, double x) {
    switch (op) {
        case OpType::Neg:    return -x;
        case OpType::Sin:    return std::sin(x);
        case OpType::Cos:    return std::cos(x);
        case OpType::Tan:    return std::tan(x);
        case OpType::Tanh:   return std::tanh(x);
        case OpType::Exp:    return std::exp(x);
        case OpType::Log:    return std::log(x);
        case OpType::Sqrt:   return std::sqrt(x);
        case OpType::Abs:    return std::abs(x);
        case OpType::Square: return x * x;
        case OpType::Asin:   return std::asin(x);
        case OpType::Acos:   return std::acos(x);
        case OpType::Atan:   return std::atan(x);
        case OpType::Sinh:   return std::sinh(x);
        case OpType::Cosh:   return std::cosh(x);
        default: return 0.0;
    }
}

} // namespace

int unary(Graph& g, OpType op, int a) {
    const Node& na = g.node(a);
    const Shape s = na.shape;

    // 1. Constant folding when operand is an immutable ConstantNode
    if (g.cse_enabled() && na.op == OpType::Constant) {
        const auto& ca = static_cast<const ConstantNode&>(na);
        Eigen::VectorXd folded(s.size());
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            folded(i) = eval_unary_scalar(op, ca.const_val(i));
        }
        if (folded.allFinite()) {
            g.record_cse_hit();
            return s.is_scalar() ? g.add_scalar_constant(folded(0)) : g.add_constant(s, folded);
        }
    }

    // 2. CSE structural hash-consing
    const std::string key = "U:" + std::to_string(static_cast<int>(op)) + ":" + std::to_string(a);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<UnaryOpNode>(op, s, a));
    g.register_cse_node(key, id);
    return id;
}

int binary(Graph& g, OpType op, int a, int b) {
    const Node& na = g.node(a);
    const Node& nb = g.node(b);
    const Shape s = broadcast_shape(na.shape, nb.shape);

    // 1. Constant folding when both operands are immutable ConstantNodes (excluding comparison masks)
    const bool is_cmp = (op == OpType::Lt || op == OpType::Le || op == OpType::Gt ||
                         op == OpType::Ge || op == OpType::Eq);
    if (g.cse_enabled() && !is_cmp && na.op == OpType::Constant && nb.op == OpType::Constant) {
        const auto& ca = static_cast<const ConstantNode&>(na);
        const auto& cb = static_cast<const ConstantNode&>(nb);
        Eigen::VectorXd folded(s.size());
        Eigen::Map<const Eigen::MatrixXd> A(ca.const_val.data(), na.shape.rows, na.shape.cols);
        Eigen::Map<const Eigen::MatrixXd> B(cb.const_val.data(), nb.shape.rows, nb.shape.cols);
        Eigen::Map<Eigen::MatrixXd> Z(folded.data(), s.rows, s.cols);
        for (Eigen::Index c = 0; c < s.cols; ++c) {
            const Eigen::Index ca_idx = (na.shape.cols == 1) ? 0 : c;
            const Eigen::Index cb_idx = (nb.shape.cols == 1) ? 0 : c;
            for (Eigen::Index r = 0; r < s.rows; ++r) {
                const Eigen::Index ra_idx = (na.shape.rows == 1) ? 0 : r;
                const Eigen::Index rb_idx = (nb.shape.rows == 1) ? 0 : r;
                Z(r, c) = eval_binary_scalar(op, A(ra_idx, ca_idx), B(rb_idx, cb_idx));
            }
        }
        if (folded.allFinite()) {
            g.record_cse_hit();
            return s.is_scalar() ? g.add_scalar_constant(folded(0)) : g.add_constant(s, folded);
        }
    }

    // 2. Algebraic simplification: x ** 2.0 -> unary Square(x)
    if (op == OpType::Pow && nb.op == OpType::Constant && nb.shape.is_scalar()) {
        const double p = static_cast<const ConstantNode&>(nb).const_val(0);
        if (p == 2.0) {
            return unary(g, OpType::Square, a);
        }
        if (p == 1.0 && na.shape == s) {
            return a;
        }
    }

    // 3. CSE structural hash-consing (canonicalizing commutative Add / Mul)
    int k_a = a;
    int k_b = b;
    if ((op == OpType::Add || op == OpType::Mul) && k_a > k_b) {
        std::swap(k_a, k_b);
    }
    const std::string key =
        "B:" + std::to_string(static_cast<int>(op)) + ":" + std::to_string(k_a) + ":" + std::to_string(k_b);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<BinaryOpNode>(op, s, a, b));
    g.register_cse_node(key, id);
    return id;
}

int matmul(Graph& g, int a, int b) {
    const Shape sa = g.node(a).shape;
    const Shape sb = g.node(b).shape;
    if (sa.cols != sb.rows) {
        throw std::invalid_argument("matmul dimension mismatch: " +
                                    sa.to_string() + " vs " + sb.to_string());
    }
    const std::string key = "MM:" + std::to_string(a) + ":" + std::to_string(b);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<MatMulNode>(Shape(sa.rows, sb.cols), a, b));
    g.register_cse_node(key, id);
    return id;
}

int linear_solve(Graph& g, int a, int b) {
    const Shape sa = g.node(a).shape;
    const Shape sb = g.node(b).shape;
    if (sa.rows != sa.cols || sa.rows < 1) {
        throw std::invalid_argument("linear_solve requires square coefficient matrix A, got " +
                                    sa.to_string());
    }
    if (sa.rows != sb.rows) {
        throw std::invalid_argument("linear_solve dimension mismatch between A " +
                                    sa.to_string() + " and B " + sb.to_string());
    }
    const std::string key = "SOLVE:" + std::to_string(a) + ":" + std::to_string(b);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<LinearSolveNode>(Shape(sb.rows, sb.cols), a, b));
    g.register_cse_node(key, id);
    return id;
}

int dot(Graph& g, int a, int b) {
    const Shape sa = g.node(a).shape;
    const Shape sb = g.node(b).shape;
    if (sa.size() != sb.size()) {
        throw std::invalid_argument("dot size mismatch: " +
                                    sa.to_string() + " vs " + sb.to_string());
    }
    const int k_a = std::min(a, b);
    const int k_b = std::max(a, b);
    const std::string key = "DOT:" + std::to_string(k_a) + ":" + std::to_string(k_b);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<DotNode>(a, b));
    g.register_cse_node(key, id);
    return id;
}

int sum(Graph& g, int a, int axis) {
    const Shape sa = g.node(a).shape;
    Shape out_s(1, 1);
    if (axis == 0) {
        out_s = Shape(1, sa.cols);
    } else if (axis == 1) {
        out_s = Shape(sa.rows, 1);
    } else if (axis != -1) {
        throw std::invalid_argument("sum: axis must be -1, 0, or 1.");
    }
    const std::string key = "SUM:" + std::to_string(a) + ":" + std::to_string(axis);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<SumNode>(out_s, a, axis));
    g.register_cse_node(key, id);
    return id;
}

int ks_logsumexp(Graph& g, int a, double rho, int axis, bool subtract_log_n) {
    if (rho == 0.0) {
        throw std::invalid_argument("ks_logsumexp: rho must be non-zero.");
    }
    const Shape sa = g.node(a).shape;
    Shape out_s(1, 1);
    if (axis == 0) {
        out_s = Shape(1, sa.cols);
    } else if (axis == 1) {
        out_s = Shape(sa.rows, 1);
    } else if (axis != -1) {
        throw std::invalid_argument("ks_logsumexp: axis must be -1, 0, or 1.");
    }
    return g.add_node(std::make_shared<KSLogSumExpNode>(out_s, a, rho, axis, subtract_log_n));
}

int transpose(Graph& g, int a) {
    const Node& na = g.node(a);
    if (na.op == OpType::Transpose && !na.inputs.empty()) {
        return na.inputs[0];
    }
    const std::string key = "TR:" + std::to_string(a);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    const Shape sa = na.shape;
    int id = g.add_node(std::make_shared<TransposeNode>(Shape(sa.cols, sa.rows), a));
    g.register_cse_node(key, id);
    return id;
}

int slice(Graph& g, int a, Eigen::Index r0, Eigen::Index r1, Eigen::Index c0, Eigen::Index c1) {
    const Shape sa = g.node(a).shape;
    if (r0 < 0 || r1 > sa.rows || r0 >= r1 || c0 < 0 || c1 > sa.cols || c0 >= c1) {
        throw std::out_of_range("slice indices out of bounds for shape " + sa.to_string());
    }
    if (r0 == 0 && r1 == sa.rows && c0 == 0 && c1 == sa.cols) {
        return a;
    }
    const std::string key = "SL:" + std::to_string(a) + ":" + std::to_string(r0) + ":" +
                            std::to_string(r1) + ":" + std::to_string(c0) + ":" + std::to_string(c1);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<SliceNode>(Shape(r1 - r0, c1 - c0), a, r0, r1, c0, c1));
    g.register_cse_node(key, id);
    return id;
}

int concat(Graph& g, const std::vector<int>& inputs, int axis) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat requires at least one input node.");
    }
    if (inputs.size() == 1) {
        return inputs[0];
    }
    const Shape s0 = g.node(inputs[0]).shape;
    Eigen::Index total_r = s0.rows;
    Eigen::Index total_c = s0.cols;
    for (std::size_t i = 1; i < inputs.size(); ++i) {
        const Shape si = g.node(inputs[i]).shape;
        if (axis == 0) {
            if (si.cols != s0.cols) {
                throw std::invalid_argument("concat(axis=0): column dimension mismatch.");
            }
            total_r += si.rows;
        } else if (axis == 1) {
            if (si.rows != s0.rows) {
                throw std::invalid_argument("concat(axis=1): row dimension mismatch.");
            }
            total_c += si.cols;
        } else {
            throw std::invalid_argument("concat: axis must be 0 or 1.");
        }
    }
    return g.add_node(std::make_shared<ConcatNode>(Shape(total_r, total_c), inputs, axis));
}

int reshape(Graph& g, int a, Shape new_shape) {
    const Shape sa = g.node(a).shape;
    if (sa.size() != new_shape.size()) {
        throw std::invalid_argument("reshape total size mismatch: " +
                                    sa.to_string() + " vs " + new_shape.to_string());
    }
    if (sa == new_shape) {
        return a;
    }
    const std::string key = "RS:" + std::to_string(a) + ":" +
                            std::to_string(new_shape.rows) + ":" + std::to_string(new_shape.cols);
    int hit = g.find_cse_node(key);
    if (hit >= 0) return hit;

    int id = g.add_node(std::make_shared<ReshapeNode>(new_shape, a));
    g.register_cse_node(key, id);
    return id;
}

int where(Graph& g, int cond_id, int true_id, int false_id) {
    const Shape sc = g.node(cond_id).shape;
    const Shape st = g.node(true_id).shape;
    const Shape sf = g.node(false_id).shape;
    const Shape stf = broadcast_shape(st, sf);
    const Shape sout = broadcast_shape(sc, stf);
    return g.add_node(std::make_shared<WhereNode>(sout, cond_id, true_id, false_id));
}

int smooth_where(Graph& g, int x_id, int true_id, int false_id, double threshold, double sharpness) {
    const Shape sx = g.node(x_id).shape;
    const Shape st = g.node(true_id).shape;
    const Shape sf = g.node(false_id).shape;
    const Shape stf = broadcast_shape(st, sf);
    const Shape sout = broadcast_shape(sx, stf);
    return g.add_node(std::make_shared<SmoothWhereNode>(
        sout, x_id, true_id, false_id, threshold, sharpness));
}

int cond_subgraph(Graph& g, int cond_id, const std::vector<int>& arg_ids,
                  std::shared_ptr<Graph> true_graph, int true_out_id,
                  std::shared_ptr<Graph> false_graph, int false_out_id) {
    if (!g.node(cond_id).shape.is_scalar()) {
        throw std::invalid_argument("cond_subgraph requires a scalar condition node.");
    }
    const Shape st = true_graph->node(true_out_id).shape;
    const Shape sf = false_graph->node(false_out_id).shape;
    if (st != sf) {
        throw std::invalid_argument("cond_subgraph: true and false branch output shapes must match.");
    }
    return g.add_node(std::make_shared<CondNode>(
        st, cond_id, arg_ids, std::move(true_graph), true_out_id,
        std::move(false_graph), false_out_id));
}

int implicit_solve(Graph& g, const std::vector<int>& outer_input_ids,
                   std::shared_ptr<Graph> res_graph, int res_out_id,
                   int inner_state_var_id, const std::vector<int>& inner_param_ids,
                   int max_iter, double tol) {
    const Shape sy = res_graph->node(inner_state_var_id).shape;
    const Shape sr = res_graph->node(res_out_id).shape;
    if (sy.size() != sr.size()) {
        throw std::invalid_argument("implicit_solve: state size must match residual size.");
    }
    return g.add_node(std::make_shared<ImplicitSolverNode>(
        sy, outer_input_ids, std::move(res_graph), res_out_id,
        inner_state_var_id, inner_param_ids, max_iter, tol));
}

int blackbox(Graph& g, Shape out_shape, const std::vector<int>& in_ids,
             BlackBoxNode::EvalFn eval_fn,
             BlackBoxNode::JacFn jac_fn,
             BlackBoxNode::HvpFn hvp_fn,
             double fd_step,
             bool fd_hessian,
             const std::string& name,
             BlackBoxNode::BatchEvalFn batch_eval_fn,
             bool is_thread_safe,
             Eigen::MatrixXi jac_sparsity_mask) {
    if (in_ids.empty()) {
        throw std::invalid_argument("blackbox requires at least one input node.");
    }
    return g.add_node(std::make_shared<BlackBoxNode>(
        out_shape, in_ids, std::move(eval_fn), std::move(jac_fn),
        std::move(hvp_fn), fd_step, fd_hessian, name, std::move(batch_eval_fn),
        is_thread_safe, std::move(jac_sparsity_mask)));
}

int map_subgraph(Graph& g, Shape out_shape, const std::vector<int>& outer_in_ids,
                 std::shared_ptr<Graph> body_graph, int body_out_id,
                 const std::vector<int>& inner_var_ids,
                 const std::vector<int>& in_axes, int out_axis, int num_batches,
                 const std::string& name) {
    return g.add_node(std::make_shared<MapNode>(
        out_shape, outer_in_ids, std::move(body_graph), body_out_id,
        inner_var_ids, in_axes, out_axis, num_batches, name));
}

int scan_subgraph(Graph& g, Shape out_shape, const std::vector<int>& outer_in_ids,
                  std::shared_ptr<Graph> step_graph, int inner_carry_id,
                  const std::vector<int>& inner_seq_ids,
                  const std::vector<int>& seq_axes,
                  const std::vector<int>& inner_shared_ids,
                  int next_carry_out_id, int step_y_out_id, int num_steps,
                  const std::string& name) {
    return g.add_node(std::make_shared<ScanNode>(
        out_shape, outer_in_ids, std::move(step_graph), inner_carry_id,
        inner_seq_ids, seq_axes, inner_shared_ids,
        next_carry_out_id, step_y_out_id, num_steps, name));
}

int mlp(Graph& g, int x_id, std::shared_ptr<const MLPEngineData> engine_data,
        const std::string& name) {
    if (!engine_data) {
        throw std::invalid_argument("ops::mlp requires non-null MLPEngineData.");
    }
    const Node& xn = g.node(x_id);
    if (xn.shape.rows != engine_data->d_in) {
        throw std::invalid_argument(
            "ops::mlp input rows (" + std::to_string(xn.shape.rows) +
            ") must match network d_in (" + std::to_string(engine_data->d_in) + ").");
    }
    Shape out_shape(engine_data->d_out, xn.shape.cols);
    return g.add_node(std::make_shared<MLPNode>(out_shape, x_id, std::move(engine_data), name));
}

int conv1d(Graph& g, int x_id, std::shared_ptr<const Conv1DEngineData> engine_data,
           const std::string& name) {
    if (!engine_data) throw std::invalid_argument("ops::conv1d requires non-null Conv1DEngineData.");
    const Node& xn = g.node(x_id);
    if (xn.shape.rows != engine_data->c_in) {
        throw std::invalid_argument("ops::conv1d: input rows must match c_in.");
    }
    const Eigen::Index l_out = engine_data->out_length(xn.shape.cols);
    return g.add_node(std::make_shared<Conv1DNode>(
        Shape(engine_data->c_out, l_out), x_id, xn.shape.cols, std::move(engine_data), name));
}

int attention(Graph& g, int x_id, std::shared_ptr<const AttentionEngineData> engine_data,
              const std::string& name) {
    if (!engine_data) throw std::invalid_argument("ops::attention requires non-null AttentionEngineData.");
    const Node& xn = g.node(x_id);
    if (xn.shape.rows != engine_data->d_model) {
        throw std::invalid_argument("ops::attention: input rows must match d_model.");
    }
    return g.add_node(std::make_shared<AttentionNode>(
        Shape(engine_data->d_model, xn.shape.cols), x_id, std::move(engine_data), name));
}

int conv2d(Graph& g, int x_id, int h_in, int w_in,
           std::shared_ptr<const Conv2DEngineData> engine_data,
           const std::string& name) {
    if (!engine_data) throw std::invalid_argument("ops::conv2d requires non-null Conv2DEngineData.");
    const Node& xn = g.node(x_id);
    if (xn.shape.rows != engine_data->c_in || xn.shape.cols != static_cast<Eigen::Index>(h_in) * w_in)
        throw std::invalid_argument("ops::conv2d: input shape must be (c_in, h_in * w_in).");
    auto [ho, wo] = engine_data->out_hw(h_in, w_in);
    return g.add_node(std::make_shared<Conv2DNode>(
        Shape(engine_data->c_out, static_cast<Eigen::Index>(ho) * wo),
        x_id, h_in, w_in, ho, wo, std::move(engine_data), name));
}

int pool2d(Graph& g, int x_id, Pool2DMode mode, int h_in, int w_in,
           int kh, int kw, int sh, int sw, int ph, int pw,
           const std::string& name) {
    if (h_in <= 0 || w_in <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || ph < 0 || pw < 0)
        throw std::invalid_argument("ops::pool2d: invalid spatial/kernel/stride/padding dimensions.");
    const Node& xn = g.node(x_id);
    if (xn.shape.cols != static_cast<Eigen::Index>(h_in) * w_in)
        throw std::invalid_argument("ops::pool2d: input cols must equal h_in * w_in.");
    const int nh = h_in + 2 * ph - kh;
    const int nw = w_in + 2 * pw - kw;
    if (nh < 0 || nw < 0) throw std::invalid_argument("ops::pool2d: input too small for kernel/padding.");
    const int ho = nh / sh + 1;
    const int wo = nw / sw + 1;
    return g.add_node(std::make_shared<Pool2DNode>(
        Shape(xn.shape.rows, static_cast<Eigen::Index>(ho) * wo),
        x_id, mode, xn.shape.rows, h_in, w_in, ho, wo, kh, kw, sh, sw, ph, pw, name));
}

int rms_norm(Graph& g, int x_id, const Eigen::VectorXd& gamma, double eps, const std::string& name) {
    const Node& xn = g.node(x_id);
    if (gamma.size() > 0 && gamma.size() != xn.shape.rows)
        throw std::invalid_argument("ops::rms_norm: gamma size must match input rows.");
    return g.add_node(std::make_shared<RMSNormNode>(xn.shape, x_id, gamma, eps, name));
}


int surrogate(Graph& g, int x_id, std::shared_ptr<const SurrogateEngineData> engine_data,
              SurrogateOutputMode mode, const std::string& name) {
    if (!engine_data) {
        throw std::invalid_argument("ops::surrogate: engine_data is null.");
    }
    const Shape xs = g.node(x_id).shape;
    if (xs.rows != engine_data->d_in) {
        throw std::invalid_argument("ops::surrogate: input rows must match surrogate d_in.");
    }
    const Shape out_shape(engine_data->d_out, xs.cols);
    return g.add_node(std::make_unique<SurrogateNode>(
        out_shape, x_id, std::move(engine_data), mode, name));
}

} // namespace ops

} // namespace fastopt
