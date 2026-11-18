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
        xdb_ptr[i] += zdb * df + zb * d2f * xd_ptr[i];
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
                Adb(ra, ca) += zdb * fa + zb * (faa * ad + fab * bd);
            }
            if (rhs.requires_grad) {
                Bdb(rb, cb) += zdb * fb + zb * (fab * ad + fbb * bd);
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
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;

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
    if (ws.bar.segment(offset, size()).squaredNorm() == 0.0) return;

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

} // namespace ops

} // namespace fastopt
