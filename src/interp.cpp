#include "fastopt/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace fastopt {

namespace {

constexpr int kMaxStackSplineDeg = 7;

// Zero-heap-allocation stack buffer for 1D B-Spline basis functions and up to 2nd derivatives
struct BasisDersStack {
    double d[3][kMaxStackSplineDeg + 1];

    BasisDersStack() { std::memset(d, 0, sizeof(d)); }

    inline double operator()(int k, int j) const { return d[k][j]; }
    inline double& operator()(int k, int j) { return d[k][j]; }
};

void validate_grid_1d(const Eigen::VectorXd& grid, const char* name) {
    if (grid.size() < 2) {
        throw std::invalid_argument(std::string(name) + ": grid must contain at least 2 points.");
    }
    for (Eigen::Index i = 1; i < grid.size(); ++i) {
        if (grid(i) <= grid(i - 1)) {
            throw std::invalid_argument(std::string(name) + ": grid must be strictly increasing.");
        }
    }
}

Eigen::Index locate_interval_1d(const Eigen::VectorXd& grid, bool extrapolate,
                                double xq, double& t, double& h, const char* node_label) {
    const Eigen::Index n = grid.size();
    const double x_min = grid(0);
    const double x_max = grid(n - 1);
    constexpr double kTol = 1e-13;

    if (xq < x_min - kTol || xq > x_max + kTol) {
        if (!extrapolate) {
            std::ostringstream oss;
            oss << node_label << ": query value " << xq
                << " is outside grid domain [" << x_min << ", " << x_max
                << "] and extrapolate=false.";
            throw std::out_of_range(oss.str());
        }
        if (xq < x_min) {
            h = grid(1) - grid(0);
            t = (xq - grid(0)) / h;
            return 0;
        }
        h = grid(n - 1) - grid(n - 2);
        t = (xq - grid(n - 2)) / h;
        return n - 2;
    }

    if (xq <= x_min) {
        h = grid(1) - grid(0);
        t = 0.0;
        return 0;
    }
    if (xq >= x_max) {
        h = grid(n - 1) - grid(n - 2);
        t = 1.0;
        return n - 2;
    }

    // O(1) direct arithmetic lookup fast path for uniform/quasi-uniform grids
    Eigen::Index idx = static_cast<Eigen::Index>(
        (xq - x_min) / (x_max - x_min) * static_cast<double>(n - 1));
    idx = std::max<Eigen::Index>(0, std::min(idx, n - 2));
    if (xq < grid(idx) || xq >= grid(idx + 1)) {
        const double* begin = grid.data();
        const double* end = begin + n;
        const double* it = std::upper_bound(begin, end, xq);
        idx = static_cast<Eigen::Index>(it - begin) - 1;
        idx = std::max<Eigen::Index>(0, std::min(idx, n - 2));
    }

    h = grid(idx + 1) - grid(idx);
    t = (xq - grid(idx)) / h;
    return idx;
}

int find_span_1d(const Eigen::VectorXd& knots, int degree, bool extrapolate,
                 double u, const char* node_label) {
    const int n_ctrl = static_cast<int>(knots.size()) - degree - 1;
    const double u_min = knots(degree);
    const double u_max = knots(n_ctrl);
    constexpr double kTol = 1e-13;

    if (u < u_min - kTol || u > u_max + kTol) {
        if (!extrapolate) {
            std::ostringstream oss;
            oss << node_label << ": query parameter " << u
                << " is outside spline domain [" << u_min << ", " << u_max
                << "] and extrapolate=false.";
            throw std::out_of_range(oss.str());
        }
        if (u < u_min) {
            int s = degree;
            while (s < n_ctrl - 1 && knots(s + 1) <= knots(s)) ++s;
            return s;
        }
        int s = n_ctrl - 1;
        while (s > degree && knots(s + 1) <= knots(s)) --s;
        return s;
    }

    if (u <= u_min) {
        int s = degree;
        while (s < n_ctrl - 1 && knots(s + 1) <= knots(s)) ++s;
        return s;
    }
    if (u >= u_max) {
        int s = n_ctrl - 1;
        while (s > degree && knots(s + 1) <= knots(s)) --s;
        return s;
    }

    // O(1) direct arithmetic guess for uniform/quasi-uniform interior knot spans
    const int n_spans = n_ctrl - degree;
    if (n_spans > 0 && u_max > u_min) {
        int guess = degree + static_cast<int>((u - u_min) / (u_max - u_min) * static_cast<double>(n_spans));
        guess = std::max(degree, std::min(guess, n_ctrl - 1));
        if (u >= knots(guess) && u < knots(guess + 1)) {
            return guess;
        }
        if (guess > degree && u >= knots(guess - 1) && u < knots(guess)) {
            return guess - 1;
        }
        if (guess + 1 < n_ctrl && u >= knots(guess + 1) && u < knots(guess + 2)) {
            return guess + 1;
        }
    }

    int low = degree;
    int high = n_ctrl;
    int mid = (low + high) / 2;
    while (u < knots(mid) || u >= knots(mid + 1)) {
        if (u < knots(mid)) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

// Stack-buffered Cox-de Boor basis and derivative evaluator (0 heap allocations for degree <= 7)
inline void ders_basis_funs_stack_1d(const Eigen::VectorXd& knots, int degree, int span,
                                     double u, int nders, BasisDersStack& ders) {
    const int p = degree;
    if (p > kMaxStackSplineDeg) {
        throw std::invalid_argument("ders_basis_funs_stack_1d: spline degree exceeds kMaxStackSplineDeg (7).");
    }

    double ndu[kMaxStackSplineDeg + 1][kMaxStackSplineDeg + 1] = {};
    double left[kMaxStackSplineDeg + 1] = {};
    double right[kMaxStackSplineDeg + 1] = {};
    ndu[0][0] = 1.0;

    for (int j = 1; j <= p; ++j) {
        left[j] = u - knots(span + 1 - j);
        right[j] = knots(span + j) - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    for (int j = 0; j <= p; ++j) {
        ders(0, j) = ndu[j][p];
        ders(1, j) = 0.0;
        ders(2, j) = 0.0;
    }

    if (nders <= 0) return;

    double a[2][kMaxStackSplineDeg + 1] = {};
    for (int r = 0; r <= p; ++r) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0;
        for (int k = 1; k <= nders; ++k) {
            double d = 0.0;
            const int rk = r - k;
            const int pk = p - k;
            if (r >= k) {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? (k - 1) : (p - r);
            for (int j = j1; j <= j2; ++j) {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk) {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders(k, r) = d;
            std::swap(s1, s2);
        }
    }

    double fac = static_cast<double>(p);
    for (int k = 1; k <= nders; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders(k, j) *= fac;
        }
        fac *= static_cast<double>(p - k);
    }
}

} // namespace

// ============================================================================
// 1. LinearInterpNode Implementation (1D Grid, Multi-Column, Exact 1st/2nd AD)
// ============================================================================
LinearInterpNode::LinearInterpNode(Shape s, int x_query_id, int y_data_id,
                                   Eigen::VectorXd grid, bool allow_extrap,
                                   std::string name)
    : Node(OpType::LinearInterp, s, {x_query_id, y_data_id}, std::move(name)),
      x_grid(std::move(grid)),
      extrapolate(allow_extrap) {
    validate_grid_1d(x_grid, "LinearInterpNode");
}

Eigen::Index LinearInterpNode::locate_interval(double xq, double& t, double& h) const {
    return locate_interval_1d(x_grid, extrapolate, xq, t, h, "LinearInterpNode");
}

void LinearInterpNode::forward(Workspace& ws) const {
    const Node& xq_node = graph->node(inputs[0]);
    const Node& y_node = graph->node(inputs[1]);

    const Eigen::Index M = xq_node.size();
    const Eigen::Index K = y_node.shape.cols;

    const double* xq_ptr = ws.val.data() + xq_node.offset;
    auto Y = ws.val_mat(y_node.offset, y_node.shape);
    auto Z = ws.val_mat(offset, shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        double t = 0.0, h = 1.0;
        const Eigen::Index idx = locate_interval(xq_ptr[m], t, h);
        const double w0 = 1.0 - t;
        const double w1 = t;
        for (Eigen::Index k = 0; k < K; ++k) {
            Z(m, k) = w0 * Y(idx, k) + w1 * Y(idx + 1, k);
        }
    }
}

void LinearInterpNode::backward(Workspace& ws) const {
    const Node& xq_node = graph->node(inputs[0]);
    const Node& y_node = graph->node(inputs[1]);
    if (!xq_node.requires_grad && !y_node.requires_grad) return;

    const Eigen::Index M = xq_node.size();
    const Eigen::Index K = y_node.shape.cols;

    const double* xq_ptr = ws.val.data() + xq_node.offset;
    double* xqb_ptr = ws.bar.data() + xq_node.offset;
    auto Y = ws.val_mat(y_node.offset, y_node.shape);
    auto Yb = ws.bar_mat(y_node.offset, y_node.shape);
    auto Zb = ws.bar_mat(offset, shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        double t = 0.0, h = 1.0;
        const Eigen::Index idx = locate_interval(xq_ptr[m], t, h);
        const double w0 = 1.0 - t;
        const double w1 = t;
        const double inv_h = 1.0 / h;

        double dxq_acc = 0.0;
        for (Eigen::Index k = 0; k < K; ++k) {
            const double zb = Zb(m, k);
            if (zb == 0.0) continue;
            if (xq_node.requires_grad) {
                dxq_acc += zb * (Y(idx + 1, k) - Y(idx, k)) * inv_h;
            }
            if (y_node.requires_grad) {
                Yb(idx, k) += zb * w0;
                Yb(idx + 1, k) += zb * w1;
            }
        }
        if (xq_node.requires_grad) {
            xqb_ptr[m] += dxq_acc;
        }
    }
}

void LinearInterpNode::forward_tangent(Workspace& ws) const {
    const Node& xq_node = graph->node(inputs[0]);
    const Node& y_node = graph->node(inputs[1]);
    auto Zd = ws.dot_mat(offset, shape);
    if (!xq_node.requires_grad && !y_node.requires_grad) {
        Zd.setZero();
        return;
    }

    const Eigen::Index M = xq_node.size();
    const Eigen::Index K = y_node.shape.cols;

    const double* xq_ptr = ws.val.data() + xq_node.offset;
    const double* xqd_ptr = ws.dot.data() + xq_node.offset;
    auto Y = ws.val_mat(y_node.offset, y_node.shape);
    auto Yd = ws.dot_mat(y_node.offset, y_node.shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        double t = 0.0, h = 1.0;
        const Eigen::Index idx = locate_interval(xq_ptr[m], t, h);
        const double w0 = 1.0 - t;
        const double w1 = t;
        const double inv_h = 1.0 / h;
        const double xqd = xq_node.requires_grad ? xqd_ptr[m] : 0.0;

        for (Eigen::Index k = 0; k < K; ++k) {
            double zd = 0.0;
            if (xq_node.requires_grad) {
                zd += ((Y(idx + 1, k) - Y(idx, k)) * inv_h) * xqd;
            }
            if (y_node.requires_grad) {
                zd += w0 * Yd(idx, k) + w1 * Yd(idx + 1, k);
            }
            Zd(m, k) = zd;
        }
    }
}

void LinearInterpNode::backward_hessian(Workspace& ws) const {
    const Node& xq_node = graph->node(inputs[0]);
    const Node& y_node = graph->node(inputs[1]);
    if (!xq_node.requires_grad && !y_node.requires_grad) return;

    const Eigen::Index M = xq_node.size();
    const Eigen::Index K = y_node.shape.cols;

    const double* xq_ptr = ws.val.data() + xq_node.offset;
    const double* xqd_ptr = ws.dot.data() + xq_node.offset;
    double* xqdb_ptr = ws.dot_bar.data() + xq_node.offset;

    auto Y = ws.val_mat(y_node.offset, y_node.shape);
    auto Yd = ws.dot_mat(y_node.offset, y_node.shape);
    auto Ydb = ws.dot_bar_mat(y_node.offset, y_node.shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        double t = 0.0, h = 1.0;
        const Eigen::Index idx = locate_interval(xq_ptr[m], t, h);
        const double w0 = 1.0 - t;
        const double w1 = t;
        const double inv_h = 1.0 / h;
        const double xqd = xq_node.requires_grad ? xqd_ptr[m] : 0.0;

        double dxqdb_acc = 0.0;
        for (Eigen::Index k = 0; k < K; ++k) {
            const double zb = Zb(m, k);
            const double zdb = Zdb(m, k);
            if (zb == 0.0 && zdb == 0.0) continue;

            if (xq_node.requires_grad) {
                const double slope = (Y(idx + 1, k) - Y(idx, k)) * inv_h;
                dxqdb_acc += zdb * slope;
                if (y_node.requires_grad) {
                    const double dslope = (Yd(idx + 1, k) - Yd(idx, k)) * inv_h;
                    dxqdb_acc += zb * dslope;
                }
            }
            if (y_node.requires_grad) {
                Ydb(idx, k) += zdb * w0 - zb * xqd * inv_h;
                Ydb(idx + 1, k) += zdb * w1 + zb * xqd * inv_h;
            }
        }
        if (xq_node.requires_grad) {
            xqdb_ptr[m] += dxqdb_acc;
        }
    }
}

// ============================================================================
// 2. BSplineEvalNode Implementation (Cox-de Boor, Stack-Buffered, 1st/2nd AD)
// ============================================================================
BSplineEvalNode::BSplineEvalNode(Shape s, int u_query_id, int ctrl_pts_id,
                                 Eigen::VectorXd knot_vec, int spline_deg,
                                 bool allow_extrap, std::string name)
    : Node(OpType::BSplineEval, s, {u_query_id, ctrl_pts_id}, std::move(name)),
      knots(std::move(knot_vec)),
      degree(spline_deg),
      extrapolate(allow_extrap) {
    if (degree < 1 || degree > kMaxStackSplineDeg) {
        throw std::invalid_argument("BSplineEvalNode: degree must be in [1, 7].");
    }
}

int BSplineEvalNode::find_span(double u) const {
    return find_span_1d(knots, degree, extrapolate, u, "BSplineEvalNode");
}

void BSplineEvalNode::forward(Workspace& ws) const {
    const Node& u_node = graph->node(inputs[0]);
    const Node& ctrl_node = graph->node(inputs[1]);

    const Eigen::Index M = u_node.size();
    const Eigen::Index K = ctrl_node.shape.cols;

    const double* u_ptr = ws.val.data() + u_node.offset;
    auto P = ws.val_mat(ctrl_node.offset, ctrl_node.shape);
    auto Z = ws.val_mat(offset, shape);

    BasisDersStack ders;
    for (Eigen::Index m = 0; m < M; ++m) {
        const int span = find_span(u_ptr[m]);
        ders_basis_funs_stack_1d(knots, degree, span, u_ptr[m], 0, ders);
        for (Eigen::Index k = 0; k < K; ++k) {
            double val = 0.0;
            for (int j = 0; j <= degree; ++j) {
                val += ders(0, j) * P(span - degree + j, k);
            }
            Z(m, k) = val;
        }
    }
}

void BSplineEvalNode::backward(Workspace& ws) const {
    const Node& u_node = graph->node(inputs[0]);
    const Node& ctrl_node = graph->node(inputs[1]);
    if (!u_node.requires_grad && !ctrl_node.requires_grad) return;

    const Eigen::Index M = u_node.size();
    const Eigen::Index K = ctrl_node.shape.cols;

    const double* u_ptr = ws.val.data() + u_node.offset;
    double* ub_ptr = ws.bar.data() + u_node.offset;
    auto P = ws.val_mat(ctrl_node.offset, ctrl_node.shape);
    auto Pb = ws.bar_mat(ctrl_node.offset, ctrl_node.shape);
    auto Zb = ws.bar_mat(offset, shape);

    BasisDersStack ders;
    const int nders = u_node.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const int span = find_span(u_ptr[m]);
        ders_basis_funs_stack_1d(knots, degree, span, u_ptr[m], nders, ders);

        double du_acc = 0.0;
        for (Eigen::Index k = 0; k < K; ++k) {
            const double zb = Zb(m, k);
            if (zb == 0.0) continue;

            if (u_node.requires_grad) {
                double dB_du = 0.0;
                for (int j = 0; j <= degree; ++j) {
                    dB_du += ders(1, j) * P(span - degree + j, k);
                }
                du_acc += zb * dB_du;
            }
            if (ctrl_node.requires_grad) {
                for (int j = 0; j <= degree; ++j) {
                    Pb(span - degree + j, k) += zb * ders(0, j);
                }
            }
        }
        if (u_node.requires_grad) {
            ub_ptr[m] += du_acc;
        }
    }
}

void BSplineEvalNode::forward_tangent(Workspace& ws) const {
    const Node& u_node = graph->node(inputs[0]);
    const Node& ctrl_node = graph->node(inputs[1]);
    auto Zd = ws.dot_mat(offset, shape);
    if (!u_node.requires_grad && !ctrl_node.requires_grad) {
        Zd.setZero();
        return;
    }

    const Eigen::Index M = u_node.size();
    const Eigen::Index K = ctrl_node.shape.cols;

    const double* u_ptr = ws.val.data() + u_node.offset;
    const double* ud_ptr = ws.dot.data() + u_node.offset;
    auto P = ws.val_mat(ctrl_node.offset, ctrl_node.shape);
    auto Pd = ws.dot_mat(ctrl_node.offset, ctrl_node.shape);

    BasisDersStack ders;
    const int nders = u_node.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const int span = find_span(u_ptr[m]);
        ders_basis_funs_stack_1d(knots, degree, span, u_ptr[m], nders, ders);
        const double ud = u_node.requires_grad ? ud_ptr[m] : 0.0;

        for (Eigen::Index k = 0; k < K; ++k) {
            double zd = 0.0;
            if (u_node.requires_grad) {
                double dB_du = 0.0;
                for (int j = 0; j <= degree; ++j) {
                    dB_du += ders(1, j) * P(span - degree + j, k);
                }
                zd += dB_du * ud;
            }
            if (ctrl_node.requires_grad) {
                for (int j = 0; j <= degree; ++j) {
                    zd += ders(0, j) * Pd(span - degree + j, k);
                }
            }
            Zd(m, k) = zd;
        }
    }
}

void BSplineEvalNode::backward_hessian(Workspace& ws) const {
    const Node& u_node = graph->node(inputs[0]);
    const Node& ctrl_node = graph->node(inputs[1]);
    if (!u_node.requires_grad && !ctrl_node.requires_grad) return;

    const Eigen::Index M = u_node.size();
    const Eigen::Index K = ctrl_node.shape.cols;

    const double* u_ptr = ws.val.data() + u_node.offset;
    const double* ud_ptr = ws.dot.data() + u_node.offset;
    double* udb_ptr = ws.dot_bar.data() + u_node.offset;

    auto P = ws.val_mat(ctrl_node.offset, ctrl_node.shape);
    auto Pd = ws.dot_mat(ctrl_node.offset, ctrl_node.shape);
    auto Pdb = ws.dot_bar_mat(ctrl_node.offset, ctrl_node.shape);
    auto Zb = ws.bar_mat(offset, shape);
    auto Zdb = ws.dot_bar_mat(offset, shape);

    BasisDersStack ders;
    const int nders = u_node.requires_grad ? std::min(2, degree) : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const int span = find_span(u_ptr[m]);
        ders_basis_funs_stack_1d(knots, degree, span, u_ptr[m], nders, ders);
        const double ud = u_node.requires_grad ? ud_ptr[m] : 0.0;

        double dudb_acc = 0.0;
        for (Eigen::Index k = 0; k < K; ++k) {
            const double zb = Zb(m, k);
            const double zdb = Zdb(m, k);
            if (zb == 0.0 && zdb == 0.0) continue;

            if (u_node.requires_grad) {
                double dB_du = 0.0;
                double d2B_du2 = 0.0;
                double dPd_du = 0.0;
                for (int j = 0; j <= degree; ++j) {
                    const double p_val = P(span - degree + j, k);
                    dB_du += ders(1, j) * p_val;
                    if (nders >= 2) {
                        d2B_du2 += ders(2, j) * p_val;
                    }
                    if (ctrl_node.requires_grad) {
                        dPd_du += ders(1, j) * Pd(span - degree + j, k);
                    }
                }
                dudb_acc += zdb * dB_du + zb * (d2B_du2 * ud + dPd_du);
            }
            if (ctrl_node.requires_grad) {
                for (int j = 0; j <= degree; ++j) {
                    const double d0 = ders(0, j);
                    const double d1 = u_node.requires_grad ? ders(1, j) : 0.0;
                    Pdb(span - degree + j, k) += zdb * d0 + zb * d1 * ud;
                }
            }
        }
        if (u_node.requires_grad) {
            udb_ptr[m] += dudb_acc;
        }
    }
}

// ============================================================================
// 3. LinearInterp2DNode Implementation (Bilinear Surface, 1st & 2nd AD)
// ============================================================================
LinearInterp2DNode::LinearInterp2DNode(Shape s, int x_query_id, int y_query_id, int z_data_id,
                                       Eigen::VectorXd xg, Eigen::VectorXd yg,
                                       bool allow_extrap, std::string name)
    : Node(OpType::LinearInterp2D, s, {x_query_id, y_query_id, z_data_id}, std::move(name)),
      x_grid(std::move(xg)),
      y_grid(std::move(yg)),
      extrapolate(allow_extrap) {
    validate_grid_1d(x_grid, "LinearInterp2DNode(x_grid)");
    validate_grid_1d(y_grid, "LinearInterp2DNode(y_grid)");
}

void LinearInterp2DNode::forward(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& z_n = graph->node(inputs[2]);

    const Eigen::Index M = size();
    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    auto Z = ws.val_mat(z_n.offset, z_n.shape);
    double* out_ptr = ws.val.data() + offset;

    for (Eigen::Index m = 0; m < M; ++m) {
        double u = 0.0, hx = 1.0, v = 0.0, hy = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], u, hx, "LinearInterp2DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], v, hy, "LinearInterp2DNode(y)");

        const double z00 = Z(ix, iy);
        const double z10 = Z(ix + 1, iy);
        const double z01 = Z(ix, iy + 1);
        const double z11 = Z(ix + 1, iy + 1);

        out_ptr[m] = (1.0 - u) * (1.0 - v) * z00 + u * (1.0 - v) * z10 +
                     (1.0 - u) * v * z01 + u * v * z11;
    }
}

void LinearInterp2DNode::backward(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& z_n = graph->node(inputs[2]);
    if (!xq_n.requires_grad && !yq_n.requires_grad && !z_n.requires_grad) return;

    const Eigen::Index M = size();
    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;
    double* xqb_ptr = ws.bar.data() + xq_n.offset;
    double* yqb_ptr = ws.bar.data() + yq_n.offset;
    auto Z = ws.val_mat(z_n.offset, z_n.shape);
    auto Zb = ws.bar_mat(z_n.offset, z_n.shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        if (fb == 0.0) continue;

        double u = 0.0, hx = 1.0, v = 0.0, hy = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], u, hx, "LinearInterp2DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], v, hy, "LinearInterp2DNode(y)");

        const double z00 = Z(ix, iy);
        const double z10 = Z(ix + 1, iy);
        const double z01 = Z(ix, iy + 1);
        const double z11 = Z(ix + 1, iy + 1);

        if (xq_n.requires_grad) {
            const double df_dx = ((1.0 - v) * (z10 - z00) + v * (z11 - z01)) / hx;
            xqb_ptr[m] += fb * df_dx;
        }
        if (yq_n.requires_grad) {
            const double df_dy = ((1.0 - u) * (z01 - z00) + u * (z11 - z10)) / hy;
            yqb_ptr[m] += fb * df_dy;
        }
        if (z_n.requires_grad) {
            Zb(ix, iy) += fb * (1.0 - u) * (1.0 - v);
            Zb(ix + 1, iy) += fb * u * (1.0 - v);
            Zb(ix, iy + 1) += fb * (1.0 - u) * v;
            Zb(ix + 1, iy + 1) += fb * u * v;
        }
    }
}

void LinearInterp2DNode::forward_tangent(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& z_n = graph->node(inputs[2]);
    const Eigen::Index M = size();
    double* fd_ptr = ws.dot.data() + offset;

    if (!xq_n.requires_grad && !yq_n.requires_grad && !z_n.requires_grad) {
        ws.dot.segment(offset, M).setZero();
        return;
    }

    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* xqd_ptr = ws.dot.data() + xq_n.offset;
    const double* yqd_ptr = ws.dot.data() + yq_n.offset;
    auto Z = ws.val_mat(z_n.offset, z_n.shape);
    auto Zd = ws.dot_mat(z_n.offset, z_n.shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        double u = 0.0, hx = 1.0, v = 0.0, hy = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], u, hx, "LinearInterp2DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], v, hy, "LinearInterp2DNode(y)");

        const double z00 = Z(ix, iy), z10 = Z(ix + 1, iy);
        const double z01 = Z(ix, iy + 1), z11 = Z(ix + 1, iy + 1);
        const double xqd = xq_n.requires_grad ? xqd_ptr[m] : 0.0;
        const double yqd = yq_n.requires_grad ? yqd_ptr[m] : 0.0;

        double fd = 0.0;
        if (xq_n.requires_grad) {
            fd += (((1.0 - v) * (z10 - z00) + v * (z11 - z01)) / hx) * xqd;
        }
        if (yq_n.requires_grad) {
            fd += (((1.0 - u) * (z01 - z00) + u * (z11 - z10)) / hy) * yqd;
        }
        if (z_n.requires_grad) {
            fd += (1.0 - u) * (1.0 - v) * Zd(ix, iy) + u * (1.0 - v) * Zd(ix + 1, iy) +
                  (1.0 - u) * v * Zd(ix, iy + 1) + u * v * Zd(ix + 1, iy + 1);
        }
        fd_ptr[m] = fd;
    }
}

void LinearInterp2DNode::backward_hessian(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& z_n = graph->node(inputs[2]);
    if (!xq_n.requires_grad && !yq_n.requires_grad && !z_n.requires_grad) return;

    const Eigen::Index M = size();
    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* xqd_ptr = ws.dot.data() + xq_n.offset;
    const double* yqd_ptr = ws.dot.data() + yq_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;
    const double* fdb_ptr = ws.dot_bar.data() + offset;
    double* xqdb_ptr = ws.dot_bar.data() + xq_n.offset;
    double* yqdb_ptr = ws.dot_bar.data() + yq_n.offset;
    auto Z = ws.val_mat(z_n.offset, z_n.shape);
    auto Zd = ws.dot_mat(z_n.offset, z_n.shape);
    auto Zdb = ws.dot_bar_mat(z_n.offset, z_n.shape);

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        const double fdb = fdb_ptr[m];
        if (fb == 0.0 && fdb == 0.0) continue;

        double u = 0.0, hx = 1.0, v = 0.0, hy = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], u, hx, "LinearInterp2DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], v, hy, "LinearInterp2DNode(y)");

        const double z00 = Z(ix, iy), z10 = Z(ix + 1, iy);
        const double z01 = Z(ix, iy + 1), z11 = Z(ix + 1, iy + 1);
        const double zd00 = z_n.requires_grad ? Zd(ix, iy) : 0.0;
        const double zd10 = z_n.requires_grad ? Zd(ix + 1, iy) : 0.0;
        const double zd01 = z_n.requires_grad ? Zd(ix, iy + 1) : 0.0;
        const double zd11 = z_n.requires_grad ? Zd(ix + 1, iy + 1) : 0.0;

        const double xqd = xq_n.requires_grad ? xqd_ptr[m] : 0.0;
        const double yqd = yq_n.requires_grad ? yqd_ptr[m] : 0.0;
        const double ud = xqd / hx;
        const double vd = yqd / hy;

        const double df_dx = ((1.0 - v) * (z10 - z00) + v * (z11 - z01)) / hx;
        const double df_dy = ((1.0 - u) * (z01 - z00) + u * (z11 - z10)) / hy;
        const double d2f_dxdy = (z11 - z10 - z01 + z00) / (hx * hy);

        if (xq_n.requires_grad) {
            const double d_dfdx_dz = ((1.0 - v) * (zd10 - zd00) + v * (zd11 - zd01)) / hx;
            xqdb_ptr[m] += fdb * df_dx + fb * (d2f_dxdy * yqd + d_dfdx_dz);
        }
        if (yq_n.requires_grad) {
            const double d_dfdy_dz = ((1.0 - u) * (zd01 - zd00) + u * (zd11 - zd10)) / hy;
            yqdb_ptr[m] += fdb * df_dy + fb * (d2f_dxdy * xqd + d_dfdy_dz);
        }
        if (z_n.requires_grad) {
            Zdb(ix, iy) += fdb * (1.0 - u) * (1.0 - v) + fb * (-ud * (1.0 - v) - (1.0 - u) * vd);
            Zdb(ix + 1, iy) += fdb * u * (1.0 - v) + fb * (ud * (1.0 - v) - u * vd);
            Zdb(ix, iy + 1) += fdb * (1.0 - u) * v + fb * (-ud * v + (1.0 - u) * vd);
            Zdb(ix + 1, iy + 1) += fdb * u * v + fb * (ud * v + u * vd);
        }
    }
}

// ============================================================================
// 4. BSplineEval2DNode Implementation (2D Tensor-Product B-Spline, Stack-Buffered)
// ============================================================================
BSplineEval2DNode::BSplineEval2DNode(Shape s, int u_query_id, int v_query_id, int ctrl_pts_id,
                                     Eigen::VectorXd ku, Eigen::VectorXd kv,
                                     int deg_u, int deg_v,
                                     bool allow_extrap, std::string name)
    : Node(OpType::BSplineEval2D, s, {u_query_id, v_query_id, ctrl_pts_id}, std::move(name)),
      knots_u(std::move(ku)),
      knots_v(std::move(kv)),
      degree_u(deg_u),
      degree_v(deg_v),
      extrapolate(allow_extrap) {
    if (degree_u < 1 || degree_v < 1 || degree_u > kMaxStackSplineDeg || degree_v > kMaxStackSplineDeg) {
        throw std::invalid_argument("BSplineEval2DNode: degree_u and degree_v must be in [1, 7].");
    }
}

void BSplineEval2DNode::forward(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& p_n = graph->node(inputs[2]);

    const Eigen::Index M = size();
    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    auto P = ws.val_mat(p_n.offset, p_n.shape);
    double* out_ptr = ws.val.data() + offset;

    BasisDersStack du, dv;
    for (Eigen::Index m = 0; m < M; ++m) {
        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval2DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval2DNode(v)");
        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], 0, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], 0, dv);

        double val = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const int iu = span_u - degree_u + a;
            double sum_v = 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                sum_v += dv(0, b) * P(iu, span_v - degree_v + b);
            }
            val += du(0, a) * sum_v;
        }
        out_ptr[m] = val;
    }
}

void BSplineEval2DNode::backward(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& p_n = graph->node(inputs[2]);
    if (!u_n.requires_grad && !v_n.requires_grad && !p_n.requires_grad) return;

    const Eigen::Index M = size();
    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;
    double* ub_ptr = ws.bar.data() + u_n.offset;
    double* vb_ptr = ws.bar.data() + v_n.offset;
    auto P = ws.val_mat(p_n.offset, p_n.shape);
    auto Pb = ws.bar_mat(p_n.offset, p_n.shape);

    BasisDersStack du, dv;
    const int nd_u = u_n.requires_grad ? 1 : 0;
    const int nd_v = v_n.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        if (fb == 0.0) continue;

        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval2DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval2DNode(v)");
        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);

        double Su = 0.0, Sv = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const int iu = span_u - degree_u + a;
            for (int b = 0; b <= degree_v; ++b) {
                const int iv = span_v - degree_v + b;
                const double p_val = P(iu, iv);
                if (u_n.requires_grad) Su += du(1, a) * dv(0, b) * p_val;
                if (v_n.requires_grad) Sv += du(0, a) * dv(1, b) * p_val;
                if (p_n.requires_grad) Pb(iu, iv) += fb * du(0, a) * dv(0, b);
            }
        }
        if (u_n.requires_grad) ub_ptr[m] += fb * Su;
        if (v_n.requires_grad) vb_ptr[m] += fb * Sv;
    }
}

void BSplineEval2DNode::forward_tangent(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& p_n = graph->node(inputs[2]);
    const Eigen::Index M = size();
    double* fd_ptr = ws.dot.data() + offset;

    if (!u_n.requires_grad && !v_n.requires_grad && !p_n.requires_grad) {
        ws.dot.segment(offset, M).setZero();
        return;
    }

    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* ud_ptr = ws.dot.data() + u_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;
    auto P = ws.val_mat(p_n.offset, p_n.shape);
    auto Pd = ws.dot_mat(p_n.offset, p_n.shape);

    BasisDersStack du, dv;
    const int nd_u = u_n.requires_grad ? 1 : 0;
    const int nd_v = v_n.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval2DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval2DNode(v)");
        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);

        const double ud = u_n.requires_grad ? ud_ptr[m] : 0.0;
        const double vd = v_n.requires_grad ? vd_ptr[m] : 0.0;

        double fd = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const int iu = span_u - degree_u + a;
            for (int b = 0; b <= degree_v; ++b) {
                const int iv = span_v - degree_v + b;
                const double p_val = P(iu, iv);
                if (u_n.requires_grad) fd += du(1, a) * dv(0, b) * p_val * ud;
                if (v_n.requires_grad) fd += du(0, a) * dv(1, b) * p_val * vd;
                if (p_n.requires_grad) fd += du(0, a) * dv(0, b) * Pd(iu, iv);
            }
        }
        fd_ptr[m] = fd;
    }
}

void BSplineEval2DNode::backward_hessian(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& p_n = graph->node(inputs[2]);
    if (!u_n.requires_grad && !v_n.requires_grad && !p_n.requires_grad) return;

    const Eigen::Index M = size();
    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* ud_ptr = ws.dot.data() + u_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;
    const double* fdb_ptr = ws.dot_bar.data() + offset;
    double* udb_ptr = ws.dot_bar.data() + u_n.offset;
    double* vdb_ptr = ws.dot_bar.data() + v_n.offset;
    auto P = ws.val_mat(p_n.offset, p_n.shape);
    auto Pd = ws.dot_mat(p_n.offset, p_n.shape);
    auto Pdb = ws.dot_bar_mat(p_n.offset, p_n.shape);

    BasisDersStack du, dv;
    const int nd_u = (u_n.requires_grad || v_n.requires_grad) ? std::min(2, degree_u) : 0;
    const int nd_v = (u_n.requires_grad || v_n.requires_grad) ? std::min(2, degree_v) : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        const double fdb = fdb_ptr[m];
        if (fb == 0.0 && fdb == 0.0) continue;

        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval2DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval2DNode(v)");
        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);

        const double ud = u_n.requires_grad ? ud_ptr[m] : 0.0;
        const double vd = v_n.requires_grad ? vd_ptr[m] : 0.0;

        double Su = 0.0, Sv = 0.0;
        double Suu = 0.0, Suv = 0.0, Svv = 0.0;
        double dSu_dP = 0.0, dSv_dP = 0.0;

        for (int a = 0; a <= degree_u; ++a) {
            const int iu = span_u - degree_u + a;
            const double Nu0 = du(0, a);
            const double Nu1 = (nd_u >= 1) ? du(1, a) : 0.0;
            const double Nu2 = (nd_u >= 2) ? du(2, a) : 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                const int iv = span_v - degree_v + b;
                const double Nv0 = dv(0, b);
                const double Nv1 = (nd_v >= 1) ? dv(1, b) : 0.0;
                const double Nv2 = (nd_v >= 2) ? dv(2, b) : 0.0;
                const double p_val = P(iu, iv);
                const double pd_val = p_n.requires_grad ? Pd(iu, iv) : 0.0;

                Su += Nu1 * Nv0 * p_val;
                Sv += Nu0 * Nv1 * p_val;
                Suu += Nu2 * Nv0 * p_val;
                Suv += Nu1 * Nv1 * p_val;
                Svv += Nu0 * Nv2 * p_val;
                dSu_dP += Nu1 * Nv0 * pd_val;
                dSv_dP += Nu0 * Nv1 * pd_val;

                if (p_n.requires_grad) {
                    Pdb(iu, iv) += fdb * (Nu0 * Nv0) + fb * (Nu1 * Nv0 * ud + Nu0 * Nv1 * vd);
                }
            }
        }

        if (u_n.requires_grad) {
            udb_ptr[m] += fdb * Su + fb * (Suu * ud + Suv * vd + dSu_dP);
        }
        if (v_n.requires_grad) {
            vdb_ptr[m] += fdb * Sv + fb * (Suv * ud + Svv * vd + dSv_dP);
        }
    }
}

// ============================================================================
// 5. LinearInterp3DNode Implementation (Trilinear Volume, 1st & 2nd AD)
// ============================================================================
LinearInterp3DNode::LinearInterp3DNode(Shape s, int x_query_id, int y_query_id,
                                       int z_query_id, int v_data_id,
                                       Eigen::VectorXd xg, Eigen::VectorXd yg, Eigen::VectorXd zg,
                                       bool allow_extrap, std::string name)
    : Node(OpType::LinearInterp3D, s, {x_query_id, y_query_id, z_query_id, v_data_id}, std::move(name)),
      x_grid(std::move(xg)),
      y_grid(std::move(yg)),
      z_grid(std::move(zg)),
      extrapolate(allow_extrap) {
    validate_grid_1d(x_grid, "LinearInterp3DNode(x_grid)");
    validate_grid_1d(y_grid, "LinearInterp3DNode(y_grid)");
    validate_grid_1d(z_grid, "LinearInterp3DNode(z_grid)");
}

void LinearInterp3DNode::forward(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& zq_n = graph->node(inputs[2]);
    const Node& v_n = graph->node(inputs[3]);

    const Eigen::Index M = size();
    const Eigen::Index nx = x_grid.size();
    const Eigen::Index ny = y_grid.size();

    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* zq_ptr = ws.val.data() + zq_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    double* out_ptr = ws.val.data() + offset;

    auto V = [v_ptr, nx, ny](Eigen::Index i, Eigen::Index j, Eigen::Index k) -> double {
        return v_ptr[i + nx * (j + ny * k)];
    };

    for (Eigen::Index m = 0; m < M; ++m) {
        double tx = 0.0, hx = 1.0, ty = 0.0, hy = 1.0, tz = 0.0, hz = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], tx, hx, "LinearInterp3DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], ty, hy, "LinearInterp3DNode(y)");
        const Eigen::Index iz = locate_interval_1d(z_grid, extrapolate, zq_ptr[m], tz, hz, "LinearInterp3DNode(z)");

        const double wx[2] = {1.0 - tx, tx};
        const double wy[2] = {1.0 - ty, ty};
        const double wz[2] = {1.0 - tz, tz};

        double val = 0.0;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    val += wx[a] * wy[b] * wz[c] * V(ix + a, iy + b, iz + c);
                }
            }
        }
        out_ptr[m] = val;
    }
}

void LinearInterp3DNode::backward(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& zq_n = graph->node(inputs[2]);
    const Node& v_n = graph->node(inputs[3]);
    if (!xq_n.requires_grad && !yq_n.requires_grad && !zq_n.requires_grad && !v_n.requires_grad) return;

    const Eigen::Index M = size();
    const Eigen::Index nx = x_grid.size();
    const Eigen::Index ny = y_grid.size();

    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* zq_ptr = ws.val.data() + zq_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;

    double* xqb_ptr = ws.bar.data() + xq_n.offset;
    double* yqb_ptr = ws.bar.data() + yq_n.offset;
    double* zqb_ptr = ws.bar.data() + zq_n.offset;
    double* vb_ptr = ws.bar.data() + v_n.offset;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        if (fb == 0.0) continue;

        double tx = 0.0, hx = 1.0, ty = 0.0, hy = 1.0, tz = 0.0, hz = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], tx, hx, "LinearInterp3DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], ty, hy, "LinearInterp3DNode(y)");
        const Eigen::Index iz = locate_interval_1d(z_grid, extrapolate, zq_ptr[m], tz, hz, "LinearInterp3DNode(z)");

        const double wx[2] = {1.0 - tx, tx};
        const double wy[2] = {1.0 - ty, ty};
        const double wz[2] = {1.0 - tz, tz};
        const double dwx[2] = {-1.0 / hx, 1.0 / hx};
        const double dwy[2] = {-1.0 / hy, 1.0 / hy};
        const double dwz[2] = {-1.0 / hz, 1.0 / hz};

        double df_dx = 0.0, df_dy = 0.0, df_dz = 0.0;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    const Eigen::Index flat_idx = (ix + a) + nx * ((iy + b) + ny * (iz + c));
                    const double v_val = v_ptr[flat_idx];
                    df_dx += dwx[a] * wy[b] * wz[c] * v_val;
                    df_dy += wx[a] * dwy[b] * wz[c] * v_val;
                    df_dz += wx[a] * wy[b] * dwz[c] * v_val;
                    if (v_n.requires_grad) {
                        vb_ptr[flat_idx] += fb * (wx[a] * wy[b] * wz[c]);
                    }
                }
            }
        }
        if (xq_n.requires_grad) xqb_ptr[m] += fb * df_dx;
        if (yq_n.requires_grad) yqb_ptr[m] += fb * df_dy;
        if (zq_n.requires_grad) zqb_ptr[m] += fb * df_dz;
    }
}

void LinearInterp3DNode::forward_tangent(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& zq_n = graph->node(inputs[2]);
    const Node& v_n = graph->node(inputs[3]);
    const Eigen::Index M = size();
    double* fd_ptr = ws.dot.data() + offset;

    if (!xq_n.requires_grad && !yq_n.requires_grad && !zq_n.requires_grad && !v_n.requires_grad) {
        ws.dot.segment(offset, M).setZero();
        return;
    }

    const Eigen::Index nx = x_grid.size();
    const Eigen::Index ny = y_grid.size();

    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* zq_ptr = ws.val.data() + zq_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;

    const double* xqd_ptr = ws.dot.data() + xq_n.offset;
    const double* yqd_ptr = ws.dot.data() + yq_n.offset;
    const double* zqd_ptr = ws.dot.data() + zq_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;

    for (Eigen::Index m = 0; m < M; ++m) {
        double tx = 0.0, hx = 1.0, ty = 0.0, hy = 1.0, tz = 0.0, hz = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], tx, hx, "LinearInterp3DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], ty, hy, "LinearInterp3DNode(y)");
        const Eigen::Index iz = locate_interval_1d(z_grid, extrapolate, zq_ptr[m], tz, hz, "LinearInterp3DNode(z)");

        const double wx[2] = {1.0 - tx, tx};
        const double wy[2] = {1.0 - ty, ty};
        const double wz[2] = {1.0 - tz, tz};
        const double dwx[2] = {-1.0 / hx, 1.0 / hx};
        const double dwy[2] = {-1.0 / hy, 1.0 / hy};
        const double dwz[2] = {-1.0 / hz, 1.0 / hz};

        const double xqd = xq_n.requires_grad ? xqd_ptr[m] : 0.0;
        const double yqd = yq_n.requires_grad ? yqd_ptr[m] : 0.0;
        const double zqd = zq_n.requires_grad ? zqd_ptr[m] : 0.0;

        double fd = 0.0;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    const Eigen::Index flat_idx = (ix + a) + nx * ((iy + b) + ny * (iz + c));
                    const double v_val = v_ptr[flat_idx];
                    const double dw_dt = (dwx[a] * xqd) * wy[b] * wz[c] +
                                         wx[a] * (dwy[b] * yqd) * wz[c] +
                                         wx[a] * wy[b] * (dwz[c] * zqd);
                    fd += dw_dt * v_val;
                    if (v_n.requires_grad) {
                        fd += (wx[a] * wy[b] * wz[c]) * vd_ptr[flat_idx];
                    }
                }
            }
        }
        fd_ptr[m] = fd;
    }
}

void LinearInterp3DNode::backward_hessian(Workspace& ws) const {
    const Node& xq_n = graph->node(inputs[0]);
    const Node& yq_n = graph->node(inputs[1]);
    const Node& zq_n = graph->node(inputs[2]);
    const Node& v_n = graph->node(inputs[3]);
    if (!xq_n.requires_grad && !yq_n.requires_grad && !zq_n.requires_grad && !v_n.requires_grad) return;

    const Eigen::Index M = size();
    const Eigen::Index nx = x_grid.size();
    const Eigen::Index ny = y_grid.size();

    const double* xq_ptr = ws.val.data() + xq_n.offset;
    const double* yq_ptr = ws.val.data() + yq_n.offset;
    const double* zq_ptr = ws.val.data() + zq_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;

    const double* xqd_ptr = ws.dot.data() + xq_n.offset;
    const double* yqd_ptr = ws.dot.data() + yq_n.offset;
    const double* zqd_ptr = ws.dot.data() + zq_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;

    const double* fb_ptr = ws.bar.data() + offset;
    const double* fdb_ptr = ws.dot_bar.data() + offset;

    double* xqdb_ptr = ws.dot_bar.data() + xq_n.offset;
    double* yqdb_ptr = ws.dot_bar.data() + yq_n.offset;
    double* zqdb_ptr = ws.dot_bar.data() + zq_n.offset;
    double* vdb_ptr = ws.dot_bar.data() + v_n.offset;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        const double fdb = fdb_ptr[m];
        if (fb == 0.0 && fdb == 0.0) continue;

        double tx = 0.0, hx = 1.0, ty = 0.0, hy = 1.0, tz = 0.0, hz = 1.0;
        const Eigen::Index ix = locate_interval_1d(x_grid, extrapolate, xq_ptr[m], tx, hx, "LinearInterp3DNode(x)");
        const Eigen::Index iy = locate_interval_1d(y_grid, extrapolate, yq_ptr[m], ty, hy, "LinearInterp3DNode(y)");
        const Eigen::Index iz = locate_interval_1d(z_grid, extrapolate, zq_ptr[m], tz, hz, "LinearInterp3DNode(z)");

        const double wx[2] = {1.0 - tx, tx};
        const double wy[2] = {1.0 - ty, ty};
        const double wz[2] = {1.0 - tz, tz};
        const double dwx[2] = {-1.0 / hx, 1.0 / hx};
        const double dwy[2] = {-1.0 / hy, 1.0 / hy};
        const double dwz[2] = {-1.0 / hz, 1.0 / hz};

        const double xqd = xq_n.requires_grad ? xqd_ptr[m] : 0.0;
        const double yqd = yq_n.requires_grad ? yqd_ptr[m] : 0.0;
        const double zqd = zq_n.requires_grad ? zqd_ptr[m] : 0.0;

        double df_dx = 0.0, df_dy = 0.0, df_dz = 0.0;
        double d_dfdx = 0.0, d_dfdy = 0.0, d_dfdz = 0.0;

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    const Eigen::Index flat_idx = (ix + a) + nx * ((iy + b) + ny * (iz + c));
                    const double v_val = v_ptr[flat_idx];
                    const double vd_val = v_n.requires_grad ? vd_ptr[flat_idx] : 0.0;

                    const double w_abc = wx[a] * wy[b] * wz[c];
                    const double wx_abc = dwx[a] * wy[b] * wz[c];
                    const double wy_abc = wx[a] * dwy[b] * wz[c];
                    const double wz_abc = wx[a] * wy[b] * dwz[c];

                    const double wxy_abc = dwx[a] * dwy[b] * wz[c];
                    const double wxz_abc = dwx[a] * wy[b] * dwz[c];
                    const double wyz_abc = wx[a] * dwy[b] * dwz[c];

                    df_dx += wx_abc * v_val;
                    df_dy += wy_abc * v_val;
                    df_dz += wz_abc * v_val;

                    d_dfdx += (wxy_abc * yqd + wxz_abc * zqd) * v_val + wx_abc * vd_val;
                    d_dfdy += (wxy_abc * xqd + wyz_abc * zqd) * v_val + wy_abc * vd_val;
                    d_dfdz += (wxz_abc * xqd + wyz_abc * yqd) * v_val + wz_abc * vd_val;

                    if (v_n.requires_grad) {
                        const double dw_dot = wx_abc * xqd + wy_abc * yqd + wz_abc * zqd;
                        vdb_ptr[flat_idx] += fdb * w_abc + fb * dw_dot;
                    }
                }
            }
        }

        if (xq_n.requires_grad) xqdb_ptr[m] += fdb * df_dx + fb * d_dfdx;
        if (yq_n.requires_grad) yqdb_ptr[m] += fdb * df_dy + fb * d_dfdy;
        if (zq_n.requires_grad) zqdb_ptr[m] += fdb * df_dz + fb * d_dfdz;
    }
}

// ============================================================================
// 5B. BSplineEval3DNode Implementation (3D Tensor-Product B-Spline Volume)
// ============================================================================
BSplineEval3DNode::BSplineEval3DNode(Shape s, int u_query_id, int v_query_id, int w_query_id, int ctrl_pts_id,
                                     Eigen::VectorXd ku, Eigen::VectorXd kv, Eigen::VectorXd kw,
                                     int deg_u, int deg_v, int deg_w,
                                     bool allow_extrap, std::string name)
    : Node(OpType::BSplineEval3D, s, {u_query_id, v_query_id, w_query_id, ctrl_pts_id}, std::move(name)),
      knots_u(std::move(ku)),
      knots_v(std::move(kv)),
      knots_w(std::move(kw)),
      degree_u(deg_u),
      degree_v(deg_v),
      degree_w(deg_w),
      extrapolate(allow_extrap) {
    if (degree_u < 1 || degree_v < 1 || degree_w < 1 ||
        degree_u > kMaxStackSplineDeg || degree_v > kMaxStackSplineDeg || degree_w > kMaxStackSplineDeg) {
        throw std::invalid_argument("BSplineEval3DNode: degree_u, degree_v, and degree_w must be in [1, 7].");
    }
}

void BSplineEval3DNode::forward(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& w_n = graph->node(inputs[2]);
    const Node& p_n = graph->node(inputs[3]);

    const Eigen::Index M = size();
    const Eigen::Index nu = knots_u.size() - degree_u - 1;
    const Eigen::Index nv = knots_v.size() - degree_v - 1;

    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* w_ptr = ws.val.data() + w_n.offset;
    const double* p_ptr = ws.val.data() + p_n.offset;
    double* out_ptr = ws.val.data() + offset;

    BasisDersStack du, dv, dw;
    for (Eigen::Index m = 0; m < M; ++m) {
        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval3DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval3DNode(v)");
        const int span_w = find_span_1d(knots_w, degree_w, extrapolate, w_ptr[m], "BSplineEval3DNode(w)");

        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], 0, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], 0, dv);
        ders_basis_funs_stack_1d(knots_w, degree_w, span_w, w_ptr[m], 0, dw);

        // Hierarchical 3D tensor contraction: w -> v -> u
        double val = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const Eigen::Index iu = static_cast<Eigen::Index>(span_u - degree_u + a);
            double sum_v = 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                const Eigen::Index iv = static_cast<Eigen::Index>(span_v - degree_v + b);
                double sum_w = 0.0;
                for (int c = 0; c <= degree_w; ++c) {
                    const Eigen::Index iw = static_cast<Eigen::Index>(span_w - degree_w + c);
                    const Eigen::Index flat_idx = iu + nu * (iv + nv * iw);
                    sum_w += dw(0, c) * p_ptr[flat_idx];
                }
                sum_v += dv(0, b) * sum_w;
            }
            val += du(0, a) * sum_v;
        }
        out_ptr[m] = val;
    }
}

void BSplineEval3DNode::backward(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& w_n = graph->node(inputs[2]);
    const Node& p_n = graph->node(inputs[3]);
    if (!u_n.requires_grad && !v_n.requires_grad && !w_n.requires_grad && !p_n.requires_grad) return;

    const Eigen::Index M = size();
    const Eigen::Index nu = knots_u.size() - degree_u - 1;
    const Eigen::Index nv = knots_v.size() - degree_v - 1;

    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* w_ptr = ws.val.data() + w_n.offset;
    const double* p_ptr = ws.val.data() + p_n.offset;
    const double* fb_ptr = ws.bar.data() + offset;

    double* ub_ptr = ws.bar.data() + u_n.offset;
    double* vb_ptr = ws.bar.data() + v_n.offset;
    double* wb_ptr = ws.bar.data() + w_n.offset;
    double* pb_ptr = ws.bar.data() + p_n.offset;

    BasisDersStack du, dv, dw;
    const int nd_u = u_n.requires_grad ? 1 : 0;
    const int nd_v = v_n.requires_grad ? 1 : 0;
    const int nd_w = w_n.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        if (fb == 0.0) continue;

        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval3DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval3DNode(v)");
        const int span_w = find_span_1d(knots_w, degree_w, extrapolate, w_ptr[m], "BSplineEval3DNode(w)");

        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);
        ders_basis_funs_stack_1d(knots_w, degree_w, span_w, w_ptr[m], nd_w, dw);

        double Su = 0.0, Sv = 0.0, Sw = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const Eigen::Index iu = static_cast<Eigen::Index>(span_u - degree_u + a);
            const double Nu0 = du(0, a);
            const double Nu1 = u_n.requires_grad ? du(1, a) : 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                const Eigen::Index iv = static_cast<Eigen::Index>(span_v - degree_v + b);
                const double Nv0 = dv(0, b);
                const double Nv1 = v_n.requires_grad ? dv(1, b) : 0.0;
                const double Nu0_Nv0 = Nu0 * Nv0;
                const double Nu1_Nv0 = Nu1 * Nv0;
                const double Nu0_Nv1 = Nu0 * Nv1;
                for (int c = 0; c <= degree_w; ++c) {
                    const Eigen::Index iw = static_cast<Eigen::Index>(span_w - degree_w + c);
                    const Eigen::Index flat_idx = iu + nu * (iv + nv * iw);
                    const double Nw0 = dw(0, c);
                    const double Nw1 = w_n.requires_grad ? dw(1, c) : 0.0;
                    const double p_val = p_ptr[flat_idx];

                    if (u_n.requires_grad) Su += (Nu1_Nv0 * Nw0) * p_val;
                    if (v_n.requires_grad) Sv += (Nu0_Nv1 * Nw0) * p_val;
                    if (w_n.requires_grad) Sw += (Nu0_Nv0 * Nw1) * p_val;
                    if (p_n.requires_grad) pb_ptr[flat_idx] += fb * (Nu0_Nv0 * Nw0);
                }
            }
        }

        if (u_n.requires_grad) ub_ptr[m] += fb * Su;
        if (v_n.requires_grad) vb_ptr[m] += fb * Sv;
        if (w_n.requires_grad) wb_ptr[m] += fb * Sw;
    }
}

void BSplineEval3DNode::forward_tangent(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& w_n = graph->node(inputs[2]);
    const Node& p_n = graph->node(inputs[3]);
    const Eigen::Index M = size();
    double* fd_ptr = ws.dot.data() + offset;

    if (!u_n.requires_grad && !v_n.requires_grad && !w_n.requires_grad && !p_n.requires_grad) {
        ws.dot.segment(offset, M).setZero();
        return;
    }

    const Eigen::Index nu = knots_u.size() - degree_u - 1;
    const Eigen::Index nv = knots_v.size() - degree_v - 1;

    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* w_ptr = ws.val.data() + w_n.offset;
    const double* p_ptr = ws.val.data() + p_n.offset;

    const double* ud_ptr = ws.dot.data() + u_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;
    const double* wd_ptr = ws.dot.data() + w_n.offset;
    const double* pd_ptr = ws.dot.data() + p_n.offset;

    BasisDersStack du, dv, dw;
    const int nd_u = u_n.requires_grad ? 1 : 0;
    const int nd_v = v_n.requires_grad ? 1 : 0;
    const int nd_w = w_n.requires_grad ? 1 : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval3DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval3DNode(v)");
        const int span_w = find_span_1d(knots_w, degree_w, extrapolate, w_ptr[m], "BSplineEval3DNode(w)");

        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);
        ders_basis_funs_stack_1d(knots_w, degree_w, span_w, w_ptr[m], nd_w, dw);

        const double ud = u_n.requires_grad ? ud_ptr[m] : 0.0;
        const double vd = v_n.requires_grad ? vd_ptr[m] : 0.0;
        const double wd = w_n.requires_grad ? wd_ptr[m] : 0.0;

        double fd = 0.0;
        for (int a = 0; a <= degree_u; ++a) {
            const Eigen::Index iu = static_cast<Eigen::Index>(span_u - degree_u + a);
            const double Nu0 = du(0, a);
            const double Nu1 = u_n.requires_grad ? du(1, a) : 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                const Eigen::Index iv = static_cast<Eigen::Index>(span_v - degree_v + b);
                const double Nv0 = dv(0, b);
                const double Nv1 = v_n.requires_grad ? dv(1, b) : 0.0;
                for (int c = 0; c <= degree_w; ++c) {
                    const Eigen::Index iw = static_cast<Eigen::Index>(span_w - degree_w + c);
                    const Eigen::Index flat_idx = iu + nu * (iv + nv * iw);
                    const double Nw0 = dw(0, c);
                    const double Nw1 = w_n.requires_grad ? dw(1, c) : 0.0;
                    const double p_val = p_ptr[flat_idx];

                    const double dW_dt = (Nu1 * Nv0 * Nw0) * ud +
                                         (Nu0 * Nv1 * Nw0) * vd +
                                         (Nu0 * Nv0 * Nw1) * wd;
                    fd += dW_dt * p_val;
                    if (p_n.requires_grad) {
                        fd += (Nu0 * Nv0 * Nw0) * pd_ptr[flat_idx];
                    }
                }
            }
        }
        fd_ptr[m] = fd;
    }
}

void BSplineEval3DNode::backward_hessian(Workspace& ws) const {
    const Node& u_n = graph->node(inputs[0]);
    const Node& v_n = graph->node(inputs[1]);
    const Node& w_n = graph->node(inputs[2]);
    const Node& p_n = graph->node(inputs[3]);
    if (!u_n.requires_grad && !v_n.requires_grad && !w_n.requires_grad && !p_n.requires_grad) return;

    const Eigen::Index M = size();
    const Eigen::Index nu = knots_u.size() - degree_u - 1;
    const Eigen::Index nv = knots_v.size() - degree_v - 1;

    const double* u_ptr = ws.val.data() + u_n.offset;
    const double* v_ptr = ws.val.data() + v_n.offset;
    const double* w_ptr = ws.val.data() + w_n.offset;
    const double* p_ptr = ws.val.data() + p_n.offset;

    const double* ud_ptr = ws.dot.data() + u_n.offset;
    const double* vd_ptr = ws.dot.data() + v_n.offset;
    const double* wd_ptr = ws.dot.data() + w_n.offset;
    const double* pd_ptr = ws.dot.data() + p_n.offset;

    const double* fb_ptr = ws.bar.data() + offset;
    const double* fdb_ptr = ws.dot_bar.data() + offset;

    double* udb_ptr = ws.dot_bar.data() + u_n.offset;
    double* vdb_ptr = ws.dot_bar.data() + v_n.offset;
    double* wdb_ptr = ws.dot_bar.data() + w_n.offset;
    double* pdb_ptr = ws.dot_bar.data() + p_n.offset;

    BasisDersStack du, dv, dw;
    const bool any_coord_grad = u_n.requires_grad || v_n.requires_grad || w_n.requires_grad;
    const int nd_u = any_coord_grad ? std::min(2, degree_u) : 0;
    const int nd_v = any_coord_grad ? std::min(2, degree_v) : 0;
    const int nd_w = any_coord_grad ? std::min(2, degree_w) : 0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double fb = fb_ptr[m];
        const double fdb = fdb_ptr[m];
        if (fb == 0.0 && fdb == 0.0) continue;

        const int span_u = find_span_1d(knots_u, degree_u, extrapolate, u_ptr[m], "BSplineEval3DNode(u)");
        const int span_v = find_span_1d(knots_v, degree_v, extrapolate, v_ptr[m], "BSplineEval3DNode(v)");
        const int span_w = find_span_1d(knots_w, degree_w, extrapolate, w_ptr[m], "BSplineEval3DNode(w)");

        ders_basis_funs_stack_1d(knots_u, degree_u, span_u, u_ptr[m], nd_u, du);
        ders_basis_funs_stack_1d(knots_v, degree_v, span_v, v_ptr[m], nd_v, dv);
        ders_basis_funs_stack_1d(knots_w, degree_w, span_w, w_ptr[m], nd_w, dw);

        const double ud = u_n.requires_grad ? ud_ptr[m] : 0.0;
        const double vd = v_n.requires_grad ? vd_ptr[m] : 0.0;
        const double wd = w_n.requires_grad ? wd_ptr[m] : 0.0;

        double Su = 0.0, Sv = 0.0, Sw = 0.0;
        double Suu = 0.0, Svv = 0.0, Sww = 0.0;
        double Suv = 0.0, Suw = 0.0, Svw = 0.0;
        double dSu_dP = 0.0, dSv_dP = 0.0, dSw_dP = 0.0;

        for (int a = 0; a <= degree_u; ++a) {
            const Eigen::Index iu = static_cast<Eigen::Index>(span_u - degree_u + a);
            const double Nu0 = du(0, a);
            const double Nu1 = (nd_u >= 1) ? du(1, a) : 0.0;
            const double Nu2 = (nd_u >= 2) ? du(2, a) : 0.0;
            for (int b = 0; b <= degree_v; ++b) {
                const Eigen::Index iv = static_cast<Eigen::Index>(span_v - degree_v + b);
                const double Nv0 = dv(0, b);
                const double Nv1 = (nd_v >= 1) ? dv(1, b) : 0.0;
                const double Nv2 = (nd_v >= 2) ? dv(2, b) : 0.0;
                for (int c = 0; c <= degree_w; ++c) {
                    const Eigen::Index iw = static_cast<Eigen::Index>(span_w - degree_w + c);
                    const Eigen::Index flat_idx = iu + nu * (iv + nv * iw);
                    const double Nw0 = dw(0, c);
                    const double Nw1 = (nd_w >= 1) ? dw(1, c) : 0.0;
                    const double Nw2 = (nd_w >= 2) ? dw(2, c) : 0.0;

                    const double p_val = p_ptr[flat_idx];
                    const double pd_val = p_n.requires_grad ? pd_ptr[flat_idx] : 0.0;

                    const double W0 = Nu0 * Nv0 * Nw0;
                    const double Wu = Nu1 * Nv0 * Nw0;
                    const double Wv = Nu0 * Nv1 * Nw0;
                    const double Ww = Nu0 * Nv0 * Nw1;

                    Su += Wu * p_val;
                    Sv += Wv * p_val;
                    Sw += Ww * p_val;

                    Suu += (Nu2 * Nv0 * Nw0) * p_val;
                    Svv += (Nu0 * Nv2 * Nw0) * p_val;
                    Sww += (Nu0 * Nv0 * Nw2) * p_val;

                    Suv += (Nu1 * Nv1 * Nw0) * p_val;
                    Suw += (Nu1 * Nv0 * Nw1) * p_val;
                    Svw += (Nu0 * Nv1 * Nw1) * p_val;

                    dSu_dP += Wu * pd_val;
                    dSv_dP += Wv * pd_val;
                    dSw_dP += Ww * pd_val;

                    if (p_n.requires_grad) {
                        pdb_ptr[flat_idx] += fdb * W0 + fb * (Wu * ud + Wv * vd + Ww * wd);
                    }
                }
            }
        }

        if (u_n.requires_grad) {
            udb_ptr[m] += fdb * Su + fb * (Suu * ud + Suv * vd + Suw * wd + dSu_dP);
        }
        if (v_n.requires_grad) {
            vdb_ptr[m] += fdb * Sv + fb * (Suv * ud + Svv * vd + Svw * wd + dSv_dP);
        }
        if (w_n.requires_grad) {
            wdb_ptr[m] += fdb * Sw + fb * (Suw * ud + Svw * vd + Sww * wd + dSw_dP);
        }
    }
}

// ============================================================================
// 6. Operator Builder Helpers
// ============================================================================
namespace ops {

int linear_interp(Graph& g, int x_query_id, int y_data_id,
                  const Eigen::VectorXd& x_grid, bool extrapolate) {
    const Shape sx = g.node(x_query_id).shape;
    const Shape sy = g.node(y_data_id).shape;
    if (sy.rows != x_grid.size()) {
        throw std::invalid_argument("linear_interp: y_data.rows must match x_grid.size().");
    }
    const Shape sout(sx.size(), sy.cols);
    return g.add_node(std::make_shared<LinearInterpNode>(
        sout, x_query_id, y_data_id, x_grid, extrapolate));
}

int bspline_eval(Graph& g, int u_query_id, int ctrl_pts_id,
                 const Eigen::VectorXd& knots, int degree, bool extrapolate) {
    const Shape su = g.node(u_query_id).shape;
    const Shape sp = g.node(ctrl_pts_id).shape;
    if (knots.size() != sp.rows + degree + 1) {
        throw std::invalid_argument("bspline_eval: knot vector length must equal n_ctrl + degree + 1.");
    }
    const Shape sout(su.size(), sp.cols);
    return g.add_node(std::make_shared<BSplineEvalNode>(
        sout, u_query_id, ctrl_pts_id, knots, degree, extrapolate));
}

int linear_interp_2d(Graph& g, int x_query_id, int y_query_id, int z_data_id,
                     const Eigen::VectorXd& x_grid, const Eigen::VectorXd& y_grid,
                     bool extrapolate) {
    const Shape sx = g.node(x_query_id).shape;
    const Shape sy = g.node(y_query_id).shape;
    const Shape sz = g.node(z_data_id).shape;
    if (sx.size() != sy.size()) {
        throw std::invalid_argument("linear_interp_2d: x_query and y_query must have the same number of elements.");
    }
    if (sz.rows != x_grid.size() || sz.cols != y_grid.size()) {
        throw std::invalid_argument("linear_interp_2d: z_data shape must match (x_grid.size(), y_grid.size()).");
    }
    return g.add_node(std::make_shared<LinearInterp2DNode>(
        Shape(sx.size(), 1), x_query_id, y_query_id, z_data_id, x_grid, y_grid, extrapolate));
}

int bspline_eval_2d(Graph& g, int u_query_id, int v_query_id, int ctrl_pts_id,
                    const Eigen::VectorXd& knots_u, const Eigen::VectorXd& knots_v,
                    int degree_u, int degree_v, bool extrapolate) {
    const Shape su = g.node(u_query_id).shape;
    const Shape sv = g.node(v_query_id).shape;
    const Shape sp = g.node(ctrl_pts_id).shape;
    if (su.size() != sv.size()) {
        throw std::invalid_argument("bspline_eval_2d: u_query and v_query must have the same number of elements.");
    }
    if (knots_u.size() != sp.rows + degree_u + 1 || knots_v.size() != sp.cols + degree_v + 1) {
        throw std::invalid_argument("bspline_eval_2d: knot vector lengths must match control net dimensions.");
    }
    return g.add_node(std::make_shared<BSplineEval2DNode>(
        Shape(su.size(), 1), u_query_id, v_query_id, ctrl_pts_id,
        knots_u, knots_v, degree_u, degree_v, extrapolate));
}

int linear_interp_3d(Graph& g, int x_query_id, int y_query_id, int z_query_id, int v_data_id,
                     const Eigen::VectorXd& x_grid, const Eigen::VectorXd& y_grid,
                     const Eigen::VectorXd& z_grid, bool extrapolate) {
    const Shape sx = g.node(x_query_id).shape;
    const Shape sy = g.node(y_query_id).shape;
    const Shape sz = g.node(z_query_id).shape;
    const Shape sv = g.node(v_data_id).shape;
    if (sx.size() != sy.size() || sx.size() != sz.size()) {
        throw std::invalid_argument("linear_interp_3d: x_query, y_query, and z_query must have the same size.");
    }
    if (sv.size() != x_grid.size() * y_grid.size() * z_grid.size()) {
        throw std::invalid_argument("linear_interp_3d: v_data size must equal len(x_grid)*len(y_grid)*len(z_grid).");
    }
    return g.add_node(std::make_shared<LinearInterp3DNode>(
        Shape(sx.size(), 1), x_query_id, y_query_id, z_query_id, v_data_id,
        x_grid, y_grid, z_grid, extrapolate));
}

int bspline_eval_3d(Graph& g, int u_query_id, int v_query_id, int w_query_id, int ctrl_pts_id,
                    const Eigen::VectorXd& knots_u, const Eigen::VectorXd& knots_v,
                    const Eigen::VectorXd& knots_w,
                    int degree_u, int degree_v, int degree_w, bool extrapolate) {
    const Shape su = g.node(u_query_id).shape;
    const Shape sv = g.node(v_query_id).shape;
    const Shape sw = g.node(w_query_id).shape;
    const Shape sp = g.node(ctrl_pts_id).shape;
    if (su.size() != sv.size() || su.size() != sw.size()) {
        throw std::invalid_argument("bspline_eval_3d: u_query, v_query, and w_query must have the same size.");
    }
    const Eigen::Index nu = knots_u.size() - degree_u - 1;
    const Eigen::Index nv = knots_v.size() - degree_v - 1;
    const Eigen::Index nw = knots_w.size() - degree_w - 1;
    if (nu < 1 || nv < 1 || nw < 1 || sp.size() != nu * nv * nw) {
        throw std::invalid_argument("bspline_eval_3d: ctrl_pts size must equal (len(ku)-du-1)*(len(kv)-dv-1)*(len(kw)-dw-1).");
    }
    return g.add_node(std::make_shared<BSplineEval3DNode>(
        Shape(su.size(), 1), u_query_id, v_query_id, w_query_id, ctrl_pts_id,
        knots_u, knots_v, knots_w, degree_u, degree_v, degree_w, extrapolate));
}

} // namespace ops

} // namespace fastopt
