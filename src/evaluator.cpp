#include "fastopt/evaluator.hpp"
#include "fastopt/ops.hpp"
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <stdexcept>

namespace fastopt {

namespace {

inline int ctz64(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int c = 0;
    while ((x & 1ULL) == 0ULL) {
        x >>= 1;
        ++c;
    }
    return c;
#endif
}

inline int popcount64(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        x &= (x - 1ULL);
        ++count;
    }
    return count;
#endif
}

inline void or_bits(std::uint64_t* dst, const std::uint64_t* src, std::size_t n_words) {
    for (std::size_t w = 0; w < n_words; ++w) {
        dst[w] |= src[w];
    }
}

inline bool bits_intersect(const std::uint64_t* a, const std::uint64_t* b, std::size_t n_words) {
    for (std::size_t w = 0; w < n_words; ++w) {
        if ((a[w] & b[w]) != 0ULL) return true;
    }
    return false;
}

inline int bits_popcount(const std::uint64_t* a, std::size_t n_words) {
    int total = 0;
    for (std::size_t w = 0; w < n_words; ++w) {
        total += popcount64(a[w]);
    }
    return total;
}

// Add clique D x D to symmetric Hessian adjacency bitsets
inline void add_clique(std::vector<std::uint64_t>& hess_bits,
                       const std::uint64_t* d,
                       std::size_t n_words) {
    for (std::size_t w = 0; w < n_words; ++w) {
        std::uint64_t word = d[w];
        while (word != 0ULL) {
            const int bit = ctz64(word);
            const std::size_t u = (w << 6) + static_cast<std::size_t>(bit);
            or_bits(&hess_bits[u * n_words], d, n_words);
            word &= (word - 1ULL);
        }
    }
}

// Add bipartite coupling Da x Db (and Db x Da) to symmetric Hessian adjacency bitsets
inline void add_bipartite(std::vector<std::uint64_t>& hess_bits,
                          const std::uint64_t* da,
                          const std::uint64_t* db,
                          std::size_t n_words) {
    for (std::size_t w = 0; w < n_words; ++w) {
        std::uint64_t word = da[w];
        while (word != 0ULL) {
            const int bit = ctz64(word);
            const std::size_t u = (w << 6) + static_cast<std::size_t>(bit);
            or_bits(&hess_bits[u * n_words], db, n_words);
            word &= (word - 1ULL);
        }
    }
    for (std::size_t w = 0; w < n_words; ++w) {
        std::uint64_t word = db[w];
        while (word != 0ULL) {
            const int bit = ctz64(word);
            const std::size_t v = (w << 6) + static_cast<std::size_t>(bit);
            or_bits(&hess_bits[v * n_words], da, n_words);
            word &= (word - 1ULL);
        }
    }
}

// Largest-Degree-First greedy graph coloring using per-color accumulator bitsets
std::vector<std::vector<int>> greedy_color_bitsets(
    const std::vector<std::uint64_t>& item_bits,
    int num_items,
    std::size_t n_words) {
    std::vector<std::vector<int>> groups;
    if (num_items <= 0 || n_words == 0) return groups;

    std::vector<int> degrees(static_cast<std::size_t>(num_items), 0);
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(num_items));

    for (int i = 0; i < num_items; ++i) {
        const int deg = bits_popcount(&item_bits[static_cast<std::size_t>(i) * n_words], n_words);
        degrees[static_cast<std::size_t>(i)] = deg;
        if (deg > 0) {
            order.push_back(i);
        }
    }

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return degrees[static_cast<std::size_t>(a)] > degrees[static_cast<std::size_t>(b)];
    });

    // Each color group maintains the bitwise OR of all its member bitsets
    std::vector<std::uint64_t> color_union_bits;

    for (int item : order) {
        const std::uint64_t* my_bits = &item_bits[static_cast<std::size_t>(item) * n_words];
        int assigned_color = -1;
        const int n_colors = static_cast<int>(groups.size());

        for (int c = 0; c < n_colors; ++c) {
            const std::uint64_t* c_bits =
                &color_union_bits[static_cast<std::size_t>(c) * n_words];
            if (!bits_intersect(my_bits, c_bits, n_words)) {
                assigned_color = c;
                break;
            }
        }

        if (assigned_color < 0) {
            assigned_color = n_colors;
            groups.emplace_back();
            color_union_bits.resize(static_cast<std::size_t>(n_colors + 1) * n_words, 0ULL);
        }

        groups[static_cast<std::size_t>(assigned_color)].push_back(item);
        or_bits(&color_union_bits[static_cast<std::size_t>(assigned_color) * n_words],
                my_bits,
                n_words);
    }

    for (auto& g : groups) {
        std::sort(g.begin(), g.end());
    }
    return groups;
}

// Compute outer workspace scalar offset corresponding to slice element `elem_idx` of batch `b`
Eigen::Index outer_slice_elem_offset(
    const Node& outer_n,
    int axis,
    int b,
    const Shape& inner_s,
    Eigen::Index elem_idx) {
    const Eigen::Index sz = inner_s.size();
    if (axis == -1) {
        return outer_n.offset + elem_idx;
    } else if (axis == 1 || outer_n.shape.cols == 1) {
        return outer_n.offset + static_cast<Eigen::Index>(b) * sz + elem_idx;
    } else {
        // axis == 0 on a 2D matrix: inner_s = (r_slice, c_slice)
        const Eigen::Index r_slice = inner_s.rows;
        const Eigen::Index r_in = elem_idx % r_slice;
        const Eigen::Index c_in = elem_idx / r_slice;
        const Eigen::Index outer_r = static_cast<Eigen::Index>(b) * r_slice + r_in;
        return outer_n.offset + outer_r + c_in * outer_n.shape.rows;
    }
}

// ============================================================================
// Exact Bitset Symbolic Sparsity Propagation Across Graph
// ============================================================================
void propagate_graph_sparsity(
    const Graph& g,
    std::vector<std::uint64_t>& deps,
    std::vector<std::uint64_t>& hess_bits,
    bool compute_hessian) {
    const Eigen::Index n_vars = g.total_var_size();
    const Eigen::Index mem_sz = g.total_memory_size();
    const std::size_t n_words = static_cast<std::size_t>((n_vars + 63) / 64);

    deps.assign(static_cast<std::size_t>(mem_sz) * n_words, 0ULL);
    if (compute_hessian) {
        hess_bits.assign(static_cast<std::size_t>(n_vars) * n_words, 0ULL);
    } else {
        hess_bits.clear();
    }

    if (n_vars == 0 || n_words == 0) return;

    auto deps_of = [&](Eigen::Index slot) -> std::uint64_t* {
        return &deps[static_cast<std::size_t>(slot) * n_words];
    };

    // 1. Seed each scalar decision variable slot i in [0, n_vars)
    for (Eigen::Index i = 0; i < n_vars; ++i) {
        deps_of(i)[static_cast<std::size_t>(i) >> 6] |= (1ULL << (i & 63));
    }

    std::vector<std::uint64_t> scratch(n_words, 0ULL);
    std::vector<std::uint64_t> scratch2(n_words, 0ULL);

    // 2. Forward topological propagation through dynamic_exec_order
    for (int id : g.dynamic_exec_order()) {
        const Node& nd = g.node(id);
        if (!nd.requires_grad) continue;

        switch (nd.op) {
            case OpType::Neg:
            case OpType::Sin:
            case OpType::Cos:
            case OpType::Tan:
            case OpType::Asin:
            case OpType::Acos:
            case OpType::Atan:
            case OpType::Sinh:
            case OpType::Cosh:
            case OpType::Tanh:
            case OpType::Exp:
            case OpType::Log:
            case OpType::Sqrt:
            case OpType::Abs:
            case OpType::Square: {
                const Node& in = g.node(nd.inputs[0]);
                const bool nonlinear = (nd.op != OpType::Neg && nd.op != OpType::Abs);
                for (Eigen::Index i = 0; i < nd.size(); ++i) {
                    const std::uint64_t* src = deps_of(in.offset + i);
                    std::uint64_t* dst = deps_of(nd.offset + i);
                    std::copy(src, src + n_words, dst);
                    if (compute_hessian && nonlinear) {
                        add_clique(hess_bits, src, n_words);
                    }
                }
                break;
            }

            case OpType::Lt:
            case OpType::Le:
            case OpType::Gt:
            case OpType::Ge:
            case OpType::Eq:
                // Step comparison masks have zero derivative everywhere
                break;

            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:
            case OpType::Pow:
            case OpType::Atan2: {
                const Node& a = g.node(nd.inputs[0]);
                const Node& b = g.node(nd.inputs[1]);
                const Eigen::Index R = nd.shape.rows;
                const Eigen::Index C = nd.shape.cols;

                for (Eigen::Index c = 0; c < C; ++c) {
                    const Eigen::Index ca = (a.shape.cols == 1) ? 0 : c;
                    const Eigen::Index cb = (b.shape.cols == 1) ? 0 : c;
                    for (Eigen::Index r = 0; r < R; ++r) {
                        const Eigen::Index ra = (a.shape.rows == 1) ? 0 : r;
                        const Eigen::Index rb = (b.shape.rows == 1) ? 0 : r;

                        const Eigen::Index slot_a = a.offset + ra + ca * a.shape.rows;
                        const Eigen::Index slot_b = b.offset + rb + cb * b.shape.rows;
                        const Eigen::Index slot_out = nd.offset + r + c * R;

                        const std::uint64_t* da = deps_of(slot_a);
                        const std::uint64_t* db = deps_of(slot_b);
                        std::uint64_t* dout = deps_of(slot_out);

                        if (a.requires_grad) or_bits(dout, da, n_words);
                        if (b.requires_grad) or_bits(dout, db, n_words);

                        if (compute_hessian) {
                            if (nd.op == OpType::Mul) {
                                if (a.requires_grad && b.requires_grad) {
                                    add_bipartite(hess_bits, da, db, n_words);
                                }
                            } else if (nd.op == OpType::Div) {
                                if (a.requires_grad && b.requires_grad) {
                                    add_bipartite(hess_bits, da, db, n_words);
                                }
                                if (b.requires_grad) {
                                    add_clique(hess_bits, db, n_words);
                                }
                            } else if (nd.op == OpType::Pow || nd.op == OpType::Atan2) {
                                add_clique(hess_bits, dout, n_words);
                            }
                        }
                    }
                }
                break;
            }

            case OpType::MatMul: {
                const Node& a = g.node(nd.inputs[0]);
                const Node& b = g.node(nd.inputs[1]);
                const Eigen::Index M = a.shape.rows;
                const Eigen::Index K = a.shape.cols;
                const Eigen::Index N = b.shape.cols;

                for (Eigen::Index j = 0; j < N; ++j) {
                    for (Eigen::Index i = 0; i < M; ++i) {
                        std::uint64_t* dout = deps_of(nd.offset + i + j * M);
                        for (Eigen::Index k = 0; k < K; ++k) {
                            const std::uint64_t* da = deps_of(a.offset + i + k * M);
                            const std::uint64_t* db = deps_of(b.offset + k + j * K);
                            if (a.requires_grad) or_bits(dout, da, n_words);
                            if (b.requires_grad) or_bits(dout, db, n_words);
                            if (compute_hessian && a.requires_grad && b.requires_grad) {
                                add_bipartite(hess_bits, da, db, n_words);
                            }
                        }
                    }
                }
                break;
            }

            case OpType::LinearSolve: {
                const Node& a = g.node(nd.inputs[0]);
                const Node& b = g.node(nd.inputs[1]);
                const Eigen::Index M = a.shape.rows;
                const Eigen::Index N = b.shape.cols;

                std::fill(scratch.begin(), scratch.end(), 0ULL);
                if (a.requires_grad) {
                    for (Eigen::Index i = 0; i < a.size(); ++i) {
                        or_bits(scratch.data(), deps_of(a.offset + i), n_words);
                    }
                }
                for (Eigen::Index j = 0; j < N; ++j) {
                    std::copy(scratch.begin(), scratch.end(), scratch2.begin());
                    if (b.requires_grad) {
                        for (Eigen::Index i = 0; i < M; ++i) {
                            or_bits(scratch2.data(), deps_of(b.offset + i + j * M), n_words);
                        }
                    }
                    for (Eigen::Index i = 0; i < M; ++i) {
                        std::uint64_t* dout = deps_of(nd.offset + i + j * M);
                        std::copy(scratch2.begin(), scratch2.end(), dout);
                    }
                    if (compute_hessian && a.requires_grad) {
                        add_clique(hess_bits, scratch2.data(), n_words);
                    }
                }
                break;
            }

            case OpType::Dot: {
                const Node& a = g.node(nd.inputs[0]);
                const Node& b = g.node(nd.inputs[1]);
                std::uint64_t* dout = deps_of(nd.offset);
                for (Eigen::Index i = 0; i < a.size(); ++i) {
                    const std::uint64_t* da = deps_of(a.offset + i);
                    const std::uint64_t* db = deps_of(b.offset + i);
                    if (a.requires_grad) or_bits(dout, da, n_words);
                    if (b.requires_grad) or_bits(dout, db, n_words);
                    if (compute_hessian && a.requires_grad && b.requires_grad) {
                        add_bipartite(hess_bits, da, db, n_words);
                    }
                }
                break;
            }

            case OpType::Sum:
            case OpType::KSLogSumExp: {
                const Node& in = g.node(nd.inputs[0]);
                const Eigen::Index R = in.shape.rows;
                const Eigen::Index C = in.shape.cols;
                int ax = -1;
                if (nd.op == OpType::Sum) {
                    ax = static_cast<const SumNode&>(nd).axis;
                } else {
                    ax = static_cast<const KSLogSumExpNode&>(nd).axis;
                }
                const bool is_ks = (nd.op == OpType::KSLogSumExp);

                if (ax == -1) {
                    std::uint64_t* dout = deps_of(nd.offset);
                    for (Eigen::Index i = 0; i < in.size(); ++i) {
                        or_bits(dout, deps_of(in.offset + i), n_words);
                    }
                    if (compute_hessian && is_ks) {
                        add_clique(hess_bits, dout, n_words);
                    }
                } else if (ax == 0) {
                    for (Eigen::Index c = 0; c < C; ++c) {
                        std::uint64_t* dout = deps_of(nd.offset + c);
                        for (Eigen::Index r = 0; r < R; ++r) {
                            or_bits(dout, deps_of(in.offset + r + c * R), n_words);
                        }
                        if (compute_hessian && is_ks) {
                            add_clique(hess_bits, dout, n_words);
                        }
                    }
                } else {
                    for (Eigen::Index r = 0; r < R; ++r) {
                        std::uint64_t* dout = deps_of(nd.offset + r);
                        for (Eigen::Index c = 0; c < C; ++c) {
                            or_bits(dout, deps_of(in.offset + r + c * R), n_words);
                        }
                        if (compute_hessian && is_ks) {
                            add_clique(hess_bits, dout, n_words);
                        }
                    }
                }
                break;
            }

            case OpType::Transpose: {
                const Node& in = g.node(nd.inputs[0]);
                const Eigen::Index R = in.shape.rows;
                const Eigen::Index C = in.shape.cols;
                for (Eigen::Index c = 0; c < C; ++c) {
                    for (Eigen::Index r = 0; r < R; ++r) {
                        const std::uint64_t* src = deps_of(in.offset + r + c * R);
                        std::uint64_t* dst = deps_of(nd.offset + c + r * C);
                        std::copy(src, src + n_words, dst);
                    }
                }
                break;
            }

            case OpType::Slice: {
                const auto& sn = static_cast<const SliceNode&>(nd);
                const Node& in = g.node(nd.inputs[0]);
                const Eigen::Index R_out = sn.r1 - sn.r0;
                const Eigen::Index C_out = sn.c1 - sn.c0;
                for (Eigen::Index c = 0; c < C_out; ++c) {
                    for (Eigen::Index r = 0; r < R_out; ++r) {
                        const std::uint64_t* src =
                            deps_of(in.offset + (sn.r0 + r) + (sn.c0 + c) * in.shape.rows);
                        std::uint64_t* dst = deps_of(nd.offset + r + c * R_out);
                        std::copy(src, src + n_words, dst);
                    }
                }
                break;
            }

            case OpType::Concat: {
                const auto& cn = static_cast<const ConcatNode&>(nd);
                if (cn.axis == 1 || nd.shape.cols == 1) {
                    Eigen::Index cur = 0;
                    for (int in_id : cn.inputs) {
                        const Node& in = g.node(in_id);
                        if (in.requires_grad) {
                            for (Eigen::Index i = 0; i < in.size(); ++i) {
                                const std::uint64_t* src = deps_of(in.offset + i);
                                std::uint64_t* dst = deps_of(nd.offset + cur + i);
                                std::copy(src, src + n_words, dst);
                            }
                        }
                        cur += in.size();
                    }
                } else {
                    Eigen::Index row_cursor = 0;
                    const Eigen::Index R_tot = nd.shape.rows;
                    const Eigen::Index C_tot = nd.shape.cols;
                    for (int in_id : cn.inputs) {
                        const Node& in = g.node(in_id);
                        const Eigen::Index r_in = in.shape.rows;
                        if (in.requires_grad) {
                            for (Eigen::Index c = 0; c < C_tot; ++c) {
                                for (Eigen::Index r = 0; r < r_in; ++r) {
                                    const std::uint64_t* src = deps_of(in.offset + r + c * r_in);
                                    std::uint64_t* dst = deps_of(nd.offset + (row_cursor + r) + c * R_tot);
                                    std::copy(src, src + n_words, dst);
                                }
                            }
                        }
                        row_cursor += r_in;
                    }
                }
                break;
            }

            case OpType::Reshape: {
                const Node& in = g.node(nd.inputs[0]);
                for (Eigen::Index i = 0; i < nd.size(); ++i) {
                    const std::uint64_t* src = deps_of(in.offset + i);
                    std::uint64_t* dst = deps_of(nd.offset + i);
                    std::copy(src, src + n_words, dst);
                }
                break;
            }

            case OpType::Where:
            case OpType::SmoothWhere: {
                const Node& c_n = g.node(nd.inputs[0]);
                const Node& t_n = g.node(nd.inputs[1]);
                const Node& f_n = g.node(nd.inputs[2]);
                const Eigen::Index R = nd.shape.rows;
                const Eigen::Index C = nd.shape.cols;
                const bool is_smooth = (nd.op == OpType::SmoothWhere);

                for (Eigen::Index col = 0; col < C; ++col) {
                    const Eigen::Index cc = (c_n.shape.cols == 1) ? 0 : col;
                    const Eigen::Index ct = (t_n.shape.cols == 1) ? 0 : col;
                    const Eigen::Index cf = (f_n.shape.cols == 1) ? 0 : col;
                    for (Eigen::Index row = 0; row < R; ++row) {
                        const Eigen::Index rc = (c_n.shape.rows == 1) ? 0 : row;
                        const Eigen::Index rt = (t_n.shape.rows == 1) ? 0 : row;
                        const Eigen::Index rf = (f_n.shape.rows == 1) ? 0 : row;

                        std::uint64_t* dst = deps_of(nd.offset + row + col * R);
                        if (is_smooth && c_n.requires_grad) {
                            or_bits(dst, deps_of(c_n.offset + rc + cc * c_n.shape.rows), n_words);
                        }
                        if (t_n.requires_grad) {
                            or_bits(dst, deps_of(t_n.offset + rt + ct * t_n.shape.rows), n_words);
                        }
                        if (f_n.requires_grad) {
                            or_bits(dst, deps_of(f_n.offset + rf + cf * f_n.shape.rows), n_words);
                        }
                        if (compute_hessian && is_smooth) {
                            add_clique(hess_bits, dst, n_words);
                        }
                    }
                }
                break;
            }

            case OpType::LinearInterp:
            case OpType::BSplineEval: {
                const Node& u_n = g.node(nd.inputs[0]);
                const Node& p_n = g.node(nd.inputs[1]);
                const Eigen::Index M = u_n.size();
                const Eigen::Index K = p_n.shape.cols;

                for (Eigen::Index k = 0; k < K; ++k) {
                    std::fill(scratch.begin(), scratch.end(), 0ULL);
                    if (p_n.requires_grad) {
                        for (Eigen::Index r = 0; r < p_n.shape.rows; ++r) {
                            or_bits(scratch.data(), deps_of(p_n.offset + r + k * p_n.shape.rows), n_words);
                        }
                    }
                    for (Eigen::Index m = 0; m < M; ++m) {
                        std::uint64_t* dst = deps_of(nd.offset + m + k * M);
                        if (u_n.requires_grad) {
                            or_bits(dst, deps_of(u_n.offset + m), n_words);
                        }
                        if (p_n.requires_grad) {
                            or_bits(dst, scratch.data(), n_words);
                        }
                        if (compute_hessian && u_n.requires_grad) {
                            if (p_n.requires_grad) {
                                add_bipartite(hess_bits, deps_of(u_n.offset + m), scratch.data(), n_words);
                            }
                            if (nd.op == OpType::BSplineEval) {
                                add_clique(hess_bits, deps_of(u_n.offset + m), n_words);
                            }
                        }
                    }
                }
                break;
            }

            case OpType::LinearInterp2D:
            case OpType::BSplineEval2D: {
                const Node& u_n = g.node(nd.inputs[0]);
                const Node& v_n = g.node(nd.inputs[1]);
                const Node& p_n = g.node(nd.inputs[2]);
                std::fill(scratch.begin(), scratch.end(), 0ULL);
                if (p_n.requires_grad) {
                    for (Eigen::Index i = 0; i < p_n.size(); ++i) {
                        or_bits(scratch.data(), deps_of(p_n.offset + i), n_words);
                    }
                }
                for (Eigen::Index m = 0; m < nd.size(); ++m) {
                    std::uint64_t* dst = deps_of(nd.offset + m);
                    if (u_n.requires_grad) or_bits(dst, deps_of(u_n.offset + m), n_words);
                    if (v_n.requires_grad) or_bits(dst, deps_of(v_n.offset + m), n_words);
                    if (p_n.requires_grad) or_bits(dst, scratch.data(), n_words);
                    if (compute_hessian) {
                        add_clique(hess_bits, dst, n_words);
                    }
                }
                break;
            }

            case OpType::LinearInterp3D:
            case OpType::BSplineEval3D: {
                const Node& x_n = g.node(nd.inputs[0]);
                const Node& y_n = g.node(nd.inputs[1]);
                const Node& z_n = g.node(nd.inputs[2]);
                const Node& v_n = g.node(nd.inputs[3]);
                std::fill(scratch.begin(), scratch.end(), 0ULL);
                if (v_n.requires_grad) {
                    for (Eigen::Index i = 0; i < v_n.size(); ++i) {
                        or_bits(scratch.data(), deps_of(v_n.offset + i), n_words);
                    }
                }
                for (Eigen::Index m = 0; m < nd.size(); ++m) {
                    std::uint64_t* dst = deps_of(nd.offset + m);
                    if (x_n.requires_grad) or_bits(dst, deps_of(x_n.offset + m), n_words);
                    if (y_n.requires_grad) or_bits(dst, deps_of(y_n.offset + m), n_words);
                    if (z_n.requires_grad) or_bits(dst, deps_of(z_n.offset + m), n_words);
                    if (v_n.requires_grad) or_bits(dst, scratch.data(), n_words);
                    if (compute_hessian) {
                        add_clique(hess_bits, dst, n_words);
                    }
                }
                break;
            }

            case OpType::Map: {
                const auto& mn = static_cast<const MapNode&>(nd);
                std::vector<std::uint64_t> sub_deps;
                std::vector<std::uint64_t> sub_hess;
                propagate_graph_sparsity(*mn.body_graph, sub_deps, sub_hess, compute_hessian);

                const Eigen::Index n_sub_vars = mn.body_graph->total_var_size();
                const std::size_t n_sub_words = static_cast<std::size_t>((n_sub_vars + 63) / 64);
                const Node& sub_out = mn.body_graph->node(mn.body_out_id);
                const Eigen::Index m_slice = sub_out.size();

                // Map each inner variable scalar slot u in [0, n_sub_vars) to (input_idx, elem_in_slice)
                std::vector<std::pair<std::size_t, Eigen::Index>> inner_slot_map(
                    static_cast<std::size_t>(n_sub_vars));
                for (std::size_t k = 0; k < mn.inner_var_ids.size(); ++k) {
                    const Node& in_v = mn.body_graph->node(mn.inner_var_ids[k]);
                    for (Eigen::Index e = 0; e < in_v.size(); ++e) {
                        inner_slot_map[static_cast<std::size_t>(in_v.offset + e)] = {k, e};
                    }
                }

                for (int b = 0; b < mn.num_batches; ++b) {
                    auto outer_slot_for_inner = [&](std::size_t u) -> Eigen::Index {
                        const auto& pr = inner_slot_map[u];
                        const Node& outer_in = g.node(mn.inputs[pr.first]);
                        const Node& inner_v = mn.body_graph->node(mn.inner_var_ids[pr.first]);
                        return outer_slice_elem_offset(outer_in, mn.in_axes[pr.first], b, inner_v.shape, pr.second);
                    };

                    for (Eigen::Index idx = 0; idx < m_slice; ++idx) {
                        const Eigen::Index out_slot =
                            (mn.out_axis == 0)
                                ? (nd.offset + static_cast<Eigen::Index>(b) + idx * nd.shape.rows)
                                : (nd.offset + idx + static_cast<Eigen::Index>(b) * nd.shape.rows);
                        std::uint64_t* dst = deps_of(out_slot);
                        const std::uint64_t* s_dep =
                            &sub_deps[static_cast<std::size_t>(sub_out.offset + idx) * n_sub_words];

                        for (std::size_t w = 0; w < n_sub_words; ++w) {
                            std::uint64_t word = s_dep[w];
                            while (word != 0ULL) {
                                const int bit = ctz64(word);
                                const std::size_t u = (w << 6) + static_cast<std::size_t>(bit);
                                const Eigen::Index o_slot = outer_slot_for_inner(u);
                                or_bits(dst, deps_of(o_slot), n_words);
                                word &= (word - 1ULL);
                            }
                        }
                    }

                    if (compute_hessian && n_sub_vars > 0) {
                        for (Eigen::Index u = 0; u < n_sub_vars; ++u) {
                            const Eigen::Index o_u = outer_slot_for_inner(static_cast<std::size_t>(u));
                            const std::uint64_t* h_row =
                                &sub_hess[static_cast<std::size_t>(u) * n_sub_words];
                            for (std::size_t w = 0; w < n_sub_words; ++w) {
                                std::uint64_t word = h_row[w];
                                while (word != 0ULL) {
                                    const int bit = ctz64(word);
                                    const Eigen::Index v = static_cast<Eigen::Index>((w << 6) + bit);
                                    if (v <= u) {
                                        const Eigen::Index o_v = outer_slot_for_inner(static_cast<std::size_t>(v));
                                        if (u == v) {
                                            add_clique(hess_bits, deps_of(o_u), n_words);
                                        } else {
                                            add_bipartite(hess_bits, deps_of(o_u), deps_of(o_v), n_words);
                                        }
                                    }
                                    word &= (word - 1ULL);
                                }
                            }
                        }
                    }
                }
                break;
            }

            case OpType::Scan: {
                const auto& sn = static_cast<const ScanNode&>(nd);
                std::vector<std::uint64_t> sub_deps;
                std::vector<std::uint64_t> sub_hess;
                propagate_graph_sparsity(*sn.step_graph, sub_deps, sub_hess, compute_hessian);

                const Eigen::Index n_sub_vars = sn.step_graph->total_var_size();
                const std::size_t n_sub_words = static_cast<std::size_t>((n_sub_vars + 63) / 64);

                const Node& outer_c0 = g.node(sn.inputs[0]);
                const Node& inner_c = sn.step_graph->node(sn.inner_carry_id);
                const Node& next_c = sn.step_graph->node(sn.next_carry_out_id);
                const Eigen::Index n_c = inner_c.size();
                const Eigen::Index n_y =
                    (sn.step_y_out_id >= 0) ? sn.step_graph->node(sn.step_y_out_id).size() : 0;
                const std::size_t n_seq = sn.inner_seq_ids.size();
                const std::size_t n_shared = sn.inner_shared_ids.size();

                // Step-level dependency buffer for all inner variables [0 .. n_sub_vars)
                std::vector<std::uint64_t> step_in_deps(
                    static_cast<std::size_t>(n_sub_vars) * n_words, 0ULL);
                std::vector<std::uint64_t> next_carry_deps(
                    static_cast<std::size_t>(n_c) * n_words, 0ULL);

                // Initialize carry dependencies from outer_c0
                for (Eigen::Index i = 0; i < n_c; ++i) {
                    or_bits(&step_in_deps[static_cast<std::size_t>(inner_c.offset + i) * n_words],
                            deps_of(outer_c0.offset + i),
                            n_words);
                }
                // Initialize shared input dependencies (constant across steps)
                for (std::size_t j = 0; j < n_shared; ++j) {
                    const Node& outer_sh = g.node(sn.inputs[1 + n_seq + j]);
                    const Node& inner_sh = sn.step_graph->node(sn.inner_shared_ids[j]);
                    if (outer_sh.requires_grad) {
                        for (Eigen::Index e = 0; e < inner_sh.size(); ++e) {
                            or_bits(&step_in_deps[static_cast<std::size_t>(inner_sh.offset + e) * n_words],
                                    deps_of(outer_sh.offset + e),
                                    n_words);
                        }
                    }
                }

                for (int t = 0; t < sn.num_steps; ++t) {
                    // Populate sequence slice dependencies for step t
                    for (std::size_t k = 0; k < n_seq; ++k) {
                        const Node& outer_seq = g.node(sn.inputs[1 + k]);
                        const Node& inner_seq = sn.step_graph->node(sn.inner_seq_ids[k]);
                        for (Eigen::Index e = 0; e < inner_seq.size(); ++e) {
                            std::uint64_t* dst =
                                &step_in_deps[static_cast<std::size_t>(inner_seq.offset + e) * n_words];
                            std::fill(dst, dst + n_words, 0ULL);
                            if (outer_seq.requires_grad) {
                                const Eigen::Index o_slot = outer_slice_elem_offset(
                                    outer_seq, sn.seq_axes[k], t, inner_seq.shape, e);
                                or_bits(dst, deps_of(o_slot), n_words);
                            }
                        }
                    }

                    // Propagate step Hessian couplings using current step_in_deps
                    if (compute_hessian && n_sub_vars > 0) {
                        for (Eigen::Index u = 0; u < n_sub_vars; ++u) {
                            const std::uint64_t* du =
                                &step_in_deps[static_cast<std::size_t>(u) * n_words];
                            const std::uint64_t* h_row =
                                &sub_hess[static_cast<std::size_t>(u) * n_sub_words];
                            for (std::size_t w = 0; w < n_sub_words; ++w) {
                                std::uint64_t word = h_row[w];
                                while (word != 0ULL) {
                                    const int bit = ctz64(word);
                                    const Eigen::Index v = static_cast<Eigen::Index>((w << 6) + bit);
                                    if (v <= u) {
                                        const std::uint64_t* dv =
                                            &step_in_deps[static_cast<std::size_t>(v) * n_words];
                                        if (u == v) {
                                            add_clique(hess_bits, du, n_words);
                                        } else {
                                            add_bipartite(hess_bits, du, dv, n_words);
                                        }
                                    }
                                    word &= (word - 1ULL);
                                }
                            }
                        }
                    }

                    // Compute step_y dependencies for step t
                    if (sn.step_y_out_id >= 0 && n_y > 0) {
                        const Node& y_n = sn.step_graph->node(sn.step_y_out_id);
                        for (Eigen::Index idx = 0; idx < n_y; ++idx) {
                            std::uint64_t* dst =
                                deps_of(nd.offset + n_c + static_cast<Eigen::Index>(t) * n_y + idx);
                            const std::uint64_t* s_dep =
                                &sub_deps[static_cast<std::size_t>(y_n.offset + idx) * n_sub_words];
                            for (std::size_t w = 0; w < n_sub_words; ++w) {
                                std::uint64_t word = s_dep[w];
                                while (word != 0ULL) {
                                    const int bit = ctz64(word);
                                    const std::size_t u = (w << 6) + static_cast<std::size_t>(bit);
                                    or_bits(dst, &step_in_deps[u * n_words], n_words);
                                    word &= (word - 1ULL);
                                }
                            }
                        }
                    }

                    // Compute next_carry dependencies
                    std::fill(next_carry_deps.begin(), next_carry_deps.end(), 0ULL);
                    for (Eigen::Index idx = 0; idx < n_c; ++idx) {
                        std::uint64_t* dst =
                            &next_carry_deps[static_cast<std::size_t>(idx) * n_words];
                        const std::uint64_t* s_dep =
                            &sub_deps[static_cast<std::size_t>(next_c.offset + idx) * n_sub_words];
                        for (std::size_t w = 0; w < n_sub_words; ++w) {
                            std::uint64_t word = s_dep[w];
                            while (word != 0ULL) {
                                const int bit = ctz64(word);
                                const std::size_t u = (w << 6) + static_cast<std::size_t>(bit);
                                or_bits(dst, &step_in_deps[u * n_words], n_words);
                                word &= (word - 1ULL);
                            }
                        }
                    }

                    // Advance carry dependencies to next step
                    for (Eigen::Index idx = 0; idx < n_c; ++idx) {
                        std::copy(
                            &next_carry_deps[static_cast<std::size_t>(idx) * n_words],
                            &next_carry_deps[static_cast<std::size_t>(idx + 1) * n_words],
                            &step_in_deps[static_cast<std::size_t>(inner_c.offset + idx) * n_words]);
                    }
                }

                // Store final_carry dependencies into first n_c output slots
                for (Eigen::Index idx = 0; idx < n_c; ++idx) {
                    or_bits(deps_of(nd.offset + idx),
                            &step_in_deps[static_cast<std::size_t>(inner_c.offset + idx) * n_words],
                            n_words);
                }
                break;
            }

            case OpType::BlackBox: {
                const auto& bb = static_cast<const BlackBoxNode&>(nd);
                Eigen::Index total_in = 0;
                for (int in_id : nd.inputs) {
                    total_in += g.node(in_id).size();
                }
                if (bb.jac_sparsity_mask.rows() == nd.size() && bb.jac_sparsity_mask.cols() == total_in) {
                    std::vector<Eigen::Index> in_slot_map;
                    in_slot_map.reserve(static_cast<std::size_t>(total_in));
                    std::vector<char> in_grad_map;
                    in_grad_map.reserve(static_cast<std::size_t>(total_in));
                    for (int in_id : nd.inputs) {
                        const Node& in = g.node(in_id);
                        for (Eigen::Index k = 0; k < in.size(); ++k) {
                            in_slot_map.push_back(in.offset + k);
                            in_grad_map.push_back(in.requires_grad ? 1 : 0);
                        }
                    }
                    for (Eigen::Index r = 0; r < nd.size(); ++r) {
                        std::uint64_t* dst = deps_of(nd.offset + r);
                        for (Eigen::Index c = 0; c < total_in; ++c) {
                            if (in_grad_map[static_cast<std::size_t>(c)] && bb.jac_sparsity_mask(r, c) != 0) {
                                or_bits(dst, deps_of(in_slot_map[static_cast<std::size_t>(c)]), n_words);
                            }
                        }
                        if (compute_hessian) {
                            add_clique(hess_bits, dst, n_words);
                        }
                    }
                } else {
                    std::fill(scratch.begin(), scratch.end(), 0ULL);
                    for (int in_id : nd.inputs) {
                        const Node& in = g.node(in_id);
                        if (!in.requires_grad) continue;
                        for (Eigen::Index i = 0; i < in.size(); ++i) {
                            or_bits(scratch.data(), deps_of(in.offset + i), n_words);
                        }
                    }
                    for (Eigen::Index i = 0; i < nd.size(); ++i) {
                        or_bits(deps_of(nd.offset + i), scratch.data(), n_words);
                    }
                    if (compute_hessian) {
                        add_clique(hess_bits, scratch.data(), n_words);
                    }
                }
                break;
            }

            case OpType::Cond:
            case OpType::ImplicitSolver:
            default: {
                std::fill(scratch.begin(), scratch.end(), 0ULL);
                for (int in_id : nd.inputs) {
                    const Node& in = g.node(in_id);
                    if (!in.requires_grad) continue;
                    for (Eigen::Index i = 0; i < in.size(); ++i) {
                        or_bits(scratch.data(), deps_of(in.offset + i), n_words);
                    }
                }
                for (Eigen::Index i = 0; i < nd.size(); ++i) {
                    or_bits(deps_of(nd.offset + i), scratch.data(), n_words);
                }
                if (compute_hessian) {
                    add_clique(hess_bits, scratch.data(), n_words);
                }
                break;
            }
        }
    }
}

} // namespace

// ============================================================================
// Evaluator Implementation
// ============================================================================
Evaluator::Evaluator(std::shared_ptr<Graph> graph, std::vector<int> output_ids)
    : graph_(std::move(graph)), output_ids_(std::move(output_ids)) {
    if (!graph_) {
        throw std::invalid_argument("Evaluator received null Graph pointer.");
    }
    graph_->compile(output_ids_);
    ensure_static_synced();

    has_blackbox_ = graph_->has_unsafe_blackbox();
}

void Evaluator::ensure_sparsity_plans() {
    if (!sparsity_ready_) {
        build_sparsity_and_coloring_plans();
        sparsity_ready_ = true;
    }
}

void Evaluator::build_sparsity_and_coloring_plans() {
    const Eigen::Index n = graph_->total_var_size();
    const std::size_t n_words = static_cast<std::size_t>((n + 63) / 64);
    const std::size_t n_rows_words_hess = n_words;

    std::vector<std::uint64_t> deps;
    std::vector<std::uint64_t> hess_bits;
    propagate_graph_sparsity(*graph_, deps, hess_bits, true);

    // 1. Build JacobianSparsityPlan for each registered output
    jac_plans_.clear();
    jac_plans_.resize(output_ids_.size());
    for (std::size_t out_idx = 0; out_idx < output_ids_.size(); ++out_idx) {
        const Node& out_n = graph_->node(output_ids_[out_idx]);
        const Eigen::Index m = out_n.size();
        JacobianSparsityPlan& plan = jac_plans_[out_idx];
        plan.m = m;
        plan.n = n;
        plan.row_ptr.assign(static_cast<std::size_t>(m + 1), 0);
        plan.col_nz_map.assign(static_cast<std::size_t>(n), {});

        if (m == 0 || n == 0 || !out_n.requires_grad) {
            plan.use_row_coloring = true;
            continue;
        }

        const std::size_t m_words = static_cast<std::size_t>((m + 63) / 64);
        std::vector<std::uint64_t> row_bits(
            static_cast<std::size_t>(m) * n_words, 0ULL);
        std::vector<std::uint64_t> col_bits(
            static_cast<std::size_t>(n) * m_words, 0ULL);

        std::vector<int> r_vec;
        std::vector<int> c_vec;
        for (Eigen::Index r = 0; r < m; ++r) {
            plan.row_ptr[static_cast<std::size_t>(r)] = static_cast<int>(r_vec.size());
            const std::uint64_t* r_dep =
                &deps[static_cast<std::size_t>(out_n.offset + r) * n_words];
            std::copy(r_dep, r_dep + n_words,
                      &row_bits[static_cast<std::size_t>(r) * n_words]);

            for (std::size_t w = 0; w < n_words; ++w) {
                std::uint64_t word = r_dep[w];
                while (word != 0ULL) {
                    const int bit = ctz64(word);
                    const int c = static_cast<int>((w << 6) + bit);
                    const int nz_idx = static_cast<int>(r_vec.size());
                    r_vec.push_back(static_cast<int>(r));
                    c_vec.push_back(c);
                    plan.col_nz_map[static_cast<std::size_t>(c)].emplace_back(static_cast<int>(r), nz_idx);
                    col_bits[static_cast<std::size_t>(c) * m_words + (static_cast<std::size_t>(r) >> 6)] |=
                        (1ULL << (r & 63));
                    word &= (word - 1ULL);
                }
            }
        }
        plan.row_ptr[static_cast<std::size_t>(m)] = static_cast<int>(r_vec.size());

        plan.rows = Eigen::Map<const Eigen::VectorXi>(r_vec.data(), static_cast<Eigen::Index>(r_vec.size()));
        plan.cols = Eigen::Map<const Eigen::VectorXi>(c_vec.data(), static_cast<Eigen::Index>(c_vec.size()));

        plan.row_color_groups = greedy_color_bitsets(row_bits, static_cast<int>(m), n_words);
        plan.col_color_groups = greedy_color_bitsets(col_bits, static_cast<int>(n), m_words);

        plan.use_row_coloring =
            (plan.row_color_groups.size() <= plan.col_color_groups.size());
    }

    // 2. Build HessianSparsityPlan for the Lagrangian
    hess_plan_.n = n;
    hess_plan_.col_tril_map.assign(static_cast<std::size_t>(n), {});
    hess_plan_.col_full_rows.assign(static_cast<std::size_t>(n), {});
    if (n > 0) {
        std::vector<int> h_rows;
        std::vector<int> h_cols;
        for (Eigen::Index r = 0; r < n; ++r) {
            const std::uint64_t* r_bits =
                &hess_bits[static_cast<std::size_t>(r) * n_rows_words_hess];
            for (std::size_t w = 0; w < n_rows_words_hess; ++w) {
                std::uint64_t word = r_bits[w];
                while (word != 0ULL) {
                    const int bit = ctz64(word);
                    const int c = static_cast<int>((w << 6) + bit);
                    hess_plan_.col_full_rows[static_cast<std::size_t>(c)].push_back(static_cast<int>(r));
                    if (r >= c) {
                        const int nz_idx = static_cast<int>(h_rows.size());
                        h_rows.push_back(static_cast<int>(r));
                        h_cols.push_back(c);
                        hess_plan_.col_tril_map[static_cast<std::size_t>(c)].emplace_back(
                            static_cast<int>(r), nz_idx);
                    }
                    word &= (word - 1ULL);
                }
            }
        }
        hess_plan_.tril_rows =
            Eigen::Map<const Eigen::VectorXi>(h_rows.data(), static_cast<Eigen::Index>(h_rows.size()));
        hess_plan_.tril_cols =
            Eigen::Map<const Eigen::VectorXi>(h_cols.data(), static_cast<Eigen::Index>(h_cols.size()));
        hess_plan_.color_groups = greedy_color_bitsets(hess_bits, static_cast<int>(n), n_rows_words_hess);
    }
}

void Evaluator::ensure_static_synced() {
    if (synced_param_version_ != graph_->param_version()) {
        graph_->initialize_static_workspace(main_ws_);
        for (auto& tws : thread_ws_) {
            graph_->initialize_static_workspace(tws);
        }
        synced_param_version_ = graph_->param_version();
        has_valid_forward_ = false;
    }
}

void Evaluator::ensure_forward_evaluated(const Eigen::Ref<const Eigen::VectorXd>& x) {
    if (x.size() != graph_->total_var_size()) {
        throw std::invalid_argument("Decision vector x size mismatch in Evaluator.");
    }
    ensure_static_synced();
    const bool same_x = has_valid_forward_ &&
                        (last_forward_param_version_ == synced_param_version_) &&
                        (last_forward_x_.size() == x.size()) &&
                        (x.size() == 0 || (last_forward_x_.array() == x.array()).all());
    if (!same_x) {
        run_forward_dynamic(main_ws_, x);
        last_forward_x_ = x;
        last_forward_param_version_ = synced_param_version_;
        has_valid_forward_ = true;
    }
}

void Evaluator::ensure_thread_workspaces(int n_threads, bool need_tier1, bool need_tier2) {
    ensure_static_synced();
    const Eigen::Index mem_size = graph_->total_memory_size();
    if (static_cast<int>(thread_ws_.size()) < n_threads) {
        std::size_t old_size = thread_ws_.size();
        thread_ws_.resize(static_cast<std::size_t>(n_threads));
        for (std::size_t i = old_size; i < thread_ws_.size(); ++i) {
            thread_ws_[i].allocate_tier0(mem_size);
            thread_ws_[i].val = main_ws_.val;
        }
    }
    for (int i = 0; i < n_threads; ++i) {
        auto& ws = thread_ws_[static_cast<std::size_t>(i)];
        if (need_tier1) ws.allocate_tier1(mem_size);
        if (need_tier2) ws.allocate_tier2(mem_size);
    }
}

void Evaluator::run_forward_dynamic(Workspace& ws, const Eigen::Ref<const Eigen::VectorXd>& x) const {
    if (x.size() != graph_->total_var_size()) {
        throw std::invalid_argument("Decision vector x size mismatch in Evaluator.");
    }
    if (&ws == &main_ws_) {
        ++forward_eval_count_;
    }
    if (x.size() > 0) {
        ws.val.head(graph_->total_var_size()) = x;
    }
    for (int id : graph_->dynamic_exec_order()) {
        graph_->node(id).forward(ws);
    }
}

std::vector<Eigen::MatrixXd> Evaluator::eval(const Eigen::Ref<const Eigen::VectorXd>& x) {
    ensure_forward_evaluated(x);

    std::vector<Eigen::MatrixXd> results;
    results.reserve(output_ids_.size());
    for (int out_id : output_ids_) {
        const Node& n = graph_->node(out_id);
        results.emplace_back(main_ws_.val_mat(n.offset, n.shape));
    }
    return results;
}

Eigen::MatrixXd Evaluator::eval_node(const Eigen::Ref<const Eigen::VectorXd>& x, int node_id) {
    ensure_forward_evaluated(x);
    const Node& n = graph_->node(node_id);
    return main_ws_.val_mat(n.offset, n.shape);
}

std::vector<Eigen::MatrixXd> Evaluator::eval_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& X_pop, int num_threads) {
    const Eigen::Index n_pop = X_pop.rows();
    const Eigen::Index n_vars = X_pop.cols();
    if (n_vars != graph_->total_var_size()) {
        throw std::invalid_argument("eval_batch: X_pop.cols() does not match total_var_size.");
    }

    ensure_static_synced();

    bool has_batch_blackbox = false;
    for (int id : graph_->dynamic_exec_order()) {
        const Node& nd = graph_->node(id);
        if (nd.op == OpType::BlackBox) {
            const auto& bb = static_cast<const BlackBoxNode&>(nd);
            if (bb.batch_eval_fn) {
                has_batch_blackbox = true;
            }
        }
    }

    std::vector<Eigen::MatrixXd> batch_outputs(output_ids_.size());
    for (std::size_t k = 0; k < output_ids_.size(); ++k) {
        const Eigen::Index out_len = graph_->node(output_ids_[k]).size();
        batch_outputs[k].resize(n_pop, out_len);
    }
    if (n_pop == 0) {
        return batch_outputs;
    }

    if (has_batch_blackbox) {
        ensure_thread_workspaces(static_cast<int>(n_pop), false, false);
        for (Eigen::Index p = 0; p < n_pop; ++p) {
            if (n_vars > 0) {
                thread_ws_[static_cast<std::size_t>(p)].val.head(n_vars) = X_pop.row(p).transpose();
            }
        }

        for (int id : graph_->dynamic_exec_order()) {
            const Node& nd = graph_->node(id);
            if (nd.op == OpType::BlackBox) {
                const auto& bb = static_cast<const BlackBoxNode&>(nd);
                bb.forward_batch_workspaces(thread_ws_, n_pop);
            } else {
                for (Eigen::Index p = 0; p < n_pop; ++p) {
                    nd.forward(thread_ws_[static_cast<std::size_t>(p)]);
                }
            }
        }

        for (Eigen::Index p = 0; p < n_pop; ++p) {
            const Workspace& ws = thread_ws_[static_cast<std::size_t>(p)];
            for (std::size_t k = 0; k < output_ids_.size(); ++k) {
                const Node& out_n = graph_->node(output_ids_[k]);
                batch_outputs[k].row(p) = ws.val.segment(out_n.offset, out_n.size()).transpose();
            }
        }
        return batch_outputs;
    }

    int max_t = (num_threads > 0) ? num_threads : omp_get_max_threads();
    if (has_blackbox_) {
        max_t = 1;
    }
    max_t = std::max(1, std::min(max_t, static_cast<int>(n_pop)));
    ensure_thread_workspaces(max_t, false, false);

    if (max_t == 1) {
        Workspace& ws = thread_ws_[0];
        for (Eigen::Index p = 0; p < n_pop; ++p) {
            Eigen::VectorXd x_row = X_pop.row(p).transpose();
            run_forward_dynamic(ws, x_row);
            for (std::size_t k = 0; k < output_ids_.size(); ++k) {
                const Node& out_n = graph_->node(output_ids_[k]);
                batch_outputs[k].row(p) = ws.val.segment(out_n.offset, out_n.size()).transpose();
            }
        }
    } else {
        #pragma omp parallel for num_threads(max_t) schedule(static)
        for (Eigen::Index p = 0; p < n_pop; ++p) {
            const int tid = omp_get_thread_num();
            Workspace& ws = thread_ws_[static_cast<std::size_t>(tid)];
            Eigen::VectorXd x_row = X_pop.row(p).transpose();
            run_forward_dynamic(ws, x_row);

            for (std::size_t k = 0; k < output_ids_.size(); ++k) {
                const Node& out_n = graph_->node(output_ids_[k]);
                batch_outputs[k].row(p) = ws.val.segment(out_n.offset, out_n.size()).transpose();
            }
        }
    }

    return batch_outputs;
}

Eigen::VectorXd Evaluator::gradient(
    const Eigen::Ref<const Eigen::VectorXd>& x, std::size_t out_idx) {
    if (out_idx >= output_ids_.size()) {
        throw std::out_of_range("gradient: out_idx out of range.");
    }
    const Node& out_n = graph_->node(output_ids_[out_idx]);
    if (out_n.size() != 1) {
        throw std::invalid_argument("gradient requires a scalar output node; use jacobian() for vectors.");
    }

    ensure_forward_evaluated(x);
    main_ws_.allocate_tier1(graph_->total_memory_size());
    main_ws_.bar(out_n.offset) = 1.0;

    const auto& dyn = graph_->dynamic_exec_order();
    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        const Node& nd = graph_->node(*it);
        if (nd.requires_grad) {
            nd.backward(main_ws_);
        }
    }

    return main_ws_.bar.head(graph_->total_var_size());
}

Eigen::VectorXd Evaluator::sparse_jacobian_values(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    std::size_t out_idx,
    int num_threads) {
    ensure_sparsity_plans();
    if (out_idx >= output_ids_.size()) {
        throw std::out_of_range("sparse_jacobian_values: out_idx out of range.");
    }
    const JacobianSparsityPlan& plan = jac_plans_[out_idx];
    const Node& out_n = graph_->node(output_ids_[out_idx]);
    Eigen::VectorXd vals = Eigen::VectorXd::Zero(plan.nnz());
    if (plan.nnz() == 0 || !out_n.requires_grad) {
        return vals;
    }

    ensure_forward_evaluated(x);
    const auto& dyn = graph_->dynamic_exec_order();
    const int n_colors = plan.num_colors();
    int max_t = (num_threads > 0) ? num_threads : omp_get_max_threads();
    if (has_blackbox_) {
        max_t = 1;
    }
    max_t = std::max(1, std::min(max_t, n_colors));

    if (plan.use_row_coloring) {
        if (max_t == 1) {
            main_ws_.allocate_tier1(graph_->total_memory_size());
            for (const auto& group : plan.row_color_groups) {
                main_ws_.bar.setZero();
                for (int r : group) {
                    main_ws_.bar(out_n.offset + r) = 1.0;
                }
                for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
                    const Node& nd = graph_->node(*it);
                    if (nd.requires_grad) {
                        nd.backward(main_ws_);
                    }
                }
                for (int r : group) {
                    const int start = plan.row_ptr[static_cast<std::size_t>(r)];
                    const int stop = plan.row_ptr[static_cast<std::size_t>(r + 1)];
                    for (int k = start; k < stop; ++k) {
                        vals(k) = main_ws_.bar(plan.cols(k));
                    }
                }
            }
        } else {
            ensure_thread_workspaces(max_t, true, false);
            for (int t = 0; t < max_t; ++t) {
                thread_ws_[static_cast<std::size_t>(t)].val = main_ws_.val;
            }
            #pragma omp parallel for num_threads(max_t) schedule(static)
            for (int g_idx = 0; g_idx < n_colors; ++g_idx) {
                const int tid = omp_get_thread_num();
                Workspace& tws = thread_ws_[static_cast<std::size_t>(tid)];
                const auto& group = plan.row_color_groups[static_cast<std::size_t>(g_idx)];
                tws.bar.setZero();
                for (int r : group) {
                    tws.bar(out_n.offset + r) = 1.0;
                }
                for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
                    const Node& nd = graph_->node(*it);
                    if (nd.requires_grad) {
                        nd.backward(tws);
                    }
                }
                for (int r : group) {
                    const int start = plan.row_ptr[static_cast<std::size_t>(r)];
                    const int stop = plan.row_ptr[static_cast<std::size_t>(r + 1)];
                    for (int k = start; k < stop; ++k) {
                        vals(k) = tws.bar(plan.cols(k));
                    }
                }
            }
        }
    } else {
        if (max_t == 1) {
            main_ws_.allocate_tier2(graph_->total_memory_size());
            for (const auto& group : plan.col_color_groups) {
                main_ws_.dot.setZero();
                for (int c : group) {
                    main_ws_.dot(c) = 1.0;
                }
                for (int id : dyn) {
                    const Node& nd = graph_->node(id);
                    if (nd.requires_grad) {
                        nd.forward_tangent(main_ws_);
                    }
                }
                for (int c : group) {
                    for (const auto& pr : plan.col_nz_map[static_cast<std::size_t>(c)]) {
                        vals(pr.second) = main_ws_.dot(out_n.offset + pr.first);
                    }
                }
            }
        } else {
            ensure_thread_workspaces(max_t, false, true);
            for (int t = 0; t < max_t; ++t) {
                thread_ws_[static_cast<std::size_t>(t)].val = main_ws_.val;
            }
            #pragma omp parallel for num_threads(max_t) schedule(static)
            for (int g_idx = 0; g_idx < n_colors; ++g_idx) {
                const int tid = omp_get_thread_num();
                Workspace& tws = thread_ws_[static_cast<std::size_t>(tid)];
                const auto& group = plan.col_color_groups[static_cast<std::size_t>(g_idx)];
                tws.dot.setZero();
                for (int c : group) {
                    tws.dot(c) = 1.0;
                }
                for (int id : dyn) {
                    const Node& nd = graph_->node(id);
                    if (nd.requires_grad) {
                        nd.forward_tangent(tws);
                    }
                }
                for (int c : group) {
                    for (const auto& pr : plan.col_nz_map[static_cast<std::size_t>(c)]) {
                        vals(pr.second) = tws.dot(out_n.offset + pr.first);
                    }
                }
            }
        }
    }

    return vals;
}

Eigen::MatrixXd Evaluator::jacobian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    std::size_t out_idx,
    int num_threads) {
    ensure_sparsity_plans();
    if (out_idx >= output_ids_.size()) {
        throw std::out_of_range("jacobian: out_idx out of range.");
    }
    const JacobianSparsityPlan& plan = jac_plans_[out_idx];
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(plan.m, plan.n);
    if (plan.nnz() == 0) {
        return J;
    }

    Eigen::VectorXd vals = sparse_jacobian_values(x, out_idx, num_threads);
    for (Eigen::Index k = 0; k < plan.nnz(); ++k) {
        J(plan.rows(k), plan.cols(k)) = vals(k);
    }
    return J;
}

Eigen::VectorXd Evaluator::hessian_vec_prod(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double obj_scale,
    const std::vector<Eigen::VectorXd>& constraint_multipliers,
    const Eigen::Ref<const Eigen::VectorXd>& v) {
    const Eigen::Index n = graph_->total_var_size();
    if (v.size() != n) {
        throw std::invalid_argument("hessian_vec_prod: perturbation vector v size mismatch.");
    }

    ensure_forward_evaluated(x);

    const Eigen::Index mem_size = graph_->total_memory_size();
    main_ws_.allocate_tier1(mem_size);
    main_ws_.allocate_tier2(mem_size);

    if (!output_ids_.empty()) {
        const Node& obj_n = graph_->node(output_ids_[0]);
        if (obj_n.size() != 1) {
            throw std::invalid_argument("hessian_vec_prod: output[0] (objective) must be scalar.");
        }
        main_ws_.bar(obj_n.offset) += obj_scale;
    }
    for (std::size_t k = 0; k < constraint_multipliers.size() && (k + 1) < output_ids_.size(); ++k) {
        const Node& c_n = graph_->node(output_ids_[k + 1]);
        if (constraint_multipliers[k].size() != c_n.size()) {
            throw std::invalid_argument("hessian_vec_prod: multiplier vector size mismatch.");
        }
        main_ws_.bar.segment(c_n.offset, c_n.size()) += constraint_multipliers[k];
    }

    const auto& dyn = graph_->dynamic_exec_order();
    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        const Node& nd = graph_->node(*it);
        if (nd.requires_grad) {
            nd.backward(main_ws_);
        }
    }

    main_ws_.dot.head(n) = v;
    for (int id : dyn) {
        const Node& nd = graph_->node(id);
        if (nd.requires_grad) {
            nd.forward_tangent(main_ws_);
        }
    }

    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        const Node& nd = graph_->node(*it);
        if (nd.requires_grad) {
            nd.backward_hessian(main_ws_);
        }
    }

    return main_ws_.dot_bar.head(n);
}

Eigen::VectorXd Evaluator::sparse_hessian_values(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double obj_scale,
    const std::vector<Eigen::VectorXd>& constraint_multipliers,
    int num_threads) {
    ensure_sparsity_plans();
    const Eigen::Index n = graph_->total_var_size();
    Eigen::VectorXd vals = Eigen::VectorXd::Zero(hess_plan_.nnz_tril());
    if (n == 0 || hess_plan_.nnz_tril() == 0 || hess_plan_.color_groups.empty()) {
        return vals;
    }

    ensure_forward_evaluated(x);

    const Eigen::Index mem_size = graph_->total_memory_size();
    main_ws_.allocate_tier1(mem_size);

    if (!output_ids_.empty()) {
        const Node& obj_n = graph_->node(output_ids_[0]);
        if (obj_n.size() != 1) {
            throw std::invalid_argument("sparse_hessian_values: output[0] must be scalar.");
        }
        main_ws_.bar(obj_n.offset) += obj_scale;
    }
    for (std::size_t k = 0; k < constraint_multipliers.size() && (k + 1) < output_ids_.size(); ++k) {
        const Node& c_n = graph_->node(output_ids_[k + 1]);
        if (constraint_multipliers[k].size() != c_n.size()) {
            throw std::invalid_argument("sparse_hessian_values: multiplier vector size mismatch.");
        }
        main_ws_.bar.segment(c_n.offset, c_n.size()) += constraint_multipliers[k];
    }

    const auto& dyn = graph_->dynamic_exec_order();
    for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
        const Node& nd = graph_->node(*it);
        if (nd.requires_grad) {
            nd.backward(main_ws_);
        }
    }

    const int n_colors = static_cast<int>(hess_plan_.color_groups.size());
    int max_t = (num_threads > 0) ? num_threads : omp_get_max_threads();
    if (has_blackbox_) {
        max_t = 1;
    }
    max_t = std::max(1, std::min(max_t, n_colors));
    ensure_thread_workspaces(max_t, true, true);

    for (int t = 0; t < max_t; ++t) {
        Workspace& tws = thread_ws_[static_cast<std::size_t>(t)];
        tws.val = main_ws_.val;
        tws.bar = main_ws_.bar;
    }

    if (max_t == 1) {
        Workspace& tws = thread_ws_[0];
        for (int g_idx = 0; g_idx < n_colors; ++g_idx) {
            tws.dot.setZero();
            tws.dot_bar.setZero();
            for (int c : hess_plan_.color_groups[static_cast<std::size_t>(g_idx)]) {
                tws.dot(c) = 1.0;
            }
            for (int id : dyn) {
                const Node& nd = graph_->node(id);
                if (nd.requires_grad) {
                    nd.forward_tangent(tws);
                }
            }
            for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
                const Node& nd = graph_->node(*it);
                if (nd.requires_grad) {
                    nd.backward_hessian(tws);
                }
            }
            for (int c : hess_plan_.color_groups[static_cast<std::size_t>(g_idx)]) {
                for (const auto& pr : hess_plan_.col_tril_map[static_cast<std::size_t>(c)]) {
                    vals(pr.second) = tws.dot_bar(pr.first);
                }
            }
        }
    } else {
        #pragma omp parallel for num_threads(max_t) schedule(static)
        for (int g_idx = 0; g_idx < n_colors; ++g_idx) {
            const int tid = omp_get_thread_num();
            Workspace& tws = thread_ws_[static_cast<std::size_t>(tid)];
            tws.dot.setZero();
            tws.dot_bar.setZero();
            for (int c : hess_plan_.color_groups[static_cast<std::size_t>(g_idx)]) {
                tws.dot(c) = 1.0;
            }
            for (int id : dyn) {
                const Node& nd = graph_->node(id);
                if (nd.requires_grad) {
                    nd.forward_tangent(tws);
                }
            }
            for (auto it = dyn.rbegin(); it != dyn.rend(); ++it) {
                const Node& nd = graph_->node(*it);
                if (nd.requires_grad) {
                    nd.backward_hessian(tws);
                }
            }
            for (int c : hess_plan_.color_groups[static_cast<std::size_t>(g_idx)]) {
                for (const auto& pr : hess_plan_.col_tril_map[static_cast<std::size_t>(c)]) {
                    vals(pr.second) = tws.dot_bar(pr.first);
                }
            }
        }
    }

    return vals;
}

Eigen::MatrixXd Evaluator::lagrangian_hessian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double obj_scale,
    const std::vector<Eigen::VectorXd>& constraint_multipliers,
    int num_threads) {
    ensure_sparsity_plans();
    const Eigen::Index n = graph_->total_var_size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
    if (n == 0 || hess_plan_.nnz_tril() == 0) return H;

    Eigen::VectorXd vals = sparse_hessian_values(x, obj_scale, constraint_multipliers, num_threads);
    for (Eigen::Index k = 0; k < hess_plan_.nnz_tril(); ++k) {
        const Eigen::Index r = hess_plan_.tril_rows(k);
        const Eigen::Index c = hess_plan_.tril_cols(k);
        const double v = vals(k);
        H(r, c) = v;
        if (r != c) {
            H(c, r) = v;
        }
    }
    return H;
}

Eigen::MatrixXd Evaluator::hessian(
    const Eigen::Ref<const Eigen::VectorXd>& x, std::size_t out_idx, int num_threads) {
    if (out_idx >= output_ids_.size()) {
        throw std::out_of_range("hessian: out_idx out of range.");
    }
    if (out_idx == 0) {
        return lagrangian_hessian(x, 1.0, {}, num_threads);
    }
    std::vector<Eigen::VectorXd> mults;
    mults.reserve(out_idx);
    for (std::size_t k = 1; k < out_idx; ++k) {
        mults.push_back(Eigen::VectorXd::Zero(output_size(k)));
    }
    mults.push_back(Eigen::VectorXd::Ones(output_size(out_idx)));
    return lagrangian_hessian(x, 0.0, mults, num_threads);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
Evaluator::parameter_sensitivities(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const std::vector<int>& param_ids,
    double obj_scale,
    const std::vector<Eigen::VectorXd>& constraint_multipliers) {
    ensure_forward_evaluated(x);

    const Eigen::Index n_vars = graph_->total_var_size();
    const std::size_t n_nodes = graph_->num_nodes();

    Eigen::Index n_p = 0;
    for (int pid : param_ids) {
        const Node& pn = graph_->node(pid);
        if (pn.op != OpType::Parameter) {
            throw std::invalid_argument("parameter_sensitivities: all param_ids must be Parameter nodes.");
        }
        n_p += pn.size();
    }

    Eigen::VectorXd dL_dp = Eigen::VectorXd::Zero(n_p);
    Eigen::MatrixXd d2L_dxp = Eigen::MatrixXd::Zero(n_vars, n_p);
    std::vector<Eigen::MatrixXd> dout_dp(output_ids_.size());
    for (std::size_t k = 0; k < output_ids_.size(); ++k) {
        dout_dp[k] = Eigen::MatrixXd::Zero(output_size(k), n_p);
    }
    if (n_p == 0) {
        return {dL_dp, d2L_dxp, dout_dp};
    }

    // RAII guard to temporarily enable requires_grad on target Parameter nodes and their downstream ops
    std::vector<char> orig_grad(n_nodes, 0);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        orig_grad[i] = graph_->node(static_cast<int>(i)).requires_grad ? 1 : 0;
    }
    struct GradRestoreGuard {
        Graph& g;
        const std::vector<char>& saved;
        ~GradRestoreGuard() {
            for (std::size_t i = 0; i < saved.size(); ++i) {
                g.node_mut(static_cast<int>(i)).requires_grad = (saved[i] != 0);
            }
        }
    } guard{*graph_, orig_grad};

    for (int pid : param_ids) {
        graph_->node_mut(pid).requires_grad = true;
    }
    const auto& all_ord = graph_->all_exec_order();
    for (int id : all_ord) {
        Node& nd = graph_->node_mut(id);
        for (int in_id : nd.inputs) {
            if (graph_->node(in_id).requires_grad) {
                nd.requires_grad = true;
                break;
            }
        }
    }

    const Eigen::Index mem_size = graph_->total_memory_size();
    main_ws_.allocate_tier1(mem_size);
    main_ws_.allocate_tier2(mem_size);

    if (!output_ids_.empty()) {
        const Node& obj_n = graph_->node(output_ids_[0]);
        main_ws_.bar(obj_n.offset) += obj_scale;
    }
    for (std::size_t k = 0; k < constraint_multipliers.size() && (k + 1) < output_ids_.size(); ++k) {
        const Node& c_n = graph_->node(output_ids_[k + 1]);
        if (constraint_multipliers[k].size() != c_n.size()) {
            throw std::invalid_argument("parameter_sensitivities: multiplier vector size mismatch.");
        }
        main_ws_.bar.segment(c_n.offset, c_n.size()) += constraint_multipliers[k];
    }

    // 1. Reverse pass over all_exec_order_ gives exact dL/dp (Envelope Theorem) and populates bar
    for (auto it = all_ord.rbegin(); it != all_ord.rend(); ++it) {
        const Node& nd = graph_->node(*it);
        if (nd.requires_grad) {
            nd.backward(main_ws_);
        }
    }

    Eigen::Index p_cursor = 0;
    for (int pid : param_ids) {
        const Node& pn = graph_->node(pid);
        dL_dp.segment(p_cursor, pn.size()) = main_ws_.bar.segment(pn.offset, pn.size());
        p_cursor += pn.size();
    }

    // 2. Forward-tangent + backward-hessian pass per scalar parameter component gives dout/dp and d^2L/(dx dp)
    p_cursor = 0;
    for (int pid : param_ids) {
        const Node& pn = graph_->node(pid);
        for (Eigen::Index e = 0; e < pn.size(); ++e, ++p_cursor) {
            main_ws_.dot.setZero();
            main_ws_.dot_bar.setZero();
            main_ws_.dot(pn.offset + e) = 1.0;

            for (int id : all_ord) {
                const Node& nd = graph_->node(id);
                if (nd.requires_grad) {
                    nd.forward_tangent(main_ws_);
                }
            }

            for (std::size_t k = 0; k < output_ids_.size(); ++k) {
                const Node& out_n = graph_->node(output_ids_[k]);
                dout_dp[k].col(p_cursor) = main_ws_.dot.segment(out_n.offset, out_n.size());
            }

            for (auto it = all_ord.rbegin(); it != all_ord.rend(); ++it) {
                const Node& nd = graph_->node(*it);
                if (nd.requires_grad) {
                    nd.backward_hessian(main_ws_);
                }
            }

            if (n_vars > 0) {
                d2L_dxp.col(p_cursor) = main_ws_.dot_bar.head(n_vars);
            }
        }
    }

    return {dL_dp, d2L_dxp, dout_dp};
}

} // namespace fastopt
