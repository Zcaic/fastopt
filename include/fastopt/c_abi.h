#ifndef FASTOPT_C_ABI_H
#define FASTOPT_C_ABI_H

/*
 * Standalone Zero-Dependency C-ABI Header for fastopt BlackBox Direct Connection.
 *
 * This header can be included directly inside external C or C++17/C++20 libraries
 * (such as `fast_bemt` or `fastgeo`) without requiring Eigen, nanobind, or internal
 * fastopt headers.
 *
 * By populating a `FastOptCBlackBox` descriptor (either wrapped in a Python PyCapsule
 * with name `FASTOPT_C_BLACKBOX_CAPSULE_NAME` or passed via raw function/instance
 * pointers), `fastopt::BlackBoxNode` executes 0th-order forward, population batch,
 * 1st-order Jacobian, and 2nd-order HVP calls in 100% native C++ without acquiring
 * the Python GIL or allocating Python/NumPy objects.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FASTOPT_C_ABI_VERSION 1U
#define FASTOPT_C_BLACKBOX_CAPSULE_NAME "fastopt.CBlackBox"

/*
 * 0th-order single-point forward evaluation callback:
 *   - u:         [in]  Contiguous input array of length `n_in`
 *   - n_in:      [in]  Total number of scalar inputs
 *   - y:         [out] Contiguous output array of length `m_out` (column-major)
 *   - m_out:     [in]  Total number of scalar outputs
 *   - user_data: [in]  Opaque pointer to stateful C++ solver instance (e.g. fast_bemt::Solver*)
 *   Returns 0 on success, non-zero on error.
 */
typedef int (*FastOptCEvalFn)(
    const double* u,
    int32_t n_in,
    double* y,
    int32_t m_out,
    void* user_data);

/*
 * Optional 1st-order analytical Jacobian callback (e.g. IFT sensitivities from fast_bemt):
 *   - u:           [in]  Contiguous input array of length `n_in`
 *   - n_in:        [in]  Total number of scalar inputs
 *   - J_col_major: [out] Contiguous column-major Jacobian matrix of shape (m_out, n_in),
 *                        where entry (r, c) is at J_col_major[r + c * m_out]
 *   - m_out:       [in]  Total number of scalar outputs
 *   - user_data:   [in]  Opaque pointer to stateful C++ solver instance
 *   Returns 0 on success, non-zero on error.
 *   If NULL, `fastopt::BlackBoxNode` automatically computes central finite differences
 *   in native C++ using `eval` with zero GIL overhead.
 */
typedef int (*FastOptCJacFn)(
    const double* u,
    int32_t n_in,
    double* J_col_major,
    int32_t m_out,
    void* user_data);

/*
 * Optional 2nd-order Hessian-Vector Product (HVP) callback:
 *   Computes hvp_out = (d/du (J(u)^T * y_bar)) * u_dot  of length `n_in`.
 *   If NULL and `jac` is provided, `fastopt::BlackBoxNode` automatically computes
 *   directional central finite differences over the analytical `jac` in native C++!
 */
typedef int (*FastOptCHvpFn)(
    const double* u,
    const double* y_bar,
    const double* u_dot,
    int32_t n_in,
    int32_t m_out,
    double* hvp_out,
    void* user_data);

/*
 * Optional 0th-order vectorized population batch evaluation callback (e.g. fast_bemt::solve_batch):
 *   - U_col_major: [in]  Column-major matrix of shape (n_pop, n_in), where individual `p`
 *                        and input `j` is at U_col_major[p + j * n_pop]
 *   - n_pop:       [in]  Number of population members in the batch
 *   - n_in:        [in]  Total number of scalar inputs per individual
 *   - Y_col_major: [out] Column-major matrix of shape (n_pop, m_out), where individual `p`
 *                        and output `k` is at Y_col_major[p + k * n_pop]
 *   - m_out:       [in]  Total number of scalar outputs per individual
 *   - user_data:   [in]  Opaque pointer to stateful C++ solver instance
 *   Returns 0 on success, non-zero on error.
 */
typedef int (*FastOptCBatchEvalFn)(
    const double* U_col_major,
    int32_t n_pop,
    int32_t n_in,
    double* Y_col_major,
    int32_t m_out,
    void* user_data);

/*
 * Optional destructor callback invoked if the descriptor owns dynamically allocated `user_data`.
 */
typedef void (*FastOptCDestroyFn)(void* user_data);

/*
 * Binary-compatible C-ABI descriptor for zero-GIL C++ BlackBox operators.
 */
typedef struct FastOptCBlackBox {
    uint32_t abi_version;           /* Set to FASTOPT_C_ABI_VERSION (1) */
    int32_t n_in;                   /* Expected input size (or <= 0 to infer from graph) */
    int32_t m_out;                  /* Expected output size (or <= 0 to infer from shape) */
    int32_t is_thread_safe;         /* 1 if safe for concurrent OpenMP threads, 0 otherwise */
    void* user_data;                /* Pointer to C++ solver instance (e.g. fast_bemt::Solver*) */
    FastOptCEvalFn eval;            /* Required: 0th-order forward evaluation */
    FastOptCJacFn jac;              /* Optional: 1st-order analytical Jacobian (NULL -> C++ FD) */
    FastOptCHvpFn hvp;              /* Optional: 2nd-order HVP (NULL -> C++ directional FD) */
    FastOptCBatchEvalFn eval_batch; /* Optional: vectorized population batch evaluation */
    FastOptCDestroyFn destroy;      /* Optional: cleanup callback for user_data */
} FastOptCBlackBox;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FASTOPT_C_ABI_H */
