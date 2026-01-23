import copy
import numpy as np
import random
import torch
import os

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    check_signal,
    ignore_env, get_arch_major
)

from generators import (
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_m_grouped_masked_transpose, enumerate_k_grouped_contiguous, enumerate_m_grouped_masked_transpose_n_group,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_m_grouped_masked_2d1d, generate_m_grouped_masked_2d1d_transpose, generate_k_grouped_contiguous, generate_m_grouped_masked_2d1d_n_group, generate_m_grouped_masked_2d1d_transpose_n_group
)


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 9)
def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe = (1, 1, 128) if kernel_type.is_1d1d() and accumulate else None

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
            func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                  f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
        t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe),
                         'fp8_gemm', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
        if cublas_t > 0:
            scores.append((cublas_t + split_k_t) / t)
    if len(scores) > 0:
        print(f"Average speedup over cuBLASLt: {float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x\n")
    else:
        print("No valid scores to compute average speedup.\n")


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, enable_overlap, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d, signal = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)
            result = deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal)

            if enable_overlap:
                print("Checking signal")
                block_m, threshold = result
                check_signal(num_groups, max_m, block_m, threshold, signal, masked_m)

            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d, signal = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}, enable_overlap={enable_overlap}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_masked_2d1d() -> None:
    print('Testing m-grouped masked 2d1d GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked_transpose(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked_2d1d(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_tn_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked_2d1d(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_tn_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_masked_2d1d_transpose() -> None:
    print('Testing m-grouped masked 2d1d transpose GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, num_groups, max_n, expected_m_per_group, n, k in enumerate_m_grouped_masked_transpose(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_n, d, ref_d, signal = generate_m_grouped_masked_2d1d_transpose(num_groups, max_n, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_tn_transpose_masked(a, b, d, masked_n, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

            for j in range(num_groups):
                if masked_n[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_n[j].item()], ref_d[j, :masked_n[j].item()])
                assert diff < 0.001, f'{max_n=}, {n=}, {k=}, {j=}, masked_n={masked_n[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_n, d, ref_d = generate_m_grouped_masked_2d1d_transpose(num_groups, max_n, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_tn_transpose_masked(a, b, d, masked_n, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_n = masked_n.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_n * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_n / (max_n * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_masked_2d1d_n_group() -> None:
    print('Testing m-grouped masked 2d1d n-group GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, num_groups, max_n, m, expected_n_per_group, k in enumerate_m_grouped_masked_transpose_n_group(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_n, d, ref_d = generate_m_grouped_masked_2d1d_n_group(num_groups, max_n, m, expected_n_per_group, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_tn_n_group_masked(a, b, d, masked_n, expected_n_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
            for j in range(num_groups):
                if masked_n[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :, :masked_n[j].item()], ref_d[j, :, :masked_n[j].item()])
                assert diff < 0.001, f'{max_n=}, {m=}, {k=}, {j=}, masked_n={masked_n[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_n, d, ref_d = generate_m_grouped_masked_2d1d_n_group(num_groups, max_n, m, expected_n_per_group, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_tn_n_group_masked(a, b, d, masked_n, expected_n_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_n = masked_n.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_n_per_group={expected_n_per_group:4}, m={m:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * valid_n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_n / (max_n * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_masked_2d1d_transpose_n_group() -> None:
    print('Testing m-grouped masked 2d1d transpose n-group GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, enable_overlap, num_groups, max_n, m, expected_n_per_group, k in enumerate_m_grouped_masked_transpose_n_group(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_n, d, ref_d, signal = generate_m_grouped_masked_2d1d_transpose_n_group(num_groups, max_n, m, expected_n_per_group, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)
            result = deep_gemm.m_grouped_fp8_gemm_tn_transpose_n_group_masked(a, b, d, masked_n, expected_n_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal)
            
            if enable_overlap:
                print("Checking signal")
                block_m, threshold = result
                check_signal(num_groups, max_n, block_m, threshold, signal, masked_n)

            for j in range(num_groups):
                if masked_n[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_n[j].item()], ref_d[j, :masked_n[j].item()])
                assert diff < 0.001, f'{max_n=}, {m=}, {k=}, {j=}, masked_n={masked_n[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_n, d, ref_d, signal = generate_m_grouped_masked_2d1d_transpose_n_group(num_groups, max_n, m, expected_n_per_group, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_tn_transpose_n_group_masked(a, b, d, masked_n, expected_n_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal)

        # Test performance with fixed shapes
        valid_n = masked_n.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_n_per_group={expected_n_per_group:4}, m={m:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * valid_n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_n / (max_n * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()

def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    k_grouped_fp8_gemm_contiguous = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if get_arch_major() == 9 \
                                    else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    for num_groups, m, n, major_a, major_b, ks, expected_k_per_group in enumerate_k_grouped_contiguous(torch.float8_e4m3fn):
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups and len(ks) > 1:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, new_ks, use_ue8m0=use_ue8m0)
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device='cuda')
            k_grouped_fp8_gemm_contiguous(a, b, d, new_ks, new_ks_tensor, c)

            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {ks=}, {diff:.5f}'

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, ks, use_ue8m0=use_ue8m0)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            k_grouped_fp8_gemm_contiguous(a, b, d, ks, ks_tensor, c)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    print('Origin DeepGEMM Optimization Config:')
    os.environ['GPS_BLOCK_M'] = str(0)
    os.environ['GPS_BLOCK_N'] = str(0)
    # test_gemm()
    # os.environ['GPS_IGNORE_STAGES_LIMIT'] = str(1)
    # test_m_grouped_gemm_masked()
    # os.environ['GPS_USE_TRANSPOSE'] = str(1)
    # test_m_grouped_gemm_masked_2d1d()
    # os.environ['GPS_USE_TRANSPOSE'] = str(2)
    # test_m_grouped_gemm_masked_2d1d_transpose()
    # os.environ['GPS_USE_TRANSPOSE'] = str(3)
    # test_m_grouped_gemm_masked_2d1d_n_group()
    os.environ['GPS_USE_TRANSPOSE'] = str(4)
    test_m_grouped_gemm_masked_2d1d_transpose_n_group()
    # print('\n' + '='*50)
    # print('Testing different BLOCK_M and BLOCK_N configurations:')
    # # 不能用的配置(64, 152)
    # blockms = [64, 128, 256]
    # blockns = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
    # failed_configs = []
    # for m in blockms:
    #     for n in blockns:
    #         os.environ['GPS_BLOCK_M'] = str(m)
    #         os.environ['GPS_BLOCK_N'] = str(n)
    #         try:
    #             test_gemm()
    #             test_m_grouped_gemm_masked()
    #         except (RuntimeError, torch.cuda.CudaError) as e:
    #             # Catch CUDA errors (including kernel asserts) and continue
    #             print(f' > FAILED config (BLOCK_M={m}, BLOCK_N={n}): {e}')
    #             failed_configs.append((m, n))
    #             # Clear CUDA error state
    #             torch.cuda.synchronize()
    #             torch.cuda.empty_cache()
    
    # print('\n' + '='*50)
    # print(f'Failed configs ({len(failed_configs)}):')
    # for m, n in failed_configs:
    #     print(f'  (BLOCK_M={m}, BLOCK_N={n})')
    # os.environ['GPS_BLOCK_M'] = str(64)
    # os.environ['GPS_BLOCK_N'] = str(152)
    # test_gemm()
    # test_m_grouped_gemm_masked()
    # test_m_grouped_gemm_contiguous()
    # test_k_grouped_gemm_contiguous()
