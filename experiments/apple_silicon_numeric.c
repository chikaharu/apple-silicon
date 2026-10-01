#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("=== Apple Silicon numerical experiment ===\n");

    // 1) Link-like unit-complex cumulative product
    const int N = 1000000;
    const double theta = 1e-5;
    double complex z = 1.0 + 0.0 * I;
    double complex step = cos(theta) + I * sin(theta);

    double t0 = now_sec();
    for (int k = 0; k < N; ++k) z *= step;
    double t1 = now_sec();

    double expected_phase = fmod(N * theta, 2.0 * M_PI);
    double phase = carg(z);
    if (expected_phase > M_PI) expected_phase -= 2.0 * M_PI;
    double phase_err = fabs(phase - expected_phase);
    double norm_err = fabs(cabs(z) - 1.0);

    printf("complex_steps=%d\n", N);
    printf("complex_time_sec=%.6f\n", t1 - t0);
    printf("final_phase=%.12f\n", phase);
    printf("phase_error=%.3e\n", phase_err);
    printf("unit_norm_error=%.3e\n", norm_err);

    // 2) Accelerate/BLAS matrix multiply
    const int M = 512;
    size_t n = (size_t)M * M;
    float *A = aligned_alloc(64, n * sizeof(float));
    float *B = aligned_alloc(64, n * sizeof(float));
    float *C = aligned_alloc(64, n * sizeof(float));
    if (!A || !B || !C) return 2;

    for (size_t i = 0; i < n; ++i) {
        A[i] = (float)((i % 97) - 48) / 97.0f;
        B[i] = (float)((i % 89) - 44) / 89.0f;
        C[i] = 0.0f;
    }

    t0 = now_sec();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, M, M, 1.0f, A, M, B, M, 0.0f, C, M);
    t1 = now_sec();

    double checksum = 0.0;
    for (size_t i = 0; i < n; ++i) checksum += C[i];

    double flops = 2.0 * M * M * M;
    printf("sgemm_size=%dx%d\n", M, M);
    printf("sgemm_time_sec=%.6f\n", t1 - t0);
    printf("sgemm_gflops=%.3f\n", flops / (t1 - t0) / 1e9);
    printf("sgemm_checksum=%.9e\n", checksum);

    free(A); free(B); free(C);
    return 0;
}
