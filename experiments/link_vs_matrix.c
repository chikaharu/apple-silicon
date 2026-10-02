#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const int N = 1000000;
    const double theta = 1e-5;
    const double c = cos(theta), s = sin(theta);

    // Complex 1x1 representation of planar rotation.
    double complex z = 1.0 + 0.0 * I;
    const double complex step = c + I*s;

    double t0 = now_sec();
    for (int k = 0; k < N; ++k) z *= step;
    double t1 = now_sec();

    const double complex_z_time = t1 - t0;
    const double complex_norm_err = fabs(cabs(z) - 1.0);

    // Explicit 2x2 real rotation matrix, accumulated by matrix multiply.
    double a=1.0,b=0.0,cc=0.0,d=1.0;
    t0 = now_sec();
    for (int k = 0; k < N; ++k) {
        const double na = a*c + b*s;
        const double nb = -a*s + b*c;
        const double nc = cc*c + d*s;
        const double nd = -cc*s + d*c;
        a=na; b=nb; cc=nc; d=nd;
    }
    t1 = now_sec();

    const double mat_time = t1 - t0;

    // Frobenius norm of R^T R - I.
    const double e00 = a*a + cc*cc - 1.0;
    const double e01 = a*b + cc*d;
    const double e10 = b*a + d*cc;
    const double e11 = b*b + d*d - 1.0;
    const double ortho_err = sqrt(e00*e00 + e01*e01 + e10*e10 + e11*e11);

    // Compare final action on x=(1,0).
    const double x_complex = creal(z);
    const double y_complex = cimag(z);
    const double x_matrix = a;
    const double y_matrix = cc;
    const double action_err = hypot(x_complex - x_matrix, y_complex - y_matrix);

    printf("steps=%d\n", N);
    printf("complex_time_sec=%.9f\n", complex_z_time);
    printf("matrix2_time_sec=%.9f\n", mat_time);
    printf("speedup_matrix_over_complex=%.6f\n", mat_time/complex_z_time);
    printf("complex_unit_norm_error=%.3e\n", complex_norm_err);
    printf("matrix2_orthogonality_error=%.3e\n", ortho_err);
    printf("final_action_error=%.3e\n", action_err);
    printf("complex_final=(%.12f, %.12f)\n", x_complex, y_complex);
    printf("matrix2_final=(%.12f, %.12f)\n", x_matrix, y_matrix);
    return 0;
}
