#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <locale.h>

#define N 2048

int main() {
    setlocale(LC_ALL, "Russian");

    static double A[N][N], B[N][N], C[N][N], Temp[N][N];
    int i, j, k, r, var, t;
    int nt;
    double start, end, min_time;
    double t1_ikj = 0.0, t1_kij = 0.0;

    srand(time(NULL));

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
        }

    printf("Умножение матриц %dx%d\n", N, N);
    printf("Порядок   | Потоков |   Время (с)   | Ускорение | Эффективность\n\n");

    int threads[] = { 1, 2, 4, 8, 12, 14, 16, 20, 24, 28, 32, 34, 64 };
    int num_tests = sizeof(threads) / sizeof(threads[0]);


    for (t = 0; t < num_tests; t++)
    {
        nt = threads[t];
        omp_set_num_threads(nt);

        for (var = 0; var < 2; var++)
        {
            const char* order;
            if (var == 0) {
                order = "i-k-j";
            }
            else {
                order = "k-i-j";
            }

            min_time = 1e9;

            for (r = 0; r < 3; r++){

                for (i = 0; i < N; i++)
                    for (j = 0; j < N; j++)
                    {
                        C[i][j] = 0.0;
                    }

                start = omp_get_wtime();

                #pragma omp parallel private(i,j,k)
                {
                    static double localC[N][N];

                    for (i = 0; i < N; i++)
                        for (j = 0; j < N; j++)
                            localC[i][j] = 0.0;

                    if (var == 0) {
                        #pragma omp for schedule(dynamic)
                        for (i = 0; i < N; i++) {
                            for (k = 0; k < N; k++) {
                                for (j = 0; j < N; j++) {
                                    localC[i][j] += A[i][k] * B[k][j];
                                }
                            }
                        }
                    }
                    else {
                        #pragma omp for schedule(dynamic)
                        for (k = 0; k < N; k++) {
                            for (i = 0; i < N; i++) {
                                for (j = 0; j < N; j++) {
                                    localC[i][j] += A[i][k] * B[k][j];
                                }
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        for (i = 0; i < N; i++)
                            for (j = 0; j < N; j++)
                                C[i][j] += localC[i][j];
                    }
                }

                end = omp_get_wtime();

                if (end - start < min_time) {
                    min_time = end - start;
                }
            }

            if (nt == 1) {
                if (var == 0) {
                    t1_ikj = min_time;
                }
                else {
                    t1_kij = min_time;
                }
            }

            double speedup;
            if (nt == 1) {
                speedup = 1.0;
            }
            else {
                if (var == 0) {
                    speedup = t1_ikj / min_time;
                }
                else {
                    speedup = t1_kij / min_time;
                }
            }

            double eff = speedup / nt * 100.0;

            printf("%-8s  | %2d      | %.6f      | %.6f  | %.6f%%\n",
                order, nt, min_time, speedup, eff);
        }
    }

    return 0;
}