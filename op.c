#include <stdio.h>
#include <math.h>
#include "matrix.h"
double func(int n) {
    double temp = 0;
    for (int i = 0; i < 11; temp += pow(-1, i) * pow(n, i), i++);
    return temp;
}

double func_prime(int n, int dim, double* coef) {
    double temp = 0;
    for (int i = 0; i < dim; temp += coef[i] * pow(n, i), i++);
    return temp;
}

int main() {
    double sum = 0;
    double seq[10];
    for (int i = 0; i < 10; i++) {
        int n = i + 1;
        seq[i] = func(n);
    }
    double** mat = matrix_alloc(10, 10);
    double** inv_mat = matrix_alloc(10, 10);
    double sol[10];
    for (int i = 2; i <= 9; i++) {
        puts("rhs:");
        print_vector(i, seq);
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < i; k++) {
                mat[j][k] = pow(j + 1, k);
            }
        }
        puts("coef:");
        print_matrix(i, i, mat);
        matrix_invert(i, mat, inv_mat);
        left_multiply(i, i, inv_mat, seq, sol);
        puts("sol:");
        print_vector(i, sol);
        double temp = func_prime(i + 1, i, sol);
        printf("num:\n%lf\n", temp);
        sum += temp;
    }
    printf("sum:\n%lf\n", sum + 1);
    free_matrix(10, mat);
    free_matrix(10, inv_mat);
}
