#define _CRT_SECURE_NO_WARNINGS

#include <complex>

#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>

#include "mkl.h"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "omp.h"
#include "mpi.h"
#include "half.hpp"

#define ALIG 64

#define ACC_HALF

#ifdef ACC_DOUBLE
	#define TComplex MKL_Complex16
	#define acc_number double
	#define MPI_number_type MPI_DOUBLE
	#define MPI_complex_type MPI_COMPLEX16
	#define acc_eta double
	#define eta_const 0.0
	#define RngUniform(eta) vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streamRand, 1, &eta, 0, 1)
	#define Tcblas_m(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) cblas_zgemm(layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)
	#define Tcblas_v(layout, transA, m, n, alpha, a, lda, x, incx, beta, y, incy) cblas_zgemv(layout, transA, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
	#define MATRIX_OUT "matrix_diag_double.txt"
	#define type_out "double"
#else
#ifdef ACC_FLOAT
	#define TComplex MKL_Complex8
	#define acc_number float
	#define MPI_number_type MPI_FLOAT
	#define MPI_complex_type MPI_COMPLEX8
	#define acc_eta float
	#define eta_const 0.0f
	#define RngUniform(eta) vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streamRand, 1, &eta, 0, 1)
	#define Tcblas_m(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) cblas_cgemm(layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)
	#define Tcblas_v(layout, transA, m, n, alpha, a, lda, x, incx, beta, y, incy) cblas_cgemv(layout, transA, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
	#define MATRIX_OUT "matrix_diag_float.txt"
	#define type_out "float"
#else
#ifdef ACC_HALF
	#define TComplex std::complex<half_float::half>
	#define acc_number half_float::half
	#define MPI_number_type MPI_SHORT
	#define MPI_complex_type MPI_FLOAT
	#define acc_eta float
	#define eta_const (half_float::half)1e-4
	#define RngUniform(eta) vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streamRand, 1, &eta, 0, 1)
	#define Tcblas_m(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) Mcblas_m(m, n, k, a, b, c)
	#define Tcblas_v(layout, transA, m, n, alpha, a, lda, x, incx, beta, y, incy) Mcblas_v(m, n, a, x, y)
	#define MATRIX_OUT "matrix_diag_half.txt"
	#define type_out "half"
#endif
#endif
#endif

// tree
struct split
{
    //bool type;            //node (true) or head //prev
    split * next;           //head: node_dt_0[counter]; node_dt_i: node_dt_i+1
    split * prev;           //head: 0; node_dt_i: node_dt_i-1
    double dt;				//head: period time; node: value current dt
    unsigned int steps;		//head: count dis operator A; node: max counters value
    unsigned int counter;	//head: counter branch; node_0: steps in period; node: number i 
    unsigned int n;			//size system (matrix)
    TComplex * matrix;	//head: A[steps]; node: exp_dt
    acc_number * g;				//head: g[steps]; node: 0
};

struct conf
{
	unsigned int before_kT;				// the number of periods relax
    unsigned int kT;					// the number of periods count
	unsigned int pT;					// periods between measurements
 	//double h;							// step int ( = 0)
    //unsigned int m;					// the number of integration steps ( = 0)
	unsigned int threads_omp;			// openmp threads
	unsigned int L;						// the number of track in one OMP thread
	unsigned int rnd_max;				// Leapfrog
	unsigned int rnd_cur;				// Leapfrog (0:rnd_max)
	char input [255];					// name inpute file
	unsigned int mkl_threads;			// mkl threads
};

struct states
{
	unsigned int size_phi;	//size system
	unsigned int n;			//the number of phi in struct
	unsigned int xx;			//phi[i] where i<xx, possible qj
	unsigned int xy;			//phi[i] where xx<i<xy, qj in dt
	TComplex * phis;	// matrix phi[i]
	acc_number * norms;			//vector ||phi[i]||
	acc_number * eta;			//vector eta[i]
	unsigned int * steps;	//vector steps on period bit_mask
	
};

void swap_states(states * src, states * dst);

int configuration(FILE * file, conf * config_data);

split * create_struct_bin (FILE * file);					//read *.bin
split * create_struct_bin_MPI (FILE * file, int numprocs, int myid);

void delete_split_struct (split * head);					//delete input data

void cmp_struct_not_member(split * head1, split * head2);	//cmp data for openmp
void delete_split_struct_not_member (split * head);			//delete data for openmp

acc_number norm_vector2(TComplex * phi, int N);			//square norm vector

void init_middle_state(TComplex * phi, unsigned int N);		//3-state in middle vector

void integ_matrix (split * head, conf config_data, VSLStreamStatePtr streamRand, void (*init)(TComplex * phi, unsigned int N), long long int * numb_QJ);

