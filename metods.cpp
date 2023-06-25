#include "header.h"

#define PART 0.03
#define IND2(a,b) ((a) * n * n + (b))

TComplex ZERO((acc_number)0.0, (acc_number)0.0), ONE((acc_number)1.0, (acc_number)0.0), I((acc_number)0.0, (acc_number)1.0);
FILE * step_file;
unsigned int number_of_QJ = 0;

void Mcblas_m(int m, int n, int k, const TComplex* a, const TComplex* b, TComplex* c)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			c[j + i * n] = TComplex((acc_number)0.0, (acc_number)0.0);
			for (int t = 0; t < k; t++)
			{
				//c[j + i * n] += a[t + i * k] * b[j + t * n];

				c[j + i * n] += a[t + i * k] * b[t + j * k];
			}
		}
	}
}

void Mcblas_v(int m, int n, const TComplex* a, const TComplex* x, TComplex* y)
{
	for (int i = 0; i < m; i++)
	{
		y[i] = TComplex((acc_number)0.0, (acc_number)0.0);
		for (int j = 0; j < n; j++)
		{
			y[i] += x[j] * a[j + i * n];
		}
	}
}

void mul_exp(TComplex * exp_i, states * phi_matrix, unsigned int from, unsigned int to, TComplex * res)
{
	if((to - from) > 0)
		//fprintf(step_file, "%d\n", (to - from));

			if(((double)(to - from))/phi_matrix->n > PART)
			{
				MKL_INT         m, n, k;
				MKL_INT         lda, ldb, ldc;
				TComplex        alpha, beta;
				TComplex        *a, *b, *c;
				CBLAS_LAYOUT    layout;
				CBLAS_TRANSPOSE transA, transB;

				layout = CblasRowMajor;
				transA = CblasNoTrans;
				transB = CblasTrans;

				m = to - from;
				k = phi_matrix->size_phi;
				n = phi_matrix->size_phi;

				alpha = ONE;
				beta = ZERO;

				a = &(phi_matrix->phis[from * phi_matrix->size_phi]);
				b = exp_i;
				c = &(res[from * phi_matrix->size_phi]);

				lda = k;
				ldb = k;
				ldc = n;

				Tcblas_m(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
				//cblas_zgemm(layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
				//printf("m:%i, k:%i, n:%i\n", m, k, n);
			}
			else
			{
				MKL_INT         m, n;
				MKL_INT         lda, incx, incy;
				TComplex        alpha, beta;
				TComplex        *a, *x, *y;
				CBLAS_LAYOUT    layout;
				CBLAS_TRANSPOSE transA;

				layout = CblasRowMajor;
				transA = CblasNoTrans;

				m = phi_matrix->size_phi;
				n = phi_matrix->size_phi;

				alpha = ONE;
				beta = ZERO;

				a = exp_i;

				lda = n;
				incx = 1;
				incy = 1;

				for (unsigned int i = from; i < to; i++)
				{
					x = &(phi_matrix->phis[n * i]);
					y = &(res[i * n]);

					Tcblas_v(layout, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
					//cblas_zgemv(layout, transA, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
				}
			}
}

acc_number norm_vector2(TComplex * phi, int n)
{
	acc_number norm = (acc_number)0.;
	for(int i = 0; i < n; i++)
	{
		norm += phi[i].real() * phi[i].real() + phi[i].imag() * phi[i].imag();
	}
	return norm;
}

acc_number Kahan_norm_vector2(TComplex* phi, int N)
{
	acc_number norm = (acc_number)0.0;
	acc_number c = (acc_number)0.0;
	for (int i = 0; i < N; i++)
	{
		acc_number y = (phi[i].real() * phi[i].real() + phi[i].imag() * phi[i].imag()) - c;
		acc_number t = norm + y;
		c = (t - norm) - y;
		norm = t;
	}
	return norm;
}

TComplex * tmp_complex_QJ;
acc_number * tmp_double_QJ;
#pragma omp threadprivate (tmp_complex_QJ, tmp_double_QJ)


inline void QJ(TComplex * phi, acc_number eta, acc_number* g, TComplex * A, unsigned int n, unsigned int g_k, acc_number random)
{// size(tmp_complex) = n; size(tmp_double) = k
	#pragma omp atomic
	number_of_QJ++;

	TComplex * tmp_complex = tmp_complex_QJ;
	acc_number * tmp_double = tmp_double_QJ;

	unsigned int num_thread = omp_get_thread_num();

	MKL_INT         m;
	MKL_INT         lda, incx, incy;
	TComplex   alpha, beta;
	TComplex  *a, *x, *y;
	CBLAS_LAYOUT    layout;
	CBLAS_TRANSPOSE transA;

	layout = CblasRowMajor;
	transA = CblasNoTrans;
	m = n;
	alpha = ONE;
	beta = ZERO;
	lda = n;
	incx = 1;
	incy = 1;
	x = phi;
	y = tmp_complex;


	int index = 0;
	TComplex coeff;

	acc_number norm = sqrt(norm_vector2(phi, n));
	acc_number t_re, t_im;

	for(int i = 0; i < n; i++)
	{
		t_re = phi[i].real();
		t_im = phi[i].imag();
		phi[i] = TComplex(t_re / norm, t_im / norm);
	}
	norm=0.;

	acc_number* gnorms = tmp_double;
	acc_number div = (acc_number)0.;
	acc_number ran = random;

	for(unsigned int i = 0; i < g_k; i++)
	{
		a = &((TComplex*)A)[IND2(i, 0)];
		Tcblas_v(layout, transA, m, m, alpha, a, lda, x, incx, beta, y, incy);
		//cblas_zgemv (layout, transA, m, m, &alpha, a, lda, x, incx, &beta, y, incy);
		gnorms[i]=(norm_vector2(tmp_complex, n));
		gnorms[i] *= g[i];

		div += gnorms[i];
	}

	ran *= div;

	while(ran - gnorms[index] > 0.)		//calculat index
	{
		ran -= gnorms[index];
		index++;
		if(index == g_k - 1)
			break;
	}

	while(gnorms[index] == 0)		// correct index
	{
		if (index == 0)
		{
			index++;
		}
		else
		{
			index--;
		}
	}


	memset(y, 0, n * sizeof(TComplex));
	a = &((TComplex*)A)[IND2(index, 0)];
	Tcblas_v(layout, transA, m, m, alpha, a, lda, x, incx, beta, y, incy);
	//cblas_zgemv (layout, transA, m, m, &alpha, a, lda, x, incx, &beta, y, incy);

	for(int i = 0; i < n; i++)
	{
		t_re = y[i].real() / sqrt(gnorms[index] / g[index]);
		t_im = y[i].imag() / sqrt(gnorms[index] / g[index]);
		phi[i] = TComplex(t_re, t_im);
	}
}

void init_middle_state(TComplex * phi, unsigned int n)
{
	memset(phi, 0, n * sizeof(TComplex));
	//phi[0].real = 1.0;
	phi[n/2] = TComplex((acc_number)sqrt(1.0/2.0));
	phi[n/2-1] = TComplex((acc_number)sqrt(1.0/4.0));
	phi[n/2+1] = TComplex((acc_number)sqrt(1.0/4.0));
}

unsigned int sort_eta(states * phi_matrix, unsigned int from, unsigned int to, TComplex * mem)
{
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	int i = from;
	int j = to - 1;

	while(j >= i)
	{
		while((phi_matrix->eta[i] < phi_matrix->norms[i]) && (i < j))
		{
			i++;
		}
		if(i == to - 1)
		{
			if(phi_matrix->eta[i] < phi_matrix->norms[i])
				i++;
		}

		while((phi_matrix->eta[j] >= phi_matrix->norms[j]) && (j > i))
		{
			j--;
		}
		if(j == from)
		{
			if(phi_matrix->eta[j] >= phi_matrix->norms[j])
				j--;
		}

		if(j >= i)			//swap
		{
			memcpy(tmp_vec, &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
			memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
			memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

			tmp_dou = phi_matrix->norms[i];
			phi_matrix->norms[i] = phi_matrix->norms[j];
			phi_matrix->norms[j] = tmp_dou;

			tmp_dou = phi_matrix->eta[i];
			phi_matrix->eta[i] = phi_matrix->eta[j];
			phi_matrix->eta[j] = tmp_dou;

			tmp_int = phi_matrix->steps[i];
			phi_matrix->steps[i] = phi_matrix->steps[j];
			phi_matrix->steps[j] = tmp_int;	

			if(i != j)
				i++;
			j--;
		}
	}
	return i;
}

unsigned int sort_level(states * phi_matrix, unsigned int from, unsigned int to, TComplex * mem, unsigned int level)
{
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	int i = from;
	int j = to - 1;

	while(j >= i)
	{
		while((phi_matrix->steps[i]&(1<<level)) && (i < j))
		{
			i++;
		}
		if(i == j)
		{
			if(phi_matrix->steps[i]&(1<<level))
				i++;
		}

		while(!(phi_matrix->steps[j]&(1<<level)) && (j > i))
		{
			j--;
		}
		if(j == i)
		{
			if(!phi_matrix->steps[j]&(1<<level))
				j--;
		}

		if(j >= i)			//swap
		{
			memcpy(tmp_vec, &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
			memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
			memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

			tmp_dou = phi_matrix->norms[i];
			phi_matrix->norms[i] = phi_matrix->norms[j];
			phi_matrix->norms[j] = tmp_dou;

			tmp_dou = phi_matrix->eta[i];
			phi_matrix->eta[i] = phi_matrix->eta[j];
			phi_matrix->eta[j] = tmp_dou;

			tmp_int = phi_matrix->steps[i];
			phi_matrix->steps[i] = phi_matrix->steps[j];
			phi_matrix->steps[j] = tmp_int;			

			if(i != j)
				i++;
			j--;
		}
	}
	return i;
}

unsigned int copy_sort_eta(states * phi_matrix, unsigned int from, unsigned int to, TComplex * mem, unsigned int level)  //return i, ||phi[i_]||>eta[i_], where i_=0:i;
{
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	acc_number norm_i, norm_j;

	int i = from;
	int j = to - 1;

	while(j >= i)
	{
		norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi); 
		while((phi_matrix->eta[i] > norm_i) && (i < j))
		{
			i++;
			norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(i == j)
		{
			if((phi_matrix->eta[i] > norm_i))
				i++;
		}

		norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		while((phi_matrix->eta[j] <= norm_j) && (j > i))
		{
			memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[j * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));
			phi_matrix->norms[j] = norm_j;
			phi_matrix->steps[j] += (1<<level);

			j--;

			norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(j == i)
		{
			if(phi_matrix->eta[j] <= norm_j)
			{
				memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[j * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));
				phi_matrix->norms[j] = norm_j;
				phi_matrix->steps[j] += (1<<level);

				j--;
			}
		}

		if(j >= i)			//swap
		{
			if( i != j)
			{	
				memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[i * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));

				phi_matrix->norms[i] = phi_matrix->norms[j];
				phi_matrix->norms[j] = norm_i;

				tmp_dou = phi_matrix->eta[i];
				phi_matrix->eta[i] = phi_matrix->eta[j];
				phi_matrix->eta[j] = tmp_dou;

				tmp_int = phi_matrix->steps[i];
				phi_matrix->steps[i] = phi_matrix->steps[j];
				phi_matrix->steps[j] = tmp_int;

				phi_matrix->steps[j] += (1<<level);

				i++;
			}

			j--;
		}

	}
	return i;
}


unsigned int copy_sort_eta_n(states * phi_matrix, unsigned int from, unsigned int to, TComplex * mem, unsigned int level)  //return i, ||phi[i_]||>eta[i_], where i_=0:i;
{
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	acc_number norm_i, norm_j;

	int i = from;
	int j = to - 1;

	while(j >= i)
	{
		norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi); 
		while((phi_matrix->eta[i] <= norm_i) && (i < j))
		{
			memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &mem[i * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));
			phi_matrix->norms[i] = norm_i;
			phi_matrix->steps[i] += (1<<level);
			i++;
			norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(i == j)
		{
			if((phi_matrix->eta[i] <= norm_i))
				i++;
		}

		norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		while((phi_matrix->eta[j] > norm_j) && (j > i))
		{
			j--;

			norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(j == i)
		{
			if(phi_matrix->eta[j] > norm_j)
			{
				j--;
			}
		}

		if(j >= i)			//swap
		{
			if( i != j)
			{	
				memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &mem[j * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));

				phi_matrix->norms[j] = phi_matrix->norms[i];
				phi_matrix->norms[i] = norm_j;

				tmp_dou = phi_matrix->eta[j];
				phi_matrix->eta[j] = phi_matrix->eta[i];
				phi_matrix->eta[i] = tmp_dou;

				tmp_int = phi_matrix->steps[j];
				phi_matrix->steps[j] = phi_matrix->steps[i];
				phi_matrix->steps[i] = tmp_int;

				phi_matrix->steps[i] += (1<<level);

				i++;
			}

			j--;
		}

	}
	return i;
}

unsigned int split_x_x(states * phi_matrix, unsigned int from, unsigned int to, TComplex * mem, unsigned int level, unsigned int max_step)  //return i, ||phi[i_]||>eta[i_], where i_=0:i;
{
	if(phi_matrix->xy == 0)
		return 0;
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	acc_number norm_i, norm_j;

	int i = from;
	int j = to - 1;
	int xy = phi_matrix->xy - 1;

	while(j >= i)
	{

		norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi); 
		while((phi_matrix->eta[i] > norm_i) && (i < j))
		{
			i++;
			norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi);
		}

		norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		while((phi_matrix->eta[j] <= norm_j) && (j > i))
		{
			if((phi_matrix->steps[j] + (1<<level)) <= max_step)
			{
				memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[j * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));
				phi_matrix->norms[j] = norm_j;
				phi_matrix->steps[j] += (1<<level);

				j--;
			}
			else
			{

				memcpy(tmp_vec, &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &(phi_matrix->phis[xy * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[xy * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

				tmp_dou = phi_matrix->norms[j];
				phi_matrix->norms[j] = phi_matrix->norms[xy];
				phi_matrix->norms[xy] = tmp_dou;

				tmp_dou = phi_matrix->eta[j];
				phi_matrix->eta[j] = phi_matrix->eta[xy];
				phi_matrix->eta[xy] = tmp_dou;

				tmp_int = phi_matrix->steps[j];
				phi_matrix->steps[j] = phi_matrix->steps[xy];
				phi_matrix->steps[xy] = tmp_int;

				xy--;
				j--;
			}
			norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(j == i)
		{
			if(phi_matrix->eta[j] <= norm_j)
			{
				if((phi_matrix->steps[j] + (1<<level)) <= max_step)
				{
					memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[j * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));
					phi_matrix->norms[j] = norm_j;
					phi_matrix->steps[j] += (1<<level);

					j--;
				}
				else
				{
					memcpy(tmp_vec, &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &(phi_matrix->phis[xy * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[xy * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

					tmp_dou = phi_matrix->norms[j];
					phi_matrix->norms[j] = phi_matrix->norms[xy];
					phi_matrix->norms[xy] = tmp_dou;

					tmp_dou = phi_matrix->eta[j];
					phi_matrix->eta[j] = phi_matrix->eta[xy];
					phi_matrix->eta[xy] = tmp_dou;

					tmp_int = phi_matrix->steps[j];
					phi_matrix->steps[j] = phi_matrix->steps[xy];
					phi_matrix->steps[xy] = tmp_int;

					xy--;
					j--;
				}
			}
			else
			{
				i++;
			}
		}
		
		if(j > i)			//swap
		{
				if((phi_matrix->steps[i] + (1<<level)) <= max_step)
				{
					memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &mem[i * phi_matrix->size_phi], phi_matrix->size_phi * sizeof(TComplex));

					phi_matrix->norms[i] = phi_matrix->norms[j];
					phi_matrix->norms[j] = norm_i;

					tmp_dou = phi_matrix->eta[i];
					phi_matrix->eta[i] = phi_matrix->eta[j];
					phi_matrix->eta[j] = tmp_dou;

					tmp_int = phi_matrix->steps[i];
					phi_matrix->steps[i] = phi_matrix->steps[j];
					phi_matrix->steps[j] = tmp_int;

					phi_matrix->steps[j] += (1<<level);

					i++;
				}
				else
				{
					memcpy(tmp_vec, &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[xy * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[xy * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

					tmp_dou = phi_matrix->norms[i];
					phi_matrix->norms[i] = phi_matrix->norms[xy];
					phi_matrix->norms[xy] = tmp_dou;

					tmp_dou = phi_matrix->eta[i];
					phi_matrix->eta[i] = phi_matrix->eta[xy];
					phi_matrix->eta[xy] = tmp_dou;

					tmp_int = phi_matrix->steps[i];
					phi_matrix->steps[i] = phi_matrix->steps[xy];
					phi_matrix->steps[xy] = tmp_int;
					xy--;

					memcpy(tmp_vec, &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

					tmp_dou = phi_matrix->norms[i];
					phi_matrix->norms[i] = phi_matrix->norms[j];
					phi_matrix->norms[j] = tmp_dou;

					tmp_dou = phi_matrix->eta[i];
					phi_matrix->eta[i] = phi_matrix->eta[j];
					phi_matrix->eta[j] = tmp_dou;

					tmp_int = phi_matrix->steps[i];
					phi_matrix->steps[i] = phi_matrix->steps[j];
					phi_matrix->steps[j] = tmp_int;

					i++;
				}
			j--;
		}

	}
	phi_matrix->xy = xy + 1;
	return i;
}

unsigned int split_x_y_n(states * phi_matrix, TComplex * mem, unsigned int level, unsigned int max_step)  //return i, ||phi[i_]||>eta[i_], where i_=0:i;
{
	TComplex * tmp_vec = mem;
	acc_number tmp_dou;
	unsigned int tmp_int;

	acc_number norm_i, norm_j;

	int i = phi_matrix->xx;
	int j = phi_matrix->xy - 1;

	while(j >= i)
	{
		norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi); 
		while((i < j) && (((phi_matrix->steps[i] + (1<<level)) <= max_step) || (phi_matrix->eta[i] > norm_i)))
		{
			i++;
			norm_i = norm_vector2(&mem[i * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(i == j)
		{
			if((phi_matrix->eta[i] > norm_i) && ((phi_matrix->steps[i] + (1<<level)) <= max_step))
				i++;
		}

		norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		while((j > i) && (((phi_matrix->steps[j] + (1<<level)) > max_step) && (phi_matrix->eta[j] <= norm_j)))
		{
			j--;
			norm_j = norm_vector2(&mem[j * phi_matrix->size_phi], phi_matrix->size_phi);
		}
		if(j == i)
		{
			if(((phi_matrix->steps[j] + (1<<level)) > max_step) && (phi_matrix->eta[j] <= norm_j))
			{
				j--;
			}
		}

		if(j >= i)			//swap
		{
			if( i != j)
			{
				if(((phi_matrix->steps[j] + (1<<level)) <= max_step) || (phi_matrix->eta[j] > norm_i))
				{
					memcpy(tmp_vec, &(phi_matrix->phis[j * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[j * phi_matrix->size_phi]), &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
					memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), tmp_vec, phi_matrix->size_phi * sizeof(TComplex));

					tmp_dou = phi_matrix->norms[j];
					phi_matrix->norms[j] = phi_matrix->norms[i];
					phi_matrix->norms[i] = tmp_dou;

					tmp_dou = phi_matrix->eta[i];
					phi_matrix->eta[i] = phi_matrix->eta[j];
					phi_matrix->eta[j] = tmp_dou;

					tmp_int = phi_matrix->steps[i];
					phi_matrix->steps[i] = phi_matrix->steps[j];
					phi_matrix->steps[j] = tmp_int;

					i++;
				}
			}
			j--;
		}

	}
	return i;
}

inline void backward(split * head, split * branch, states * phi_matrix, conf config_data, VSLStreamStatePtr streamRand, TComplex * mem)
{
	
	unsigned int k = 0;
	while(branch->prev != head)
	{
		k = sort_level(phi_matrix, phi_matrix->xy, phi_matrix->n, mem, branch->steps - branch->counter - 1);
		mul_exp(branch->matrix, phi_matrix, phi_matrix->xy, k, mem);
		memcpy(&(phi_matrix->phis[phi_matrix->xy * phi_matrix->size_phi]), &(mem[phi_matrix->xy * phi_matrix->size_phi]), sizeof(TComplex) * (k - phi_matrix->xy) * phi_matrix->size_phi);
		for(unsigned int i = phi_matrix->xy; i < k; i++)
		{
			phi_matrix->norms[i] = norm_vector2(&(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi);
			phi_matrix->steps[i] += 1<<(branch->steps - branch->counter - 1);
		}
		branch = branch->prev;
	}

	unsigned int max_step = (1<<(branch->steps - 1)) * branch->counter;
	mul_exp(branch->matrix, phi_matrix, phi_matrix->xy, phi_matrix->n, mem);
	for(unsigned int i = phi_matrix->xy; i < phi_matrix->n; i++)
	{
		if(phi_matrix->steps[i] + (1<<(branch->steps - 1)) <= max_step)
		{
			phi_matrix->steps[i] += 1<<(branch->steps - 1);
			memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(mem[i * phi_matrix->size_phi]), sizeof(TComplex) * phi_matrix->size_phi);
			phi_matrix->norms[i] = norm_vector2(&(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi);
		}
	}
	branch = branch->prev;
}

inline void forward(split * head, split * branch, states * phi_matrix, conf config_data, VSLStreamStatePtr streamRand, TComplex * mem)
{
	unsigned int max_step = (1<<(branch->steps - 1)) * branch->counter;
	for(unsigned int i=0; i<branch->counter; i++)
	{
		mul_exp(branch->matrix, phi_matrix, phi_matrix->xx, phi_matrix->xy, mem);
		phi_matrix->xx = split_x_x(phi_matrix, phi_matrix->xx, phi_matrix->xy, mem, branch->steps - 1, max_step);
		//phi_matrix->xy = split_x_y_n(phi_matrix, mem, branch->steps - 1, max_step);
		//phi_matrix->xx = copy_sort_eta_n(phi_matrix, 0, phi_matrix->xy, mem, branch->steps - 1);
	}
	phi_matrix->xy = phi_matrix->xx;
	if (phi_matrix->xx > 0)
	{
		while(branch->next)
		{
			mul_exp(branch->matrix, phi_matrix, 0, phi_matrix->xx, mem);
			//copy_sort_eta(phi_matrix, 0, phi_matrix->xx, mem, branch->steps - branch->counter - 1);
			split_x_x(phi_matrix, 0, phi_matrix->xy, mem, branch->steps - branch->counter - 1, max_step);
			phi_matrix->xx = phi_matrix->xy;
			branch = branch->next;
		}

		mul_exp(branch->matrix, phi_matrix, 0, phi_matrix->xx, mem);
		//copy_sort_eta(phi_matrix, 0, phi_matrix->xx, mem, branch->steps - branch->counter - 1);
		split_x_x(phi_matrix, 0, phi_matrix->xy, mem, branch->steps - branch->counter - 1, max_step);
		phi_matrix->xx = phi_matrix->xy;
		
		for(unsigned int i = 0; i < phi_matrix->xy; i++)
		{
			if(phi_matrix->steps[i] + 1 > max_step)
			{
				memcpy(mem, &(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[i * phi_matrix->size_phi]), &(phi_matrix->phis[phi_matrix->xy * phi_matrix->size_phi]), phi_matrix->size_phi * sizeof(TComplex));
				memcpy(&(phi_matrix->phis[phi_matrix->xy * phi_matrix->size_phi]), mem, phi_matrix->size_phi * sizeof(TComplex));

				acc_number tmp_dou = phi_matrix->norms[i];
				phi_matrix->norms[i] = phi_matrix->norms[phi_matrix->xy];
				phi_matrix->norms[phi_matrix->xy] = tmp_dou;

				tmp_dou = phi_matrix->eta[i];
				phi_matrix->eta[i] = phi_matrix->eta[phi_matrix->xy];
				phi_matrix->eta[phi_matrix->xy] = tmp_dou;

				unsigned int tmp_int = phi_matrix->steps[i];
				phi_matrix->steps[i] = phi_matrix->steps[phi_matrix->xy];
				phi_matrix->steps[phi_matrix->xy] = tmp_int;
				phi_matrix->xy--;
			}
		}
		phi_matrix->xx = phi_matrix->xy;

		mul_exp(branch->matrix, phi_matrix, 0, phi_matrix->xx, mem);
		memcpy(phi_matrix->phis, mem, sizeof(TComplex) * phi_matrix->size_phi * phi_matrix->xx);

		for(unsigned int i = 0; i < phi_matrix->xx; i++)
		{
			phi_matrix->steps[i] += (1<<(branch->steps - branch->counter - 1));
			/*acc_number random;
			RngUniform(random);*/

			acc_number random;
			acc_eta rand1;
			RngUniform(rand1);
			random = (acc_number)rand1;

			QJ(&(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->eta[i], head->g, head->matrix, phi_matrix->size_phi, head->steps, random);
			
			acc_number eta = (acc_number)0.0;
			acc_eta eta1;
			
			/*RngUniform(phi_matrix->eta[i]);
			while (phi_matrix->eta[i] == 0.0)
			{
				RngUniform(phi_matrix->eta[i]);
			}*/

			while (eta <= eta_const)
			{
				RngUniform(eta1);
				eta = (acc_number)eta1;
			}

			phi_matrix->eta[i] = eta;
			phi_matrix->norms[i] = 1.0;

		}
	}
}

inline void integ_one_branch(split * head, split * branch, states * phi_matrix, conf config_data, VSLStreamStatePtr streamRand, TComplex * mem)
{
	memset(phi_matrix->steps, 0, sizeof(unsigned int) * phi_matrix->n);

	TComplex * tmp_res = mem;
	unsigned int max_step = (1<<(branch->steps - 1)) * branch->counter;

	while(phi_matrix->xy > 0)
	{
		forward(head, branch, phi_matrix, config_data, streamRand, mem);
		phi_matrix->xx = 0;
	}
	
	while (branch->next)
	{
		branch = branch->next;
	}
	backward(head, branch, phi_matrix, config_data, streamRand, mem);
}

inline void integ_one_period(split * head, states * phi_matrix, conf config_data, VSLStreamStatePtr streamRand, TComplex * mem)
{
	for(unsigned int i = 0; i < head->counter; i++)
	{
		memset(phi_matrix->steps, 0, sizeof(unsigned int) * phi_matrix->n);
		phi_matrix->xx = 0;
		phi_matrix->xy = phi_matrix->n;

		integ_one_branch(head, &(head->next[i]), phi_matrix, config_data, streamRand, mem);
	}
}

inline TComplex Complex_mul(TComplex a, acc_number b)
{
	return TComplex(a.real() * b, a.imag() * b);
}
inline TComplex Complex_scalar_mul(TComplex * a, TComplex * b, int N)
{
	TComplex res = ((acc_number)0.0, (acc_number)0.0);
	acc_number t_re, t_im;
	for (int i = 0; i < N; i++)
	{
		t_re = res.real() + a[i].real() * b[i].real() + a[i].imag() * b[i].imag();
		t_im = res.imag() + b[i].real() * a[i].imag() - a[i].real() * b[i].imag();
		res = TComplex(t_re, t_im);
	}
	return res;
}
void count_ro(states phi_matrix, MKL_Complex16 * ro)
{
	MKL_Complex16 tmp;
	for(int k = 0; k < phi_matrix.n; k++)
	{
		double norm = (double)sqrt(norm_vector2(&(phi_matrix.phis[k * phi_matrix.size_phi]), phi_matrix.size_phi));
		double t_re, t_im;
		for(int i = 0; i < phi_matrix.size_phi; i++)
			for(int j = 0; j < phi_matrix.size_phi; j++)
			{
				tmp = Complex_mul(Complex_scalar_mul(&phi_matrix.phis[k * phi_matrix.size_phi + i], &phi_matrix.phis[k * phi_matrix.size_phi + j], 1), (acc_number)(1/norm));
				t_re = ro[i * phi_matrix.size_phi + j].real();
				t_im = ro[i * phi_matrix.size_phi + j].imag();
				ro[i * phi_matrix.size_phi + j] = MKL_Complex16(t_re + tmp.real() / phi_matrix.n, t_im + tmp.imag() / phi_matrix.n);
			}
	}
}

void integ_matrix (split * head, conf config_data, VSLStreamStatePtr streamRand, void (*init)(TComplex * phi, unsigned int N), long long int* numb_QJ)
{
	states * phi_matrix = (states *) mkl_malloc(sizeof(states), ALIG);
	TComplex * mem;


	//char filename[80];
	//sprintf(filename, "step_%i.txt", config_data.rnd_cur + omp_get_thread_num());
	//step_file = fopen(filename, "w");

	phi_matrix->size_phi = head->n;
	phi_matrix->n = config_data.L;
	phi_matrix->xx = phi_matrix->n;
	phi_matrix->xy = phi_matrix->n;

	phi_matrix->phis = (TComplex *) mkl_malloc(sizeof(TComplex) * phi_matrix->n * phi_matrix->size_phi, ALIG);
	phi_matrix->norms = (acc_number*) mkl_malloc(sizeof(acc_number) * phi_matrix->n, ALIG);
	phi_matrix->steps = (unsigned int *) mkl_malloc(sizeof(unsigned int) * phi_matrix->n, ALIG);
	phi_matrix->eta = (acc_number*) mkl_malloc(sizeof(acc_number) * phi_matrix->n, ALIG);
	mem = (TComplex *) mkl_malloc(sizeof(TComplex) * phi_matrix->n * phi_matrix->size_phi, ALIG);
	memset(phi_matrix->eta, 0, sizeof(acc_number) * phi_matrix->n);
	memset(phi_matrix->steps, 0, sizeof(unsigned int) * phi_matrix->n);

	tmp_complex_QJ = (TComplex *) mkl_malloc(sizeof(TComplex) * phi_matrix->size_phi, ALIG);
	tmp_double_QJ = (acc_number*) mkl_malloc(sizeof(acc_number) * head->steps, ALIG);


	//out
	MKL_Complex16 * ro = (MKL_Complex16*) mkl_malloc(sizeof(MKL_Complex16) * phi_matrix->size_phi * phi_matrix->size_phi, ALIG);
	memset(ro, 0, sizeof(MKL_Complex16) * phi_matrix->size_phi * phi_matrix->size_phi);


	for(unsigned int i = 0; i < phi_matrix->n; i++)
	{
		init(&(phi_matrix->phis[i * phi_matrix->size_phi]), phi_matrix->size_phi);
		phi_matrix->norms[i] = 1.0;
		/*while (phi_matrix->eta[i] == 0.0)
		{
			RngUniform(phi_matrix->eta[i]);
		}*/

		acc_number eta = (acc_number)0.0;
		acc_eta eta1;

		while (eta <= eta_const)
		{
			RngUniform(eta1);
			eta = (acc_number)eta1;
		}

		phi_matrix->eta[i] = eta;
	}

	int num_thread = omp_get_thread_num();

	if (num_thread == 0 && (config_data.before_kT > 0))
	{
		printf("before_kT iterations\n\n");
		fflush(stdout);
	}

	for(unsigned int i = 0; i < config_data.before_kT; i++)
	{
		//printf("beforKT:%i\n",i);
		integ_one_period(head, phi_matrix, config_data, streamRand, mem);

		if (num_thread == 0)
		{
			printf("%d / %d\n", i + 1, config_data.kT);
			fflush(stdout);
		}
	}

	if (num_thread == 0 && (config_data.kT > 0))
	{
		printf("kT iterations\n\n");
		fflush(stdout);
	}

	for(unsigned int i = 0; i < config_data.kT; i++)
	{
		integ_one_period(head, phi_matrix, config_data, streamRand, mem);
		//printf("kT:%i\n",i);

		if (num_thread == 0)
		{
			printf("%d / %d\n", i + 1, config_data.kT);
			fflush(stdout);
		}

		if(i % config_data.pT == 0)
		{
			//output

			//count_ro(phi_matrix[0], ro);
		}
	}

	count_ro(phi_matrix[0], ro);
	double temp1;

	if (num_thread == 0)
	{
		//printf("\nDiag elements:\n\n");
		FILE* file = fopen(MATRIX_OUT, "w");
		for (int i = 0; i < phi_matrix->size_phi; i++)
		{
			temp1 = (double)ro[i + i * phi_matrix->size_phi].real();
			//printf("%lf\n", temp1);
			fprintf(file, "%lf\n", temp1);
		}
		fclose(file);

		*numb_QJ = number_of_QJ;
	}

	//char filename[80];
	//sprintf(filename, "out_ro_%i.bin", config_data.rnd_cur + omp_get_thread_num());
	//FILE * file = fopen(filename, "wb");
	//fwrite(ro, sizeof(TComplex), phi_matrix->size_phi * phi_matrix->size_phi, file);
	//fclose(file);

	mkl_free(ro);

	mkl_free(tmp_complex_QJ);
	mkl_free(tmp_double_QJ);

	mkl_free(phi_matrix->phis);
	mkl_free(phi_matrix->norms);
	mkl_free(phi_matrix->steps);
	mkl_free(phi_matrix->eta);
	mkl_free(phi_matrix);
	mkl_free(mem);
}