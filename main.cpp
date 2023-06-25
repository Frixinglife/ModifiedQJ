#include "header.h"

int main(int argc, char **argv)
{
    if(argc < 3)
	{
		printf("No argument filenames");
		fflush(stdout);
		return -2;
	}
	conf config_data;
    int numprocs, myid;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	double start_time = MPI_Wtime();;
	if (myid == 0)
    {
		FILE * config = fopen(argv[1], "r");
		if (configuration(config, &config_data)) 
		{ 
			fclose (config);
			return 1;
		}
		fclose (config);

		printf("Number of processes : %i\n", numprocs);
		fflush(stdout);
		printf("Number of threads : %i\n\n", config_data.threads_omp);
		fflush(stdout);
	}
	MPI_Bcast(&config_data, sizeof(conf), MPI_BYTE, 0, MPI_COMM_WORLD);

	FILE * input;
	if (myid == 0)
    {
		input = fopen(config_data.input, "rb");
	}
    split * head = create_struct_bin_MPI(input, numprocs, myid);
	if (myid == 0)
    {
		fclose(input);
	}
   
    split * heads = (split *) mkl_malloc(sizeof(split) * config_data.threads_omp, ALIG);
    for(unsigned int i = 0; i < config_data.threads_omp; i++)
    {
        cmp_struct_not_member (head, &heads[i]);
    }

	config_data.rnd_cur += config_data.threads_omp * myid;

	VSLStreamStatePtr * streamsRand = (VSLStreamStatePtr *) mkl_malloc(sizeof(VSLStreamStatePtr) * config_data.threads_omp, ALIG);
    vslNewStream (&streamsRand[0], VSL_BRNG_MCG59, 777);
    for(unsigned int i = 1; i < config_data.threads_omp; i++)
    {
        vslCopyStream (&streamsRand[i], streamsRand[0]);
    }
	int status = VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED;
    for(unsigned int i = 0; i < config_data.threads_omp; i++)
    {
        status = vslLeapfrogStream (streamsRand[i], config_data.rnd_cur + i, config_data.rnd_max);
    }

    omp_set_dynamic(0);	omp_set_nested(1);
	mkl_set_dynamic(0);
	unsigned int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
	//mkl_set_num_threads(mkl_get_max_threads());
	mkl_domain_set_num_threads(max_threads, MKL_DOMAIN_BLAS);

	if (myid == 0)
	{
		printf("Begin\n\n");
	}

	long long int number_of_QJ_local = 0, number_of_QJ_global = 0;

#pragma omp parallel num_threads(config_data.threads_omp)
	{
        unsigned int num_thread = omp_get_thread_num();
		mkl_set_num_threads_local(config_data.mkl_threads);
		integ_matrix (&heads[num_thread], config_data, streamsRand[num_thread], init_middle_state, &number_of_QJ_local);
	}

	MPI_Reduce(&number_of_QJ_local, &number_of_QJ_global, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

    mkl_free(heads);
    delete_split_struct (head);
    mkl_free(streamsRand);

	double time = MPI_Wtime()- start_time;
	 if (myid == 0)
    {
		FILE* file1 = fopen(argv[2], "a+");
		FILE* file2 = fopen("QJ.txt", "a+");
		
		fprintf(file1, type_out);
		fprintf(file1, " %lf\n", time);
		fprintf(file2, type_out);
		fprintf(file2, " %lld\n", number_of_QJ_global);

		fclose(file1);
		fclose(file2);
		printf("\nEnd\n");
		fflush(stdout);
	 }
    MPI_Finalize();


    return 0;
}