#include "header.h"

int configuration(FILE * file, conf * config_data)
{
	int count = 0;
	char line[255];
	char name[255], key[255];
	while(fgets(line, 255, file))
	{
		sscanf(line, "%s = %s", name, key);
		if(strncmp("before_kT", name, 9) == 0)
		{
			sscanf(key, "%d", &(config_data->before_kT));
			count = count|(1<<0);
		}
		if(strncmp("kT", name, 2) == 0)
		{
			sscanf(key, "%d", &(config_data->kT));
			count = count|(1<<1);
		}
		if(strncmp("pT", name, 2) == 0)
		{
			sscanf(key, "%d", &(config_data->pT));
			count = count|(1<<2);
		}
		if(strncmp("Num_threads_omp", name, 15) == 0)
		{
			sscanf(key, "%d", &(config_data->threads_omp));
			count = count|(1<<3);
		}
		if(strncmp("L_in_one_thread", name, 15) == 0)
		{
			sscanf(key, "%d", &(config_data->L));
			count = count|(1<<4);
		}
		if(strncmp("RND_max", name, 7) == 0)
		{
			sscanf(key, "%d", &(config_data->rnd_max));
			count = count|(1<<5);
		}
		if(strncmp("RND_cur", name, 7) == 0)
		{
			sscanf(key, "%d", &(config_data->rnd_cur));
			count = count|(1<<6);
		}
		if(strncmp("file", name, 4) == 0)
		{
			sscanf(key, "%s", &(config_data->input));
			count = count|(1<<7);
		}
		
		if(strncmp("mkl_threads", name, 11) == 0)
		{
			sscanf(key, "%d", &(config_data->mkl_threads));
			count = count|(1<<8);
		}
	}

	count = count ^ 511;

	return count;
}

void read_matrix_bin (FILE * file, TComplex * matrix, unsigned int n)
{
	MKL_Complex16* temp = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * n * n, ALIG);
	fread(temp, sizeof(MKL_Complex16), n * n, file);

	acc_number t_re, t_im;

	for (int i = 0; i < n * n; i++)
	{
		t_re = (acc_number)temp[i].real();
		t_im = (acc_number)temp[i].imag();
		matrix[i] = TComplex(t_re, t_im);
	}

	mkl_free(temp);
}

void read_branch_bin (FILE * file, split * branch)
{
    unsigned int deep, n;
    split * node = branch;
    fread(&deep, sizeof(int), 1, file);
    n = (branch->prev)->n;
    for(unsigned int i = 0; i < deep - 1; i++)
    {
        fread(&(node->dt), sizeof(double), 1, file);
        fread(&(node->counter), sizeof(int), 1, file); 
		if(i != 0)
			node->counter = i;
		node->steps = deep;
        node->g = 0;
        node->n = n;
        node->matrix = (TComplex *)mkl_malloc(sizeof(TComplex)* n * n, ALIG);
        read_matrix_bin(file, node->matrix, n);
        node->next = (split *)mkl_malloc(sizeof(split), ALIG);
        node->next->prev = node;
        node = node->next;
    }
    fread(&(node->dt), sizeof(double), 1, file);
    fread(&(node->steps), sizeof(int), 1, file);
    node->counter = deep - 1;
	node->steps = deep;
    node->g = 0;
    node->n = n;
    node->matrix = (TComplex *)mkl_malloc(sizeof(TComplex)* n * n, ALIG);
    read_matrix_bin(file, node->matrix, n);
    node->next = 0;
}

split * create_struct_bin (FILE * file)
{
    split * head = (split *)mkl_malloc(sizeof(split), ALIG);
    unsigned int q_branch, n;
    double t;
    fread(&(t), sizeof(double), 1, file);
    fread(&(n), sizeof(int), 1, file);
    fread(&(q_branch), sizeof(int), 1, file);
    head->prev = 0;
    head->dt = t;
    head->counter = q_branch;
    head->n = n;
    head->next = (split *)mkl_malloc(sizeof(split) * q_branch, ALIG);
    for(unsigned int i = 0; i < q_branch; i++)
    {
        (head->next)[i].prev = head;
        read_branch_bin(file, &((head->next)[i]));
    }
    
    fread(&(head->steps), sizeof(int), 1, file);;
    head->matrix = (TComplex *)mkl_malloc(sizeof(TComplex) * head->steps * n * n, ALIG);
    head->g = (acc_number *)mkl_malloc(sizeof(acc_number) * head->steps, ALIG);
	double gtemp;
	for (unsigned int k = 0; k < head->steps; k++)
	{
		fread(&gtemp, sizeof(double), 1, file);
		head->g[k] = (acc_number)gtemp;
        read_matrix_bin(file, &(head->matrix[k * n * n]), n);
    }
    return head;
}



void read_branch_bin_MPI (FILE * file, split * branch, int numprocs, int myid)
{
    unsigned int deep, n;
    split * node = branch;

    if(myid == 0)
    {
        fread(&deep, sizeof(int), 1, file);
    }
    MPI_Bcast(&deep, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    n = (branch->prev)->n;
    for(unsigned int i = 0; i < deep - 1; i++)
    {
        if(myid == 0)
        {
            fread(&(node->dt), sizeof(double), 1, file);
            fread(&(node->counter), sizeof(int), 1, file);
			if(i != 0)
				node->counter = i;
        }
        MPI_Bcast(&(node->dt), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(node->counter), 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        node->steps = deep;
        node->g = 0;
        node->n = n;
        node->matrix = (TComplex *)mkl_malloc(sizeof(TComplex)* n * n, ALIG);

        if(myid == 0)
        {
            read_matrix_bin(file, node->matrix, n);
        }
        MPI_Bcast(node->matrix, n * n, MPI_complex_type, 0, MPI_COMM_WORLD);

        node->next = (split *)mkl_malloc(sizeof(split), ALIG);
        node->next->prev = node;
        node = node->next;
    }

    if(myid == 0)
    {
        fread(&(node->dt), sizeof(double), 1, file);
        fread(&(node->steps), sizeof(int), 1, file);
    }
    MPI_Bcast(&(node->dt), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    node->counter = deep - 1;
	node->steps = deep;
    node->g = 0;
    node->n = n;
    node->matrix =  (TComplex *)mkl_malloc(sizeof(TComplex)* n * n, ALIG);

    if(myid == 0)
    {
        read_matrix_bin(file, node->matrix, n);
    }
    MPI_Bcast(node->matrix, n * n, MPI_complex_type, 0, MPI_COMM_WORLD);

    node->next = 0;
}

split * create_struct_bin_MPI (FILE * file, int numprocs, int myid)
{
    split * head = (split *)mkl_malloc(sizeof(split), ALIG);
    unsigned int q_branch, n;
    double t;

    if(myid == 0)
    {
        fread(&(t), sizeof(double), 1, file);
        fread(&(n), sizeof(int), 1, file);
        fread(&(q_branch), sizeof(int), 1, file);
    }
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&q_branch, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    head->prev = 0;
    head->dt = t;
    head->counter = q_branch;
    head->n = n;
    head->next = (split *)mkl_malloc(sizeof(split) * q_branch, ALIG);
    for(unsigned int i = 0; i < q_branch; i++)
    {
        (head->next)[i].prev = head;
        read_branch_bin_MPI(file, &((head->next)[i]), numprocs, myid);
    }
    

    if(myid == 0)
    {
        fread(&(head->steps), sizeof(int), 1, file);;
    }
    MPI_Bcast(&(head->steps), 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    head->matrix = (TComplex *)mkl_malloc(sizeof(TComplex) * head->steps * n * n, ALIG);
    head->g = (acc_number *)mkl_malloc(sizeof(acc_number) * head->steps, ALIG);
	double gtemp;
    if(myid == 0)
    {
		for (unsigned int k = 0; k < head->steps; k++)
		{
			fread(&gtemp, sizeof(double), 1, file);
			head->g[k] = (acc_number)gtemp;
			read_matrix_bin(file, &(head->matrix[k * n * n]), n);
        }
    }
    MPI_Bcast(head->g, head->steps, MPI_number_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(head->matrix, head->steps * n * n, MPI_complex_type, 0, MPI_COMM_WORLD);

    return head;
}

void cmp_branch_not_member(split * src, split * dst)
{
    split * node1 = src;
    split * node2 = dst;
    while (node1->next)
    {
        node2->dt = node1->dt;
        node2->steps = node1->steps;
        node2->counter = node1->counter;
        node2->g = node1->g;
        node2->n = node1->n;
        node2->matrix = node1->matrix;
        node2->next = (split *)mkl_malloc(sizeof(split), ALIG);
        node2->next->prev = node2;
        node2 = node2->next;
        node1 = node1->next;
    }
    node2->dt = node1->dt;
    node2->steps = node1->steps;
    node2->counter = node1->counter;
    node2->g = node1->g;
    node2->n = node1->n;
    node2->matrix = node1->matrix;
    node2->next = 0;
}

void cmp_struct_not_member(split * src, split * dst)
{
    dst->prev = src->prev;
    dst->dt = src->dt;
    dst->counter = src->counter;
    dst->n = src->n;
    dst->next = (split *)mkl_malloc(sizeof(split) * dst->counter, ALIG);
    for(unsigned int i = 0; i < dst->counter; i++)
    {
        (dst->next)[i].prev = dst;
        cmp_branch_not_member(&((src->next)[i]), &((dst->next)[i]));
    }
    dst->steps = src->steps;
    dst->matrix = src->matrix;
    dst->g = src->g;
}

void delete_branch (split * branch)
{
    mkl_free(branch->matrix);
    branch->prev = 0;
    branch->matrix = 0;
    branch->steps = 0;
    branch->dt = 0;
    branch->counter = 0;
    branch->n = 0;

    branch = branch->next;
    while (branch->next->next == 0)
    {
        mkl_free(branch->matrix);
        branch->prev = 0;
        branch->matrix = 0;
        branch->steps = 0;
        branch->dt = 0;
        branch->counter = 0;
        branch->n = 0;

        branch = branch->next;
        mkl_free(branch->prev);
    }
    mkl_free(branch->matrix);
    branch->prev = 0;
    branch->matrix = 0;
    branch->steps = 0;
    branch->dt = 0;
    branch->counter = 0;
    branch->n = 0;

    mkl_free(branch);
}

void delete_split_struct (split * head)
{
    for(unsigned int i = 0; i < head->counter; i++)
    {
        delete_branch (&(head->next)[i]);
    }
    mkl_free(head->next);
    mkl_free(head->matrix);
    mkl_free(head->g);
}

void delete_branch_not_member (split * branch)
{
    branch->prev = 0;
    branch->matrix = 0;
    branch->steps = 0;
    branch->dt = 0;
    branch->counter = 0;
    branch->n = 0;

    branch = branch->next;
    while (branch->next->next == 0)
    {
        branch->prev = 0;
        branch->matrix = 0;
        branch->steps = 0;
        branch->dt = 0;
        branch->counter = 0;
        branch->n = 0;

        branch = branch->next;
        mkl_free(branch->prev);
    }
    branch->prev = 0;
    branch->matrix = 0;
    branch->steps = 0;
    branch->dt = 0;
    branch->counter = 0;
    branch->n = 0;

    mkl_free(branch);
}

void delete_split_struct_not_member (split * head)
{
    for(unsigned int i = 0; i < head->counter; i++)
    {
        delete_branch_not_member (&(head->next)[i]);
    }
    mkl_free(head->next);
    head->matrix = 0;
    head->g = 0;
}