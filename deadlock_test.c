#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
    int my_rank;
    int p;
    int source;
    int dest;
    int tag=0;
    char message[10000];
    int i;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &p);

    for(i=0;i<10000;i++){
        if(my_rank==0){
            message[i]='a';
            printf("\n Size of data in buffer = %ld",sizeof(char)*(i+1) + 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(my_rank == 0)
        {
            MPI_Send(message, i+1, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(message, 10000, MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status);
        }
        else
        if(my_rank == 1)
        {
            MPI_Send(message, i+1, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
            MPI_Recv(message, 10000, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Finalize();

    return 0;
}