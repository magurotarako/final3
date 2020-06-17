#include <mpi.h>
#include <cstdio>
#include <unistd.h>

int main(int argc, char**argv){
  MPI_Init(&argc, &argv);
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("%02d / %02d at %s\n",rank, procs, hostname);
  sleep(10);
  MPI_Finalize();
}
