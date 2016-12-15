/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
** ORIGINAL
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
** IMPROVED (opposing side always accessible with +/- 4)
** 1 8 7
**  \|/
** 2-0-6
**  /|\
** 3 4 5
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
**
** Grid strides are modified, now: (Major first)
** (Velocity, Y, X)
** This makes vectors a lot faster to load/store
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/
#define _POSIX_C_SOURCE (200112L)
#include<omp.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

#define VECTOR_SIZE 8
#define L(X, Y, V, NX) ((X) + ((V)+(Y)*18)*(NX)) //Requires array to be offset correctly, and X to be x+4, this ensures 32byte alignment for vector intrinics
#define _SCHEDULE_ schedule(static, 4)

#define START(RANK, SIZE, NY) ((RANK)*(NY)/(SIZE)+( (RANK)<((NY)%(SIZE))?(RANK):((NY)%(SIZE)) ))
#define END(RANK, SIZE, NY) (START((RANK), (SIZE), (NY)) + ((NY)/(SIZE)) - ( (RANK)<(NY)%(SIZE)?0:1 ))
#define CHUNK(RANK, SIZE, NY) ( (END(RANK, SIZE, NY)) - (START(RANK, SIZE, NY)) + 1 )

typedef float t_speed;
typedef float t_obstacle;
/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  t_speed density;       /* density per link */
  t_speed accel;         /* density redistribution */
  t_speed omega;         /* relaxation parameter */
  int nx_pad;
  int ny_pad;
  int rank;   /* Proc # */
  int size;   /* # of Procs */
  int available_cells;
  int north;
  int south;
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  t_speed speeds[NSPEEDS];
} t_cell;

typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  propagate;
  cl_kernel  lbm;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
} t_ocl;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               t_obstacle** obstacles_ptr, t_speed** av_vels_ptr, int* available_cells, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param* params, t_speed* cells, t_speed* tmp_cells, t_obstacle* obstacles, t_speed* av_vels, t_speed inverse_available_cells, t_ocl ocl);
int accelerate_flow(const t_param* params, t_speed* cells, t_obstacle* obstacles);
int propagate(const t_param* params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param* params, t_speed* tmp_cells, t_obstacle* obstacles);
//int collision(const t_param* params, t_cell* cells, t_cell* tmp_cells, t_obstacle* obstacles, t_speed* av_vels, int time, int available_cells);
int write_values(const t_param* params, t_speed* cells, t_obstacle* obstacles, t_speed* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             t_obstacle** obstacles_ptr, t_speed** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
t_speed total_density(const t_param* params, t_speed* cells);

/* compute average velocity */
t_speed av_velocity(const t_param* params, t_speed* cells, t_obstacle* obstacles);

/* calculate Reynolds number */
t_speed calc_reynolds(const t_param* params, t_speed* cells, t_obstacle* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  int provided, flag;
  t_param* params = (t_param*) malloc(sizeof(t_param));  /* struct to hold parameter values */

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_ocl ocl;
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  t_obstacle* obstacles = NULL;    /* grid indicating which cells are blocked */
  t_speed* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int available_cells = 0;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  
  // DOESN'T NEED MPI SPECIALISATION AS OUTSIDE TIMED REGION, WOULD BE BENEFICIAL OTHERWISE
  initialise(paramfile, obstaclefile, params, &cells, &tmp_cells, &obstacles, &av_vels, &available_cells, &ocl);
  params->available_cells = available_cells;
  t_speed *av_vels_recv = malloc(sizeof *av_vels_recv * (params->maxIters+1));
  //printf("available cells = %d\n", available_cells);
  int offset = 4 + (9 * (params->nx_pad));
  //printf("P%d sample cell value: %f\n", params->rank, cells[0 * params->ny * params->nx_pad + 5   * params->nx_pad + 5]);

  //Ensure same start state ()
  // DOESN'T NEED MPI SPECIALISATION AS OUTSIDE TIMED REGION, WOULD BE BENEFICIAL OTHERWISE
  accelerate_flow(params, cells+offset, obstacles);
  // DOESN'T NEED MPI SPECIALISATION AS OUTSIDE TIMED REGION, WOULD BE BENEFICIAL OTHERWISE
  propagate(params, cells+offset, tmp_cells+offset); //TODO: this function may not set up grid correctly as does start->end (inclusive), may require extra row each side


  /* iterate for maxIters timesteps */
  if(params->rank == 0){
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }

/*
**  BEGIN TIMING STEP
*/
 
  //Lattice-Bolzmann Iterations (this function contains the loop, no need to loop this call)
  timestep(params, cells+offset, tmp_cells+offset, obstacles, av_vels, 1.0/available_cells, ocl);

  //TODO: Pass chunks back to master from other nodes

/*
**  END TIMING STEP
*/

  if(params->rank == 0){
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  } 

  t_speed reynold_value;
  t_speed reynold_sum = calc_reynolds(params, cells+offset, obstacles);

  /* write final values and free memory */
  if(params->rank == 0){
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", reynold_sum); //TODO: make calc_reynolds MPI
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells+offset, obstacles, av_vels);
  }

  finalise(params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);
  return EXIT_SUCCESS;
}

/*int timestep(const t_param* params, t_cell* cells, t_cell* tmp_cells, int* obstacles, double* av_vels, int time, int available_cells)
{
  //accelerate_flow(params, cells, obstacles);
  //propagate(params, cells, tmp_cells); //requires all cells, updates all tmp_cells
  collision(params, cells, tmp_cells, obstacles, av_vels, time, available_cells);
  return EXIT_SUCCESS;
}*/

int accelerate_flow(const t_param* params, t_speed* cells, t_obstacle* obstacles)
{
  /* compute weighting factors */
  t_speed w1 = params->density * params->accel / 9.0;
  t_speed w2 = params->density * params->accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = params->ny - 2;

  for (int jj = 4; jj < params->nx + 4; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if ( ((int*)obstacles)[ii * params->nx + jj - 4] == -1
        && (cells[L(jj, ii, 2, params->nx_pad)] - w1) > 0.0
        && (cells[L(jj, ii, 5, params->nx_pad)] - w2) > 0.0
        && (cells[L(jj, ii, 6, params->nx_pad)] - w2) > 0.0)
    {
      /* decrease 'west-side' densities */
      cells[L(jj, ii, 2, params->nx_pad)] -= w1;
      cells[L(jj, ii, 5, params->nx_pad)] -= w2;
      cells[L(jj, ii, 6, params->nx_pad)] -= w2;
      /* increase 'east-side' densities */
      cells[L(jj, ii, 1, params->nx_pad)] += w1;
      cells[L(jj, ii, 3, params->nx_pad)] += w2;
      cells[L(jj, ii, 8, params->nx_pad)] += w2;
      
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param* params, t_speed* cells, t_speed* tmp_cells)
{
  for (int y = 0; y < params->ny; y++)
  {
    int s = y == 0 ? params->ny-1 : y-1;
    #pragma ivdep
    for (int x = 4; x < params->nx+4; x++)
    {
      tmp_cells[L(x  , y               , 0, params->nx_pad)] = cells[L(x  , y  , 0, params->nx_pad)];
      tmp_cells[L(x+1, y               , 1, params->nx_pad)] = cells[L(x  , y  , 1, params->nx_pad)];
      tmp_cells[L(x-1, y               , 2, params->nx_pad)] = cells[L(x  , y  , 2, params->nx_pad)];
      tmp_cells[L(x+1, (y+1)%params->ny, 3, params->nx_pad)] = cells[L(x  , y  , 3, params->nx_pad)];
      tmp_cells[L(x  , (y+1)%params->ny, 4, params->nx_pad)] = cells[L(x  , y  , 4, params->nx_pad)];
      tmp_cells[L(x-1, (y+1)%params->ny, 5, params->nx_pad)] = cells[L(x  , y  , 5, params->nx_pad)];
      tmp_cells[L(x-1, (s), 6, params->nx_pad)] = cells[L(x  , y  , 6, params->nx_pad)];
      tmp_cells[L(x  , (s), 7, params->nx_pad)] = cells[L(x  , y  , 7, params->nx_pad)];
      tmp_cells[L(x+1, (s), 8, params->nx_pad)] = cells[L(x  , y  , 8, params->nx_pad)];
    }
  }
  return EXIT_SUCCESS;
}

int rebound(const t_param* params, t_speed* tmp_cells, t_obstacle* obstacles)
{
  /* loop over the cells in the grid */
  for (int y = 0; y < params->ny; y++)
  {
    for (int x = 0; x < params->nx; x++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[y * params->nx + x])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        float t1 = tmp_cells[1*params->ny*params->nx_pad + y * params->nx_pad + x];
        float t2 = tmp_cells[2*params->ny*params->nx_pad + y * params->nx_pad + x];
        float t5 = tmp_cells[5*params->ny*params->nx_pad + y * params->nx_pad + x];
        float t6 = tmp_cells[6*params->ny*params->nx_pad + y * params->nx_pad + x];
        
        tmp_cells[1*params->ny*params->nx_pad + y * params->nx_pad + x] = tmp_cells[3*params->ny*params->nx_pad + y * params->nx_pad + x];
        tmp_cells[2*params->ny*params->nx_pad + y * params->nx_pad + x] = tmp_cells[4*params->ny*params->nx_pad + y * params->nx_pad + x];
        tmp_cells[3*params->ny*params->nx_pad + y * params->nx_pad + x] = t1;
        tmp_cells[4*params->ny*params->nx_pad + y * params->nx_pad + x] = t2;
        
        tmp_cells[5*params->ny*params->nx_pad + y * params->nx_pad + x] = tmp_cells[7*params->ny*params->nx_pad + y * params->nx_pad + x];
        tmp_cells[6*params->ny*params->nx_pad + y * params->nx_pad + x] = tmp_cells[8*params->ny*params->nx_pad + y * params->nx_pad + x];
        tmp_cells[7*params->ny*params->nx_pad + y * params->nx_pad + x] = t5;
        tmp_cells[8*params->ny*params->nx_pad + y * params->nx_pad + x] = t6;
      }
    }
  }

  return EXIT_SUCCESS;
}

int timestep(const t_param* restrict params, t_speed* restrict cells, t_speed* restrict tmp_cells, t_obstacle* restrict obstacles, t_speed* restrict av_vels, t_speed inverse_available_cells, t_ocl ocl)
{
  cl_int err;
  //Have loop inside kernel
  err = clSetKernelArg(ocl.lbm, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting lbm arg 0", __LINE__);
  err = clSetKernelArg(ocl.lbm, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting lbm arg 1", __LINE__);
  err = clSetKernelArg(ocl.lbm, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting lbm arg 2", __LINE__);
  err = clSetKernelArg(ocl.lbm, 3, sizeof(cl_int), &(params->nx));
  checkError(err, "setting lbm arg 3", __LINE__);
  err = clSetKernelArg(ocl.lbm, 4, sizeof(cl_int), &(params->ny));
  checkError(err, "setting lbm arg 4", __LINE__);
  err = clSetKernelArg(ocl.lbm, 5, sizeof(cl_float), &inverse_available_cells);
  checkError(err, "setting lbm arg 5", __LINE__);
  err = clSetKernelArg(ocl.lbm, 6, sizeof(cl_float), &(params->density) );
  checkError(err, "setting lbm arg 6", __LINE__);
  err = clSetKernelArg(ocl.lbm, 7, sizeof(cl_float), &(params->accel) );
  checkError(err, "setting lbm arg 7", __LINE__);

  size_t global[2] = {params->nx, params->ny};//maybe divide nx by vectorsize
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.lbm, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueuing lbm kernel", __LINE__);

  err = clFinish(ocl.queue);
  checkError(err, "waiting for lbm kernel", __LINE__);




  return EXIT_SUCCESS;
  //const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const t_speed w0 = 4.0 / 9.0;  /* weighting factor */
  const t_speed w1 = 1.0 / 9.0;  /* weighting factor */
  const t_speed w2 = 1.0 / 36.0; /* weighting factor */
  
  //t_speed u[4]; /* directional velocity components */
  t_speed ux, uy;
  /* equilibrium densities */
  //t_speed d_equ[NSPEEDS];
  t_speed local_density;

  t_speed tot_u = 0.0;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

  /* Efficient Paralisation Rough Work
    Node Layout
    C=CPU (2 cores each, Intel SandyBridge E5-2670, 2.6GHz)
    [C][C]---[C][C]
    [C][C]---[C][C]
    [C][C]RAM[C][C]
    [C][C]---[C][C]
    Main speedups will come from fewer cache-misses and false-sharing
  */

  
}

t_speed av_velocity(const t_param* params, t_speed* cells, t_obstacle* obstacles)
{
  t_speed tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;
  const int start = START(params->rank, params->size, params->ny);
  const int end = END(params->rank, params->size, params->ny);

  /* loop over all non-blocked cells */
  for (int ii = start; ii <= end; ii++)
  {
    cells[L(4, ii, 1, params->nx_pad)] = cells[L(params->nx+4, ii, 1, params->nx_pad)];
    cells[L(4, ii, 3, params->nx_pad)] = cells[L(params->nx+4, ii, 3, params->nx_pad)];
    cells[L(4, ii, 8, params->nx_pad)] = cells[L(params->nx+4, ii, 8, params->nx_pad)];
    
    cells[L(params->nx+3, ii, 2, params->nx_pad)] = cells[L(3, ii, 2, params->nx_pad)];
    cells[L(params->nx+3, ii, 5, params->nx_pad)] = cells[L(3, ii, 5, params->nx_pad)];
    cells[L(params->nx+3, ii, 6, params->nx_pad)] = cells[L(3, ii, 6, params->nx_pad)];

    for (int jj = 4; jj < params->nx + 4; jj++)
    {
      /* ignore occupied cells */
      if (((int*)obstacles)[ii * params->nx + jj - 4] == -1)
      {
        /* local density total */
        t_speed local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[L(jj, ii, kk, params->nx_pad)];
        }
        if(local_density == 0){
          printf("Local density = 0\n");
        }
        t_speed inverse_local_density = 1/local_density;

        /* x-component of velocity */
        t_speed u_x = cells[L(jj, ii, 1, params->nx_pad)]
                      + cells[L(jj, ii, 3, params->nx_pad)]
                      + cells[L(jj, ii, 8, params->nx_pad)]
                      - (cells[L(jj, ii, 2, params->nx_pad)]
                         + cells[L(jj, ii, 5, params->nx_pad)]
                         + cells[L(jj, ii, 6, params->nx_pad)]);

        /* compute y velocity component */
        t_speed u_y = cells[L(jj, ii, 3, params->nx_pad)]
                      + cells[L(jj, ii, 4, params->nx_pad)]
                      + cells[L(jj, ii, 5, params->nx_pad)]
                      - (cells[L(jj, ii, 6, params->nx_pad)]
                         + cells[L(jj, ii, 7, params->nx_pad)]
                         + cells[L(jj, ii, 8, params->nx_pad)]);
        u_x *= inverse_local_density;
        u_y *= inverse_local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y)); //TODO: Expensive line
        /* increase counter of inspected cells */
      }
    }
  }

  return tot_u / (t_speed)(params->available_cells);
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               t_obstacle** obstacles_ptr, t_speed** av_vels_ptr, int* available_cells, t_ocl* ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  double temp;
  retval = fscanf(fp, "%lf\n", &temp);
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  params->density = (t_speed) temp;

  retval = fscanf(fp, "%lf\n", &temp);
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  params->accel = (t_speed) temp;

  retval = fscanf(fp, "%lf\n", &temp);
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
  params->omega = (t_speed) temp;
  //static const int vcopy[9][2] = { {0, 0}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}};
  //memcpy( params->v, vcopy, sizeof(vcopy) );

  /* and close up the file */
  //fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  //*cells_ptr = (t_cell*)malloc(sizeof(t_cell) * (params->ny * params->nx));

  //Array stride (largest first): velocity, y, x
  //Number of x needs to be padded so we always get 32 byte boundaries
  //Space for 4 floats before and after each row
  params->nx_pad = params->nx + 8; //for ghost columns (bigger than necessary but provides alignment)
  params->ny_pad = params->ny + 2; //for ghost rows

  //MUST START 16 bytes before boundary
  posix_memalign((void**) cells_ptr, 32, sizeof(t_speed) * (NSPEEDS * ((params->ny_pad) * (params->nx_pad)) * 2 + 4) );

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  //*tmp_cells_ptr = (t_cell*)malloc(sizeof(t_cell) * (params->ny * params->nx));
  //posix_memalign((void**) tmp_cells_ptr, 32, sizeof(t_speed) * (NSPEEDS * ((params->ny_pad) * params->nx_pad) + 4) );
  *tmp_cells_ptr = *cells_ptr + (9*params->nx_pad);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(t_obstacle) * ((params->ny) * params->nx));
  //posix_memalign((void**) obstacles_ptr, 16, sizeof(t_cell) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  t_speed w0 = params->density * 4.0 / 9.0;
  t_speed w1 = params->density      / 9.0;
  t_speed w2 = params->density      / 36.0;

  int offset = 4 + (9 * (params->nx_pad));

  #pragma omp parallel
  for (int ii = 0; ii < params->ny; ii++)
  {
    #pragma omp for simd _SCHEDULE_ //Using same scheduling as timestep for first-touch NUMA policy
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[L(jj+4, ii, 0, params->nx_pad) + offset] = w0;
      /* axis directions */
      (*cells_ptr)[L(jj+4, ii, 1, params->nx_pad) + offset] = w1;
      (*cells_ptr)[L(jj+4, ii, 2, params->nx_pad) + offset] = w1;
      (*cells_ptr)[L(jj+4, ii, 4, params->nx_pad) + offset] = w1;
      (*cells_ptr)[L(jj+4, ii, 7, params->nx_pad) + offset] = w1;
      /* diagonals */
      (*cells_ptr)[L(jj+4, ii, 3, params->nx_pad) + offset] = w2;
      (*cells_ptr)[L(jj+4, ii, 5, params->nx_pad) + offset] = w2;
      (*cells_ptr)[L(jj+4, ii, 6, params->nx_pad) + offset] = w2;
      (*cells_ptr)[L(jj+4, ii, 8, params->nx_pad) + offset] = w2;

      (*tmp_cells_ptr)[L(jj+4, ii, 0, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 1, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 2, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 3, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 4, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 5, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 6, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 7, params->nx_pad) + offset] = 0;
      (*tmp_cells_ptr)[L(jj+4, ii, 8, params->nx_pad) + offset] = 0;
    }
  }

  /* first set all cells in obstacle array to zero */
  int clear_mask = -1;
  #pragma omp parallel
  for (int ii = 0; ii < params->ny; ii++)
  {
    #pragma omp for simd _SCHEDULE_
    for (int jj = 0; jj < params->nx; jj++)
    {
      *(int*)&(*obstacles_ptr)[ii * params->nx + jj] = clear_mask; //Sets all bits to 1, looks messy but should work
    }
  }
  (*available_cells) = params->nx * params->ny;

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    if(blocked){
      (*obstacles_ptr)[yy * params->nx + xx] = 0.0;
    }
  }

  for(int x = 0; x < params->nx; ++x){
    for(int y = 0; y < params->ny; ++y){
      if((*obstacles_ptr)[y * params->nx +x] == 0.0){
        (*available_cells)--;
      }
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->maxIters+1));
  //posix_memalign((void**) av_vels_ptr, 16, sizeof(t_cell) * (params->ny * params->nx));

  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
  checkError(err, "creating propagate kernel", __LINE__);
  ocl->lbm = clCreateKernel(ocl->program, "lbm", &err);
  checkError(err, "creating lbm kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             t_obstacle** obstacles_ptr, t_speed** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  //free(*tmp_cells_ptr);
  //*tmp_cells_ptr = NULL;

  //free(*cells_ptr);
  //*cells_ptr = NULL;

  

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.propagate);
  clReleaseKernel(ocl.lbm);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


t_speed calc_reynolds(const t_param* params, t_speed* cells, t_obstacle* obstacles)
{
  const t_speed viscosity = 1.0 / 6.0 * (2.0 / params->omega - 1.0);

  return av_velocity(params, cells, obstacles) * params->reynolds_dim / viscosity;
}

t_speed total_density(const t_param* params, t_speed* cells)
{
  t_speed total = 0.0;  /* accumulator */

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[kk*params->ny*params->nx_pad + ii * params->nx_pad + jj];
      }
    }
  }

  return total;
}

int write_values(const t_param* params, t_speed* cells, t_obstacle* obstacles, t_speed* av_vels)
{
  FILE* fp;                     /* file pointer */
  const t_speed c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  t_speed local_density;         /* per grid cell sum of densities */
  t_speed pressure;              /* fluid pressure in grid cell */
  t_speed u_x;                   /* x-component of velocity in grid cell */
  t_speed u_y;                   /* y-component of velocity in grid cell */
  t_speed u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 4; jj < params->nx + 4; jj++)
    {
      /* an occupied cell */
      if ( ((int*)obstacles)[ii * params->nx + jj - 4] == 0 )
      {
        u_x = u_y = u = 0.0;
        pressure = params->density * c_sq;
      }
      /* no obstacle */
      else
      {
        /* local density total */
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[L(jj, ii, kk, params->nx_pad)];
        }
        t_speed inverse_local_density = 1/local_density;

        /* x-component of velocity */
        t_speed u_x = cells[L(jj, ii, 1, params->nx_pad)]
                      + cells[L(jj, ii, 3, params->nx_pad)]
                      + cells[L(jj, ii, 8, params->nx_pad)]
                      - (cells[L(jj, ii, 2, params->nx_pad)]
                         + cells[L(jj, ii, 5, params->nx_pad)]
                         + cells[L(jj, ii, 6, params->nx_pad)]);

        /* compute y velocity component */
        t_speed u_y = cells[L(jj, ii, 3, params->nx_pad)]
                      + cells[L(jj, ii, 4, params->nx_pad)]
                      + cells[L(jj, ii, 5, params->nx_pad)]
                      - (cells[L(jj, ii, 6, params->nx_pad)]
                         + cells[L(jj, ii, 7, params->nx_pad)]
                         + cells[L(jj, ii, 8, params->nx_pad)]);
        u_x *= inverse_local_density;
        u_y *= inverse_local_density;
        /* accumulate the norm of x- and y- velocity components */
        u = sqrt((u_x * u_x) + (u_y * u_y)); //TODO: Expensive line
        pressure = local_density * c_sq;
        /* increase counter of inspected cells */
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj-4, ii, u_x, u_y, u, pressure, -((int*)obstacles)[ii * params->nx + jj - 4]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params->maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}