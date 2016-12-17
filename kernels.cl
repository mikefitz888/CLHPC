#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
//int offset = 4 + (9 * (params->nx_pad));
//#define L(X, Y, V, NX) (4+(X) + ((V)+(Y)*18+18)*(NX)) //Offsets built in
#define L(X, Y, V, NW) ((X) + ((V)+(Y)*9)*(NW))
#define L2(X, Y, V, NX) (4+(X) + ((V)+(Y)*18+27)*(NX))
#define VEC_SIZE 8
#define floatv float
#define int int
//#define VEC_LOAD(ADDR) (*(ADDR))//vload8(0, (ADDR))
//#define VEC_STORE(ADDR, DATA) *(ADDR)=DATA//vstore8((DATA), 0, ADDR)

typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int nx_pad;
  int ny_pad;
  int available_cells;
} t_param;

//Utility constants
constant floatv Vo25 = (floatv)(0.25f);
constant floatv V1o5 = (floatv)(1.5f);
constant floatv V2   = (floatv)(2.0f);
constant floatv V3   = (floatv)(3.0f);
constant floatv V4o5 = (floatv)(4.5f);

constant float w0 = 4.0f / 9.0f;  /* weighting factor */
constant float w1 = 1.0f / 9.0f;  /* weighting factor */
constant float w2 = 1.0f / 36.0f; /* weighting factor */

//Command line constants
constant float inverse_available_cells = INV_CELL_COUNT;
constant float acceleration = ACCEL;
constant float density = DENSITY;
constant float omega = OMEGA;
constant int nx = NX;
constant int ny = NY;
constant int nx_pad = NXPAD;

//Pre-calculated vector constants
constant floatv Vnomega = (floatv)(1-OMEGA);
constant floatv start_weight = (floatv)(STARTW);


typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

kernel void swapGhostCellsLR(global float* grid, int temp){
  int y = get_global_id(0);
  if(temp){
    grid[L2(4, y, 1, NXPAD)] = grid[L2(NX+4, y, 1, NXPAD)];
    grid[L2(4, y, 3, NXPAD)] = grid[L2(NX+4, y, 3, NXPAD)];
    grid[L2(4, y, 8, NXPAD)] = grid[L2(NX+4, y, 8, NXPAD)];
  
    grid[L2(NX+3, y, 2, NXPAD)] = grid[L2(3, y, 2, NXPAD)];
    grid[L2(NX+3, y, 5, NXPAD)] = grid[L2(3, y, 5, NXPAD)];
    grid[L2(NX+3, y, 6, NXPAD)] = grid[L2(3, y, 6, NXPAD)];
  }else{
    grid[L(4, y, 1, NXPAD)] = grid[L(NX+4, y, 1, NXPAD)];
    grid[L(4, y, 3, NXPAD)] = grid[L(NX+4, y, 3, NXPAD)];
    grid[L(4, y, 8, NXPAD)] = grid[L(NX+4, y, 8, NXPAD)];
  
    grid[L(NX+3, y, 2, NXPAD)] = grid[L(3, y, 2, NXPAD)];
    grid[L(NX+3, y, 5, NXPAD)] = grid[L(3, y, 5, NXPAD)];
    grid[L(NX+3, y, 6, NXPAD)] = grid[L(3, y, 6, NXPAD)];
  }
}

kernel void swapGhostCellsTB(global float* grid, int temp){
  int x = get_global_id(0);
  if(temp){
    grid[L2(x, 0, 3, NXPAD)] = grid[L2(x, NY, 3, NXPAD)];
    grid[L2(x, 0, 4, NXPAD)] = grid[L2(x, NY, 4, NXPAD)];
    grid[L2(x, 0, 5, NXPAD)] = grid[L2(x, NY, 5, NXPAD)];
    
    grid[L2(x, NY-1, 6, NXPAD)] = grid[L2(x, -1, 6, NXPAD)];
    grid[L2(x, NY-1, 7, NXPAD)] = grid[L2(x, -1, 7, NXPAD)];
    grid[L2(x, NY-1, 8, NXPAD)] = grid[L2(x, -1, 8, NXPAD)];
  }else{
    grid[L(x, 0, 3, NXPAD)] = grid[L(x, NY, 3, NXPAD)];
    grid[L(x, 0, 4, NXPAD)] = grid[L(x, NY, 4, NXPAD)];
    grid[L(x, 0, 5, NXPAD)] = grid[L(x, NY, 5, NXPAD)];
    
    grid[L(x, NY-1, 6, NXPAD)] = grid[L(x, -1, 6, NXPAD)];
    grid[L(x, NY-1, 7, NXPAD)] = grid[L(x, -1, 7, NXPAD)];
    grid[L(x, NY-1, 8, NXPAD)] = grid[L(x, -1, 8, NXPAD)];
  }
}

kernel void reduce(int w, global float* partial_sums){
  int x = get_global_id(0);
  int y = get_global_id(1);

  partial_sums[y*w+x] = 0;
}


kernel void lbm(global float* input_grid, global float* output_grid, global float* obstacles, global float* partial_sums, int it, global t_param* params)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  int y_n = (y + 1) % params->ny;
  int y_s = y - 1; if(y_s < 0) y_s = params->ny-1;
  int x_e = (x + 1) % params->nx;
  int x_w = x - 1; if(x_w < 0) x_w = params->nx-1;

  if(obstacles[y*params->nx+x] != 0){
    float xneg = input_grid[L(x, y, 2, params->nx)] + input_grid[L(x, y, 6, params->nx)] + input_grid[L(x, y, 5, params->nx)];
    float yneg = input_grid[L(x, y, 6, params->nx)] + input_grid[L(x, y, 7, params->nx)] + input_grid[L(x, y, 8, params->nx)];
    float xpos = input_grid[L(x, y, 1, params->nx)] + input_grid[L(x, y, 3, params->nx)] + input_grid[L(x, y, 8, params->nx)];
    float ypos = input_grid[L(x, y, 4, params->nx)] + input_grid[L(x, y, 3, params->nx)] + input_grid[L(x, y, 5, params->nx)];
    float local_density = ypos + yneg + input_grid[L(x, y, 0, params->nx)] + input_grid[L(x, y, 1, params->nx)] + input_grid[L(x, y, 2, params->nx)];
    float inverse_local_density = 1.0/local_density;

    float ux = xpos - xneg;
          ux*= inverse_local_density;
    float uy = ypos - yneg;
          uy*= inverse_local_density;

    float uxsq = ux*ux;
    float uysq = uy*uy;
    //sum = sqrt(uxsq + uysq);

    float ux3 = 3.0 * ux;
    float uy3 = 3.0 * uy;

    float uxsq3 = 3.0 * uxsq;
    float uxsq15= 1.5 * uxsq;

    float uysq3 = 3.0 * uysq;
    float uysq15= 1.5 * uysq;

    float u_sq = uxsq15 + uysq15;

    float leading_diag  = 4.5 * (ux-uy)*(ux-uy); // = 4.5*(x-y)^2 == 4.5*(y-x)^2
    float trailing_diag = 4.5 * (ux+uy)*(ux+uy);

    local_density *= (4.0 / 9.0) * params->omega;

    output_grid[L(x, y, 0, params->nx)] = input_grid[L(x, y, 0, params->nx)] * (1 - params->omega) + (local_density + local_density * (-uxsq15 - uysq15));


    local_density *= 0.25;
    output_grid[L(x_w, y, 2, params->nx)] = input_grid[L(x, y, 2, params->nx)] * (1 - params->omega) + (local_density + local_density * (-ux3 + uxsq3 - uysq15));
    output_grid[L(x, y_s, 7, params->nx)] = input_grid[L(x, y, 7, params->nx)] * (1 - params->omega) + (local_density + local_density * (-uy3 + uysq3 - uxsq15));
    output_grid[L(x_e, y, 1, params->nx)] = input_grid[L(x, y, 1, params->nx)] * (1 - params->omega) + (local_density + local_density * ( ux3 + uxsq3 - uysq15));
    output_grid[L(x, y_n, 4, params->nx)] = input_grid[L(x, y, 4, params->nx)] * (1 - params->omega) + (local_density + local_density * ( uy3 + uysq3 - uxsq15));

    local_density *= 0.25;
    output_grid[L(x_w, y_n, 5, params->nx)] = input_grid[L(x, y, 5, params->nx)] * (1 - params->omega) + (local_density + local_density * (-ux3 + uy3 + leading_diag  -u_sq));
    output_grid[L(x_w, y_s, 6, params->nx)] = input_grid[L(x, y, 6, params->nx)] * (1 - params->omega) + (local_density + local_density * (-ux3 - uy3 + trailing_diag -u_sq));
    output_grid[L(x_e, y_s, 8, params->nx)] = input_grid[L(x, y, 8, params->nx)] * (1 - params->omega) + (local_density + local_density * ( ux3 - uy3 + leading_diag  -u_sq));
    output_grid[L(x_e, y_n, 3, params->nx)] = input_grid[L(x, y, 3, params->nx)] * (1 - params->omega) + (local_density + local_density * ( ux3 + uy3 + trailing_diag -u_sq));
    if(y == params->ny - 2){
      float m1 = params->density * params->accel / 9.0;
      float m2 = params->density * params->accel / 36.0;
      if (output_grid[L(x_w, y_n, 5, params->nx)] > m2
          && output_grid[L(x_w, y, 2, params->nx)] > m1
          && output_grid[L(x_w, y_s, 6, params->nx)] > m2)
      {
        output_grid[L(x_w, y_n, 5, params->nx)] -= m2;
        output_grid[L(x_w, y, 2, params->nx)] -= m1;
        output_grid[L(x_w, y_s, 6, params->nx)] -= m2;

        output_grid[L(x_e, y_n, 3, params->nx)] += m2;
        output_grid[L(x_e, y, 1, params->nx)] += m1;
        output_grid[L(x_e, y_s, 8, params->nx)] += m2; 
      }
    }
  }else{
    output_grid[L(x_e, y, 1, params->nx)] = input_grid[L(x, y, 2, params->nx)];
    output_grid[L(x_w, y, 2, params->nx)] = input_grid[L(x, y, 1, params->nx)];
    output_grid[L(x_e, y_n, 3, params->nx)] = input_grid[L(x, y, 6, params->nx)];
    output_grid[L(x, y_n, 4, params->nx)] = input_grid[L(x, y, 7, params->nx)];
    output_grid[L(x_w, y_n, 5, params->nx)] = input_grid[L(x, y, 8, params->nx)];
    output_grid[L(x_w, y_s, 6, params->nx)] = input_grid[L(x, y, 3, params->nx)];
    output_grid[L(x, y_s, 7, params->nx)] = input_grid[L(x, y, 4, params->nx)];
    output_grid[L(x_e, y_s, 8, params->nx)] = input_grid[L(x, y, 5, params->nx)]; 
  }
}


















kernel void accelerate_flow(global t_speed* othercells,
                            global int* obstacles,
                            int nx, int ny,
                            double density, double accel)
{
  /* compute weighting factors */
  double w1 = density * accel / 9.0;
  double w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (othercells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (othercells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (othercells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    othercells[ii * nx + jj].speeds[1] += w1;
    othercells[ii * nx + jj].speeds[5] += w2;
    othercells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    othercells[ii * nx + jj].speeds[3] -= w1;
    othercells[ii * nx + jj].speeds[6] -= w2;
    othercells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* othercells,
                      global t_speed* tmp_othercells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring othercells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_othercells[ii  * nx + jj ].speeds[0] = othercells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_othercells[ii  * nx + x_e].speeds[1] = othercells[ii * nx + jj].speeds[1]; /* east */
  tmp_othercells[y_n * nx + jj ].speeds[2] = othercells[ii * nx + jj].speeds[2]; /* north */
  tmp_othercells[ii  * nx + x_w].speeds[3] = othercells[ii * nx + jj].speeds[3]; /* west */
  tmp_othercells[y_s * nx + jj ].speeds[4] = othercells[ii * nx + jj].speeds[4]; /* south */
  tmp_othercells[y_n * nx + x_e].speeds[5] = othercells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_othercells[y_n * nx + x_w].speeds[6] = othercells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_othercells[y_s * nx + x_w].speeds[7] = othercells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_othercells[y_s * nx + x_e].speeds[8] = othercells[ii * nx + jj].speeds[8]; /* south-east */
}
