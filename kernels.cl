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


kernel void lbm(global float* input_grid, global float* output_grid, global float* obstacles, global float* partial_sums, int it, global t_param* params)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  /*if(x == 12 & y == 72){
    printf("=======================\n");
    printf("cells test: %f %d\n", grid[L(x, y, 5, nx_pad)], L(x, y, 5, nx_pad));
    printf("tmp_cells test: %f %d\n", grid[L2(x, y, 5, nx_pad)], L2(x, y, 5, nx_pad));
    printf("NXPAD = %d, NX = %d, NY = %d\n", NXPAD, NX, NY);
  }*/
  //int offset = 4 + (2 * 9 * (params->nx_pad));
  //int offset = 4 + (9 * nx_pad);

   floatv u0_o = input_grid[L(x, y, 0, params->nx)];
   floatv u1_o = input_grid[L(x, y, 1, params->nx)];
   floatv u2_o = input_grid[L(x, y, 2, params->nx)];
   floatv u3_o = input_grid[L(x, y, 3, params->nx)];
   floatv u4_o = input_grid[L(x, y, 4, params->nx)];
   floatv u5_o = input_grid[L(x, y, 5, params->nx)];
   floatv u6_o = input_grid[L(x, y, 6, params->nx)];
   floatv u7_o = input_grid[L(x, y, 7, params->nx)];
   floatv u8_o = input_grid[L(x, y, 8, params->nx)];

   floatv o_mask2 = obstacles[y*params->nx+x];

  floatv xneg = u2_o + u5_o + u6_o;
  floatv xpos = u1_o + u3_o + u8_o;
  floatv yneg = u6_o + u7_o + u8_o;
  floatv ypos = u3_o + u4_o + u5_o;

  floatv local_density = u0_o + u1_o + u2_o + yneg + ypos;
  double u_x = native_divide((xpos - xneg),local_density);
  double u_y = native_divide((ypos - yneg),local_density);

  double u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

  /* equilibrium densities */
  double c_sq = 1.0 / 3.0;
  double d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
             * (1.0 - u_sq / (2.0 * c_sq));
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                                   + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                                   + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                                   + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                                   + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                                   + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                                   + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                                   + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));
  d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                                   + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                                   - u_sq / (2.0 * c_sq));

   /*if(it == 1){
    if(u0_o < 0){printf("(%d, %d)[0] < 0\n", x, y);}
    if(u1_o < 0){printf("(%d, %d)[1] < 0\n", x, y);}
    if(u2_o < 0){printf("(%d, %d)[2] < 0\n", x, y);}
    if(u3_o < 0){printf("(%d, %d)[3] < 0\n", x, y);}
    if(u4_o < 0){printf("(%d, %d)[4] < 0\n", x, y);}
    if(u5_o < 0){printf("(%d, %d)[5] < 0\n", x, y);}
    if(u6_o < 0){printf("(%d, %d)[6] < 0\n", x, y);}
    if(u7_o < 0){printf("(%d, %d)[7] < 0\n", x, y);}
    if(u8_o < 0){printf("(%d, %d)[8] < 0\n", x, y);}
   }*/

  
  

  /*if(x == 40 && y == 80){
    printf("u0=%f e0=%f\n", u0, e0);
  }*/
  u0 = (u0 + d_equ[0]);
  u1 = (u1 + d_equ[1]);
  u2 = (u2 + d_equ[2]);
  u3 = (u3 + d_equ[3]);
  u4 = (u4 + d_equ[4]);
  u5 = (u5 + d_equ[5]);
  u6 = (u6 + d_equ[6]);
  u7 = (u7 + d_equ[7]);
  u8 = (u8 + d_equ[8]);
  /*if(x == 40 && y == 80){
    printf("u0=%f e0=%f\n, u0_o=%f, omega=%f", u0, e0, u0_o, params->omega);
    //printf("(%d, %d): %f %f %f %f %f %f %f %f %f\n", x, y, e0, e1, e2, e3, e4, e5, e6, e7, e8);
  }*/
  /* End: Collision */

   /* Begin: Accelerate */
  float wt1, wt2;
  if(y == params->ny - 2){
    wt1 = native_divide(params->accel * params->density , 9.0f);
    wt2 = native_divide(params->accel * params->density , 36.0f);
    if(o_mask2 != 0.0f && u2 > wt1 && u5 > wt2 && u6 > wt2){
      
      u1 += wt1;
      u3 += wt2;
      u8 += wt2;

      u2 -= wt1;
      u5 -= wt2;
      u6 -= wt2;
    }
  }
  /* End: Accelerate */

  /* Begin: Rebound */
  if(o_mask2 == 0.0f){
    u0 = u0_o;
    u1 = u2_o;
    u2 = u1_o;
    u3 = u6_o;
    u4 = u7_o;
    u5 = u8_o;
    u6 = u3_o;
    u7 = u4_o;
    u8 = u5_o;
  }
  /* End: Rebound */
  
  /* Begin: Propogate */
  int e = (x+1)%params->nx;
  int w = (x==0)?params->nx-1:x-1;

  int n = (y+1)%params->ny;
  int s = (y==0)?params->ny-1:y-1;

  output_grid[L(x  , y  , 0, params->nx)] = u0; // Does not propogate
  output_grid[L(e  , y  , 1, params->nx)] = u1; // Does not propogate
  output_grid[L(w  , y  , 2, params->nx)] = u2; // Does not propogate
  output_grid[L(e  , n  , 3, params->nx)] = u3; // Does not propogate
  output_grid[L(x  , n  , 4, params->nx)] = u4; // Does not propogate
  output_grid[L(w  , n  , 5, params->nx)] = u5; // Does not propogate
  output_grid[L(w  , s  , 6, params->nx)] = u6; // Does not propogate
  output_grid[L(x  , s  , 7, params->nx)] = u7; // Does not propogate
  output_grid[L(e  , s  , 8, params->nx)] = u8; // Does not propogate

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
