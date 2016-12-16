#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
//int offset = 4 + (9 * (params->nx_pad));
//#define L(X, Y, V, NX) (4+(X) + ((V)+(Y)*18+18)*(NX)) //Offsets built in
#define L(X, Y, V, NX) ((X) + ((V)+(Y)*9)*(NX))
#define L2(X, Y, V, NX) (4+(X) + ((V)+(Y)*18+27)*(NX))
#define VEC_SIZE 8
#define floatv float
#define int int
#define VEC_LOAD(ADDR) (*(ADDR))//vload8(0, (ADDR))
#define VEC_STORE(ADDR, DATA) *(ADDR)=DATA//vstore8((DATA), 0, ADDR)

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


kernel void lbm(global float* input_grid, global float* output_grid, global float* obstacles, global float* partial_sums, int it)
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

   floatv u0_o = input_grid[L(x, y, 0, NX)];
   floatv u1_o = input_grid[L(x, y, 1, NX)];
   floatv u2_o = input_grid[L(x, y, 2, NX)];
   floatv u3_o = input_grid[L(x, y, 3, NX)];
   floatv u4_o = input_grid[L(x, y, 4, NX)];
   floatv u5_o = input_grid[L(x, y, 5, NX)];
   floatv u6_o = input_grid[L(x, y, 6, NX)];
   floatv u7_o = input_grid[L(x, y, 7, NX)];
   floatv u8_o = input_grid[L(x, y, 8, NX)];

  floatv o_mask2 = obstacles[y*NX+x];

  floatv xneg = u2_o + u5_o + u6_o;
  floatv xpos = u1_o + u3_o + u8_o;
  floatv yneg = u6_o + u7_o + u8_o;
  floatv ypos = u3_o + u4_o + u5_o;

  floatv density = u0_o + u1_o + u2_o + yneg + ypos;
  /*if(density <= 0){
    printf("Density is negative: %d, x = %d, y = %d\n", it, x-4, y);
    printf("%f %f %f %f %f %f %f %f %f", u0_o, u1_o, u2_o, u3_o, u4_o, u5_o, u6_o, u7_o, u8_o);
  }*/
  xpos = (xpos - xneg)/density;
  ypos = (ypos - yneg)/density;
  
  floatv x_sq = xpos*xpos;
  floatv y_sq = ypos*ypos;

  /*floatv sum =  sqrt(x_sq + y_sq) * o_mask2; //Ignore obstacles in the summation

  //float tot_u = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7 + sum.s8 + sum.s9 + sum.s10 + sum.s11 + sum.s12 + sum.s13 + sum.s14 + sum.s15;
  float tot_u = dot(sum.s0123, (float4)(1));
  tot_u += dot(sum.s4567, (float4)(1));
  //TODO: division
  partial_sums[y*NX+get_global_id(0)] = tot_u;*/

  floatv u0 = (u0_o * Vnomega);
  floatv u1 = (u1_o * Vnomega);
  floatv u2 = (u2_o * Vnomega);
  floatv u3 = (u3_o * Vnomega);
  floatv u4 = (u4_o * Vnomega);
  floatv u5 = (u5_o * Vnomega);
  floatv u6 = (u6_o * Vnomega);
  floatv u7 = (u7_o * Vnomega);
  floatv u8 = (u8_o * Vnomega);

  floatv ux3 = (V3 * xpos);
  floatv uy3 = (V3 * ypos);


  floatv uxsq3 = (V3 * x_sq);
  floatv uysq3 = (V3 * y_sq);


  floatv uxsq15 = (V1o5 * x_sq);
  floatv uysq15 = (V1o5 * y_sq);

  floatv u_sq = (uxsq15 + uysq15);

  floatv leading_diag  = (V4o5*((xpos - ypos) * (xpos - ypos)));
  floatv trailing_diag = (V4o5*((xpos + ypos) * (xpos + ypos)));

  //The general equation to follow (slightly optimized) is: u_next = u * (1 - omega) + (density + density * (3u + 4.5u^2 - 1.5(ux^2 + uy^2))) * w * omega
  density = (density * start_weight); //density *= w0 * omega

  floatv e0 = (density - (density * (uxsq15 + uysq15)));

  //Axis
  density = (density * Vo25);
  floatv px = density * ux3 * V2;
  floatv py = density * uy3 * V2;
  floatv e1 = (density + (density * ((ux3 + uxsq3) - uysq15))); //East
  floatv e4 = (density + (density * ((uy3 + uysq3) - uxsq15))); //North
  floatv e2 = (e1 - px); //West
  floatv e7 = (e4 - py); //South

  //Diagonals
  density = (density * Vo25);
  px = (px * Vo25);
  py = (py * Vo25);

  floatv e3 = (density + (density * ((trailing_diag + (ux3 + uy3)) - u_sq)));
  floatv e5 = (density + (density * ((leading_diag + uy3) - (ux3 + u_sq) )));
  floatv e6 = ((e3 - px) - py);
  floatv e8 = ((e5 + px) - py);

  u0 = (u0 + e0);
  u1 = (u1 + e1);
  u2 = (u2 + e2);
  u3 = (u3 + e3);
  u4 = (u4 + e4);
  u5 = (u5 + e5);
  u6 = (u6 + e6);
  u7 = (u7 + e7);
  u8 = (u8 + e8);
  /* End: Collision */

  /* Add Acceleration */
  if(o_mask2 > 0){
    u1 = u1 + (floatv)(1);
    u3 = u3 + (floatv)(1);
    u8 = u8 + (floatv)(1);
  }

  /* Begin: Rebound: openCL mix */
  /*u0 = mix(u0_o, u0, o_mask2); //zero where obstacle
  u1 = mix(u2_o, u1, o_mask2);
  u2 = mix(u1_o, u2, o_mask2);
  u3 = mix(u6_o, u3, o_mask2);
  u4 = mix(u7_o, u4, o_mask2);
  u5 = mix(u8_o, u5, o_mask2);
  u6 = mix(u3_o, u6, o_mask2);
  u7 = mix(u4_o, u7, o_mask2);
  u8 = mix(u5_o, u8, o_mask2);*/
  if(o_mask2 == 0){
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
  /* None of these swap nodes as y != end && y != start */
  int e = (x+1)%NX;
  int w = (x==0)?NX-1:x-1;

  int n = (y+1)%NY;
  int s = (y==0)?NY-1:y-1;

  output_grid[L(x  , y  , 0, NX)] = u0; // Does not propogate
  output_grid[L(e  , y  , 1, NX)] = u1; // Does not propogate
  output_grid[L(w  , y  , 2, NX)] = u2; // Does not propogate
  output_grid[L(e  , n  , 3, NX)] = u3; // Does not propogate
  output_grid[L(x  , n  , 4, NX)] = u4; // Does not propogate
  output_grid[L(w  , n  , 5, NX)] = u5; // Does not propogate
  output_grid[L(w  , s  , 6, NX)] = u6; // Does not propogate
  output_grid[L(x  , s  , 7, NX)] = u7; // Does not propogate
  output_grid[L(e  , s  , 8, NX)] = u8; // Does not propogate
    

    /*VEC_STORE(&output_grid[L(x  , y  , 0, nx)], u0); // Does not propogate
    VEC_STORE(&output_grid[L((x+1)%(NX+4), y  , 1, nx)], u1);
    VEC_STORE(&output_grid[L((x==4)?NX+3:x-1, y  , 2, nx)], u2);
    VEC_STORE(&output_grid[L((x+1)%(NX+4), y+1, 3, nx)], u3);
    VEC_STORE(&output_grid[L(x  , y+1, 4, nx)], u4);
    VEC_STORE(&output_grid[L((x==4)?NX+3:x-1, y+1, 5, nx)], u5);
    VEC_STORE(&output_grid[L((x==4)?NX+3:x-1, (y==0)?NY-1:y-1, 6, nx)], u6);
    VEC_STORE(&output_grid[L(x  , (y==0)?NY-1:y-1, 7, nx)], u7);
    VEC_STORE(&output_grid[L((x+1)%(NX+4), (y==0)?NY-1:y-1, 8, nx)], u8);*/
  /* End: Propogate */

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
