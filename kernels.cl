#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define L(X, Y, V, NX) ((X) + ((V)+(Y)*18)*(NX))
#define VEC_SIZE 8
#define floatv float8
#define intv int8
#define VEC_LOAD(ADDR) vload8(0, (ADDR))
#define VEC_STORE(ADDR, DATA) vstore8((DATA), 0, ADDR)

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

kernel void swapGhostCellsLR(global float* cells){
  int y = get_global_id(0);
  cells[L(4, y, 1, NXPAD)] = cells[L(NX+4, y, 1, NXPAD)];
  cells[L(4, y, 3, NXPAD)] = cells[L(NX+4, y, 3, NXPAD)];
  cells[L(4, y, 8, NXPAD)] = cells[L(NX+4, y, 8, NXPAD)];
  
  cells[L(NX+3, y, 2, NXPAD)] = cells[L(3, y, 2, NXPAD)];
  cells[L(NX+3, y, 5, NXPAD)] = cells[L(3, y, 5, NXPAD)];
  cells[L(NX+3, y, 6, NXPAD)] = cells[L(3, y, 6, NXPAD)];
}

kernel void swapGhostCellsTB(global float* cells){
  int x = get_global_id(0);
  cells[L(x, 0, 3, NXPAD)] = cells[L(x, NY, 3, NXPAD)];
  cells[L(x, 0, 4, NXPAD)] = cells[L(x, NY, 4, NXPAD)];
  cells[L(x, 0, 5, NXPAD)] = cells[L(x, NY, 5, NXPAD)];
  
  cells[L(x, NY-1, 6, NXPAD)] = cells[L(x, -1, 6, NXPAD)];
  cells[L(x, NY-1, 7, NXPAD)] = cells[L(x, -1, 7, NXPAD)];
  cells[L(x, NY-1, 8, NXPAD)] = cells[L(x, -1, 8, NXPAD)];
}


kernel void lbm(global float* cells, global float* tmp_cells, global float* obstacles, global float* partial_sums)
{
  int x = get_global_id(0)*16 + 4;
  int y = get_global_id(1);
  int offset = 4 + (9 * nx_pad);

  //if(x == 4 && y == 86){
    printf("==========================\n");
    printf("cells[0]=%f\n", cells[0]);
    cells[0] = 81;
    printf("tmp_cells[0]=%f\n", tmp_cells[0]);
    tmp_cells[0] = 82;
    printf("Test value: %f %p\n", tmp_cells[L(x+1, y+1, 3, nx_pad)], &tmp_cells[L(x+1, y+1, 3, nx_pad)]);
    printf("Test value: %f %p\n", cells[L(x+1, y+1, 3, nx_pad)], &cells[L(x+1, y+1, 3, nx_pad)]);
    printf("%p %p\n", cells, tmp_cells);
 // }


  floatv u0_o = VEC_LOAD(&tmp_cells[L(x, y, 0, nx_pad)]);
  floatv u1_o = VEC_LOAD(&tmp_cells[L(x, y, 1, nx_pad)]);
  floatv u2_o = VEC_LOAD(&tmp_cells[L(x, y, 2, nx_pad)]);
  floatv u3_o = VEC_LOAD(&tmp_cells[L(x, y, 3, nx_pad)]);
  floatv u4_o = VEC_LOAD(&tmp_cells[L(x, y, 4, nx_pad)]);
  floatv u5_o = VEC_LOAD(&tmp_cells[L(x, y, 5, nx_pad)]);
  floatv u6_o = VEC_LOAD(&tmp_cells[L(x, y, 6, nx_pad)]);
  floatv u7_o = VEC_LOAD(&tmp_cells[L(x, y, 7, nx_pad)]);
  floatv u8_o = VEC_LOAD(&tmp_cells[L(x, y, 8, nx_pad)]);

  floatv o_mask2 = VEC_LOAD(&obstacles[y*nx+x-4]);

  floatv xneg = u3_o + u6_o + u7_o;
  floatv xpos = u1_o + u5_o + u8_o;
  floatv yneg = u4_o + u7_o + u8_o;
  floatv ypos = u2_o + u5_o + u6_o;

  floatv density = u0_o + u1_o + u3_o + yneg + ypos;

  xpos = (xpos - xneg)/density;
  ypos = (ypos - yneg)/density;
  
  floatv x_sq = xpos*xpos;
  floatv y_sq = ypos*ypos;

  floatv sum =  sqrt(x_sq + y_sq) * o_mask2; //Ignore obstacles in the summation

  //float tot_u = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7 + sum.s8 + sum.s9 + sum.s10 + sum.s11 + sum.s12 + sum.s13 + sum.s14 + sum.s15;
  float tot_u = dot(sum.s0123, (float4)(1));
  tot_u += dot(sum.s4567, (float4)(1));
  //TODO: division
  partial_sums[y*NX+get_global_id(0)] = tot_u;

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
  if(y == 2){
    intv msk2 = (u2 > w1); // check that u2 > w1, 0 if false, all bits 1 if true
    intv msk5 = (u5 > w2); 
    intv msk6 = (u6 > w2);
    intv omsk = (o_mask2 > 0);

    intv wmask = (msk2 & msk5) & (msk6 & omsk); //Set mask to 0 for inappropriate cells

    floatv w1_masked = (w1 * convert_float16(wmask) )*(floatv)(-1);
    floatv w2_masked = (w2 * convert_float16(wmask) )*(floatv)(-1);

    u1 = (u1 + w1_masked);
    u3 = (u3 + w2_masked);
    u8 = (u8 + w2_masked);
    u2 = (u2 - w1_masked);
    u5 = (u5 - w2_masked);
    u6 = (u6 - w2_masked);
  }

  /* Begin: Rebound: openCL mix */
  u0 = mix(u0_o, u0, o_mask2); //zero where obstacle
  u1 = mix(u2_o, u1, o_mask2);
  u2 = mix(u1_o, u2, o_mask2);
  u3 = mix(u6_o, u3, o_mask2);
  u4 = mix(u7_o, u4, o_mask2);
  u5 = mix(u8_o, u5, o_mask2);
  u6 = mix(u3_o, u6, o_mask2);
  u7 = mix(u4_o, u7, o_mask2);
  u8 = mix(u5_o, u8, o_mask2);
  /* End: Rebound */
  
  /* Begin: Propogate */
  /* None of these swap nodes as y != end && y != start */
  VEC_STORE(&cells[L(x  , y  , 0, nx_pad)], u0); // Does not propogate
  VEC_STORE(&cells[L(x+1, y  , 1, nx_pad)], u1);
  VEC_STORE(&cells[L(x-1, y  , 2, nx_pad)], u2);
  VEC_STORE(&cells[L(x+1, y+1, 3, nx_pad)], (floatv)(8));
  VEC_STORE(&cells[L(x  , y+1, 4, nx_pad)], u4);
  VEC_STORE(&cells[L(x-1, y+1, 5, nx_pad)], u5);
  VEC_STORE(&cells[L(x-1, y-1, 6, nx_pad)], u6);
  VEC_STORE(&cells[L(x  , y-1, 7, nx_pad)], u7);
  VEC_STORE(&cells[L(x+1, y-1, 8, nx_pad)], u8);

  //if(x == 4 && y == 86){
    printf("Running kernel on (%d, %d); Sample = [%f, %f, %f, %f, %f, %f, %f, %f, %f] => [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n", x, y, u0_o.s1, u1_o.s1, u2_o.s1, u3_o.s1, u4_o.s1, u5_o.s1, u6_o.s1, u7_o.s1, u8_o.s1, u0.s1, u1.s1, u2.s1, u3.s1, u4.s1, u5.s1, u6.s1, u7.s1, u8.s1);
    printf("Test value: %f %p\n", cells[L(x+1, y+1, 3, nx_pad)], &cells[L(x+1, y+1, 3, nx_pad)]);
    printf("%p %p\n", cells, tmp_cells);
  //}
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
