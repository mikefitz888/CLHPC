#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define L(X, Y, V, NX) ((X) + ((V)+(Y)*18)*(NX))
#define VEC_SIZE 16
#define floatv float16
#define VEC_LOAD(ADDR) vload16(0, (ADDR))


typedef struct
{
  double speeds[NSPEEDS];
} t_speed;


kernel void lbm(global float* cells,
                global float* obstacles,
                int nx, int ny, int nx_pad,
                float inverse_available_cells, float pdensity, float accel,
                float16 Vnomega, float16 V3, float16 Vo25, float16 V1o5, float16 V4o5)
{
  int x = get_global_id(0)*16;
  int y = get_global_id(1);
  int offset = 4 + (9 * nx_pad);
  global float* tmp_cells = cells + offset;

  floatv u0_o = VEC_LOAD(&cells[L(x, y, 0, nx_pad)]);
  floatv u1_o = VEC_LOAD(&cells[L(x, y, 1, nx_pad)]);
  floatv u2_o = VEC_LOAD(&cells[L(x, y, 2, nx_pad)]);
  floatv u3_o = VEC_LOAD(&cells[L(x, y, 3, nx_pad)]);
  floatv u4_o = VEC_LOAD(&cells[L(x, y, 4, nx_pad)]);
  floatv u5_o = VEC_LOAD(&cells[L(x, y, 5, nx_pad)]);
  floatv u6_o = VEC_LOAD(&cells[L(x, y, 6, nx_pad)]);
  floatv u7_o = VEC_LOAD(&cells[L(x, y, 7, nx_pad)]);
  floatv u8_o = VEC_LOAD(&cells[L(x, y, 8, nx_pad)]);

  floatv o_mask2 = VEC_LOAD(&obstacles[y*nx+x]);

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

  float tot_u = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7 + sum.s8 + sum.s9 + sum.s10 + sum.s11 + sum.s12 + sum.s13 + sum.s14 + sum.s15;

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


  uxsq3 = (V3 * x_sq);
  uysq3 = (V3 * y_sq);


  uxsq15 = (V1o5 * x_sq);
  uysq15 = (V1o5 * y_sq);

  floatv u_sq = (uxsq15 + uysq15);

  __m256 leading_diag  = _mm256_mul_ps(V4o5, _mm256_mul_ps(_mm256_sub_ps(xpos, ypos), _mm256_sub_ps(xpos, ypos)));
  __m256 trailing_diag = _mm256_mul_ps(V4o5, _mm256_mul_ps(_mm256_add_ps(xpos, ypos), _mm256_add_ps(xpos, ypos)));

  //The general equation to follow (slightly optimized) is: u_next = u * (1 - omega) + (density + density * (3u + 4.5u^2 - 1.5(ux^2 + uy^2))) * w * omega
  density = _mm256_mul_ps(density, start_weight); //density *= w0 * omega

  __m256 e0 = _mm256_sub_ps(density, _mm256_mul_ps(density, _mm256_add_ps(uxsq15, uysq15)));

  //Axis
  density = _mm256_mul_ps(density, Vo25);
  __m256 px = _mm256_mul_ps(_mm256_mul_ps(density, ux3), V2);
  __m256 py = _mm256_mul_ps(_mm256_mul_ps(density, uy3), V2);
  __m256 e1 = _mm256_add_ps(density, _mm256_mul_ps(density, _mm256_sub_ps(_mm256_add_ps(ux3, uxsq3), uysq15))); //East
  __m256 e4 = _mm256_add_ps(density, _mm256_mul_ps(density, _mm256_sub_ps(_mm256_add_ps(uy3, uysq3), uxsq15))); //North
  __m256 e2 = _mm256_sub_ps(e1, px); //West
  __m256 e7 = _mm256_sub_ps(e4, py); //South

  //Diagonals
  density = _mm256_mul_ps(density, Vo25);
  px = _mm256_mul_ps(px, Vo25);
  py = _mm256_mul_ps(py, Vo25);

  __m256 e3 = _mm256_add_ps(density, _mm256_mul_ps(density, _mm256_sub_ps(_mm256_add_ps(trailing_diag, _mm256_add_ps(ux3, uy3)), u_sq)));
  __m256 e5 = _mm256_add_ps(density, _mm256_mul_ps(density, _mm256_sub_ps(_mm256_add_ps(leading_diag, uy3), _mm256_add_ps(ux3, u_sq) )));
  __m256 e6 = _mm256_sub_ps(_mm256_sub_ps(e3, px), py);
  __m256 e8 = _mm256_sub_ps(_mm256_add_ps(e5, px), py);

  u0 = _mm256_add_ps(u0, e0);
  u1 = _mm256_add_ps(u1, e1);
  u2 = _mm256_add_ps(u2, e2);
  u3 = _mm256_add_ps(u3, e3);
  u4 = _mm256_add_ps(u4, e4);
  u5 = _mm256_add_ps(u5, e5);
  u6 = _mm256_add_ps(u6, e6);
  u7 = _mm256_add_ps(u7, e7);
  u8 = _mm256_add_ps(u8, e8);
  /* End: Collision */

  /* Add Acceleration */

  /* Begin: Rebound */
  u0 = _mm256_blendv_ps(u0_o, u0, o_mask2); //zero where obstacle
  u1 = _mm256_blendv_ps(u2_o, u1, o_mask2);
  u2 = _mm256_blendv_ps(u1_o, u2, o_mask2);
  u3 = _mm256_blendv_ps(u6_o, u3, o_mask2);
  u4 = _mm256_blendv_ps(u7_o, u4, o_mask2);
  u5 = _mm256_blendv_ps(u8_o, u5, o_mask2);
  u6 = _mm256_blendv_ps(u3_o, u6, o_mask2);
  u7 = _mm256_blendv_ps(u4_o, u7, o_mask2);
  u8 = _mm256_blendv_ps(u5_o, u8, o_mask2);
  /* End: Rebound */


  /* Begin: Propogate */
  /* None of these swap nodes as y != end && y != start */
  _mm256_store_ps( &cells[L(x  , y  , 0, params->nx_pad)], u0); // Does not propogate
  _mm256_storeu_ps(&cells[L(x+1, y  , 1, params->nx_pad)], u1);
  _mm256_storeu_ps(&cells[L(x-1, y  , 2, params->nx_pad)], u2);
  _mm256_storeu_ps(&cells[L(x+1, y+1, 3, params->nx_pad)], u3);
  _mm256_store_ps( &cells[L(x  , y+1, 4, params->nx_pad)], u4);
  _mm256_storeu_ps(&cells[L(x-1, y+1, 5, params->nx_pad)], u5);
  _mm256_storeu_ps(&cells[L(x-1, y-1, 6, params->nx_pad)], u6);
  _mm256_store_ps( &cells[L(x  , y-1, 7, params->nx_pad)], u7);
  _mm256_storeu_ps(&cells[L(x+1, y-1, 8, params->nx_pad)], u8);        
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
