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
                float inverse_available_cells, float density, float accel)
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

  float o_mask2 = VEC_LOAD(&obstacles[y*nx+x]);

  floatv xneg = u3_o + u6_o + u7_o;
  floatv xpos = u1_o + u5_o + u8_o;
  floatv yneg = u4_o + u7_o + u8_o;
  floatv ypos = u2_o + u5_o + u6_o;

  floatv density = u0_o + u1_o + u3_o + yneg + ypos;

  xpos = (xpos - xneg)/density;
  ypos = (ypos - yneg)/density;
  
  floatv x_sq = xpos*xpos;
  floatv y_sq = ypos*ypos;

  floatv sum =  sqrt(x_sq + y_sq) & o_mask2; //Ignore obstacles in the summation

  floatv tot_u = dot(sum, (float16)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f));



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
