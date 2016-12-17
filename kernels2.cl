#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

/*typedef struct
{
  double speeds[NSPEEDS];
} t_speed;*/
typedef t_speed float*;

constant double c_sq = 1.0 / 3.0; /* square of speed of sound */
constant double w0 = 4.0 / 9.0;  /* weighting factor */
constant double w1 = 1.0 / 9.0;  /* weighting factor */
constant double w2 = 1.0 / 36.0; /* weighting factor */

kernel void accelerate_flow(global t_speed* cells,
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
      && (cells[L(jj, ii, 3, nx)] - w1) > 0.0
      && (cells[L(jj, ii, 6, nx)] - w2) > 0.0
      && (cells[L(jj, ii, 7, nx)] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[L(jj, ii, 1, nx)] += w1;
    cells[L(jj, ii, 5, nx)] += w2;
    cells[L(jj, ii, 8, nx)] += w2;
    /* decrease densities */
    cells[L(jj, ii, 3, nx)] -= w1;
    cells[L(jj, ii, 6, nx)] -= w2;
    cells[L(jj, ii, 7, nx)] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
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
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[L(jj, ii, 0, nx)] = cells[L(jj, ii, 0, nx)]; /* central cell, no movement */
  tmp_cells[L(x_e, ii, 1, nx)] = cells[L(jj, ii, 1, nx)]; /* east */
  tmp_cells[L(jj, y_n, 2, nx)] = cells[L(jj, ii, 2, nx)]; /* north */
  tmp_cells[L(x_w, ii, 3, nx)] = cells[L(jj, ii, 3, nx)]; /* west */
  tmp_cells[L(jj, y_s, 4, nx)] = cells[L(jj, ii, 4, nx)]; /* south */
  tmp_cells[L(x_e, y_n, 5, nx)] = cells[L(jj, ii, 5, nx)]; /* north-east */
  tmp_cells[L(x_w, y_n, 6, nx)] = cells[L(jj, ii, 6, nx)]; /* north-west */
  tmp_cells[L(x_w, y_s, 7, nx)] = cells[L(jj, ii, 7, nx)]; /* south-west */
  tmp_cells[L(x_e, y_s, 8, nx)] = cells[L(jj, ii, 8, nx)]; /* south-east */
}

kernel void rebound(global t_speed* cells,
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    int nx, int ny){
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  if (obstacles[ii * nx + jj])
  {
    cells[L(jj, ii, 1, nx)] = tmp_cells[L(jj, ii, 3, nx)];
    cells[L(jj, ii, 2, nx)] = tmp_cells[L(jj, ii, 4, nx)];
    cells[L(jj, ii, 3, nx)] = tmp_cells[L(jj, ii, 1, nx)];
    cells[L(jj, ii, 4, nx)] = tmp_cells[L(jj, ii, 2, nx)];
    cells[L(jj, ii, 5, nx)] = tmp_cells[L(jj, ii, 7, nx)];
    cells[L(jj, ii, 6, nx)] = tmp_cells[L(jj, ii, 8, nx)];
    cells[L(jj, ii, 7, nx)] = tmp_cells[L(jj, ii, 5, nx)];
    cells[L(jj, ii, 8, nx)] = tmp_cells[L(jj, ii, 6, nx)];
  }
}

kernel void collision(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, double omega){
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  if (!obstacles[ii * nx + jj])
  {
    /* compute local density total */
    double local_density = 0.0;

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cells[ii * nx + jj].speeds[kk];
    }

    /* compute x velocity component */
    double u_x = (cells[L(jj, ii, 1, nx)]
           + cells[L(jj, ii, 5, nx)]
           + cells[L(jj, ii, 8, nx)]
           - (cells[L(jj, ii, 3, nx)]
              + cells[L(jj, ii, 6, nx)]
              + cells[L(jj, ii, 7, nx)]))
          / local_density;
    /* compute y velocity component */
    double u_y = (cells[L(jj, ii, 2, nx)]
           + cells[L(jj, ii, 5, nx)]
           + cells[L(jj, ii, 6, nx)]
           - (cells[L(jj, ii, 4, nx)]
              + cells[L(jj, ii, 7, nx)]
              + cells[L(jj, ii, 8, nx)]))
          / local_density;

    /* velocity squared */
    double u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
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

    /* relaxation step */
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      cells[L(jj, ii, kk, nx)] = tmp_cells[L(jj, ii, kk, nx)]
                                              + omega
                                              * (d_equ[kk] - tmp_cells[L(jj, ii, kk, nx)]);
    }
  }
}
