#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

constant float c_sq = 1.0 / 3.0; /* square of speed of sound */
constant float w0 = 4.0 / 9.0;  /* weighting factor */
constant float w1 = 1.0 / 9.0;  /* weighting factor */
constant float w2 = 1.0 / 36.0; /* weighting factor */

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
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
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
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
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}

kernel void rebound(global t_speed* cells,
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    int nx, int ny){
  int jj = get_global_id(0);
  int ii = get_global_id(1);

}

kernel void collision(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, float omega, float density, float accel){
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  int yn = (ii + 1) % ny;
  int xe = (jj + 1) % nx;
  int ys = (ii == 0) ? (ny - 1) : (ii - 1);
  int xw = (jj == 0) ? (nx - 1) : (jj - 1);
  if (!obstacles[ii * nx + jj])
  {

    /* compute x velocity component */
    global float* speeds = tmp_cells[ii * nx + jj].speeds;

    float xneg = speeds[3] + speeds[6] + speeds[7];
    float yneg = speeds[4] + speeds[7] + speeds[8]; 
    float xpos = speeds[1] + speeds[5] + speeds[8]; //048
    float ypos = speeds[2] + speeds[5] + speeds[6]; //026

    float local_density = xpos + xneg + speeds[0] + speeds[2] + speeds[4];
    float inverse_local_density = 1.0f/local_density;

    float ux = (xpos - xneg)*inverse_local_density;
    /* compute y velocity component */
    float uy = (ypos - yneg)*inverse_local_density;

    /* velocity squared */
    //float u_sq = ux * ux + uy * uy;


    float uxsq = ux*ux;
    float uysq = uy*uy;
    //tot_u += sqrt(uxsq + uysq);

    float ux3 = 3.0 * ux;
    float uy3 = 3.0 * uy;

    float uxsq3 = 3.0 * uxsq;
    float uxsq15= 1.5 * uxsq;

    float uysq3 = 3.0 * uysq;
    float uysq15= 1.5 * uysq;

    float u_sq = uxsq15 + uysq15;

    float leading_diag  = 4.5 * (ux-uy)*(ux-uy); // = 4.5*(x-y)^2 == 4.5*(y-x)^2
    float trailing_diag = 4.5 * (ux+uy)*(ux+uy);

    local_density *= w0 * omega;

    //local_density is no longer local density
    cells[y * nx + x].speeds[0] = speeds[0] * (1 - omega) + (local_density + local_density * (-uxsq15 - uysq15));

    local_density *= 0.25;
    cells[ii * nx + xe].speeds[1] = speeds[1] * (1 - omega) + (local_density + local_density * ( ux3 + uxsq3 - uysq15));
    cells[yn * nx + jj].speeds[2] = speeds[2] * (1 - omega) + (local_density + local_density * ( uy3 + uysq3 - uxsq15));
    cells[ii * nx + xw].speeds[3] = speeds[3] * (1 - omega) + (local_density + local_density * (-ux3 + uxsq3 - uysq15));
    cells[ys * nx + jj].speeds[4] = speeds[4] * (1 - omega) + (local_density + local_density * (-uy3 + uysq3 - uxsq15));

    local_density *= 0.25;
    cells[yn * nx + xe].speeds[5] = speeds[5] * (1 - omega) + (local_density + local_density * ( ux3 + uy3 + trailing_diag -u_sq));
    cells[yn * nx + xw].speeds[6] = speeds[6] * (1 - omega) + (local_density + local_density * (-ux3 + uy3 + leading_diag  -u_sq));
    cells[ys * nx + xw].speeds[7] = speeds[7] * (1 - omega) + (local_density + local_density * (-ux3 - uy3 + trailing_diag -u_sq));
    cells[ys * nx + xe].speeds[8] = speeds[8] * (1 - omega) + (local_density + local_density * ( ux3 - uy3 + leading_diag  -u_sq));
    


    /* directional velocity components */
    //float u[NSPEEDS];
    //u[1] =   u_x;        /* east */
    //u[2] =         u_y;  /* north */
    //u[3] = - u_x;        /* west */
    //u[4] =       - u_y;  /* south */
    //u[5] =   u_x + u_y;  /* north-east */
    //u[6] = - u_x + u_y;  /* north-west */
    //u[7] = - u_x - u_y;  /* south-west */
    //u[8] =   u_x - u_y;  /* south-east */

    /* equilibrium densities */
    //float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    //d_equ[0] = w0 * local_density
    //           * (1.0 - u_sq / (2.0 * c_sq));
    /* axis speeds: weight w1 */
    /*d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
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
                                     - u_sq / (2.0 * c_sq));*/
    /* diagonal speeds: weight w2 */
    /*d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
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
                                     - u_sq / (2.0 * c_sq));*/

    /* relaxation step */
    /*cells[ii * nx + jj].speeds[0] = tmp_cells[ii * nx + jj].speeds[0] + omega * (d_equ[0] - tmp_cells[ii * nx + jj].speeds[0]);
    cells[ii * nx + xe].speeds[1] = tmp_cells[ii * nx + jj].speeds[1] + omega * (d_equ[1] - tmp_cells[ii * nx + jj].speeds[1]);
    cells[yn * nx + jj].speeds[2] = tmp_cells[ii * nx + jj].speeds[2] + omega * (d_equ[2] - tmp_cells[ii * nx + jj].speeds[2]);
    cells[ii * nx + xw].speeds[3] = tmp_cells[ii * nx + jj].speeds[3] + omega * (d_equ[3] - tmp_cells[ii * nx + jj].speeds[3]);
    cells[ys * nx + jj].speeds[4] = tmp_cells[ii * nx + jj].speeds[4] + omega * (d_equ[4] - tmp_cells[ii * nx + jj].speeds[4]);
    cells[yn * nx + xe].speeds[5] = tmp_cells[ii * nx + jj].speeds[5] + omega * (d_equ[5] - tmp_cells[ii * nx + jj].speeds[5]);
    cells[yn * nx + xw].speeds[6] = tmp_cells[ii * nx + jj].speeds[6] + omega * (d_equ[6] - tmp_cells[ii * nx + jj].speeds[6]);
    cells[ys * nx + xw].speeds[7] = tmp_cells[ii * nx + jj].speeds[7] + omega * (d_equ[7] - tmp_cells[ii * nx + jj].speeds[7]);
    cells[ys * nx + xe].speeds[8] = tmp_cells[ii * nx + jj].speeds[8] + omega * (d_equ[8] - tmp_cells[ii * nx + jj].speeds[8]);*/


    float w1 = density * accel / 9.0;
    float w2 = density * accel / 36.0;

    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (ii == ny-2
        && (cells[ii * nx + xw].speeds[3] - w1) > 0.0
        && (cells[yn * nx + xw].speeds[6] - w2) > 0.0
        && (cells[ys * nx + xw].speeds[7] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[ii * nx + xe].speeds[1] += w1;
      cells[yn * nx + xe].speeds[5] += w2;
      cells[ys * nx + xe].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii * nx + xw].speeds[3] -= w1;
      cells[yn * nx + xw].speeds[6] -= w2;
      cells[ys * nx + xw].speeds[7] -= w2;
    }

  }else{
    cells[ii * nx + xe].speeds[1] = tmp_cells[ii * nx + jj].speeds[3];
    cells[yn * nx + jj].speeds[2] = tmp_cells[ii * nx + jj].speeds[4];
    cells[ii * nx + xw].speeds[3] = tmp_cells[ii * nx + jj].speeds[1];
    cells[ys * nx + jj].speeds[4] = tmp_cells[ii * nx + jj].speeds[2];
    cells[yn * nx + xe].speeds[5] = tmp_cells[ii * nx + jj].speeds[7];
    cells[yn * nx + xw].speeds[6] = tmp_cells[ii * nx + jj].speeds[8];
    cells[ys * nx + xw].speeds[7] = tmp_cells[ii * nx + jj].speeds[5];
    cells[ys * nx + xe].speeds[8] = tmp_cells[ii * nx + jj].speeds[6];
  }
}
