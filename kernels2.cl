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

    float in[9];
    for(int i = 0; i < 9; i++){
      in[i] = speeds[i];
    }

    float xneg = in[3] + in[6] + in[7];
    float yneg = in[4] + in[7] + in[8]; 
    float xpos = in[1] + in[5] + in[8]; //048
    float ypos = in[2] + in[5] + in[6]; //026

    float local_density = xpos + xneg + in[0] + in[2] + in[4];
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
    cells[ii * nx + jj].speeds[0] = in[0] * (1 - omega) + (local_density + local_density * (-uxsq15 - uysq15));

    local_density *= 0.25;
    cells[ii * nx + xe].speeds[1] = in[1] * (1 - omega) + (local_density + local_density * ( ux3 + uxsq3 - uysq15));
    cells[yn * nx + jj].speeds[2] = in[2] * (1 - omega) + (local_density + local_density * ( uy3 + uysq3 - uxsq15));
    cells[ii * nx + xw].speeds[3] = in[3] * (1 - omega) + (local_density + local_density * (-ux3 + uxsq3 - uysq15));
    cells[ys * nx + jj].speeds[4] = in[4] * (1 - omega) + (local_density + local_density * (-uy3 + uysq3 - uxsq15));

    local_density *= 0.25;
    cells[yn * nx + xe].speeds[5] = in[5] * (1 - omega) + (local_density + local_density * ( ux3 + uy3 + trailing_diag -u_sq));
    cells[yn * nx + xw].speeds[6] = in[6] * (1 - omega) + (local_density + local_density * (-ux3 + uy3 + leading_diag  -u_sq));
    cells[ys * nx + xw].speeds[7] = in[7] * (1 - omega) + (local_density + local_density * (-ux3 - uy3 + trailing_diag -u_sq));
    cells[ys * nx + xe].speeds[8] = in[8] * (1 - omega) + (local_density + local_density * ( ux3 - uy3 + leading_diag  -u_sq));


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
