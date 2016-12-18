#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define L(X, Y, V, W) ((X) + ((V)*(W)) + ((Y)*(W)*(9))) 

/*typedef struct
{
  float speeds[NSPEEDS];
} t_speed;*/
typedef float t_speed;

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
      && (cells[L(jj, ii, 3, nx)] - w1) > 0.0
      && (cells[L(jj, ii, 6, nx)] - w2) > 0.0
      && (cells[L(jj, ii, 7, nx)] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[L(jj, ii, 1, nx)] += w1;
    cells[L(jj, ii, 5, nx)] += w2;
    cells[L(jj, ii, 8, nx)] += w2;
    /* decrease 'west-side' densities */
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

}

void reduceGlobal(global float* lbuffer, global float* av_vels, int iteration){
  int i = get_group_id(0);
}

void reduce(global float* lbuffer, local volatile float* datastr, global float* av_vels, int iteration){
  int i = get_local_id(0);
    //if(i < 64){datastr[i] += datastr[i+64];}
    if(i < 32){datastr[i] += datastr[i+32];}
    if(i < 16){ datastr[i] += datastr[i+16]; }
    if(i < 8){ datastr[i] += datastr[i+8]; }
    if(i < 4){ datastr[i] += datastr[i+4]; } 
    if(i < 2){ datastr[i] += datastr[i+2]; }
    if(i < 1){ datastr[i] += datastr[i+1]; }

    if(i == 0) {
      lbuffer[get_group_id(0)] = datastr[0];
    }
    //barrier(CLK_GLOBAL_MEM_FENCE);
    if(i == 0) reduceGlobal(lbuffer, av_vels, iteration);
}

kernel void collision(global t_speed* restrict cells,
                      global t_speed* restrict tmp_cells,
                      global int* restrict obstacles,
                      int nx, int ny, float omega, float density, float accel, global float* restrict av_vels, local volatile float* restrict datastr, global float* restrict lbuffer, int iteration){
  int gid = get_global_id(0);
  int jj = gid & (nx-1); // y*nx+x
  int ii = (get_global_id(0)-jj)/nx;
  int yn = (ii + 1) & (ny-1);
  int xe = (jj + 1) & (nx-1);
  int ys = (ii == 0) ? (ny - 1) : (ii - 1);
  int xw = (jj == 0) ? (nx - 1) : (jj - 1);
  float sum = 0.0f;
  //int iteration = 4;
  /*if(get_local_id(0) == 0){
    printf("group_id=%d, size=%d\n", get_group_id(0), get_local_size(0)=64);
  }*/
  //t_speed cell = tmp_cells[gid];
  float* in;// = cell.speeds;
  for(int i = 0; i < 9; i++){
    in[i] = cells[L(jj, ii, i, nx)];
  }

  if (!obstacles[gid])
  {

    /* compute x velocity component */
    //global float* speeds = tmp_cells[gid].speeds;

    //float in[9];
    /*for(int i = 0; i < 9; i++){
      in[i] = speeds[i];
    }*/
    

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

    float uxsq = ux*ux;
    float uysq = uy*uy;
    datastr[get_local_id(0)] = sqrt(uxsq + uysq);
    //lbuffer[get_global_id(0)] = sqrt(uxsq + uysq);
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
    float e[9];
    e[0] = local_density - local_density*(uxsq15 + uysq15);

    local_density *= 0.25;
    float px = local_density * ux3 * 2;
    float py = local_density * uy3 * 2;
    e[1] = (local_density +(local_density * ((ux3 + uxsq3) - uysq15))); //East
    e[2] = (local_density +(local_density * ((uy3 + uysq3) - uxsq15))); //North
    e[3] = (e[1] - px); //West
    e[4] = (e[2] - py); //South

    //Diagonals
    local_density *= 0.25;
    px *= 0.25;
    py *= 0.25;
    //NE NW SW SE
    e[5] = (local_density + (local_density * ((trailing_diag + (ux3 + uy3)) - u_sq)));
    e[6] = (local_density + (local_density * ((leading_diag + uy3) - (ux3 + u_sq) )));
    e[7] = ((e[5] - px) - py);
    e[8] = ((e[6] + px) - py);

    for(int i = 0; i < 9; i++){
      in[i] *= (1 - omega);
      in[i] += e[i];
    }

    //Acceleration
    float w1 = density * accel / 9.0;
    float w2 = density * accel / 36.0;
    if (ii == ny-2
        && (in[3] - w1) > 0.0
        && (in[6] - w2) > 0.0
        && (in[7] - w2) > 0.0)
    {
      in[1] += w1;
      in[5] += w2;
      in[8] += w2;
      in[3] -= w1;
      in[6] -= w2;
      in[7] -= w2;
    }

    cells[L(jj, ii, 0, nx)]  =in[0]; /* central cell, no movement */
    cells[L(xe, ii, 1, nx)] =in[1]; /* east */
    cells[L(jj, yn, 2, nx)] =in[2]; /* north */
    cells[L(xw, ii, 3, nx)] =in[3]; /* west */
    cells[L(jj, y_s, 4, nx)] =in[4]; /* south */
    cells[L(xe, yn, 5, nx)]=in[5]; /* north-east */
    cells[L(xw, yn, 6, nx)]=in[6]; /* north-west */
    cells[L(xw, ys, 7, nx)]=in[7]; /* south-west */
    cells[L(xe, ys, 8, nx)]=in[8]; /* south-east */
  }else{
    datastr[get_local_id(0)] = 0.0f;
    //lbuffer[get_global_id(0)] = 0.0f;

    cells[L(xe, ii, 1, nx)] =in[3]; /* east */
    cells[L(jj, yn, 2, nx)] =in[4]; /* north */
    cells[L(xw, ii, 3, nx)] =in[1]; /* west */
    cells[L(jj, ys, 4, nx)] =in[2]; /* south */
    cells[L(xe, yn, 5, nx)]=in[7]; /* north-east */
    cells[L(xw, yn, 6, nx)]=in[8]; /* north-west */
    cells[L(xw, ys, 7, nx)]=in[5]; /* south-west */
    cells[L(xe, ys, 8, nx)]=in[6]; /* south-east */
  }

  /* Reduction */
  //int num_wrk_items  = get_local_size(0);                 
  //int local_id       = get_local_id(0);                   
  //int group_id       = get_group_id(0); 

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(lbuffer, datastr, av_vels, iteration);
}
