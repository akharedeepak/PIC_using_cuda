#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <fstream>
#include "cuda_wrapper.h"


#define PI 3.14159265

using namespace std;

double getAcc( double*, double*, double*, int, int, double, int, double**, double* );

int main()
{   
    // Generate Initial Conditions
	srand(42);  

    default_random_engine generator;
    uniform_real_distribution<double> uniform_distribution(0.0,1.0);
    normal_distribution<double> norm_distribution(0.0,1.0);

	// Simulation parameters
	int N         = 40000;     // Number of particles
	int Nx        = 100;      // Number of mesh cells
	int tEnd      = 50;      // time at which simulation ends
	int dt        = 1;       // timestep
	int n0        = 1;       // electron number density
	int vb        = 3;       // beam velocity
	int vth       = 1;       // beam width
	double A          = 0.1;     // perturbation
	double t          = 0;       // current time of the simulation
	double boxsize    = 50;      // periodic domain [0,boxsize]
	bool plotRealTime = 1;   // switch on for plotting as the simulation goes along
	
    double dx = boxsize/Nx;

    int i, j, Nh, Nt;
    double random01, randn, *pos, *vel, *E, *phi_grid, **Gmtx, *bvec, **Lmtx, **tempA;

    pos  = new double[N];
    vel  = new double[N];
    E    = new double[N];
    bvec = new double[Nx];
    phi_grid = new double[Nx];
    Gmtx = new double*[Nx];
    // Lmtx = new double*[Nx];
    // tempA = new double*[Nx];
    for(i = 0; i < Nx; ++i)
    {
        Gmtx[i] = new double[Nx+1];
        // Lmtx[i] = new double[Nx+1];
        // tempA[i] = new double[Nx+1];
    }

    // construct 2 opposite-moving Guassian beams
    Nh = (int) N/2;
    for (i=0 ; i < N ; i++)
    {   
        pos[i] = uniform_distribution(generator) * boxsize;
        vel[i] = vth * norm_distribution(generator) + vb;
    }

    for (i=Nh ; i < N ; i++)
        vel[i] *= -1;

    // add perturbation
    for (i=0 ; i < N ; i++)
    {   
        vel[i] *= (1 + A*sin(2*PI*pos[i]/boxsize));
    } 

    
	// Construct matrix G to computer Gradient  (1st derivative)
    for (i=0 ; i < Nx ; i++)
    {   
        for (j=0 ; j < Nx ; j++)
        {   
            Gmtx[i][j] = 0;
            if (i == j+1) Gmtx[i][j] =  1/(2*dx);
            if (i+1 == j) Gmtx[i][j] = -1/(2*dx);
        }
    } 
    Gmtx[0][Nx-1] =  1/(2*dx);
	Gmtx[Nx-1][0] = -1/(2*dx);

    // Construct matrix L to computer Laplacian (2nd derivative)
    // for (i=0 ; i < Nx ; i++)
    // {   
    //     for (j=0 ; j < Nx ; j++)
    //     {   
    //         Lmtx[i][j] = 0;
    //         if (i == j+1) Lmtx[i][j] =  1/(dx*dx);
    //         if (i+1 == j) Lmtx[i][j] =  1/(dx*dx);
    //         if (i == j)   Lmtx[i][j] = -2/(dx*dx);
    //     }
    // } 
    // Lmtx[0][Nx-1] =  1/(dx*dx);
	// Lmtx[Nx-1][0] =  1/(dx*dx);
    
    // Deep copy
    // for (i=0 ; i < Nx ; i++)
    //     for (j=0 ; j < Nx ; j++)
    //         tempA[i][j] = Lmtx[i][j];

    getAcc( E, phi_grid, pos, N, Nx, boxsize, n0, Gmtx, bvec );

	// number of timesteps
	Nt = (int)(ceil(tEnd/dt));


    // Simulation main loop
    for (int niter = 0 ; niter < Nt ; niter++)
    {
        // cout << "E"<<endl;
        // for (i = 0 ; i < N ; i++)
        //     cout << -E[i]<<",";
        // cin.get();
        // cout << "pos"<<endl;
        // for(j = 0 ; j < N ; j++)
        //     cout << pos[j]<< ",";
        // cout << endl;
        // cout << "vel"<<endl;
        // for(j = 0 ; j < N ; j++)
        //     cout << vel[j]<< ",";
        // cout << endl << endl;
        // cin.get();

        for(j = 0 ; j < N ; j++)
        {
            vel[j] += 0.5*dt * (-E[j]); // (1/2) kick
            // cout << -E[j]<< ",";
            //  drift (and apply periodic boundary conditions)
            pos[j] += vel[j] * dt;
            pos[j] = fmod(pos[j] , boxsize);
            if(pos[j] < 0)
                pos[j] = pos[j] + 50.0;
        }
        
        // cout << "pos"<<endl;
        // for(j = 0 ; j < N ; j++)
        //     cout << pos[j]<< ",";
        // cout << endl;
        // cout << "vel"<<endl;
        // for(j = 0 ; j < N ; j++)
        //     cout << vel[j]<< ",";
        // cout << endl << endl;
        // cin.get();


        // Deep copy
        // for (i=0 ; i < Nx ; i++)
        //     for (j=0 ; j < Nx ; j++)
        //         tempA[i][j] = Lmtx[i][j];
        // update accelerations
        getAcc( E, phi_grid, pos, N, Nx, boxsize, n0, Gmtx, bvec );

        // cout << "E"<<endl;
        // for (i = 0 ; i < N ; i++)
        //     cout << -E[i]<<",";
        // cin.get();


        for(j = 0 ; j < N ; j++)
        {
            vel[j] += 0.5*dt * (-E[j]); // (1/2) kick

            // update time
            t += dt;
        }
    }

    // Create and open a text file
    ofstream MyFile("PICresult.txt");

    // Write to the file
    cout << "Writing data"<<endl;
    for(j = 0 ; j < N ; j++)
        MyFile << pos[j] << "\t" << vel[j] << endl;
    
    // Close the file
    MyFile.close();
    
    // cout << "pos"<<endl;
    // for(j = 0 ; j < N ; j++)
    //     cout << pos[j]<< ",";
    // cout << endl;
    // cout << "vel"<<endl;
    // for(j = 0 ; j < N ; j++)
    //     cout << vel[j]<< ",";
    // cout << endl << endl;
    // cin.get();



    

    // cleanup
    for(i = 0; i < Nx; ++i)
    {
        delete[] Gmtx[i];
        // delete[] Lmtx[i];
        // delete[] tempA[i];
    }
    delete[] Gmtx;
    // delete[] Lmtx;
    delete[] pos;
    delete[] vel;
    delete[] E;
    delete[] bvec;
    delete[] phi_grid;
    // delete[] tempA;

} 

double getAcc( double* E, double* phi_grid, double* pos, int N, int Nx, double boxsize, int n0, double** Gmtx, double* bvec )
{   
    /*
    Calculate the acceleration on each particle due to electric field
	pos      is an Nx1 matrix of particle positions
	Nx       is the number of mesh cells
	boxsize  is the domain [0,boxsize]
	n0       is the electron number density
	Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
	Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
	a        is an Nx1 matrix of accelerations
	*/

    // Calculate Electron Number Density on the Mesh by 
	// placing particles into the 2 nearest bins (j & j+1, with proper weights)
	// and normalizing
    
    int i, j, pidx[N], pidx1[N];
    double weight_j[N], weight_jp1[N], n[Nx], E_grid[Nx], phi_grid_old[Nx], error, phi_temp[Nx];

    double dx  = (double)boxsize / Nx;
    for (i = 0 ; i < Nx ; i++)
        n[i] = 0;

    for (i = 0 ; i < N ; i++)
    {
        pidx[i] = (int)floor(pos[i]/dx);
        pidx1[i] = pidx[i] + 1;
        weight_j[i]   = ( pidx1[i]*dx - pos[i]  )/dx;
        weight_jp1[i] = ( pos[i]      - pidx[i]*dx )/dx;
        pidx1[i] = pidx1[i] % Nx;

        n[pidx[i]] += weight_j[i];
        n[pidx1[i]] += weight_jp1[i];
        // cout <<pidx[i]<<" "<<pidx1[i] <<endl;
        // if(pidx[i] == 8 )
        // cout << pidx[i]<<" "<<weight_j[i]<<" "<< n[pidx[i]]<<endl;
        // if(pidx1[i] == 8)
        // cout << pidx[i]<<" "<<weight_j[i]<<" "<< n[pidx1[i]]<<endl;
    }

    for (i = 0 ; i < Nx ; i++)
    {
        n[i] *= (double)n0 * boxsize / N / dx;
        bvec[i] = (n[i] - n0)*dx*dx;
        phi_grid_old[i]=phi_grid[i];
        // cout << n[i] << ",";
    }

    // Solve Poisson's Equation: laplacian(phi) = n-n0

    get_grid_Potential(phi_grid ,bvec);

    // for (i = 0 ; i < Nx ; i++)
    // {
    //     phi_temp[i]=phi_grid[i];
    // }

    // for (j = 0 ; j < 10000 ; j++)
    // {
    //     error = 0;
    //     for (i = 1 ; i < Nx-1 ; i++)
    //     {   
    //         phi_grid[i] = 0.5*(phi_grid_old[i-1] + phi_grid_old[i+1] - bvec[i]);
    //         error += abs(phi_grid_old[i] - phi_grid[i]);
    //     }
    //     error += abs(phi_grid_old[0] - 0.5*(phi_grid_old[Nx-1] + phi_grid_old[1] - bvec[0])) + abs(phi_grid_old[Nx-1] - 0.5*(phi_grid_old[Nx-2] + phi_grid_old[0] - bvec[Nx-1]));
    //     phi_grid[0]    = 0.5*(phi_grid_old[Nx-1] + phi_grid_old[1] - bvec[0]);
    //     phi_grid[Nx-1] = 0.5*(phi_grid_old[Nx-2] + phi_grid_old[0] - bvec[Nx-1]);

    //     // cout << phi_grid[1] << " " << error << endl;
    //     for (i = 0 ; i < Nx ; i++)
    //         phi_grid_old[i] = 0.8*phi_grid[i] + (1-0.8)*phi_grid_old[i];

    //     // if(error < 1e-6) break;

    // }
    // cout << j <<" "<< "error = "<< error << endl;

    // for (i = 0 ; i < Nx ; i++)
    // {
    //     cout<< phi_temp[i] <<"  " << phi_grid[i]<<endl;
    // }
    // cin.get();


    // //E_grid = - Gmtx*phi;
    // for (i = 0 ; i < Nx ; i++)
    // {
    //     E_grid[i] = 0;
    //     for (j = 0 ; j < Nx ; j++)
    //     {
    //         E_grid[i] += Gmtx[i][j]*phi_grid[j];
    //     }
    //     // cout<<E_grid[i] << ",";
    // }

    get_grid_Efield(phi_grid, Gmtx);
    for (i = 0 ; i < Nx ; i++)
    {   
        E_grid[i] = phi_grid[i];
        // cout<< phi_grid[i] <<"  " << E_grid[i]<<endl;
    }
    // cin.get();

    // get_Efield_at_pos(phi_grid, Gmtx);
    for (i = 0 ; i < N ; i++)
    {
        E[i] = weight_j[i] * E_grid[pidx[i]] + weight_jp1[i] * E_grid[pidx1[i]];
    }


    return 0;
}

