#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <omp.h>
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
#include <sstream>

//Получил lambda 6* 10-4 м
//Получил из среднеквадратичного D = 0.134 м^2/с без перевода 133,5
//Получил из АКФС до перевода в си 122,5 

int timenumbers = 5000;
double rad = 0.56; // = 1.54*10^(-10)м
double dt0 = 0.0001; // = 2.27*10^(-16)с
double den0 = 0.8; // = 2.42*10^(23)м^(-3) = 5e-2
int N0 = 125;
double Temperature = 300;//K
double kBoltzmann  = 28.1e-3; 
double mass = 1; // = 33.55*10^(-27)кг
double sigmal = 1; // = 2.75*10^(-10)м
double epsilon = 1; // = 4.91*10^(-22)Дж

void progress(int total, int ready) {
	system("clear");
	int totaldotz = 40;
	double progr = double(ready) / double(total);
	int dots = int(progr * totaldotz);
	int ii = 0;
	std::cout << int(progr * 100) << "% [";
	for (int i = 0; i < dots; i++) std::cout << "#";
	for (int i = 0; i < totaldotz - dots - 1 ; i++) std::cout << " ";
	std::cout << "]" << std::endl;
}

class Model{
public:
    double *x;
    double *y;
    double *z;
    double *x0;
    double *y0;
    double *z0;
    double *vx;
    double *vy;
    double *vz;
    double *wx;
    double *wy;
    double *wz;
    double *nx;
    double *ny;
    double *nz;
    double *v0;
    double L;
    double dt;
    double den;
    double rc;
    double a;
    double b;
    double K;
    double U;
    double Temperature;
    double kBoltzmann; 
    double mass;
    double sigmal;
    double rad;
    int N;
    
    Model(double dt0, double den0, double rc0, double a0, double b0, int N0, double Temperature0, double kBoltzmann0, double mass0, double sigmal0, double rad0){
        dt = dt0;
        den = den0;
        rc = rc0;
        a = a0;
        b = b0;
        N = N0;
        Temperature= Temperature0;
        kBoltzmann = kBoltzmann0; 
        mass = mass0;
        sigmal = sigmal0;
        rad = rad0;
    }

    void init(){
        x = new double [N];
        y = new double [N];
        z = new double [N];
        x0 = new double [N];
        y0 = new double [N];
        z0 = new double [N];
        vx = new double [N];
        vy = new double [N];
        vz = new double [N];
        wx = new double [N];
        wy = new double [N];
        wz = new double [N];
        nx = new double [N];
        ny = new double [N];
        nz = new double [N];
        v0 = new double [N];
        L = pow(N/den, 1.0/3.0);
        std::cout << L;
    }

    void nullv(){
        for (int i = 0; i < N; i++) {
			vx[i] = 0; 
			vy[i] = 0;
            vz[i] = 0;
        }
    }

    void randomv(){
        K = 0;

        double sigma = sqrt(kBoltzmann*Temperature/mass);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> X(0, sigma);
        std::normal_distribution<> Y(0, sigma);
        std::normal_distribution<> Z(0, sigma);
    
        for (int i = 0; i < N; i++) {
            vx[i] = X(gen);
            vy[i] = Y(gen);
            vz[i] = Z(gen);
            nx[i] = 0;
            ny[i] = 0;
            nz[i] = 0;
            v0[i] = sqrt(vx[i]*vx[i] * vy[i]*vy[i] + vz[i]*vz[i]);
            K += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        }
        std::cout << std::endl << vx[100] << "speed" << std::endl;

        /*gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long int Seed = 23410981;
		gsl_rng_set(r, Seed);
		for (int i = 0; i < N; i++) {
			vx[i] = gsl_ran_exponential(r, 3.0);
			vy[i] = gsl_ran_exponential(r, 3.0);
			vz[i] = gsl_ran_exponential(r, 3.0);
            nx[i] = 0;
            ny[i] = 0;
            nz[i] = 0;
            v0[i] = sqrt(vx[i]*vx[i] * vy[i]*vy[i] + vz[i]*vz[i]);
            K += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
		}*/
		
    }
    
    void placement(){
        int n = int(pow(N, 1.0/3.0));
        int m = 0;
        double dist = pow(1.0/den, 1.0/3.0);
        for(int i = 0; i <= n; i++){
            for(int j = 0; j <= n; j++){
                for(int k = 0; k <= n; k++){
                    x[m] = double((i + 0.5)*dist);
                    y[m] = double((j + 0.5)*dist);
                    z[m] = double((k + 0.5)*dist);
                    x0[m] = x[m];
                    y0[m] = y[m];
                    z0[m] = z[m];
                    m++;
                }
            }
        }
        N = m;
    }
    
    void interaction(){
        U = 0;
        for(int i = 0; i < N; i++){
            wx[i] = 0.0;
            wy[i] = 0.0;
            wz[i] = 0.0;
        }
        
        double dx, dy, dz, r2, r6, r12, f;
        #pragma omp parallel for private(dx, dy, dz, r2, r6, f) reduction(+,-:U,wx,wy,wz)
        for (int i = 0; i < (N - 1); i++) {
			for (int j = i + 1; j < N; j++) {
				dx  = (x[i] - x[j]);
				dy  = (y[i] - y[j]);
				dz  = (z[i] - z[j]);
				if (dx > L/2)       dx -= L;
				else if (dx < -L/2) dx += L;
				if (dy > L/2)       dy -= L;
				else if (dy < -L/2) dy += L;
				if (dz > L/2)       dz -= L;
				else if (dz < -L/2) dz += L;
				r2 = dx * dx + dy * dy + dz * dz;
				if (r2 < rc*rc){ 
					r6 = 1.0 / (r2 * r2 * r2);
					f = (a*r6 * r6 - b * r6)/mass;
                    U += f;
                    #pragma omp atomic
					wx[i] += dx * f / r2;
                    #pragma omp atomic
					wx[j] -= dx * f / r2;
                    #pragma omp atomic
					wy[i] += dy * f / r2;
                    #pragma omp atomic
					wy[j] -= dy * f / r2;
                    #pragma omp atomic
					wz[i] += dz * f / r2;
                    #pragma omp atomic
					wz[j] -= dz * f / r2;
				}
			}
		}
    }
    
    void show(){
        std::cout << vx[100] <<" " << wx[100] << std::endl;
    }
        
    void motion(){
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] += vx[i]*dt + wx[i]*dt*dt/2;
            y[i] += vy[i]*dt + wy[i]*dt*dt/2;
            z[i] += vz[i]*dt + wz[i]*dt*dt/2;
            if (x[i] >= L) {x[i] = (x[i] - L); nx[i] += 1;}
            if (y[i] >= L) {y[i] = (y[i] - L); ny[i] += 1;}
            if (z[i] >= L) {z[i] = (z[i] - L); nz[i] += 1;}
            if (x[i] < 0) {x[i] = L + x[i]; nx[i] -= 1;}
            if (y[i] < 0) {y[i] = L + y[i]; ny[i] -= 1;}
            if (z[i] < 0) {z[i] = L + z[i]; nz[i] -= 1;}
        }
    }
    
    void changev(){
        K = 0;
        #pragma omp parallel for reduction(+:K)
        for (int i = 0; i < N; i++) {
            vx[i] += 0.5*wx[i] * dt;
            vy[i] += 0.5*wy[i] * dt;
            vz[i] += 0.5*wz[i] * dt;
            K += (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]) / 2;
        }
    }
    
    void print(int t){
        double lambda = 1;
        
        std::ofstream F;
        std::string name = std::to_string(t) + ".xyz";
        F.open(name);
		F << N << std::endl;
        F << "Lattice=\" " << L << " 0.0 0.0 0.0 " << L << " 0.0 0.0 0.0 " << L << " \"\n";
        for (int i = 0; i < N; i++) {
            F << 16 << '\t' << x[i] / lambda << '\t' << y[i] / lambda << '\t' << z[i] / lambda << '\t' << vx[i] / lambda << '\t' << vy[i] / lambda << '\t' << vz[i] / lambda << std::endl;
        }
        F.close();
    }
    
    void printf(){
        int *distr;
        double Vx, Vy, Vz;
        
        
        std::ofstream foute;
        foute.open("energy.txt", std::ios::app);
        foute << U + K << '\t' << U << '\t' << K << std::endl;
        foute.close();
        
        std::ofstream foutep;
        foutep.open("momentum.txt", std::ios::app);
        for(int i = 0; i < N; i++){
            Vx += vx[i];
            Vy += vy[i];
            Vz += vz[i];
        }
        foutep << Vx << '\t' << Vy << '\t' << Vz << std::endl;
        foutep.close();
        Vx = Vy = Vz = 0;
        
        std::ofstream foutex;
        double avedevx, avedevy, avedevz;
        foutex.open("deviation.txt", std::ios::app);
        for(int i = 0; i < N; i++){
            avedevx += pow(x[i] + nx[i] * L - x0[i], 2);
            avedevy += pow(y[i] + ny[i] * L - y0[i], 2);
            avedevz += pow(z[i] + nz[i] * L - z0[i], 2);
        }
        foutex << avedevx/N << '\t' << avedevy/N << '\t' << avedevz/N <<std::endl;
        foutex.close();
        
        
        std::ofstream fouter;
        double vv_0, vv, v_i;
        fouter.open("akfs.txt", std::ios::app);
        for(int i = 0; i < N; i++){
            v_i = sqrt(vx[i]*vx[i] * vy[i]*vy[i] + vz[i]*vz[i]);
            vv_0 += v_i*v0[i];
            vv += v_i*v_i;
            v_i = 0;
        }
        fouter << vv_0/vv <<std::endl;
        fouter.close();
        
    }
    
    ~Model(){
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] x0;
        delete[] y0;
        delete[] z0;
        delete[] v0;
        delete[] vx;
        delete[] vy;
        delete[] vz;
        delete[] wx;
        delete[] wy;
        delete[] wz; 
        delete[] nx;
        delete[] ny;
        delete[] nz;
    }
};



int main(  int argc, char * argv[]) {
    double rc0 = 10*sigmal;
    double a0 = 48;//4*epsilon*pow(sigmal, 12);
    double b0 = 24;//4*epsilon*pow(sigmal, 6);
    Model *mod;
    mod = new Model(dt0, den0, rc0, a0, b0, N0, Temperature, kBoltzmann, mass, sigmal, rad);
    mod->init();
    mod->placement();
    mod->randomv();
    mod->interaction();
    for(int t = 0; t <= timenumbers; t++){
        if(t%50 == 0){mod->print(t);}
        mod->motion();
        mod->changev();
        mod->interaction();
        mod->changev();
        if(t%100 == 0){mod->printf();}
        
        progress(timenumbers, t);
    }
    delete mod;
    //system("python graph.py");
    return 0;
}
