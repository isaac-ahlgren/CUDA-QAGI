#include <stdio.h>
#include <float.h>
#include <stdlib.h>

#define NEGATIVE_INF -1 //(-inf, b)
#define POSITIVE_INF 1 //(a, inf)
#define BOTH_INF 2 //(-inf, inf)
#define MAX_ITERATIONS 50 //Number of cycles allowed before quit
#define MAX_SUBINTERVALS_ALLOWED 10
#define MAX_TOTALDIVISIONS_ALLOWED 10

#define ABS(x) ((x < 0) ? -x : x)
#define MAX(x, y) ((x < y) ? y : x)
#define MIN(x, y) ((x < y) ? x : y)

enum {
    NORMAL, MAX_ITERATIONS_ALLOWED = 0x1, ROUNDOFF_ERROR = 0x2, BAD_INTEGRAND_BEHAVIOR = 0x4,
    TOLERANCE_CANNOT_BE_ACHIEVED = 0x8, DIVERGENT = 0x10, INVALID_INPUT = 0x20
};

typedef struct sintegral {
    double a; //left end point of interval
    double b; //right end point of interval
    double error; //error over the interval
    double result; //result over the interval
    double resasc; //approximation of F-I/(B-A) over transformed integrand
    double resabs; //approximation of the integral over the absolute value of the function
} Subintegral;

typedef struct extr {
    double list[52]; //lower two tables of the epsilon table
    double prevlist[3]; //list of three most recent elements
    int index; //Index of epsilon table
    double error; //error found in epsilon extrapolation
    double result;
    int calls; //number of calls to the extrapolation procedure
} Epsilontable;

typedef struct inte {
    char equation[50]; //String for equation to be parsed
    int evaluations; //number of evaluations of the integrand
    double result; //total result calculated
    double abserror; //total error in result calculated
    int ier; //bit for error flagging
    int iroff1, iroff2, iroff3; //flags for the amount of round off error detected through three different types
    int extrap; //logical variable denoting whether the algorithm is attempting extrapolation or not
    int noext; //logical variable denoting whether extrapolation is no longer allowed
} Integrand;

__device__ double f(double x) {
    return 1 / (1 + (x * x));
}

//Host Functions
void flagError(Integrand*, int);
void pqk15i(Subintegral*, double, int);
void setvalues(Subintegral*, Integrand*, double, int, int);
//Device Functions
__device__ double dbl_atomicAdd(double*, double);

void qagi(Integrand* integrand, int inf, double bound, double abserror_thresh, double relerror_thresh)
{
    Subintegral sublist[MAX_SUBINTERVALS_ALLOWED]; //list containing left end point, right end points, result, and error estimate
    int errori; //index for the largest interval error estimate
    double maxerror; //the maximum error found so far
    double lasterror; //the previous error estimate
    double errorsum; //total error so far
    double errorbound; //the max error requested
    double resultsum; //total results
    int index; //Index for the current subinterval in the sublist
    int signchange; //logical variable indicating that there was a sign change over the interval

    integrand->ier = 0;
    integrand->evaluations = 0;
    integrand->result = 0;
    integrand->abserror = 0;
    resultsum = 0;
    errorsum = 0;
    index = 0;

    if (abserror_thresh < 0 && relerror_thresh < 0) { //test for invalid input
        flagError(integrand, INVALID_INPUT);
        return;
    }

    if (inf == BOTH_INF) //bound autoset to 0 for unbounded intervals
        bound = 0;

    /* Creating first interval from 1 to 0 */
    sublist[index].a = 0;
    sublist[index].b = 1;
    /* Parallel Gauss-Kronrod Quadrature */
    pqk15i(sublist, bound, inf);
    /* Set result and error as sum of all results and errors */
    resultsum = sublist[index].result;
    errorsum = sublist[index].error;
    errori = index;

    /* Test of Accuracy */
    errorbound = MAX(abserror_thresh, relerror_thresh * ABS(resultsum));

    if (errorsum <= 100 * DBL_EPSILON * sublist[index].resabs && errorbound < errorsum) //checks if round off error and if the error is above threshhold
        flagError(integrand, ROUNDOFF_ERROR);
    if (index == MAX_ITERATIONS - 1)
        flagError(integrand, MAX_ITERATIONS_ALLOWED);
    if (integrand->ier != 0 || (errorsum <= errorbound && errorsum != sublist[index].resasc) || errorsum == 0) { //ends if it has an error, within the bounds of error threshhold, or if error is zero
        integrand->result = resultsum;
        integrand->evaluations = (inf == BOTH_INF) ? 2 * (15 + index * 30) : 15 + index * 30;
        integrand->abserror = errorsum;
        return;
    }
    if ((1 - DBL_EPSILON / 2) * sublist[index].resabs <= ABS(resultsum))
        signchange = 0;
    else
        signchange = 1;

    /* Start bisecting interval*/

    /* Variables for quadrature defined (1 and 2 are left and right subintervel respectively) */
    int nextit; //Logical variable for determining if the next iteration should start
    /* Variables for extrapolation defined */
    Epsilontable epsiltable; //epsilon table used for extrapolation
    double smallest; //Size of the smallest interval considered up to now multiplied by 1.5
    double large_errorsum; //Sum of errors besides the smallest one so far 
    double ex_errorbound; //Errorbound used in extrapolation
    int ktmin; //Amount of times extrapolated with no decrease in error
    double correc; //The amount of error added in total error if roundoff detected in extrapolation

    epsiltable.list[0] = resultsum;
    integrand->result = resultsum;
    integrand->abserror = DBL_MAX;
    integrand->iroff1 = 0;
    integrand->iroff2 = 0;
    integrand->iroff3 = 0;
    maxerror = errorsum;
    integrand->extrap = 0;
    integrand->noext = 0;
    ktmin = 0;
    nextit = 0;

    for (index = 1; index < MAX_ITERATIONS; index++) {
        lasterror = maxerror;


       /* A fucking sick ass function will be here */

        
    }
}

/*
    Function used to preform addition atomically with
    double precision.
    Parameters:
        address - pointer to memory address of double to be incremented
        val - the value to increment with
*/
__device__ double dbl_atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/*
    Function used to preform the 15-point Gauss-Kronrod quadrature
    over the desired inteval using parallel computation.
    Parameters:
        bound - bound for semi-infinite integrals
        inf - constant denoting which direction the integral
              is infinite over
        interval - structure denoting the area of the interval and
        the results and error
        resabs - the integral of the absolute value
                 of the integrals
        resasc -  the integral of the value of the integral
                  subtracted by the mean value of the integral
*/
__global__ void CUDA_qk15i(double, int, Subintegral*);

void pqk15i(Subintegral* interval, double bound, int inf) {
    
    /* Allocate memory space to the GPU */
    Subintegral* d_interval; cudaMalloc((void**)&d_interval, sizeof(Subintegral));
    /* Copy host memory to device */
    cudaMemcpy(d_interval, interval, sizeof(Subintegral), cudaMemcpyHostToDevice);
    /* Perform  Gaussian-Kronrod Quadrature */
    CUDA_qk15i << <1, 15 >> > (bound, inf, d_interval);
    /* Copy results back to host variables */
    cudaMemcpy(interval, d_interval, sizeof(Subintegral), cudaMemcpyDeviceToHost);
    /* Free Memory */
    cudaFree(d_interval);
}

__global__ void qk15i(Subintegral, Subintegral*, int, int, int);
__device__ int findDivisions(double, double, int, int);
__device__ double sumResults(Subintegral*, int);
__device__ double sumError(Subintegral*, int);
__device__ int checkRoundOff(Subintegral*, int);

__device__ Subintegral* allocmem[MAX_SUBINTERVALS_ALLOWED]; //Array of allocated memory for each subinterval
__device__ int allocl = 0; //Length of allocmem

__global__ void dqk15i(Integrand* integrand, Subintegral* list, int bound, int inf, double* errorsum, double* resultsum, int index)
{
    int tindex; //Unique Thread index
    int divisions; //Amount of divisions allocated to subinterval
    double toterror; //total error over subinterval
    double totresult; //total result over the interval
    Subintegral* results; //memory for each subsection of the subinterval

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    /* Find the amount of divisions and allocate amount of corresponding memory */
    divisions = findDivisions(list[tindex].error, *errorsum, index, MAX_TOTALDIVISIONS_ALLOWED);
    cudaMalloc((void**)&results, sizeof(Subintegral) * divisions);
    /* Perform Dynamic Gauss-Kronrod Quadrature*/
    qk15i <<<divisions, 1>>> (list[tindex], results, bound, inf, divisions);

    /* Improve previous approximations to integral and error and test for accuracy  */
    totresult = sumResults(results, divisions);
    toterror = sumError(results, divisions);
    dbl_atomicAdd(errorsum, toterror - list[tindex].error);
    dbl_atomicAdd(resultsum, totresult - list[tindex].result);
 
    //errorbound = MAX(abserror_thresh, relerror_thresh * ABS(resultsum)); move outside function

    //Checking roundoff error
    if (checkRoundOff(results, divisions)) {
        if (ABS(list[tindex].result - totresult) <= 1.0E-05 * ABS(totresult)
            && 9.9E-01 * list[tindex].error <= toterror)
            if (integrand->extrap)
                atomicAdd(&(integrand->iroff2), 1);
            else
                atomicAdd(&(integrand->iroff1), 1);
        if (index > 10 && list[tindex].error < toterror)
            atomicAdd(&(integrand->iroff3), 1);
    }
    //Set error flags - move all these out of function
    /*
    if (10 <= integrand->iroff1 + integrand->iroff2 || 20 <= integrand->iroff3)
        flagError(integrand, ROUNDOFF_ERROR);
    if (index == MAX_ITERATIONS - 1) //move outside of function
        flagError(integrand, MAX_ITERATIONS_ALLOWED);
   if (MAX(ABS(a), ABS(b)) <= (1 + 1000 * DBL_EPSILON) * (ABS(m) + 1000 * DBL_MIN)) //dont know how to translate
        flagError(integrand, BAD_INTEGRAND_BEHAVIOR);
    */
}


__global__ void qk15i(Subintegral initial, Subintegral* interval, int bound, int inf, int divisions)
{
    double delx; //the distance inbetween each point evaluated
    int tindex; //unique thread identifier
    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    delx = (initial.b - initial.a) / divisions;
    /* Creating interval to be disected */
    interval[tindex].a = delx * tindex; 
    interval[tindex].b = delx * (tindex + 1);
    /* Multiple Gauss-Kronrod Quadrature */
    CUDA_qk15i <<<1, 15>>> (bound, inf, interval + tindex);
}

/*
    CUDA translated 15-point Gauss Quadrature
*/
__global__ void CUDA_qk15i(double bound, int inf, Subintegral* interval)
{
    double xk[] = { //arguments for Gauss-Kronod quadrature
        0.0, 9.491079123427585E-01,
        8.648644233597691E-01, 7.415311855993944E-01,
        5.860872354676911E-01, 4.058451513773972E-01,
        2.077849550078985E-01, 9.914553711208126E-01
    };
    double wg[] = { //weight for Gauss rule
        4.179591836734694E-01, 1.294849661688697E-01,
        0.0, 2.797053914892767E-01,
        0.0, 3.818300505051189E-01,
        0.0, 0.0
    };
    double wgk[] = { //weight for Gauss-Kronrod rule
        2.094821410847278E-01, 6.309209262997855E-02,
        1.047900103222502E-01, 1.406532597155259E-01,
        1.690047266392679E-01, 1.903505780647854E-01,
        2.044329400752989E-01, 2.293532201052922E-02
    };

    __shared__ double resultg, resultk, resulta; //results for Gauss and Kronrod rules and the absolute value of Kronrod rule
    __shared__ double resultasc; //the integral of the value of the integral subtracted by the mean value of the integral
    double fval; //function evaluated at transformed arguments
    double x, sx; //arguments for the center of the subintervals and right or left subinterval depending on sign
    double transx; //transformed arguments
    double center, hlength; //Center of transformed integral and half length of integral
    int dinf; //Variable that changes the tranformation equation depending on the orientation of the infinite portion
    int tindex; //Index for thread
    double mean_value; //approximation of mean value over tranformed integrand
    int sign; //Determines if shifted argument is shifted right or left, determined by index

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    sign = (tindex > 7) ? -1 : 1; //If above 7, shift argument left, else shift right
    tindex -= (tindex > 7) ? 7 : 0; //Index's above 7 share same elements with the index 7 behind it
    dinf = MIN(1, inf);
    hlength = (interval->b - interval->a) / 2;
    center = (interval->a + interval->b) / 2;
    if (tindex == 0) {
        resultg = 0;
        resultk = 0;
        resulta = 0;
        resultasc = 0;
    }
    __syncthreads();

    /* Start computing the 15 point Kronrod estimation */

    x = hlength * xk[tindex]; //Shift center of subinterval
    sx = center + sign * x; //Shift either right or left
    transx = bound + dinf * (1 - sx) / sx; //tranform and evaluate using tranformation equation
    fval = f(transx);
    if (inf == BOTH_INF)
        fval += f(-transx);
    fval /= (sx * sx);
    dbl_atomicAdd(&resultg, wg[tindex] * fval);
    dbl_atomicAdd(&resultk, wgk[tindex] * fval);
    dbl_atomicAdd(&resulta, wgk[tindex] * ABS(fval));
    __syncthreads();

    /* Calculate resasc */
    mean_value = resultk / 2;
    dbl_atomicAdd(&resultasc, wgk[tindex] * ABS(fval - mean_value));
    __syncthreads();

    if (tindex == 0) {
        interval->result = resultk * hlength;
        interval->resasc = resultasc * hlength;
        interval->resabs = resulta * hlength;

        /* Calculating error */
        interval->error = ABS((resultk - resultg) * hlength);

        if (interval->resasc != 0 && interval->error != 0) //traditonal way to calculate error
            interval->error = interval->resasc * MIN(1, pow(200 * interval->error / interval->resasc, 1.5));
        if (interval->resabs > DBL_MIN / (DBL_EPSILON * 50)) //Checks roundoff error
            interval->error = MAX((DBL_EPSILON / 50) * interval->resabs, interval->error);
    }
}

/* 
    Could be optimized better, has a tendancy to pick
    maxallowed-1 divisions over maxallowed because of
    truncation.
*/
__device__ int findDivisions(double error, double errorsum, int index, int maxallowed) {

    int allowed = maxallowed - ((index + 1) * 2) + 1; //Amount of extra divisions to be distributed  
    return (int) ((error / errorsum) * allowed) + 2; //Gives out a default of 2 threads and gives excess to intervals with high error

}

__device__ double sumResults(Subintegral* results, int num)
{
    double res = 0;
    for (int i = 0; i < num; i++)
        res += results[i].result;
    return res;
}

__device__ double sumError(Subintegral* results, int num)
{
    double err = 0;
    for (int i = 0; i < num; i++)
        err += results[i].error;
    return err;
}

__device__ int checkRoundOff(Subintegral* results, int num)
{
    for (int i = 0; i < num; i++)
        if (results[i].resasc == results[i].error)
            return 0;
    return 1;
}

/*
    Function turns on bits to flag errors over
    the integrand.
    Parameters:
        integrand - the structure repesenting the integrand
        error - desired error to be flagged
*/
void flagError(Integrand* integrand, int error)
{
    integrand->ier |= error;
}


/*
    Function used to finish up the program by setting the correct
    values to the integrand
    
    Parameters:
        list - list of subintervals bisected
        integrand - structure representing the bundle of variables associated with
                    the integrand
        index - current index of the list
        inf - constant denoting which direction the integral
              is infinite 
*/
void setvalues(Subintegral* list, Integrand* integrand, double errorsum, int index, int inf)
{
    double res = 0;
    for (int i = 0; i <= index; i++)
        res += list[i].result;
    integrand->result = res;
    integrand->evaluations = (inf == BOTH_INF) ? 2 * (15 + index * 30) : 15 + index * 30;
    integrand->abserror = errorsum;
}

__global__ void test(Subintegral* list, double errorsum, int index)
{
    int tindex = tindex = threadIdx.x + blockIdx.x * blockDim.x;
    int div = findDivisions(list[tindex].error, errorsum, index, 30);
    printf("%f %d\n", list[tindex].error, div);
}

int main()
{
    /*
    Subintegral initial;
    initial.a = 0;
    initial.b = 1;
    Subintegral results[10];
    Subintegral* d_results;
    */
    /*
    cudaMalloc((void**)&d_results, sizeof(Subintegral)*10);
    qk15i<<<10, 1>>>(initial, d_results, 0, BOTH_INF, 10);
    cudaMemcpy(results, d_results, sizeof(Subintegral)*10, cudaMemcpyDeviceToHost);
    
    double result = 0;
    for (int i = 0; i < 10; i++)
        result += results[i].result;

    printf("%f\n", result);
    */

    Subintegral list[30];
    /*
    list[0].error = 8922.3440895228996;
    list[1].error = 24.824634333689687;
    list[2].error = 0.014152200425590322;
    list[3].error = 1.2654399636772547e-07;
    list[4].error = 5.9988279370659382e-08;
    list[5].error = 5.133955049180035e-08;
    list[6].error = 2.6745188194879989e-09;
    list[7].error = 1.6867272160903006e-10;
    list[8].error = 1.6826417701967357e-11;
    list[9].error = 9.4300300284172862e-15;
    */
    //double errorsum = 8947.1828762977511;
    /*
    list[0].error = 2130.3232293756005;
    list[1].error = 2.6745188194879989e-09;
    */
    //double errorsum = 2130.3232293782748;

    list[0].error = 100;
    list[1].error = 150;
    list[2].error = 300;
    list[3].error = 500;
    double errorsum = 1050;
    int index = 3;
    Subintegral* d_interval; cudaMalloc((void**)&d_interval, sizeof(Subintegral)*30);
    cudaMemcpy(d_interval, list, sizeof(Subintegral)*30, cudaMemcpyHostToDevice);
    test <<<1, 4 >>> (d_interval, errorsum, index);
    cudaFree(d_interval);

    
}
