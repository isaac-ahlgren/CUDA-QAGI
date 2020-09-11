#include <stdio.h>
#include <float.h>
#include <stdlib.h>

#define NEGATIVE_INF -1 //(-inf, b)
#define POSITIVE_INF 1 //(a, inf)
#define BOTH_INF 2 //(-inf, inf)
#define MAX_ITERATIONS 50
#define MAX_ARRAY_LENGTH 100
#define MAX_SUBINTERVALS_ALLOWED 100
#define MAX_DIVISIONS_ALLOCATED 50

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
void pqk15i(double, int, Subintegral*, double*, double*);
void setvalues(Subintegral*, Integrand*, double, int, int);
//Device Functions
__device__ double dbl_atomicAdd(double*, double);

void qagi(Integrand* integrand, int inf, double bound, double abserror_thresh, double relerror_thresh)
{
    Subintegral sublist[MAX_ARRARY_LENGTH]; //list containing left end point, right end points, result, and error estimate
    int errori; //index for the largest interval error estimate
    double maxerror; //the maximum error found so far
    double lasterror; //the previous error estimate
    double errorsum; //total error so far
    double errorbound; //the max error requested
    double resultsum; //total results
    double resasc; //approximation of F-I/(B-A) over transformed integrand
    double resabs; //approximation of the integral over the absolute value of the function
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
    pqk15i(bound, inf, sublist, &resabs, &resasc);
    /* Set result and error as sum of all results and errors */
    resultsum = sublist[index].result
    errorsum = sublist[index].error;
    errori = index;

    /* Test of Accuracy */
    errorbound = MAX(abserror_thresh, relerror_thresh * ABS(resultsum));

    if (errorsum <= 100 * DBL_EPSILON * resabs && errorbound < errorsum) //checks if round off error and if the error is above threshhold
        flagError(integrand, ROUNDOFF_ERROR);
    if (index == MAX_ITERATIONS - 1)
        flagError(integrand, MAX_ITERATIONS_ALLOWED);
    if (integrand->ier != 0 || (errorsum <= errorbound && errorsum != resasc) || errorsum == 0) { //ends if it has an error, within the bounds of error threshhold, or if error is zero
        integrand->result = resultsum;
        integrand->evaluations = (inf == BOTH_INF) ? 2 * (15 + index * 30) : 15 + index * 30;
        integrand->abserror = errorsum;
        return;
    }
    if ((1 - DBL_EPSILON / 2) * resabs <= ABS(resultsum))
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
    integrand.extrap = 0;
    integrand.noext = 0;
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
__global__ void CUDA_qk15i(double, int, Subintegral*, double*, double*);

void pqk15i(double bound, int inf, Subintegral* interval, double* resabs, double* resasc) {
    
    /* Allocate memory space to the GPU */
    double* d_interval; cudaMalloc((void**)&d_interval, sizeof(Subintegral));
    double* d_resabs; cudaMalloc((void**)&d_resabs, sizeof(double));
    double* d_resasc; cudaMalloc((void**)&d_resasc, sizeof(double));
    /* Perform  Gaussian-Kronrod Quadrature */
    CUDA_qk15i << <1, 15 >> > (bound, inf, interval, d_resabs, d_resasc);
    /* Copy results back to host variables */
    cudaMemcpy(interval, d_interval, sizeof(Subintegral), cudaMemcpyDeviceToHost);
    cudaMemcpy(resabs, d_resabs, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resasc, d_resasc, sizeof(double), cudaMemcpyDeviceToHost);
    /* Free Memory */
    cudaFree(d_interval); cudaFree(d_resabs); cudaFree(d_resasc);
}

typedef struct interv {
    Subintegral interval; //Container holding a, b, result, and error
    double resasc; //the integral of I - I/(A-B)
    double resabs; //the integral of the absolute value of the function
} Part;

__global__ void qk15i(double, double, Part*, int, int, int);
__device__ int findDivisions(double, double, int);

__device__ Part* allocmem[MAX_SUBINTERVALS_ALLOWED]; //Array of allocated memory for each subinterval
__device__ int allocl = 0; //Length of allocmem

__global__ void dqk15i(Integrand* integrand, Subintegral* list, int bound, int inf, double* errorsum, double* resultsum, double maxerror, int index)
{
    int tindex; //Unique Thread index
    int divisions; //Amount of divisions allocated to subinterval
    double toterror; //total error over subinterval
    double totresult; //total result over the interval
    Part* results; //memory for each subsection of the subinterval

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    /* Find the amount of divisions and allocate amount of corresponding memory */
    divisions = findDivisions(list[tindex].error, *errorsum, MAX_DIVISIONS_ALLOCATED); 
    cudaMalloc((void**)&results, sizeof(Part) * divisions);
    /* Perform Dynamic Gauss-Kronrod Quadrature*/
    qk15i <<<divisions, 1>>> (list[tindex].a, list[tindex].b, results, bound, inf, divisions);

    /* Improve previous approximations to integral and error and test for accuracy  */
    totresult = result1 + result2;
    toterror = error1 + error2;
    dbl_atomicAdd(error, toterror - maxerror);
    dbl_atomicAdd(resultsum, totresult - list[tindex].result);
 
    errorbound = MAX(abserror_thresh, relerror_thresh * ABS(resultsum));//move outside function

    //Checking roundoff error
    if (resasc1 != error1 && resasc2 != error2) {
        if (ABS(list[tindex].result - totresult) <= 1.0E-05 * ABS(totresult)
            && 9.9E-01 * maxerror <= toterror)
            if (integrand->extrap)
                atomicAdd(&(integrand->iroff2), 1);
            else
                atomicAdd(&(integrand->iroff1), 1);
        if (index > 10 && maxerror < toterror)
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


__global__ void qk15i(double a, double b, Part* results, int bound, int inf, int divisions)
{
    double delx; //the distance inbetween each point evaluated
    int tindex; //unique thread identifier
    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    delx = (b - a) / divisions;
    /* Creating interval to be disected */
    results[tindex].interval.a = delx * tindex; 
    results[tindex].interval.b = delx * (tindex + 1);
    /* Multiple Gauss-Kronrod Quadrature */
    CUDA_qk15i <<<1, 15>>> (bound, inf, &(results[tindex].interval), &(results[tindex].resabs), &(results[tindex].resasc));
}

/*
    CUDA translated 15-point Gauss Quadrature
*/
__global__ void CUDA_qk15i(double bound, int inf, Subintegral* interval, double* resabs, double* resasc)
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
    sign = (index > 7) ? -1 : 1; //If above 7, shift argument left, else shift right
    tindex -= (index > 7) ? 7 : 0; //Index's above 7 share same elements with the index 7 behind it
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
        *resasc = resultasc * hlength;
        *resabs = resulta * hlength;

        /* Calculating error */
        interval->abserr = ABS((resultk - resultg) * hlength);

        if (*resasc != 0 && *abserr != 0) //traditonal way to calculate error
            interval->abserr = *resasc * MIN(1, pow(200 * *abserr / *resasc, 1.5));
        if (*resabs > DBL_MIN / (DBL_EPSILON * 50)) //Checks roundoff error
            interval->abserr = MAX((DBL_EPSILON / 50) * *resabs, *abserr);
    }
}

__device__ int findDivisions(double error, double errorsum, int maxallowed) {

    return (int)(((error / errorsum) * maxallowed < 2) ? (error / errorsum) * maxallowed : 2);

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

int main()
{
    Subintegral integral;
    integral.a = 0;
    integral.b = 1;
    Part results[10];
    Part* d_results;

    cudaMalloc((void**)&d_results, sizeof(Part)*10);
    qk15i<<<10, 1>>>(0, BOTH_INF, 10, integral, d_results);
    cudaMemcpy(results, d_results, sizeof(Part)*10, cudaMemcpyDeviceToHost);

    double result = 0;
    for (int i = 0; i < 10; i++)
        result += results[i].result;

    printf("%f\n", result);
    
}
