#include <stdio.h>
#include <float.h>

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
    unsigned skimmed : 1; //logical variable denoting whether subinterval is skimmed
} Subintegral;

typedef struct extr {
    double list[52]; //Lower two tables of the epsilon table
    double prevlist[3]; //List of three most recent elements
    int index; //Index of epsilon table
    double error; //Error found in epsilon extrapolation
    double result; //Result from epsilon table
    int calls; //Number of calls to the extrapolation procedure
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

typedef struct res {
    Subintegral original;
    Subintegral* results;
    double totalerror;
    double totalresult;
    int divisions;
    int nskimmed; 
} Result;

typedef struct dev {
    Subintegral* list; //list containing left end point, right end points, result, and error estimate 
    Result* result; //list containing results of the singular intervals
    double* totalresult; //total results over the list
    double* totalerror; //total error over the list
    int* index; //index on the device side
} Device;

__device__ double f(double x) {
    return 1 / (1 + (x * x));
}

void flagError(Integrand*, int);
void setvalues(Subintegral*, Integrand*, double, int, int);


__global__ void CUDA_qk15i(double, int, Subintegral*);
__global__ void setTotals(Subintegral* list, double* errorsum, double* resultsum);

void fqk15i(Device device, int bound, int inf, double* resultsum, double* errorsum)
{

    CUDA_qk15i <<<1, 15>>> (bound, inf, device.list);

    cudaMemcpy(resultsum, &device.list[0].result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(errorsum, &device.list[0].error, sizeof(double), cudaMemcpyDeviceToHost);
    setTotals<<<1,1>>>(device.list, device.totalerror, device.totalresult);
    
}

__global__ void dqk15i(Subintegral*, Result*, int, int, int, int*, double*, double*);
__global__ void checkRoundOff(Integrand*, Result*, int*);
__global__ void skimValues(Subintegral*, Result*, int, int*, double, double, double*);

void wqk15i(Device device, Integrand* integrand, int bound, int inf, int* index, double abserr_thresh, double relerr_thresh, double* errorsum, double* resultsum)
{
    int oindex; //Original index before quadrature

    /* Reset necissary arguments */
    oindex = *index;
    cudaMemset(device.index, 0, sizeof(int)); //Resets device index for allocating memory

    /* Perform Dynamic Gauss-Kronrod Quadrature */
    dqk15i <<<oindex+1, 1>>> (device.list, device.result, bound, inf, oindex, device.index, device.totalerror, device.totalresult);
    /* Check round off error */
    //checkRoundOff <<<oindex+1, 1>>> (integrand, device.result, device.index);
    /* Skim results */
    cudaMemset(device.index, 0, sizeof(int)); //Resets device index for allocating memory
    skimValues <<<oindex+1, 1>>> (device.list, device.result, oindex, device.index, abserr_thresh, relerr_thresh, device.totalresult);

    /* Copy results necissary to the CPU side */
    cudaMemcpy(index, device.index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultsum, device.totalresult, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(errorsum, device.totalerror, sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void qk15i(Subintegral, Subintegral*, int, int, int);
__device__ int findDivisions(double, double, int, int);
__device__ double sumResults(Subintegral*, int);
__device__ double sumError(Subintegral*, int);
__device__ Subintegral* alloclist(Subintegral*, int*, int);
__device__ double dbl_atomicAdd(double*, double);

/* Memory to be dynamically allocated */
__device__ Subintegral allocmem[MAX_SUBINTERVALS_ALLOWED];

__global__ void dqk15i(Subintegral* list, Result* rlist, int bound, int inf, int oindex, int* nindex, double* errorsum, double* resultsum)
{
    int tindex; //Unique Thread index
    int divisions; //Amount of divisions allocated to subinterval
    double toterror; //Total error over subinterval
    double totresult; //Total result over the interval
    Subintegral original; //Original interval before divisions
    Subintegral* memindex; //Position in global memory to return results

    tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex <= oindex) {
        original = list[tindex];
        /* Find the amount of divisions and allocate amount of corresponding memory */
        divisions = findDivisions(list[tindex].error, *errorsum, oindex, MAX_TOTALDIVISIONS_ALLOWED);
        memindex = alloclist(allocmem, nindex, divisions); 
        /* Perform Dynamic Gauss-Kronrod Quadrature*/
        qk15i <<<divisions, 1>>> (original, memindex, bound, inf, divisions);
        cudaDeviceSynchronize();
    
    /* Improve previous approximations to integral and error and test for accuracy  */
        totresult = sumResults(memindex, divisions);
        toterror = sumError(memindex, divisions);
        dbl_atomicAdd(errorsum, toterror - original.error);
        dbl_atomicAdd(resultsum, totresult - original.result);
    /* Append results to interfunctional list */
        rlist[tindex].original = original;
        rlist[tindex].results = memindex;
        rlist[tindex].totalresult = totresult;
        rlist[tindex].totalerror = toterror;
        rlist[tindex].divisions = divisions;
        rlist[tindex].nskimmed = 0;
    }
}

__device__ int checkRO(Subintegral*, int);

__global__ void checkRoundOff(Integrand* integrand, Result* rlist, int* index) 
{
    int tindex; //Unique Thread index
    int divisions; //Amount of divisions allocated to subinterval
    double toterror; //Total error over subinterval
    double totresult; //Total result over the interval
    Subintegral original; //Original subintegral split
    Subintegral* list; //Position in global memory to return results

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tindex <= *index) {
        divisions = rlist[tindex].divisions;
        toterror = rlist[tindex].totalerror;
        totresult = rlist[tindex].totalresult;
        list = rlist[tindex].results;
        original = rlist[tindex].original;
        /* Checking roundoff error */
        if (checkRO(list, divisions)) {
            if (ABS(original.result - totresult) <= 1.0E-05 * ABS(totresult)
                && 9.9E-01 * original.error <= toterror)
                if (integrand->extrap)
                    atomicAdd(&(integrand->iroff2), 1);
                else
                    atomicAdd(&(integrand->iroff1), 1);
            if (*index > 10 && original.error < toterror)
                atomicAdd(&(integrand->iroff3), 1);
        }
    }
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
        interval->skimmed = 0;

        /* Calculating error */
        interval->error = ABS((resultk - resultg) * hlength);

        if (interval->resasc != 0 && interval->error != 0) //traditonal way to calculate error
            interval->error = interval->resasc * MIN(1, pow(200 * interval->error / interval->resasc, 1.5));
        if (interval->resabs > DBL_MIN / (DBL_EPSILON * 50)) //Checks roundoff error
            interval->error = MAX((DBL_EPSILON / 50) * interval->resabs, interval->error);
    }
}

__device__ int findDivisions(double error, double errorsum, int index, int maxallowed) {

    int allowed = maxallowed - ((index + 1) * 2); //Amount of extra divisions to be distributed  
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

__device__ int checkRO(Subintegral* results, int num)
{
    for (int i = 0; i < num; i++)
        if (results[i].resasc == results[i].error)
            return 0;
    return 1;
}

__device__ Subintegral* alloclist(Subintegral* list, int* index, int amount)
{
    int old; //Older copy of variable
    int mindex;

    old = *index;
    do {
        mindex = old;
        old = atomicCAS(index, mindex, mindex + amount);
    } while (mindex != old);
    return mindex + list;
}

__global__ void flag(Subintegral* results, int index, double errorbound, int* nskimmed) 
{
    int tindex; //Unique thread identifier

    tindex = threadIdx.x + blockIdx.x * blockDim.x;

    /* If results within toleration, mark them for skimming */
    if (tindex <= index && results[tindex].error <= errorbound * (results[tindex].b - results[tindex].a)) {
        results[tindex].skimmed = 1;
        *nskimmed++;
    }
}

__global__ void skimValues(Subintegral* list, Result* results, int oindex, int* index, double abserr_thresh, double relerr_thresh, double* resultsum)
{
    int tindex; //Unique thread identifier
    int slength; //Length of skimmed list
    int length; //Length of the non-skimmed list
    Subintegral* slist; //Skimmed list
    Subintegral* nslist; //Non-Skimmed list
    double errorbound; //Error bound used for flagging intervals

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    length = results[tindex].divisions;
 
    /* Start flagging results */
    errorbound = MAX(abserr_thresh, relerr_thresh * ABS(*resultsum));
    flag <<<length, 1>>> (results[tindex].results, oindex, errorbound, &results[tindex].nskimmed);
    nslist = results[tindex].results;
    slength = length - results[tindex].nskimmed;
    if (slength == 0) {
        /* Find Positions in Global Memory */
        slist = alloclist(list, index, slength);
        /* Place intervals that aren't skimmed into global */
        while (slength > 0) {
            if (!nslist[length].skimmed) {
                slist[slength-1] = nslist[length-1];
                slength--; length--;
            }
        }
    }

}

__global__ void setTotals(Subintegral* list, double* errorsum, double* resultsum) 
{
    *errorsum = list[0].error;
    *resultsum = list[0].result;
}

__global__ void setInterval(Subintegral* list) 
{
    list[0].a = 0;
    list[0].b = 1;
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
    Integrand integrand;
    Device device;
    int index;
    double errorsum;
    double resultsum;

    integrand.ier = 0;
    integrand.evaluations = 0;
    integrand.result = 0;
    integrand.abserror = 0;
    index = 0;

    /* Allocate device side memory */
    Subintegral* d_list; cudaMalloc((void**) &d_list, sizeof(Subintegral) * MAX_SUBINTERVALS_ALLOWED);
    Result* d_results; cudaMalloc((void**) &d_results, sizeof(Result) * MAX_SUBINTERVALS_ALLOWED);
    double* d_toterror;cudaMalloc((void**) &d_toterror, sizeof(double));
    double* d_totresult; cudaMalloc((void**) &d_totresult, sizeof(double));
    int* d_index; cudaMalloc((void**) &d_index, sizeof(double));
    device.list = d_list;
    device.result = d_results;
    device.totalerror = d_toterror;
    device.totalresult = d_totresult;
    device.index = d_index;
    /* Set first interval to (1,0) */
    setInterval<<<1,1>>>(device.list);
    /* Parallel Gauss-Kronrod Quadrature */
    fqk15i(device, 0, BOTH_INF, &resultsum, &errorsum);

    wqk15i(device, &integrand, 0, BOTH_INF, &index, 0.1, 0.1, &errorsum, &resultsum);
    Subintegral list[MAX_SUBINTERVALS_ALLOWED];
    cudaMemcpy(list, device.list, sizeof(Subintegral) * MAX_SUBINTERVALS_ALLOWED, cudaMemcpyDeviceToHost);

    printf("%d\n", index);
    for (int i = 0; i < index; i++)
        printf("%f %f\n", list[i].result, list[i].error);
    printf("\n");
    printf("%f %f\n", resultsum, errorsum);
}
