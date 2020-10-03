#include <stdio.h>
#include <float.h>

#define NEGATIVE_INF -1 //(-inf, b)
#define POSITIVE_INF 1 //(a, inf)
#define BOTH_INF 2 //(-inf, inf)
#define MAX_ITERATIONS 50 //Number of cycles allowed before quit
#define MAX_SUBINTERVALS_ALLOWED 30
#define MAX_EXTRADIVISIONS_ALLOWED 10

enum {
    NORMAL, MAX_ITERATIONS_ALLOWED = 0x1, ROUNDOFF_ERROR = 0x2, BAD_INTEGRAND_BEHAVIOR = 0x4,
    TOLERANCE_CANNOT_BE_ACHIEVED = 0x8, DIVERGENT = 0x10, INVALID_INPUT = 0x20, NO_SPACE = 0x40
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
    int currentevals; //Total number of evaluations over entire program
    int ier; //bits for error flagging
    int iroff1, iroff2, iroff3; //flags for the amount of round off error detected through three different types
    int extrap; //logical variable denoting whether the algorithm is attempting extrapolation or not
    int noext; //logical variable denoting whether extrapolation is no longer allowed
} Integrand;

typedef struct res {
    Subintegral original; //Original result before quadrature
    Subintegral* results; //List of intervals from the quadrature
    double totalerror; //total error over list
    double totalresult; //total result over list
    int divisions; //total divisions allocated
    int nskimmed; //number of results to be skimmed in list
} Result;

typedef struct dev {
    Subintegral* list; //list containing left end point, right end points, result, and error estimate 
    Result* result; //list containing results of the singular intervals
    Integrand* integrand; //structure representing the integrand
    double* totalresult; //total results over the list
    double* totalerror; //total error over the list
    int* index; //index on the device side
} Device;

__device__ double f(double x) {
    return  1 / (1 + x * x);
}

void flagError(Integrand*, int);
void setvalues(Integrand*, double, double, int, double*, double*, int*, int*);
void fqk15i(Device, int, int, double*, double*, double*, double*);
void wqk15i(Device, Integrand*, int, int, int*, double, double, double*, double*);
void extrapolate(Epsilontable*);
__host__ __device__ double _max(double a, double b);
__host__ __device__ double _min(double a, double b);
__host__ __device__ double _abs(double a);

void qagi(double bound, int inf, double abserror_thresh, double relerror_thresh, double* result, double* abserror, int* evaluations, int* ier)
{
    Integrand integrand; //structure representing variables related to the integrand or its state
    Device device; //structure representing reusable device side memory
    double errorsum; //total error so far
    double resultsum; //total results
    double errorbound; //max error requested
    double resasc; //approximation of F-I/(B-A) over transformed integrand
    double resabs; //approximation of the integral over the absolute value of the function
    int index; //index for the current subinterval in the sublist
    int iterations; //Amount of times iterated through loop
    int signchange; //logical variable indicating that there was a sign change over the interval

    /* Initializing Variables */
    *ier = NORMAL;
    *evaluations = 0;
    *result = 0;
    *abserror = 0;
    resultsum = 0;
    errorsum = 0;
    index = 0;
    iterations = 1;
    integrand.currentevals = (inf == BOTH_INF) ? 15 : 30;
    integrand.ier = NORMAL;
    integrand.iroff1 = 0;
    integrand.iroff2 = 0;
    integrand.iroff3 = 0;
    integrand.noext = 0;
    integrand.extrap = 0;
    /* Allocate device side memory and storing struct for reusable memory */
    Subintegral* d_list;    cudaMalloc((void**)&d_list, sizeof(Subintegral) * MAX_SUBINTERVALS_ALLOWED); device.list = d_list;
    Result* d_results;      cudaMalloc((void**)&d_results, sizeof(Result) * MAX_SUBINTERVALS_ALLOWED);   device.result = d_results;
    Integrand* d_integrand; cudaMalloc((void**)&d_integrand, sizeof(Integrand));                         device.integrand = d_integrand;
    double* d_toterror;     cudaMalloc((void**)&d_toterror, sizeof(double));                             device.totalerror = d_toterror;
    double* d_totresult;    cudaMalloc((void**)&d_totresult, sizeof(double));                            device.totalresult = d_totresult;
    int* d_index;           cudaMalloc((void**)&d_index, sizeof(double));                                device.index = d_index;
    /* Copy integrand data to device side */
    cudaMemcpy(device.integrand, &integrand, sizeof(Integrand), cudaMemcpyHostToDevice);

    /* Test for valid input */
    if (abserror_thresh < 0 && relerror_thresh < 0) {
        flagError(&integrand, INVALID_INPUT);
        return;
    }
    /* Autoset bound to 0 for unbounded intervals */
    if (inf == BOTH_INF)
        bound = 0;

    /* Calculate first quadrature */
    fqk15i(device, 0, inf, &resabs, &resasc, &resultsum, &errorsum);

    /* Test of accuracy */
    errorbound = _max(abserror_thresh, relerror_thresh * _abs(resultsum));

    if (errorsum <= 100 * DBL_EPSILON * resabs && errorbound < errorsum) //checks if round off error and if the error is above threshhold
                flagError(&integrand, ROUNDOFF_ERROR);
        if (iterations == MAX_ITERATIONS - 1)
                flagError(&integrand, MAX_ITERATIONS);
        if (integrand.ier != NORMAL || (errorsum <= errorbound && errorsum != resasc) || errorsum == 0) { //ends if it has an error, within the bounds of error threshhold, or if error is zero
                *result = resultsum;
                *evaluations = integrand.currentevals;
                *abserror = errorsum;
                return;
        }
        if ((1 - DBL_EPSILON / 2) * resabs <= _abs(resultsum))
                signchange = 0;
        else
                signchange = 1;
        
    /* Start disecting interval */

    /* Variables for extrapolation defined */
    Epsilontable epsiltable; //epsilon table used for extrapolation
    double epsilerror; //error calculated in epsilon table
    double epsilresult; //results calculated in epsilon table
    double ex_errorbound; //errorbound used in extrapolation
    int ktmin; //amount of times extrapolated with no decrease in error
    double correc; //the amount of error added in total error if roundoff detected in extrapolation

    epsilresult = resultsum;
    epsilerror = DBL_MAX;
    epsiltable.list[0] = resultsum;
    ktmin = 0;

    for (iterations = 2; iterations < MAX_ITERATIONS; iterations++) {

        /* Start Waterfall Quadrature */
        wqk15i(device, &integrand, bound, inf, &index, abserror_thresh, relerror_thresh, &errorsum, &resultsum);

        //Set error flags
        if (10 <= integrand.iroff1 + integrand.iroff2 || 20 <= integrand.iroff3)
            flagError(&integrand, ROUNDOFF_ERROR);
        if (iterations >= MAX_ITERATIONS)
            flagError(&integrand, MAX_ITERATIONS_ALLOWED);
        if ((index+1) * 2 >= MAX_SUBINTERVALS_ALLOWED)
            flagError(&integrand, NO_SPACE);
        //if (_max(_abs(a1), _abs(b2)) <= (1 + 1000 * DBL_EPSILON) * (_abs(a2) + 1000 * DBL_MIN))
            //flagError(integrand, BAD_INTEGRAND_BEHAVIOR);
        
        if (errorsum <= errorbound) { //If error is under requested threshhold, add up all results and end program
            setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
            return;
        }

        if (integrand.ier != NORMAL) //If error detected, break out of loop and let error handling there
            break;

        if (iterations == 1) {//If first iteration, initialize extrapolation variables and start next iteration
            epsiltable.list[1] = resultsum;
            epsiltable.index = 1;
            epsiltable.calls = 0;
            ex_errorbound = errorbound;
            continue;
        }

        if (integrand.noext) //start next iteration if extrapolation not allowed
            continue;

        /* Extrapolation */
        int epsili = ++epsiltable.index;
        epsiltable.list[epsili] = resultsum;
        extrapolate(&epsiltable);
        ktmin++;

        if (5 < ktmin && epsilerror < 1.0E-03 * errorsum)
                flagError(&integrand, TOLERANCE_CANNOT_BE_ACHIEVED);
        if (epsiltable.error < epsilerror) { //Check if error found was less than previous error
                ktmin = 0; //Reset ktmin
                epsilerror = epsiltable.error;
                epsilresult = epsiltable.result;
                ex_errorbound = max(abserror_thresh, relerror_thresh * _abs(epsiltable.result));

                if (*abserror <= ex_errorbound)
                        break;
        }

        if (epsiltable.index == 0)
            integrand.noext = 1;
        if (integrand.ier == TOLERANCE_CANNOT_BE_ACHIEVED)
            break;
    }

    /* Set final results and error */
	if (epsilerror == DBL_MAX) { //if no extrapolation was necissary, set values
		setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
		return;
	}

	if (integrand.ier == 0 && 5 > integrand.iroff2) { //if no problems arise in extrapolation, evaluate if divergent and set values

		if (signchange && _max(_abs(epsilresult), _abs(resultsum)) <= resabs * 1.0E-02)
            setvalues(&integrand, epsilresult, epsilerror, inf, result, abserror, evaluations, ier);
		else if (1.0E-02 > epsilresult / resultsum || (epsilresult / resultsum) > 1.0E+02 || errorsum > _abs(resultsum)) {
			flagError(&integrand, DIVERGENT);
			setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
        }
        else setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);

		return;
	}

	if (5 <= integrand.iroff2) //If round off error flaged in extrapolation table, modify error
		epsilerror += correc;

	if (integrand.ier == 0)
		flagError(&integrand, ROUNDOFF_ERROR);

	if (epsilresult != 0 && resultsum != 0) {

		if (errorsum / _abs(resultsum) >= epsilerror / _abs(epsilresult)) {

			if (signchange && max(_abs(epsilresult), _abs(resultsum)) <= resabs * 1.0E-02) 
                setvalues(&integrand, epsilresult, epsilerror, inf, result, abserror, evaluations, ier);
			else if (1.0E-02 > epsilresult / resultsum || (epsilresult / resultsum) > 1.0E+02 || errorsum > _abs(resultsum)) {
				flagError(&integrand, DIVERGENT);
			    setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
			}
			return;
		}
        else setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
		
		return;
	}

	if (errorsum < epsilerror) {
		setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
		return;
	}

	if (resultsum == 0) {
		setvalues(&integrand, epsilresult, epsilerror, inf, result, abserror, evaluations, ier);
		return;
	}

	if (signchange && _max(_abs(epsilresult), _abs(resultsum)) <= resabs * 1.0E-02) { //Test if divergent
        setvalues(&integrand, epsilresult, epsilerror, inf, result, abserror, evaluations, ier);
        return;
	}
	else if (1.0E-02 > epsilresult / resultsum || (epsilresult / resultsum) > 1.0E+02 || errorsum > _abs(resultsum)) {
		flagError(&integrand, DIVERGENT);
		setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);
		return;
    }
    else setvalues(&integrand, resultsum, errorsum, inf, result, abserror, evaluations, ier);

    return;
}



/**********************************************/
/**********************************************/
/************HOST FUNCTIONS********************/
/**********************************************/
/**********************************************/

/*
    Evaluates initial interval using parallelized Gauss-Kronrod
    Quadrature. It then intitializes errorsum and resultsum on
    both host and device side.

    Parameters:
        device - structure meant to represent reusable device memory
        bound - finite bound for semi-infinite integrals.
                Default is 0.
        inf - constant used to denote which direction the
                integral is infinite
        resabs - integral of the absolute value of the function
        resasc - integral of F-I/(B-A) over transformed integrand
        errorsum - total error over the entire list
        resultsum - total results over the entire list
*/

__global__ void CUDA_qk15i(double, int, Subintegral*);
__global__ void setTotals(Subintegral*, double*, double*);
__global__ void setInterval(Subintegral*);

void fqk15i(Device device, int bound, int inf, double* resabs, double* resasc, double* resultsum, double* errorsum)
{
    /* Set first interval to (0,1) */
    setInterval<<<1, 1>>>(device.list);
    /* Perform Initial Gauss-Kronrod Calculation */
    CUDA_qk15i <<<1, 15 >>> (bound, inf, device.list);
    /* Copy result and error to device side total error and results */
    setTotals <<<1, 1 >>> (device.list, device.totalerror, device.totalresult);
    /* Copy result and error to host side total error and results */
    cudaMemcpy(resultsum, &device.list[0].result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(errorsum, &device.list[0].error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resasc, &device.list[0].resasc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resabs, &device.list[0].resabs, sizeof(double), cudaMemcpyDeviceToHost);
}

/*
    Evaluates entire list of interval using a "water fall" method.
    This method uses multiple threads to evluate each interval and
    determine the appropriate amount of divisions which depends on
    the percentage of error in each interval. Round off error is
    then checked and flagged accordingly. Intervals that have
    a tolerable amount of error are then "skimmed" also known as
    being taken out of circulation.

    Parameters:
        device - structure meant to represent reusable device memory
        integrand - structure meant to represent variables associated
                    with the integrand as a whole
        bound - finite bound for semi-infinite integrals.
                Default is 0.
        inf - constant used to denote which direction the
                integral is infinite
        index - current index of list
        abserr_thresh - absolute error threshold
        relerr_thresh - relative error threshold
        errorsum - total error over the entire list
        resultsum - total results over the entire list
*/

__global__ void dqk15i(Subintegral*, Result*, int, int, int, int*, double*, double*);
__global__ void checkRoundOff(Integrand*, Result*, int, int*);
__global__ void skimValues(Subintegral*, Result*, int, int*, double, double, double*);
__global__ void fixindex(int*);
void updateEvaluations(int*, int*, int);

void wqk15i(Device device, Integrand* integrand, int bound, int inf, int* index, double abserr_thresh, double relerr_thresh, double* errorsum, double* resultsum)
{
    int currentevals; //Current evaluations
    int oindex; //Original index before quadrature

    oindex = *index;
    currentevals = integrand->currentevals;
    cudaMemcpy(device.integrand, integrand, sizeof(Integrand), cudaMemcpyHostToDevice);

    /* Perform Dynamic Gauss-Kronrod Quadrature */
    cudaMemset(device.index, 0, sizeof(int)); //Resets device index for allocating memory
    dqk15i <<<oindex + 1, 1>>> (device.list, device.result, bound, inf, oindex, device.index, device.totalerror, device.totalresult);
    fixindex <<<1,1>>> (device.index);

    updateEvaluations(device.index, &currentevals, inf);

    /* Check round off error */
    checkRoundOff <<<oindex + 1, 1>>> (device.integrand, device.result, oindex, device.index);
    /* Skim results */
    cudaMemset(device.index, 0, sizeof(int)); //Resets device index for allocating memory
    skimValues <<<oindex + 1, 1>>> (device.list, device.result, oindex, device.index, abserr_thresh, relerr_thresh, device.totalresult);
    fixindex <<<1,1>>> (device.index);

    /* Copy results necissary to the CPU side */
    cudaMemcpy(index, device.index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(integrand, device.integrand, sizeof(Integrand), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultsum, device.totalresult, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(errorsum, device.totalerror, sizeof(double), cudaMemcpyDeviceToHost);
    /* Set total evaluations */
    integrand->currentevals = currentevals;
}

/*
    Extrapolates series convergence of result
    of integrand using epsilon algorithm.

    Parameters:
        table - the epsilon table for Wynn's
                epsilon algorithm
*/
void extrapolate(Epsilontable* table)
{
	double error; //Absolute error calulated by the table
	double result; //Result calculated by the table
	int n; //Amount of elements in the Epsilontable
	int k; //Index of the Epsilontable 
	int num; //Number of elements originally in the Epsilon table
	int extraplim; //Maximum number of elements allowed in the table
	int newelements; //Number of new elements added to the table
	double e0, e1, e2, e3; //Elements of the Lozenge used to arrange the Hankel matrix
	double error1, error2, error3; //Absolute value of the difference between e3 and e1 ,e2 and e1, e1 and e0 respectively
	double tol1, tol2, tol3; //Tolerance allowed in the calculations determining convergence
	double ss;
	double epsinf;

	table->calls++;
	table->error = DBL_MAX;
	n = table->index;
	table->result = table->list[n];

	if (n < 2) { //Not enough elements to make an extrapolation
		table->error = max(table->error, (DBL_EPSILON / 2) * _abs(table->result)); //Not enough elements to make an extrapolation
		return;
	}

	extraplim = 50;
	table->list[n + 2] = table->list[n];
	table->list[n] = DBL_MAX;
	newelements = n / 2;
	k = num = n;

	for (int i = 0; i < newelements; i++) {
		e0 = table->list[k - 2];
		e1 = table->list[k - 1];
		e2 = table->list[k + 2];
		error2 = _abs(e2 - e1);
		error3 = _abs(e1 - e0);
		tol2 = DBL_EPSILON * max(_abs(e1), _abs(e2));
		tol3 = DBL_EPSILON * max(_abs(e1), _abs(e0));

		/* If e0, e1, and e2 are equal to within machine accuracy, convergence is assumed */
		if (error2 <= tol2 && error3 <= tol3) {
			table->result = e2;
			table->error = max(error2 + error3, DBL_EPSILON / 2 * _abs(table->result));
			return;
		}

		e3 = table->list[k];
		table->list[k] = e1;
		error1 = _abs(e1 - e3);
		tol1 = DBL_EPSILON * max(_abs(e1), _abs(e3));

		/* If two elements are very close to eachother, omit a portion of the table */
		if (error1 <= tol1 || error2 <= tol2 || error3 <= tol3) {
			n = i + i;
			break;
		}

		ss = 1 / (e1 - e3) + 1 / (e2 - e1) - 1 / (e1 - e0);
		epsinf = _abs(ss * e1);

		/* Testing for irregular behavior */
		if (epsinf <= 1.0E-04) {
			n = i + i;
			break;
		}
		
		/* Compute new element */
		result = e1 + 1 / ss;
		table->list[k] = result;
		k -= 2;
		error = error2 + _abs(result - e2) + error3;

		if (error <= table->error) {
			table->error = error;
			table->result = result;
		}
	}

	/* Shift table */
	n = (n == (extraplim - 1)) ? 2 * ((extraplim-1) / 2) - 1 : n;
	int ib = ((num+1) % 2 == 0) ? 1 : 0;
	int ie = newelements + 1;

	for (int i = 0; i < ie; i++) {
		table->list[ib] = table->list[ib + 2];
		ib += 2;
	}

	if (num != n) { //If portion of table was detected to be irregular, shift over table to get rid of irregular portion
		int index = num - n;
		for (int i = 0; i < n; i++) {
			table->list[i] = table->list[index];
			index++;
		}
	}

	if (table->calls < 4) { //Not enough extrapolated values to make an error estimate
		int calls = table->calls-1;
		table->prevlist[calls] = table->result;
		table->error = DBL_MAX;
	}
	else { //If enough extrapolated values, estimate the error with the 3 previous results
		table->error = _abs(table->result - table->prevlist[2]) + _abs(table->result - table->prevlist[1]) + 
			_abs(table->result - table->prevlist[0]);
		table->prevlist[0] = table->prevlist[1];
		table->prevlist[1] = table->prevlist[2];
		table->prevlist[2] = result;
	}
	table->error = max(table->error, (DBL_EPSILON / 2) * _abs(table->result));
	table->index = n;

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
    Function used to update number of evaluations

    Parameters:
        d_index - device side index pointer
        currentevals - the current evaluations
        inf - constant used to denote which direction the
              integral is infinite
*/
void updateEvaluations(int* d_index, int* currentevals, int inf)
{
    int* index; //Index found

    cudaMemcpy(index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
    *currentevals += (inf == BOTH_INF) ? 2 * ((*index+1) * 15) : (*index+1) * 15;
}

/*
    Function used to finish up the program by setting the correct
    values to the integrand

    Parameters:
        list - list of subintervals bisected
        integrand - structure representing the bundle of variables associated with
                    the integrand
        resultsum - total results
        errorsum - total error
        totali - total intervals over entire program
        inf - constant denoting which direction the integral
              is infinite
*/
void setvalues(Integrand* integrand, double resultsum, double errorsum, int inf, double* result, double* abserror, int* evaluations, int* ier)
{
    *result = resultsum;
    *evaluations = integrand->currentevals;
    *abserror = errorsum;
    *ier = integrand->ier;
}



/**********************************************/
/**********************************************/
/************GLOBAL FUNCTIONS******************/
/**********************************************/
/**********************************************/

/*
    Uses multiple threads to launch Gauss-Kronrod Quadrature over
    every interval. Each interval gets a dynamically allocated
    amount of divisions with each one getting at least 2. These
    results are then appended to rlist and used checkRoundOff and
    skimValues.

    Parameters:
        list - list of device sided intervals to be divided
        rlist - list of results from original intervals that are
                used in other kernels
        bound - finite bound for semi-infinite integrals.
                Default is 0.
        inf - constant used to denote which direction the
                integral is infinite
        oindex - original index before Gauss-Krondrod Quadrature
        nindex - new index after Gauss-Kronrod Quadrature
        errorsum - total error over the entire list
        resultsum - total results over the entire list
*/

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
        divisions = findDivisions(list[tindex].error, *errorsum, oindex, MAX_EXTRADIVISIONS_ALLOWED);
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

/*
    Uses multiple threads to evaluate round of error in results
    found from each interval divided in dqk15i.

    Parameters:
        integrand - structure meant to represent variables associated
                    with the integrand as a whole
        rlist - list of results from original intervals that are
                used in other kernels
        oindex - original index of list
        nindex - new index of list
*/
__device__ int checkRO(Subintegral*, int);

__global__ void checkRoundOff(Integrand* integrand, Result* rlist, int oindex, int* nindex)
{
    int tindex; //Unique Thread index
    int divisions; //Amount of divisions allocated to subinterval
    double toterror; //Total error over subinterval
    double totresult; //Total result over the interval
    Subintegral original; //Original subintegral split
    Subintegral* list; //Position in global memory to return results

    tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex <= oindex) {
        divisions = rlist[tindex].divisions;
        toterror = rlist[tindex].totalerror;
        totresult = rlist[tindex].totalresult;
        list = rlist[tindex].results;
        original = rlist[tindex].original;
        /* Checking roundoff error */
        if (checkRO(list, divisions)) {
            if (_abs(original.result - totresult) <= 1.0E-05 * _abs(totresult)
                && 9.9E-01 * original.error <= toterror)
                if (integrand->extrap)
                    atomicAdd(&(integrand->iroff2), 1);
                else
                    atomicAdd(&(integrand->iroff1), 1);
            if (*nindex > 10 && original.error < toterror)
                atomicAdd(&(integrand->iroff3), 1);
        }
    }
}

/*
    Used to apply multiple Gauss-Kronrod Quadratures over
    one interval. The amount is dependent on amount of
    divisions allocated in dqk15i. First divides initial
    interval, performs Gauss-Kronrod Quadrature, then puts
    result into array of Subintegrals.

    Parameters:
        intitial - intitial interval to be divided
        list - list to return results to
        bound - finite bound for semi-infinite integrals.
                Default is 0.
        inf - constant used to denote which direction the
                integral is infinite
        divisions - number of divisions to be done on the
                    initial interval
*/

__global__ void qk15i(Subintegral initial, Subintegral* list, int bound, int inf, int divisions)
{
    double delx; //the distance inbetween each point evaluated
    int tindex; //unique thread identifier

    tindex = threadIdx.x + blockIdx.x * blockDim.x;
    delx = (initial.b - initial.a) / divisions;
    /* Creating interval to be disected */
    list[tindex].a = delx * tindex + initial.a;
    list[tindex].b = delx * (tindex + 1) + initial.a;
    /* Multiple Gauss-Kronrod Quadrature */
    CUDA_qk15i <<<1, 15>>> (bound, inf, list + tindex);
}

/*
    CUDA translated 15-point Gauss Quadrature.

    Parameters:
        bound - finite bound for semi-infinite integrals.
                Default is 0.
        inf - constant used to denote which direction the
                integral is infinite
        interval - interval to be evaluated
*/
__global__ void CUDA_qk15i(double bound, int inf, Subintegral* interval)
{
    double xk[] = { //arguments for Gauss-Kronrod quadrature
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
    dinf = _min(1, inf);
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
    dbl_atomicAdd(&resulta, wgk[tindex] * _abs(fval));
    __syncthreads();

    /* Calculate resasc */
    mean_value = resultk / 2;
    dbl_atomicAdd(&resultasc, wgk[tindex] * _abs(fval - mean_value));
    __syncthreads();

    if (tindex == 0) {
        interval->result = resultk * hlength;
        interval->resasc = resultasc * hlength;
        interval->resabs = resulta * hlength;
        interval->skimmed = 0;

        /* Calculating error */
        interval->error = _abs((resultk - resultg) * hlength);

        if (interval->resasc != 0 && interval->error != 0) //traditonal way to calculate error
            interval->error = interval->resasc * _min(1, pow(200 * interval->error / interval->resasc, 1.5));
        if (interval->resabs > DBL_MIN / (DBL_EPSILON * 50)) //Checks roundoff error
            interval->error = _max((DBL_EPSILON / 50) * interval->resabs, interval->error);
    }
}

/*
    Flags list of results to be "skimmed" or also known as
    taking an interval out of circulation.

    Parameters:
        results - results to be flagged
        index - current index of list
        errorbound - tolerable error over total
                     error in list
        nskimmed - number of skimmed results found
*/
__global__ void flag(Subintegral* results, int index, double errorbound, int* nskimmed)
{
    int tindex; //Unique thread identifier

    tindex = threadIdx.x + blockIdx.x * blockDim.x;

    /* If results within toleration, mark them for skimming */
    if (tindex <= index && results[tindex].error <= errorbound * (results[tindex].b - results[tindex].a)) {
        results[tindex].skimmed = 1;
        atomicAdd(nskimmed, 1);
    }
}

/*
    Skims values from results and returns them to
    the main list.

    Parameters:
        list - main list to returb intervals to
        results - struct representing important variables to each
                  divided interval
        oindex - original index before Gauss-Kronrod Quadrature
        nindex - new index after Gauss-Kronrod Quadrature and skimming
        abserr_thresh - absolute error threshold
        relerr_thresh - relative error threshold
        errorsum - total error over the entire list
        resultsum - total results over the entire list
*/

__global__ void skimValues(Subintegral* list, Result* results, int oindex, int* nindex, double abserr_thresh, double relerr_thresh, double* resultsum)
{
    int tindex; //Unique thread identifier
    int slength; //Length of skimmed list
    int length; //Length of the non-skimmed list
    Subintegral* slist; //Skimmed list
    Subintegral* nslist; //Non-Skimmed list
    double errorbound; //Error bound used for flagging intervals

    tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex <= oindex) {
        length = results[tindex].divisions;
        /* Start flagging results */
        errorbound = _max(abserr_thresh, relerr_thresh * _abs(*resultsum));
        flag <<<length, 1 >>> (results[tindex].results, length, errorbound, &results[tindex].nskimmed);
        cudaDeviceSynchronize();
        nslist = results[tindex].results;
        slength = length - results[tindex].nskimmed;
        if (slength != 0) {
            /* Find Positions in Global Memory */
            slist = alloclist(list, nindex, slength);
            /* Place intervals that aren't skimmed into global */
            while (slength > 0) {
                if (!nslist[length].skimmed) {
                    slist[slength - 1] = nslist[length - 1];
                    slength--;
                }
                length--;
            }
        }
    }
}

/*
    Kernel used to set the reuseable device memory of
    totalerror and totalresults to the result and error
    of the first intitial value after fqk15i.

    Parameters:
        list - device sided list to hold subintegrals
        totalerror - device sided memory for total error
                     in the list
        totalresult - device sided memory for total results
                      in the list
*/

__global__ void setTotals(Subintegral* list, double* totalerror, double* totalresult)
{
    *totalerror = list[0].error;
    *totalresult = list[0].result;
}

/*
    Kernel used to set the first interval to boundary of
    (0,1). Used in fqk15i.

    Parameters:
        list - device sided list to hold subintegrals
*/
__global__ void setInterval(Subintegral* list)
{
    list[0].a = 0;
    list[0].b = 1;
}

/*
    Kernel used to fix nindex by subtracting one
    
    Parameters:
        index - device sided index
*/
__global__ void fixindex(int* index) {
        (*index)--;
}



/**********************************************/
/**********************************************/
/************DEVICE FUNCTIONS******************/
/**********************************************/
/**********************************************/

/*
    Finds the appropriate amount of divisions to be allocated
    depending on percentage of error in interval.

    Parameters:
        error - amount of error in interval
        errorsum - total error over entire list
        index - current index of list
        extrallowed - extra divisions allowed after minimum of two are allowed
*/
__device__ int findDivisions(double error, double errorsum, int index, int extrallowed) {
    int extraspace; //how much extra space is left

    /*extraspace = MAX_SUBINTERVALS_ALLOWED - ((index+1) * 2 + extrallowed);
    extrallowed = (extraspace >= 0) ? extrallowed : extrallowed + extraspace;
    return (int)((error / errorsum) * extrallowed) + 2; //Gives out a default of 2 threads and gives excess to intervals with high error
    */
    return 2;
}

/*
    Sums all results calulated.

    Parameters:
        results - list of calculated intervals
        num - amount of intervals
*/
__device__ double sumResults(Subintegral* results, int num)
{
    double res = 0;
    for (int i = 0; i < num; i++)
        res += results[i].result;
    return res;
}

/*
    Sums all error found.

    Parameters:
        results - list of calculated intervals
        num - amount of intervals
*/
__device__ double sumError(Subintegral* results, int num)
{
    double err = 0;
    for (int i = 0; i < num; i++)
        err += results[i].error;
    return err;
}

/*
    Checks if every member in list satisfies this
    condition: resasc == error

    Parameters:
        results - list of calculated intervals
        num - amount of intervals
*/
__device__ int checkRO(Subintegral* results, int num)
{
    for (int i = 0; i < num; i++)
        if (results[i].resasc == results[i].error)
            return 0;
    return 1;
}

/*
    Allocates space in a list.

    Parameters:
        list - list to be allocated
        index - current index of list
        amount - amount of space needed
*/
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

__host__ __device__ double _max(double a, double b)
{
    return ((a < b) ? b : a);
}

__host__ __device__ double _min(double a, double b) 
{
    return ((a < b) ? a : b);
}

__host__ __device__ double _abs(double a)
{
    return ((a < 0) ? -a : a);
}

int main()
{
    double result;
    double abserror;
    int evaluations;
    int ier;

    qagi(0, BOTH_INF, 0.1, 0.1, &result, &abserror, &evaluations, &ier);

    printf("Results: %f\nError: %f\nEvaluations: %d\n", result, abserror, evaluations);

}
