#include <stdlib.h>
#include "ocl.h"

#define USE_RINTERNALS 1
#include <Rinternals.h>

/* Implementation of oclPlatforms */
SEXP ocl_platforms() {
    SEXP res;
    cl_uint np;
    cl_platform_id *pid;
    cl_int last_ocl_error;

    if ((last_ocl_error = clGetPlatformIDs(0, 0, &np)) != CL_SUCCESS)
	ocl_err("clGetPlatformIDs", last_ocl_error);
    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	pid = (cl_platform_id *) malloc(sizeof(cl_platform_id) * np);
        if (pid == NULL)
            Rf_error("Out of memory");
	if ((last_ocl_error = clGetPlatformIDs(np, pid, 0)) != CL_SUCCESS) {
	    free(pid);
	    ocl_err("clGetPlatformIDs", last_ocl_error);
	}
	PROTECT(res);
	for (i = 0; i < np; i++)
	    SET_VECTOR_ELT(res, i, mkPlatformID(pid[i]));
	free(pid);
	UNPROTECT(1);
    }
    return res;
}

/* Implementation of oclDevices */
SEXP ocl_devices(SEXP platform, SEXP sDevType) {
    cl_platform_id pid = getPlatformID(platform);
    SEXP res;
    cl_uint np;
    cl_device_id *did;
    cl_device_type dt = CL_DEVICE_TYPE_DEFAULT;
    const char *dts;
    cl_int last_ocl_error;

    if (TYPEOF(sDevType) != STRSXP || LENGTH(sDevType) != 1)
	Rf_error("invalid device type - must be a character vector of length one");
    dts = CHAR(STRING_ELT(sDevType, 0));
    if (dts[0] == 'C' || dts[0] == 'c')
	dt = CL_DEVICE_TYPE_CPU;
    else if (dts[0] == 'G' || dts[0] == 'g')
	dt = CL_DEVICE_TYPE_GPU;
    else if (dts[0] == 'A' || dts[0] == 'a') {
	if (dts[1] == 'C' || dts[1] == 'c')
	    dt = CL_DEVICE_TYPE_ACCELERATOR;
	else if (dts[1] == 'L' || dts[1] == 'l')
	    dt = CL_DEVICE_TYPE_ALL;
    }
    if (dt == CL_DEVICE_TYPE_DEFAULT && dts[0] != 'D' && dts[0] != 'd')
	Rf_error("invalid device type - must be one of 'cpu', 'gpu', 'accelerator', 'default', 'all'.");

    last_ocl_error = clGetDeviceIDs(pid, dt, 0, 0, &np);
    if (last_ocl_error != CL_SUCCESS &&
        last_ocl_error != CL_DEVICE_NOT_FOUND)
	ocl_err("clGetDeviceIDs", last_ocl_error);

    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	did = (cl_device_id *) malloc(sizeof(cl_device_id) * np);
        if (did == NULL)
            Rf_error("Out of memory");
	if ((last_ocl_error = clGetDeviceIDs(pid, dt, np, did, 0)) != CL_SUCCESS) {
	    free(did);
	    ocl_err("clGetDeviceIDs", last_ocl_error);
	}
	PROTECT(res);
	for (i = 0; i < np; i++)
	    SET_VECTOR_ELT(res, i, mkDeviceID(did[i]));
	free(did);
	UNPROTECT(1);
    }
    return res;
}

/* Implementation of oclContext */
SEXP ocl_context(SEXP device_exp)
{
    cl_device_id device_id = getDeviceID(device_exp);
    cl_context ctx;
    cl_command_queue queue;
    SEXP ctx_exp, queue_exp;
    cl_int last_ocl_error;

    ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &last_ocl_error);
    if (!ctx)
        ocl_err("clCreateContext", last_ocl_error);
    ctx_exp = PROTECT(mkContext(ctx));
    Rf_setAttrib(ctx_exp, oclDeviceSymbol, device_exp);

    // Add command queue
    queue = clCreateCommandQueue(ctx, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &last_ocl_error);
    if (!queue)
        ocl_err("clCreateCommandQueue", last_ocl_error);
    queue_exp = PROTECT(mkCommandQueue(queue));
    Rf_setAttrib(ctx_exp, oclQueueSymbol, queue_exp);

    UNPROTECT(2);
    return ctx_exp;
}

static SEXP getDeviceInfo(cl_device_id device_id, cl_device_info di) {
    char infobuf[2048];
    cl_int last_ocl_error = clGetDeviceInfo(device_id, di, sizeof(infobuf), &infobuf, NULL);
    if (last_ocl_error != CL_SUCCESS)
	ocl_err("clGetDeviceInfo", last_ocl_error);
    return Rf_mkString(infobuf);
}

static SEXP getPlatformInfo(cl_platform_id platform_id, cl_device_info di) {
    char infobuf[2048];
    cl_int last_ocl_error = clGetPlatformInfo(platform_id, di, sizeof(infobuf), &infobuf, NULL);
    if (last_ocl_error != CL_SUCCESS)
	ocl_err("clGetPlatformInfo", last_ocl_error);
    return Rf_mkString(infobuf);
}

/* Implementation of print.clDeviceID and oclInfo.clDeviceID */
SEXP ocl_get_device_info(SEXP device) {
    SEXP res;
    cl_device_id device_id = getDeviceID(device);
    const char *names[] = { "name", "vendor", "version", "profile", "exts", "driver.ver", "max.frequency" };
    size_t numAttr = sizeof(names) / sizeof(const char *);

    SEXP nv = PROTECT(Rf_allocVector(STRSXP, numAttr));
    int i;
    for (i = 0; i < LENGTH(nv); i++) SET_STRING_ELT(nv, i, mkChar(names[i]));

    res = PROTECT(Rf_allocVector(VECSXP, numAttr));
    Rf_setAttrib(res, R_NamesSymbol, nv);
    SET_VECTOR_ELT(res, 0, getDeviceInfo(device_id, CL_DEVICE_NAME));
    SET_VECTOR_ELT(res, 1, getDeviceInfo(device_id, CL_DEVICE_VENDOR));
    SET_VECTOR_ELT(res, 2, getDeviceInfo(device_id, CL_DEVICE_VERSION));
    SET_VECTOR_ELT(res, 3, getDeviceInfo(device_id, CL_DEVICE_PROFILE));
    SET_VECTOR_ELT(res, 4, getDeviceInfo(device_id, CL_DEVICE_EXTENSIONS));
    SET_VECTOR_ELT(res, 5, getDeviceInfo(device_id, CL_DRIVER_VERSION));
    cl_uint max_freq;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_freq), &max_freq, NULL);
    SET_VECTOR_ELT(res, 6, Rf_ScalarInteger(max_freq));

    UNPROTECT(2);
    return res;
}

/* Implementation of print.clPlatformID and oclInfo.clPlatformID */
SEXP ocl_get_platform_info(SEXP platform) {
    SEXP res;
    cl_platform_id platform_id = getPlatformID(platform);
    const char *names[] = { "name", "vendor", "version", "profile", "exts" };
    SEXP nv = PROTECT(Rf_allocVector(STRSXP, 5));
    int i;
    for (i = 0; i < LENGTH(nv); i++) SET_STRING_ELT(nv, i, mkChar(names[i]));
    res = PROTECT(Rf_allocVector(VECSXP, LENGTH(nv)));
    Rf_setAttrib(res, R_NamesSymbol, nv);
    SET_VECTOR_ELT(res, 0, getPlatformInfo(platform_id, CL_PLATFORM_NAME));
    SET_VECTOR_ELT(res, 1, getPlatformInfo(platform_id, CL_PLATFORM_VENDOR));
    SET_VECTOR_ELT(res, 2, getPlatformInfo(platform_id, CL_PLATFORM_VERSION));
    SET_VECTOR_ELT(res, 3, getPlatformInfo(platform_id, CL_PLATFORM_PROFILE));
    SET_VECTOR_ELT(res, 4, getPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS));
    UNPROTECT(2);
    return res;
}

static char buffer[2048]; /* kernel build error buffer */

/* Implementation of oclSimpleKernel */
SEXP ocl_ez_kernel(SEXP context, SEXP k_name, SEXP code, SEXP mode) {
    cl_context ctx = getContext(context);
    cl_device_id device = getDeviceID(getAttrib(context, oclDeviceSymbol));
    cl_program program;
    cl_kernel kernel;
    cl_int last_ocl_error;

    if (TYPEOF(k_name) != STRSXP || LENGTH(k_name) != 1)
	Rf_error("invalid kernel name");
    if (TYPEOF(code) != STRSXP || LENGTH(code) < 1)
	Rf_error("invalid kernel code");
    if (TYPEOF(mode) != STRSXP || LENGTH(mode) != 1)
	Rf_error("invalid output mode specification");

    {
	int sn = LENGTH(code), i;
	const char **cptr;
	cptr = (const char **) malloc(sizeof(char*) * sn);
        if (cptr == NULL)
            Rf_error("Out of memory");
	for (i = 0; i < sn; i++)
	    cptr[i] = CHAR(STRING_ELT(code, i));
	program = clCreateProgramWithSource(ctx, sn, cptr, NULL, &last_ocl_error);
	free(cptr);
	if (!program)
	    ocl_err("clCreateProgramWithSource", last_ocl_error);
    }

    last_ocl_error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (last_ocl_error != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	clReleaseProgram(program);
	Rf_error("clBuildProgram failed (with %d): %s", last_ocl_error, buffer);
    }

    kernel = clCreateKernel(program, CHAR(STRING_ELT(k_name, 0)), &last_ocl_error);
    clReleaseProgram(program);
    if (!kernel)
	ocl_err("clCreateKernel", last_ocl_error);

    {
	SEXP sk = PROTECT(mkKernel(kernel));
	Rf_setAttrib(sk, oclContextSymbol, context);
	Rf_setAttrib(sk, oclModeSymbol, mode);
	Rf_setAttrib(sk, oclNameSymbol, k_name);
	UNPROTECT(1);
	return sk;
    }
}

/* Implementation of oclRun */
/* .External */
SEXP ocl_call(SEXP args) {
    int on, an = 0;
    ClType ftype = CLT_DOUBLE;
    SEXP ker = CADR(args), olen, arg, dimVec;
    cl_kernel kernel = getKernel(ker);
    SEXP context_exp = getAttrib(ker, oclContextSymbol);
    cl_command_queue commands = getCommandQueue(getAttrib(context_exp, oclQueueSymbol));
    cl_mem output;
    cl_event output_wait;
    SEXP output_wait_exp;
    size_t wdims[3] = {0, 0, 0};
    cl_uint wdim = 1;
    cl_int last_ocl_error;

    /* Get (optional) arguments */
    args = CDDR(args);
    /* Get kernel precision */
    ftype = get_type(Rf_getAttrib(ker, oclModeSymbol));

    olen = CAR(args);  /* size */
    args = CDR(args);
    on = Rf_asInteger(olen);
    if (on < 0)
	Rf_error("invalid output length");

    dimVec = coerceVector(CAR(args), INTSXP);  /* dim */
    wdim = LENGTH(dimVec);
    if (wdim > 3)
	Rf_error("OpenCL standard only supports up to three work item dimensions - use index vectors for higher dimensions");
    if (wdim) {
	int i; /* we don't use memcpy in case int and size_t are different */
	for (i = 0; i < wdim; i++)
	    wdims[i] = INTEGER(dimVec)[i];
    }
    if (wdim < 1 || wdims[0] < 1 || (wdim > 1 && wdims[1] < 1) || (wdim > 2 && wdims[2] < 1))
	Rf_error("invalid dimensions - must be a numeric vector with positive values");
    args = CDR(args);

    cl_uint num_args, wait_events = 0;
    clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL);
    cl_event *input_wait = calloc(num_args - 2, sizeof(cl_event));
    if (input_wait == NULL)
        Rf_error("Out of memory");

    SEXP resultbuf = PROTECT(cl_create_buffer(context_exp, olen, Rf_getAttrib(ker, oclModeSymbol)));
    output = (cl_mem)R_ExternalPtrAddr(resultbuf);
    if (clSetKernelArg(kernel, an++, sizeof(cl_mem), &output) != CL_SUCCESS)
	Rf_error("failed to set first kernel argument as output in clSetKernelArg");
    if (clSetKernelArg(kernel, an++, sizeof(on), &on) != CL_SUCCESS)
	Rf_error("failed to set second kernel argument as output length in clSetKernelArg");
    while ((arg = CAR(args)) != R_NilValue) {
        if (TYPEOF(arg) == EXTPTRSXP) {
            // buffer argument
            cl_mem argument = getBuffer(arg);
            SEXP wait_exp = Rf_getAttrib(arg, oclEventSymbol);

            last_ocl_error = clSetKernelArg(kernel, an++, sizeof(cl_mem), &argument);
            if (last_ocl_error != CL_SUCCESS)
                Rf_error("Failed to set vector kernel argument %d (length=%d, error %d)", an, Rf_asInteger(cl_get_buffer_length(arg)), last_ocl_error);

            if (wait_events >= num_args - 2)
                Rf_error("More arguments than expected");
            if (TYPEOF(wait_exp) == EXTPTRSXP)
                input_wait[wait_events++] = getEvent(wait_exp);
        } else {
            // single-value argument
            if (LENGTH(arg) != 1)
                Rf_error("Non-buffer arguments must be scalar values");
            size_t size;
            void* data;
            float intermediate;
            switch (TYPEOF(arg)) {
                case REALSXP:
                    if (ftype == CLT_FLOAT) {
                        size = sizeof(float);
                        intermediate = (float)*REAL(arg);
                        data = &intermediate;
                    }
                    else {
                        size = sizeof(double);
                        data = REAL(arg);
                    }
                    break;
                case INTSXP:
                    size = sizeof(int);
                    data = INTEGER(arg);
                    break;
                default:
                    Rf_error("only numeric or integer scalar kernel arguments are supported");
            }

            last_ocl_error = clSetKernelArg(kernel, an++, size, data);
            if (last_ocl_error != CL_SUCCESS)
                Rf_error("Failed to set scalar kernel argument %d (error %d)", an, last_ocl_error);
        }
	args = CDR(args);
    }

    last_ocl_error = clEnqueueNDRangeKernel(commands, kernel, wdim, NULL, wdims, NULL,
        wait_events, wait_events ? input_wait : NULL, &output_wait);
    if (last_ocl_error != CL_SUCCESS)
	ocl_err("Kernel execution", last_ocl_error);
    free(input_wait);

    // Attach event to output buffer
    output_wait_exp = mkEvent(output_wait);
    Rf_setAttrib(resultbuf, oclEventSymbol, output_wait_exp);

    UNPROTECT(1);
    return resultbuf;
}
