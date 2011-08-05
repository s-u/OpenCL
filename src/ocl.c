#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <Rinternals.h>

void ocl_err(const char *str) {
    Rf_error("%s failed", str);
}

static void clFreeFin(SEXP ref) {
    free(R_ExternalPtrAddr(ref));
}

static SEXP mkPlatformID(cl_platform_id id) {
    SEXP ptr;
    cl_platform_id *pp = (cl_platform_id*) malloc(sizeof(cl_platform_id));
    pp[0] = id;
    ptr = PROTECT(R_MakeExternalPtr(pp, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeFin, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clPlatformID"));
    UNPROTECT(1);
    return ptr;
}

static cl_platform_id getPlatformID(SEXP platform) {
    if (!Rf_inherits(platform, "clPlatformID") || TYPEOF(platform) != EXTPTRSXP)
	Rf_error("invalid platform");
    return ((cl_platform_id*)R_ExternalPtrAddr(platform))[0];
}

static SEXP mkDeviceID(cl_device_id id) {
    SEXP ptr;
    cl_device_id *pp = (cl_device_id*) malloc(sizeof(cl_device_id));
    pp[0] = id;
    ptr = PROTECT(R_MakeExternalPtr(pp, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeFin, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clDeviceID"));
    UNPROTECT(1);
    return ptr;
}

static cl_device_id getDeviceID(SEXP device) {
    if (!Rf_inherits(device, "clDeviceID") ||
	TYPEOF(device) != EXTPTRSXP)
	Rf_error("invalid device");
    return ((cl_device_id*)R_ExternalPtrAddr(device))[0];
}

static void clFreeContext(SEXP ctx) {
    clReleaseContext((cl_context)R_ExternalPtrAddr(ctx));
}

static SEXP mkContext(cl_context ctx) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeContext, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clContext"));
    UNPROTECT(1);
    return ptr;
}

static cl_context getContext(SEXP ctx) {
    if (!Rf_inherits(ctx, "clContext") ||
	TYPEOF(ctx) != EXTPTRSXP)
	Rf_error("invalid OpenCL context");
    return (cl_context)R_ExternalPtrAddr(ctx);
}

static void clFreeKernel(SEXP k) {
    clReleaseKernel((cl_kernel)R_ExternalPtrAddr(k));
}

static SEXP mkKernel(cl_kernel k) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(k, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeKernel, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clKernel"));
    UNPROTECT(1);
    return ptr;
}

static cl_kernel getKernel(SEXP k) {
    if (!Rf_inherits(k, "clKernel") ||
	TYPEOF(k) != EXTPTRSXP)
	Rf_error("invalid OpenCL kernel");
    return (cl_kernel)R_ExternalPtrAddr(k);
}

SEXP ocl_platforms() {
    SEXP res;
    cl_uint np;
    cl_platform_id *pid;
    if (clGetPlatformIDs(0, 0, &np) != CL_SUCCESS)
	ocl_err("clGetPlatformIDs");
    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	pid = (cl_platform_id *) malloc(sizeof(cl_platform_id) * np);
	if (clGetPlatformIDs(np, pid, 0) != CL_SUCCESS) {
	    free(pid);
	    ocl_err("clGetPlatformIDs");
	}
	PROTECT(res);
	for (i = 0; i < np; i++)
	    SET_VECTOR_ELT(res, i, mkPlatformID(pid[i]));
	free(pid);
	UNPROTECT(1);
    }
    return res;
}

SEXP ocl_devices(SEXP platform, SEXP sDevType) {
    cl_platform_id pid = getPlatformID(platform);
    SEXP res;
    cl_uint np;
    cl_device_id *did;
    cl_device_type dt = CL_DEVICE_TYPE_DEFAULT;
    if (clGetDeviceIDs(pid, dt, 0, 0, &np) != CL_SUCCESS)
	ocl_err("clGetDeviceIDs");
    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	did = (cl_device_id *) malloc(sizeof(cl_device_id) * np);
	if (clGetDeviceIDs(pid, dt, np, did, 0) != CL_SUCCESS) {
	    free(did);
	    ocl_err("clGetDeviceIDs");
	}
	PROTECT(res);
	for (i = 0; i < np; i++)
	    SET_VECTOR_ELT(res, i, mkDeviceID(did[i]));
	free(did);
	UNPROTECT(1);
    }
    return res;
}

SEXP ocl_get_device_info_char(SEXP device, SEXP item) {
    char buf[512];
    cl_device_id device_id = getDeviceID(device);
    cl_device_info pn = (cl_device_info) Rf_asInteger(item);
    buf[0] = 0;
    if (clGetDeviceInfo(device_id, pn, sizeof(buf), &buf, NULL) != CL_SUCCESS)
	ocl_err("clGetDeviceInfo");
    return Rf_mkString(buf);
}

static SEXP getDeviceInfo(cl_device_id device_id, cl_device_info di) {
    char buf[512];
    if (clGetDeviceInfo(device_id, di, sizeof(buf), &buf, NULL) != CL_SUCCESS)
	ocl_err("clGetDeviceInfo");
    return mkString(buf);
}

static SEXP getPlatformInfo(cl_platform_id platform_id, cl_device_info di) {
    char buf[512];
    if (clGetPlatformInfo(platform_id, di, sizeof(buf), &buf, NULL) != CL_SUCCESS)
	ocl_err("clGetPlatformInfo");
    return mkString(buf);
}

SEXP ocl_get_device_info(SEXP device) {
    SEXP res;
    cl_device_id device_id = getDeviceID(device);
    const char *names[] = { "name", "vendor", "version", "profile", "exts", "driver.ver" };
    SEXP nv = PROTECT(Rf_allocVector(STRSXP, 6));
    int i;
    for (i = 0; i < LENGTH(nv); i++) SET_STRING_ELT(nv, i, mkChar(names[i]));
    res = PROTECT(Rf_allocVector(VECSXP, LENGTH(nv)));
    Rf_setAttrib(res, R_NamesSymbol, nv);
    SET_VECTOR_ELT(res, 0, getDeviceInfo(device_id, CL_DEVICE_NAME));
    SET_VECTOR_ELT(res, 1, getDeviceInfo(device_id, CL_DEVICE_VENDOR));
    SET_VECTOR_ELT(res, 2, getDeviceInfo(device_id, CL_DEVICE_VERSION));
    SET_VECTOR_ELT(res, 3, getDeviceInfo(device_id, CL_DEVICE_PROFILE));
    SET_VECTOR_ELT(res, 4, getDeviceInfo(device_id, CL_DEVICE_EXTENSIONS));
    SET_VECTOR_ELT(res, 5, getDeviceInfo(device_id, CL_DRIVER_VERSION));
    UNPROTECT(2);
    return res;
}

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

SEXP ocl_ez_kernel(SEXP device, SEXP k_name, SEXP code) {
    cl_context ctx;
    int err;
    SEXP sctx;
    cl_device_id device_id = getDeviceID(device);
    cl_program program;
    cl_kernel kernel;

    if (TYPEOF(k_name) != STRSXP || LENGTH(k_name) != 1)
	Rf_error("invalid kernel name");
    if (TYPEOF(code) != STRSXP || LENGTH(code) < 1)
	Rf_error("invalid kernel code");
    ctx = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!ctx)
	ocl_err("clCreateContext");
    sctx = PROTECT(mkContext(ctx));
    {
	int sn = LENGTH(code), i;
	const char **cptr;
	cptr = (const char **) malloc(sizeof(char*) * sn);
	for (i = 0; i < sn; i++)
	    cptr[i] = CHAR(STRING_ELT(code, i));
	program = clCreateProgramWithSource(ctx, sn, cptr, NULL, &err);
	free(cptr);
	if (!program)
	    ocl_err("clCreateProgramWithSource");
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	clReleaseProgram(program);
	Rf_error("clGetProgramBuildInfo failed: %s", buffer);
    }

    kernel = clCreateKernel(program, CHAR(STRING_ELT(k_name, 0)), &err);
    if (!kernel) {
	clReleaseProgram(program);
	ocl_err("clCreateKernel");
    }

    /* FIXME: do we need to retain/release queue and program? */
    {
	SEXP sk = PROTECT(mkKernel(kernel));
	Rf_setAttrib(sk, Rf_install("device"), device);
	UNPROTECT(2); /* sk + context */
	return sk;
    }
}

SEXP ocl_call_double(SEXP args) {
    int on, an = 0;
    size_t global;
    SEXP ker = CADR(args), olen, arg, res;
    cl_kernel kernel = getKernel(ker);
    cl_context context;
    cl_command_queue commands;
    cl_device_id device_id = getDeviceID(getAttrib(ker, Rf_install("device")));
    cl_mem output;
    int err;

    if (clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(context), &context, NULL) != CL_SUCCESS || !context)
	Rf_error("cannot obtain kernel context via clGetKernelInfo");
    args = CDDR(args);
    olen = CAR(args);
    args = CDR(args);
    on = Rf_asInteger(olen);
    if (on < 0)
	Rf_error("invalid output length");
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * on, NULL, NULL);
    if (!output)
	Rf_error("failed to create output buffer via clCreateBuffer");
    if (clSetKernelArg(kernel, an++, sizeof(cl_mem), &output) != CL_SUCCESS)
	Rf_error("failed to set first kernel argument as output in clSetKernelArg");
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
	ocl_err("clCreateCommandQueue");
    while ((arg = CAR(args)) != R_NilValue) {
	int n;
	void *ptr;
	size_t al;
	
	switch (TYPEOF(arg)) {
	case REALSXP:
	    ptr = REAL(arg);
	    al = sizeof(double);
	    break;
	case INTSXP:
	    ptr = INTEGER(arg);
	    al = sizeof(int);
	    break;
	case LGLSXP:
	    ptr = LOGICAL(arg);
	    al = sizeof(int);
	    break;
	default:
	    Rf_error("only numeric or logical kernel arguments are supported");
	    /* no-ops but needed to make the compiler happy */
	    ptr = 0;
	    al = 0;
	}
	n = LENGTH(arg);
	if (n == 1) {/* scalar */
	    if (clSetKernelArg(kernel, an++, al, ptr) != CL_SUCCESS)
		Rf_error("Failed to set scalar kernel argument %d (size=%d)", an, al);
	} else {
	    cl_mem input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  al * n, NULL, NULL);
	    if (!input)
		Rf_error("Unable to create buffer for vector argument %d", an);
	    if (clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, al * n, ptr, 0, NULL, NULL) != CL_SUCCESS)
		Rf_error("Failed to transfer data (%d elements) for vector argument %d", n, an);
	    if (clSetKernelArg(kernel, an++, sizeof(cl_mem), &input) != CL_SUCCESS)
		Rf_error("Failed to set vector kernel argument %d (size=%d, length=%d)", an, al, n);
	    /* clReleaseMemObject(input); */
	}
	args = CDR(args);
    }

    global = on;
    if (clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL) != CL_SUCCESS)
	Rf_error("Error during kernel execution");
    clFinish(commands);

    res = Rf_allocVector(REALSXP, on);
    if (clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(double) * on, REAL(res), 0, NULL, NULL ) != CL_SUCCESS)
	Rf_error("Unable to transfer results");

    clReleaseMemObject(output);
    clReleaseCommandQueue(commands);
    return res;
}
