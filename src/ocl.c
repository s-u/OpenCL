#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
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
    const char *dts;
    if (clGetDeviceIDs(pid, dt, 0, 0, &np) != CL_SUCCESS)
	ocl_err("clGetDeviceIDs");
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

#define FT_SINGLE 0
#define FT_DOUBLE 1

SEXP ocl_ez_kernel(SEXP device, SEXP k_name, SEXP code, SEXP prec) {
    cl_context ctx;
    int err;
    SEXP sctx;
    cl_device_id device_id = getDeviceID(device);
    cl_program program;
    cl_kernel kernel;
    int ftype;

    if (TYPEOF(k_name) != STRSXP || LENGTH(k_name) != 1)
	Rf_error("invalid kernel name");
    if (TYPEOF(code) != STRSXP || LENGTH(code) < 1)
	Rf_error("invalid kernel code");
    if (TYPEOF(prec) != STRSXP || LENGTH(prec) != 1)
	Rf_error("invalid precision specification");
    ftype = (CHAR(STRING_ELT(prec, 0))[0] == 'd') ? FT_DOUBLE : FT_SINGLE;
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
	Rf_setAttrib(sk, Rf_install("precision"), prec);
	UNPROTECT(2); /* sk + context */
	return sk;
    }
}


/*--- generic blockwise argument list that can be freed in one go ---*/
typedef void (*afin_t)(void*);

struct arg_chain {
    struct arg_chain *next;
    afin_t fin;
    int args, size;
    void *arg[1];
};

static struct arg_chain *arg_alloc(struct arg_chain *parent, int size) {
    struct arg_chain *c = (struct arg_chain*) malloc(sizeof(*c) + sizeof(void*) * size);
    if (!c)
	Rf_error("unable to allocate argument chain");
    c->next = 0;
    c->size = size;
    c->args = 0;
    c->fin = 0;
    if (parent)
	parent->next = c;
    return c;
}

static struct arg_chain *arg_add(struct arg_chain *where, void *arg) {
    if (!where)
	where = arg_alloc(0, 32);
    if (where->args >= where->size) {
	while (where->next) where = where->next;
	where = where->next = arg_alloc(where, 32);
    }
    where->arg[where->args++] = arg;
    return where;
}

static void arg_free(struct arg_chain *chain, afin_t fin) {
    int i, n = chain->args;
    if (chain->next)
	arg_free(chain->next, fin);
    for (i = 0; i < n; i++)
	if (fin) fin(chain->arg[i]); else free(chain->arg[i]);
    free(chain);
}

#if 0 /* unused - we use it as part of the call context instead */
static void free_protected_args(SEXP o) {
    arg_free((struct arg_chain*)R_ExternalPtrAddr(o), 0);
}

static SEXP protected_args(struct arg_chain *chain, afin_t fin) {
    SEXP res = R_MakeExternalPtr(chain, R_NilValue, R_NilValue);
    chain->fin = fin;
    R_RegisterCFinalizerEx(res, free_protected_args, TRUE);
    return res;
}
#endif

/* in order to clean up all the temporary OpenCL objects we
   keep them in this structure which we allocate as an external
   pointer so it will be clean up either by the garbage collector
   in case of an error or by us at the end of the call.
   This is rather important for memory obejcts since GPUs have
   only limited memory.
*/
typedef struct ocl_call_context {
    cl_mem output;
    cl_command_queue commands;
    void *float_out;
    struct arg_chain *float_args, *mem_objects;
} ocl_call_context_t;

static void ocl_call_context_fin(SEXP context) {
    ocl_call_context_t *ctx = (ocl_call_context_t*) R_ExternalPtrAddr(context);
    if (ctx) {
	if (ctx->output) clReleaseMemObject(ctx->output);
	if (ctx->float_args) arg_free(ctx->float_args, 0);
	if (ctx->float_out) free(ctx->float_out);
	if (ctx->mem_objects) arg_free(ctx->mem_objects, (afin_t) clReleaseMemObject);
	if (ctx->commands) clReleaseCommandQueue(ctx->commands);
	free(ctx);
	CAR(context) = 0; /* this allows us to call the finalizer manually */
    }
}

SEXP ocl_call(SEXP args) {
    struct arg_chain *float_args = 0;
    ocl_call_context_t *occ;
    int on, an = 0, ftype = FT_DOUBLE, ftsize, ftres;
    size_t global;
    SEXP ker = CADR(args), olen, arg, res, octx;
    cl_kernel kernel = getKernel(ker);
    cl_context context;
    cl_command_queue commands;
    cl_device_id device_id = getDeviceID(getAttrib(ker, Rf_install("device")));
    cl_mem output;
    cl_int err;

    if (clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(context), &context, NULL) != CL_SUCCESS || !context)
	Rf_error("cannot obtain kernel context via clGetKernelInfo");
    args = CDDR(args);
    res = Rf_getAttrib(ker, install("precision"));
    if (TYPEOF(res) == STRSXP && LENGTH(res) == 1 && CHAR(STRING_ELT(res, 0))[0] != 'd')
	ftype = FT_SINGLE;
    ftsize = (ftype == FT_DOUBLE) ? sizeof(double) : sizeof(float);
    olen = CAR(args);
    args = CDR(args);
    on = Rf_asInteger(olen);
    if (on < 0)
	Rf_error("invalid output length");
    ftres = (Rf_asInteger(CAR(args)) == 1) ? 1 : 0;
    if (ftype != FT_SINGLE) ftres = 0;
    args = CDR(args);    
    occ = (ocl_call_context_t*) calloc(1, sizeof(ocl_call_context_t));
    if (!occ) Rf_error("unable to allocate ocl_call context");
    octx = PROTECT(R_MakeExternalPtr(occ, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(octx, ocl_call_context_fin, TRUE);

    occ->output = output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ftsize * on, NULL, &err);
    if (!output)
	Rf_error("failed to create output buffer of %d elements via clCreateBuffer (%d)", on, err);
    if (clSetKernelArg(kernel, an++, sizeof(cl_mem), &output) != CL_SUCCESS)
	Rf_error("failed to set first kernel argument as output in clSetKernelArg");
    if (clSetKernelArg(kernel, an++, sizeof(on), &on) != CL_SUCCESS)
	Rf_error("failed to set second kernel argument as output length in clSetKernelArg");
    occ->commands = commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
	ocl_err("clCreateCommandQueue");
    if (ftype == FT_SINGLE) /* need conversions, create floats buffer */
	occ->float_args = float_args = arg_alloc(0, 32);
    while ((arg = CAR(args)) != R_NilValue) {
	int n, ndiv = 1;
	void *ptr;
	size_t al;
	
	switch (TYPEOF(arg)) {
	case REALSXP:
	    if (ftype == FT_SINGLE) {
		int i;
		float *f;
		double *d = REAL(arg);
		n = LENGTH(arg);
		f = (float*) malloc(sizeof(float) * n);
		if (!f)
		    Rf_error("unable to allocate temporary single-precision memory for conversion from a double-precision argument vector of length %d", n);
		for (i = 0; i < n; i++) f[i] = d[i];
		ptr = f;
		al = sizeof(float);
		arg_add(float_args, ptr);
	    } else {
		ptr = REAL(arg);
		al = sizeof(double);
	    }
	    break;
	case INTSXP:
	    ptr = INTEGER(arg);
	    al = sizeof(int);
	    break;
	case LGLSXP:
	    ptr = LOGICAL(arg);
	    al = sizeof(int);
	    break;
	case RAWSXP:
	    if (inherits(arg, "clFloat")) {
		ptr = RAW(arg);
		ndiv = al = sizeof(float);
		break;
	    }
	default:
	    Rf_error("only numeric or logical kernel arguments are supported");
	    /* no-ops but needed to make the compiler happy */
	    ptr = 0;
	    al = 0;
	}
	n = LENGTH(arg);
	if (ndiv != 1) n /= ndiv;
	if (n == 1) {/* scalar */
	    if (clSetKernelArg(kernel, an++, al, ptr) != CL_SUCCESS)
		Rf_error("Failed to set scalar kernel argument %d (size=%d)", an, al);
	} else {
	    cl_mem input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  al * n, NULL, NULL);
	    if (!input)
		Rf_error("Unable to create buffer for vector argument %d", an);
	    if (!occ->mem_objects)
		occ->mem_objects = arg_alloc(0, 32);
	    arg_add(occ->mem_objects, input);
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

    res = ftres ? Rf_allocVector(RAWSXP, on * sizeof(float)) : Rf_allocVector(REALSXP, on);
    if (ftype == FT_SINGLE) {
	if (ftres) {
	    if (clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * on, RAW(res), 0, NULL, NULL ) != CL_SUCCESS)
		Rf_error("Unable to transfer results");
	    PROTECT(res);
	    Rf_setAttrib(res, R_ClassSymbol, mkString("clFloat"));
	    UNPROTECT(1);
	} else {
	    /* float - need a temporary buffer */
	    float *fr = (float*) malloc(sizeof(float) * on);
	    double *r = REAL(res);
	    int i;
	    if (!fr)
		Rf_error("unable to allocate memory for temporary single-precision output buffer");
	    occ->float_out = fr;
	    if (clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * on, fr, 0, NULL, NULL ) != CL_SUCCESS)
		Rf_error("Unable to transfer results");
	    for (i = 0; i < on; i++)
		r[i] = fr[i];
	}
    } else if (clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(double) * on, REAL(res), 0, NULL, NULL ) != CL_SUCCESS)
	Rf_error("Unable to transfer results");

    ocl_call_context_fin(octx);
    UNPROTECT(1);
    return res;
}
