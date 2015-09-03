#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
#include <Rinternals.h>

static cl_int last_ocl_error;

void ocl_err(const char *str) {
    Rf_error("%s failed (oclError %d)", str, last_ocl_error);
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

#if 0 /* currently unused so disable for now to avoid warnings ... */
static cl_context getContext(SEXP ctx) {
    if (!Rf_inherits(ctx, "clContext") ||
	TYPEOF(ctx) != EXTPTRSXP)
	Rf_error("invalid OpenCL context");
    return (cl_context)R_ExternalPtrAddr(ctx);
}
#endif

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

/* Implementation of oclPlatforms */
SEXP ocl_platforms() {
    SEXP res;
    cl_uint np;
    cl_platform_id *pid;
    if ((last_ocl_error = clGetPlatformIDs(0, 0, &np)) != CL_SUCCESS)
	ocl_err("clGetPlatformIDs");
    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	pid = (cl_platform_id *) malloc(sizeof(cl_platform_id) * np);
	if ((last_ocl_error = clGetPlatformIDs(np, pid, 0)) != CL_SUCCESS) {
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

/* Implementation of oclDevices */
SEXP ocl_devices(SEXP platform, SEXP sDevType) {
    cl_platform_id pid = getPlatformID(platform);
    SEXP res;
    cl_uint np;
    cl_device_id *did;
    cl_device_type dt = CL_DEVICE_TYPE_DEFAULT;
    const char *dts;
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
    if ((last_ocl_error = clGetDeviceIDs(pid, dt, 0, 0, &np)) != CL_SUCCESS)
	ocl_err("clGetDeviceIDs");

    res = Rf_allocVector(VECSXP, np);
    if (np > 0) {
	int i;
	did = (cl_device_id *) malloc(sizeof(cl_device_id) * np);
	if ((last_ocl_error = clGetDeviceIDs(pid, dt, np, did, 0)) != CL_SUCCESS) {
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

static char infobuf[2048];

SEXP ocl_get_device_info_char(SEXP device, SEXP item) {
    cl_device_id device_id = getDeviceID(device);
    cl_device_info pn = (cl_device_info) Rf_asInteger(item);
    *infobuf = 0;
    if ((last_ocl_error = clGetDeviceInfo(device_id, pn, sizeof(infobuf), &infobuf, NULL)) != CL_SUCCESS)
	ocl_err("clGetDeviceInfo");
    return Rf_mkString(infobuf);
}

static SEXP getDeviceInfo(cl_device_id device_id, cl_device_info di) {
    if ((last_ocl_error = clGetDeviceInfo(device_id, di, sizeof(infobuf), &infobuf, NULL)) != CL_SUCCESS)
	ocl_err("clGetDeviceInfo");
    return Rf_mkString(infobuf);
}

static SEXP getPlatformInfo(cl_platform_id platform_id, cl_device_info di) {
    if ((last_ocl_error = clGetPlatformInfo(platform_id, di, sizeof(infobuf), &infobuf, NULL)) != CL_SUCCESS)
	ocl_err("clGetPlatformInfo");
    return Rf_mkString(infobuf);
}

/* Implementation of print.clDeviceID and oclInfo.clDeviceID */
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

#define FT_SINGLE 0
#define FT_DOUBLE 1

/* Implementation of oclSimpleKernel */
SEXP ocl_ez_kernel(SEXP device, SEXP k_name, SEXP code, SEXP prec) {
    cl_context ctx;
    SEXP sctx;
    cl_device_id device_id = getDeviceID(device);
    cl_program program;
    cl_kernel kernel;

    if (TYPEOF(k_name) != STRSXP || LENGTH(k_name) != 1)
	Rf_error("invalid kernel name");
    if (TYPEOF(code) != STRSXP || LENGTH(code) < 1)
	Rf_error("invalid kernel code");
    if (TYPEOF(prec) != STRSXP || LENGTH(prec) != 1)
	Rf_error("invalid precision specification");
    ctx = clCreateContext(0, 1, &device_id, NULL, NULL, &last_ocl_error);
    if (!ctx)
	ocl_err("clCreateContext");
    sctx = PROTECT(mkContext(ctx));
    {
	int sn = LENGTH(code), i;
	const char **cptr;
	cptr = (const char **) malloc(sizeof(char*) * sn);
	for (i = 0; i < sn; i++)
	    cptr[i] = CHAR(STRING_ELT(code, i));
	program = clCreateProgramWithSource(ctx, sn, cptr, NULL, &last_ocl_error);
	free(cptr);
	if (!program)
	    ocl_err("clCreateProgramWithSource");
    }
    
    last_ocl_error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (last_ocl_error != CL_SUCCESS) {
        size_t len;
        last_ocl_error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	clReleaseProgram(program);
	Rf_error("clGetProgramBuildInfo failed (with %d): %s", last_ocl_error, buffer);
    }

    kernel = clCreateKernel(program, CHAR(STRING_ELT(k_name, 0)), &last_ocl_error);
    clReleaseProgram(program);
    if (!kernel)
	ocl_err("clCreateKernel");

    {
	SEXP sk = PROTECT(mkKernel(kernel));
	Rf_setAttrib(sk, Rf_install("device"), device);
	Rf_setAttrib(sk, Rf_install("precision"), prec);
	Rf_setAttrib(sk, Rf_install("context"), sctx);
	Rf_setAttrib(sk, Rf_install("name"), k_name);
	UNPROTECT(2); /* sk + context */
	return sk;
    }
}


/*--- generic blockwise argument list that can be freed in one go ---*/
typedef void (*afin_t)(void*);

/* Wrapper around clReleaseMemObject for afin_t, because it adheres to a
   different calling convention on Windows.
   (OpenCL API calls have CL_API_CALL = __stdcall, afin_t assumes __cdecl.)
 */
void free_clmem(void *data)
{
    clReleaseMemObject(data);
}

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
   This is rather important for memory objects since GPUs have
   only limited memory.
*/
typedef struct ocl_call_context {
    cl_mem output;
    cl_command_queue commands;
    cl_event event;
    int finished;   /* finished - results have been retrieved */
#if USE_OCL_COMPLETE_CALLBACK
    int completed;  /* completed - event has been triggered but not picked up */
#endif
    int ftres, ftype, on; /* these are only set if the context leaves ocl_call */
    void *float_out;
    struct arg_chain *float_args, *mem_objects;
} ocl_call_context_t;

static void ocl_call_context_fin(SEXP context) {
    ocl_call_context_t *ctx = (ocl_call_context_t*) R_ExternalPtrAddr(context);
    if (ctx) {
	/* if this was an asynchronous call, we must wait for it to finish */
	if (!ctx->finished) clFinish(ctx->commands);
	if (ctx->event) clReleaseEvent(ctx->event);
	if (ctx->output) clReleaseMemObject(ctx->output);
	if (ctx->float_args) arg_free(ctx->float_args, 0);
	if (ctx->float_out) free(ctx->float_out);
	if (ctx->mem_objects) arg_free(ctx->mem_objects, (afin_t) free_clmem);
	if (ctx->commands) clReleaseCommandQueue(ctx->commands);
	free(ctx);
	CAR(context) = 0; /* this allows us to call the finalizer manually */
    }
}

#if USE_OCL_COMPLETE_CALLBACK /* we are currently not using the callback since it raises memory management issues and increases complexity */
static void CL_CALLBACK ocl_complete_callback(cl_event event, cl_int status, void *ucc) {
    ocl_call_context_t *ctx = (ocl_call_context_t*) ucc;
    if (ctx) { /* just signal completion - we could be on any thread, so we don't want to issue a callback into R */
	ctx->completed = 1;
    }
}
#endif

/* Implementation of oclRun */
/* .External */
SEXP ocl_call(SEXP args) {
    struct arg_chain *float_args = 0;
    ocl_call_context_t *occ;
    int on, an = 0, ftype = FT_DOUBLE, ftsize, ftres, async;
    SEXP ker = CADR(args), olen, arg, res, octx, dimVec;
    cl_kernel kernel = getKernel(ker);
    cl_context context;
    cl_command_queue commands;
    cl_device_id device_id = getDeviceID(getAttrib(ker, Rf_install("device")));
    cl_mem output;
    size_t wdims[3] = {0, 0, 0};
    int wdim = 1;

    if (clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(context), &context, NULL) != CL_SUCCESS || !context)
	Rf_error("cannot obtain kernel context via clGetKernelInfo");
    args = CDDR(args);
    res = Rf_getAttrib(ker, install("precision"));
    if (TYPEOF(res) == STRSXP && LENGTH(res) == 1 && CHAR(STRING_ELT(res, 0))[0] != 'd')
	ftype = FT_SINGLE;
    ftsize = (ftype == FT_DOUBLE) ? sizeof(double) : sizeof(float);
    olen = CAR(args);  /* size */
    args = CDR(args);
    on = Rf_asInteger(olen);
    if (on < 0)
	Rf_error("invalid output length");
    ftres = (Rf_asInteger(CAR(args)) == 1) ? 1 : 0;  /* native.result */
    if (ftype != FT_SINGLE) ftres = 0;
    args = CDR(args);
    async = (Rf_asInteger(CAR(args)) == 1) ? 0 : 1;  /* wait */
    args = CDR(args);
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
	Rf_error("invalid dimensions - muse be a numeric vector with positive values");

    args = CDR(args);
    occ = (ocl_call_context_t*) calloc(1, sizeof(ocl_call_context_t));
    if (!occ) Rf_error("unable to allocate ocl_call context");
    octx = PROTECT(R_MakeExternalPtr(occ, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(octx, ocl_call_context_fin, TRUE);

    occ->output = output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ftsize * on, NULL, &last_ocl_error);
    if (!output)
	Rf_error("failed to create output buffer of %d elements via clCreateBuffer (%d)", on, last_ocl_error);
    if (clSetKernelArg(kernel, an++, sizeof(cl_mem), &output) != CL_SUCCESS)
	Rf_error("failed to set first kernel argument as output in clSetKernelArg");
    if (clSetKernelArg(kernel, an++, sizeof(on), &on) != CL_SUCCESS)
	Rf_error("failed to set second kernel argument as output length in clSetKernelArg");
    occ->commands = commands = clCreateCommandQueue(context, device_id, 0, &last_ocl_error);
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
	    if ((last_ocl_error = clSetKernelArg(kernel, an++, al, ptr)) != CL_SUCCESS)
		Rf_error("Failed to set scalar kernel argument %d (size=%d, error code %d)", an, al, last_ocl_error);
	} else {
	    cl_mem input = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  al * n, ptr, &last_ocl_error);
	    if (!input)
		Rf_error("Unable to create buffer (%d elements, %d bytes each) for vector argument %d (oclError %d)", n, al, an, last_ocl_error);
	    if (!occ->mem_objects)
		occ->mem_objects = arg_alloc(0, 32);
	    arg_add(occ->mem_objects, input);
#if 0 /* we used this before CL_MEM_USE_HOST_PTR */
	    if ((last_ocl_error = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, al * n, ptr, 0, NULL, NULL)) != CL_SUCCESS)
		Rf_error("Failed to transfer data (%d elements) for vector argument %d (oclError %d)", n, an, last_ocl_error);
#endif
	    if ((last_ocl_error = clSetKernelArg(kernel, an++, sizeof(cl_mem), &input)) != CL_SUCCESS)
		Rf_error("Failed to set vector kernel argument %d (size=%d, length=%d, error %d)", an, al, n, last_ocl_error);
	    /* clReleaseMemObject(input); */
	}
	args = CDR(args);
    }

    if ((last_ocl_error = clEnqueueNDRangeKernel(commands, kernel, wdim, NULL, wdims, NULL, 0, NULL, async ? &occ->event : NULL)) != CL_SUCCESS)
	ocl_err("Kernel execution");

    if (async) { /* asynchronous call -> get out and return the context */
#if USE_OCL_COMPLETE_CALLBACK
	last_ocl_error = clSetEventCallback(occ->event, CL_COMPLETE, ocl_complete_callback, occ);
#endif
	clFlush(commands); /* the specs don't guarantee execution unless clFlush is called */
	occ->ftres = ftres;
	occ->ftype = ftype;
	occ->on = on;
	Rf_setAttrib(octx, R_ClassSymbol, mkString("clCallContext"));
	UNPROTECT(1);
	return octx;
    }

    clFinish(commands);
    occ->finished = 1;

    /* we can release input memory objects now */
    if (occ->mem_objects) {
      arg_free(occ->mem_objects, (afin_t) free_clmem);
      occ->mem_objects = 0;
    }
    if (float_args) {
      arg_free(float_args, 0);
      float_args = occ->float_args = 0;
    }

    res = ftres ? Rf_allocVector(RAWSXP, on * sizeof(float)) : Rf_allocVector(REALSXP, on);
    if (ftype == FT_SINGLE) {
	if (ftres) {
	  if ((last_ocl_error = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * on, RAW(res), 0, NULL, NULL )) != CL_SUCCESS)
		Rf_error("Unable to transfer result vector (%d float elements, oclError %d)", on, last_ocl_error);
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
	    if ((last_ocl_error = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * on, fr, 0, NULL, NULL )) != CL_SUCCESS)
		Rf_error("Unable to transfer result vector (%d float elements, oclError %d)", on, last_ocl_error);
	    for (i = 0; i < on; i++)
		r[i] = fr[i];
	}
    } else if ((last_ocl_error = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(double) * on, REAL(res), 0, NULL, NULL )) != CL_SUCCESS)
	Rf_error("Unable to transfer result vector (%d double elements, oclError %d)", on, last_ocl_error);

    ocl_call_context_fin(octx);
    UNPROTECT(1);
    return res;
}

/* Implementation of oclResult */
SEXP ocl_collect_call(SEXP octx, SEXP wait) {
    SEXP res = R_NilValue;
    ocl_call_context_t *occ;
    int on;

    if (!Rf_inherits(octx, "clCallContext"))
	Rf_error("Invalid call context");
    occ = (ocl_call_context_t*) R_ExternalPtrAddr(octx);
    if (!occ || occ->finished)
	Rf_error("The call results have already been collected, they cannot be retrieved twice");

    if (Rf_asInteger(wait) == 0 && occ->event) {
	cl_int status;
	if ((last_ocl_error = clGetEventInfo(occ->event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL)) != CL_SUCCESS)
	    ocl_err("querying event object for the supplied context");
	
	if (status < 0)
	    Rf_error("Asynchronous call failed with error code 0x%x", (int) -status);

	if (status != CL_COMPLETE)
	    return R_NilValue;
    }

    clFinish(occ->commands);
    occ->finished = 1;
    
    /* we can release input memory objects now */
    if (occ->mem_objects) {
      arg_free(occ->mem_objects, (afin_t) free_clmem);
      occ->mem_objects = 0;
    }
    if (occ->float_args) {
      arg_free(occ->float_args, 0);
      occ->float_args = 0;
    }

    on = occ->on;
    res = occ->ftres ? Rf_allocVector(RAWSXP, on * sizeof(float)) : Rf_allocVector(REALSXP, on);
    if (occ->ftype == FT_SINGLE) {
	if (occ->ftres) {
	    if ((last_ocl_error = clEnqueueReadBuffer( occ->commands, occ->output, CL_TRUE, 0, sizeof(float) * on, RAW(res), 0, NULL, NULL )) != CL_SUCCESS)
		Rf_error("Unable to transfer result vector (%d float elements, oclError %d)", on, last_ocl_error);
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
	    if ((last_ocl_error = clEnqueueReadBuffer( occ->commands, occ->output, CL_TRUE, 0, sizeof(float) * on, fr, 0, NULL, NULL )) != CL_SUCCESS)
		Rf_error("Unable to transfer result vector (%d float elements, oclError %d)", on, last_ocl_error);
	    for (i = 0; i < on; i++)
		r[i] = fr[i];
	}
    } else if ((last_ocl_error = clEnqueueReadBuffer( occ->commands, occ->output, CL_TRUE, 0, sizeof(double) * on, REAL(res), 0, NULL, NULL )) != CL_SUCCESS)
	Rf_error("Unable to transfer result vector (%d double elements, oclError %d)", on, last_ocl_error);

    ocl_call_context_fin(octx);
    return res;
}
