#include "ocl.h"

#include <Rinternals.h>
#include <R_ext/Visibility.h>

/* Define symbols */
SEXP oclDeviceSymbol;
SEXP oclQueueSymbol;
SEXP oclContextSymbol;
SEXP oclNameSymbol;
SEXP oclModeSymbol;
SEXP oclEventSymbol;
SEXP oclMessageSymbol;

/* from reg.c */
void R_register_OpenCL(DllInfo *dll);

/* Install symbols */
attribute_visible void R_init_R_OpenCL(DllInfo *dll)
{
    oclDeviceSymbol = Rf_install("device");
    oclQueueSymbol = Rf_install("queue");
    oclContextSymbol = Rf_install("context");
    oclNameSymbol = Rf_install("name");
    oclModeSymbol = Rf_install("mode");
    oclEventSymbol = Rf_install("event");
    oclMessageSymbol = Rf_install("message");

    R_register_OpenCL(dll);
}

void ocl_err(const char *str, cl_int error_code) {
    Rf_error("%s failed, oclError %d: %s", str, error_code, ocl_errstr(error_code));
}

void ocl_warn(const char *str, cl_int error_code) {
    Rf_warning("%s failed, oclError %d: %s", str, error_code, ocl_errstr(error_code));
}

/* this is not actually used, but just in case we ever decide to... */
void ocl_message(const char *msg) {
    SEXP msg_call = PROTECT(lang2(oclMessageSymbol, PROTECT(Rf_mkString(msg))));
    Rf_eval(msg_call, R_GlobalEnv);
    UNPROTECT(2);
}

/* Encapsulation of a cl_platform_id as SEXP */
SEXP mkPlatformID(cl_platform_id id) {
    SEXP platform_exp;
    platform_exp = Rf_protect(R_MakeExternalPtr(id, R_NilValue, R_NilValue));
    Rf_setAttrib(platform_exp, R_ClassSymbol, Rf_mkString("clPlatformID"));
    Rf_unprotect(1);
    return platform_exp;
}

cl_platform_id getPlatformID(SEXP platform) {
    if (!Rf_inherits(platform, "clPlatformID") || TYPEOF(platform) != EXTPTRSXP)
	Rf_error("Expected OpenCL platform");
    return (cl_platform_id)R_ExternalPtrAddr(platform);
}

/* Encapsulation of a cl_device_id as SEXP */
SEXP mkDeviceID(cl_device_id id) {
    SEXP device_exp;
    device_exp = Rf_protect(R_MakeExternalPtr(id, R_NilValue, R_NilValue));
    Rf_setAttrib(device_exp, R_ClassSymbol, Rf_mkString("clDeviceID"));
    Rf_unprotect(1);
    return device_exp;
}

cl_device_id getDeviceID(SEXP device) {
    if (!Rf_inherits(device, "clDeviceID") ||
	TYPEOF(device) != EXTPTRSXP)
	Rf_error("Expected OpenCL device");
    return (cl_device_id)R_ExternalPtrAddr(device);
}

/* Encapsulation of a cl_context as SEXP */
static void clFreeContext(SEXP ctx) {
    clReleaseContext((cl_context)R_ExternalPtrAddr(ctx));
}

SEXP mkContext(cl_context ctx) {
    SEXP ptr;
    ptr = Rf_protect(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeContext, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("clContext"));
    Rf_unprotect(1);
    return ptr;
}

cl_context getContext(SEXP ctx) {
    if (!Rf_inherits(ctx, "clContext") ||
	TYPEOF(ctx) != EXTPTRSXP)
	Rf_error("Expected OpenCL context");
    return (cl_context)R_ExternalPtrAddr(ctx);
}

/* Encapsulation of a cl_command_queue as SEXP */
static void clFreeCommandQueue(SEXP k) {
    clReleaseCommandQueue((cl_command_queue)R_ExternalPtrAddr(k));
}

SEXP mkCommandQueue(cl_command_queue queue) {
    SEXP ptr;
    ptr = Rf_protect(R_MakeExternalPtr(queue, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeCommandQueue, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("clCommandQueue"));
    Rf_unprotect(1);
    return ptr;
}

cl_command_queue getCommandQueue(SEXP queue_exp) {
    if (!Rf_inherits(queue_exp, "clCommandQueue") ||
        TYPEOF(queue_exp) != EXTPTRSXP)
        Rf_error("Expected OpenCL command queue");
    return (cl_command_queue)R_ExternalPtrAddr(queue_exp);
}

/* keep track of allocations so we can trigger GC if needed */
static size_t allocated_buffer_size = 0;
/* size to trigger R-side garbage collection - 0 = disabled */
static size_t gc_trigger_size = 0;
/* high-mark to trigger gc unconditionally */
static size_t gc_high_mark = 0;
/* if last tigger GC didn't get out of trigger zone we set
   this flags and won't attempt furhter GC until high mark is reached */
static int trigger_zone = 0;

size_t R2size(SEXP sWhat, int which) {
    if (TYPEOF(sWhat) == INTSXP &&
	XLENGTH(sWhat) >= which && INTEGER(sWhat)[which] >= 0)
	return (size_t) INTEGER(sWhat)[which];

    if (TYPEOF(sWhat) == REALSXP &&
	XLENGTH(sWhat) >= which && (REAL(sWhat)[which] >= 0))
	return (size_t) (REAL(sWhat)[which]);

    if (TYPEOF(sWhat) == STRSXP &&
	XLENGTH(sWhat) >= which) {
	const char *c = CHAR(STRING_ELT(sWhat, which)), *s = c;
	while (*s >= '0' && *s <= '9') s++;
	long long l = atoll(c);
	switch (*s) {
	case 'g':
	case 'G':
	    l *= 1024;
	case 'm':
	case 'M':
	    l *= 1024;
	case 'k':
	case 'K':
	    l *= 1024;
	    break;
	default:
	    Rf_error("Invalid unit suffix in size specification: %s", c);
	}
	if (l >= 0)
	    return (size_t) l;
    }
    Rf_error("Size specification must be a valid, positive integer numeric");
    /* unreachable */
    return 0.0;
}

attribute_visible SEXP ocl_mem_limits(SEXP sTrigger, SEXP sHigh) {
    SEXP res;
    size_t tri = gc_trigger_size, hm = gc_high_mark;
    int do_set = 0;
    if (sTrigger != R_NilValue) {
	do_set = 1;
	tri = R2size(sTrigger, 0);
    }
    if (sHigh != R_NilValue) {
	do_set = 1;
	hm = R2size(sHigh, 0);
    }
    if ((tri && !hm) || (hm && !tri))
	Rf_error("The limits must be either both set or both zero to disable");
    if (hm < tri)
	Rf_error("The high mark cannot be smaller than the trigger mark");
    if (do_set) {
	gc_trigger_size = tri;
	gc_high_mark = hm;
	/* clear trigger zone */
	trigger_zone = 0;
    }
    res = Rf_protect(Rf_mkNamed(VECSXP, (const char*[]) { "trigger", "high", "used", "in.zone", "" }));
    SET_VECTOR_ELT(res, 0, Rf_ScalarReal((double) gc_trigger_size));
    SET_VECTOR_ELT(res, 1, Rf_ScalarReal((double) gc_high_mark));
    SET_VECTOR_ELT(res, 2, Rf_ScalarReal((double) allocated_buffer_size));
    SET_VECTOR_ELT(res, 3, Rf_ScalarLogical(trigger_zone));
    Rf_unprotect(1);
    return res;
}

/* Encapsulation of a cl_mem as SEXP */
static void clFreeBuffer(SEXP buffer_exp) {
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    size_t size = 0;

    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    allocated_buffer_size -= size;
    clReleaseMemObject(buffer);
}

SEXP mkBuffer(cl_mem buffer, ClType type) {
    SEXP ptr;
    if (gc_trigger_size) {
	/* should we do a gc ? */
	if ((gc_high_mark && allocated_buffer_size > gc_high_mark) ||
	    (allocated_buffer_size > gc_trigger_size && !trigger_zone)) {
	    R_gc();
	    /* if we didn't free enough to get under gc_trigger_size
	       then we are in the trigger zone which means no more GCs
	       until it gets critical */
	    if (allocated_buffer_size > gc_trigger_size)
		trigger_zone = 1;
	}
    }
    SEXP itype = Rf_protect(Rf_ScalarInteger(type));
    ptr = Rf_protect(R_MakeExternalPtr(buffer, itype, R_NilValue));
    size_t size = 0;
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    allocated_buffer_size += size;
    R_RegisterCFinalizerEx(ptr, clFreeBuffer, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("clBuffer"));
    Rf_unprotect(2);
    return ptr;
}

cl_mem getBuffer(SEXP buffer_exp) {
    if (!Rf_inherits(buffer_exp, "clBuffer") ||
        TYPEOF(buffer_exp) != EXTPTRSXP)
        Rf_error("Expected OpenCL buffer");
    return (cl_mem)R_ExternalPtrAddr(buffer_exp);
}

/* Encapsulation of a cl_kernel as SEXP */
static void clFreeKernel(SEXP k) {
    clReleaseKernel((cl_kernel)R_ExternalPtrAddr(k));
}

SEXP mkKernel(cl_kernel k) {
    SEXP ptr;
    ptr = Rf_protect(R_MakeExternalPtr(k, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeKernel, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("clKernel"));
    Rf_unprotect(1);
    return ptr;
}

cl_kernel getKernel(SEXP k) {
    if (!Rf_inherits(k, "clKernel") ||
	TYPEOF(k) != EXTPTRSXP)
	Rf_error("Expected OpenCL kernel");
    return (cl_kernel)R_ExternalPtrAddr(k);
}

/* Encapsulation of a cl_event as SEXP */
static void clFreeEvent(SEXP event_exp) {
    clReleaseEvent((cl_event)R_ExternalPtrAddr(event_exp));
}

SEXP mkEvent(cl_event event) {
    SEXP ptr;
    ptr = Rf_protect(R_MakeExternalPtr(event, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeEvent, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("clEvent"));
    Rf_unprotect(1);
    return ptr;
}

cl_event getEvent(SEXP event_exp) {
    if (!Rf_inherits(event_exp, "clEvent") ||
        TYPEOF(event_exp) != EXTPTRSXP)
        Rf_error("Expected OpenCL event");
    return (cl_event)R_ExternalPtrAddr(event_exp);
}
