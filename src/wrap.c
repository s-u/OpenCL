#include "ocl.h"

#include <Rinternals.h>

/* Define symbols */
SEXP oclDeviceSymbol;
SEXP oclQueueSymbol;
SEXP oclContextSymbol;
SEXP oclNameSymbol;
SEXP oclModeSymbol;

/* Install symbols */
__attribute__((constructor)) static void installSymbols()
{
    oclDeviceSymbol = Rf_install("device");
    oclQueueSymbol = Rf_install("queue");
    oclContextSymbol = Rf_install("context");
    oclNameSymbol = Rf_install("name");
    oclModeSymbol = Rf_install("mode");
}

void ocl_err(const char *str, cl_int error_code) {
    Rf_error("%s failed (oclError %d)", str, error_code);
}

/* Encapsulation of a cl_platform_id as SEXP */
SEXP mkPlatformID(cl_platform_id id) {
    SEXP platform_exp;
    platform_exp = PROTECT(R_MakeExternalPtr(id, R_NilValue, R_NilValue));
    Rf_setAttrib(platform_exp, R_ClassSymbol, mkString("clPlatformID"));
    UNPROTECT(1);
    return platform_exp;
}

cl_platform_id getPlatformID(SEXP platform) {
    if (!Rf_inherits(platform, "clPlatformID") || TYPEOF(platform) != EXTPTRSXP)
	Rf_error("invalid platform");
    return (cl_platform_id)R_ExternalPtrAddr(platform);
}

/* Encapsulation of a cl_device_id as SEXP */
SEXP mkDeviceID(cl_device_id id) {
    SEXP device_exp;
    device_exp = PROTECT(R_MakeExternalPtr(id, R_NilValue, R_NilValue));
    Rf_setAttrib(device_exp, R_ClassSymbol, mkString("clDeviceID"));
    UNPROTECT(1);
    return device_exp;
}

cl_device_id getDeviceID(SEXP device) {
    if (!Rf_inherits(device, "clDeviceID") ||
	TYPEOF(device) != EXTPTRSXP)
	Rf_error("invalid device");
    return (cl_device_id)R_ExternalPtrAddr(device);
}

/* Encapsulation of a cl_context as SEXP */
static void clFreeContext(SEXP ctx) {
    clReleaseContext((cl_context)R_ExternalPtrAddr(ctx));
}

SEXP mkContext(cl_context ctx) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeContext, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clContext"));
    UNPROTECT(1);
    return ptr;
}

cl_context getContext(SEXP ctx) {
    if (!Rf_inherits(ctx, "clContext") ||
	TYPEOF(ctx) != EXTPTRSXP)
	Rf_error("invalid OpenCL context");
    return (cl_context)R_ExternalPtrAddr(ctx);
}

/* Encapsulation of a cl_command_queue as SEXP */
static void clFreeCommandQueue(SEXP k) {
    clReleaseCommandQueue((cl_command_queue)R_ExternalPtrAddr(k));
}

SEXP mkCommandQueue(cl_command_queue queue) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(queue, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeCommandQueue, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clCommandQueue"));
    UNPROTECT(1);
    return ptr;
}

cl_command_queue getCommandQueue(SEXP queue_exp) {
    if (!Rf_inherits(queue_exp, "clCommandQueue") ||
        TYPEOF(queue_exp) != EXTPTRSXP)
        Rf_error("invalid OpenCL command queue");
    return (cl_command_queue)R_ExternalPtrAddr(queue_exp);
}

/* Encapsulation of a cl_mem as SEXP */
static void clFreeBuffer(SEXP buffer_exp) {
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    clReleaseMemObject(buffer);
}

SEXP mkBuffer(cl_mem buffer, ClType type) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(buffer, Rf_ScalarInteger(type), R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeBuffer, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clBuffer"));
    UNPROTECT(1);
    return ptr;
}

cl_mem getBuffer(SEXP buffer_exp) {
    if (!Rf_inherits(buffer_exp, "clBuffer") ||
        TYPEOF(buffer_exp) != EXTPTRSXP)
        Rf_error("invalid OpenCL buffer");
    return (cl_mem)R_ExternalPtrAddr(buffer_exp);
}

/* Encapsulation of a cl_kernel as SEXP */
static void clFreeKernel(SEXP k) {
    clReleaseKernel((cl_kernel)R_ExternalPtrAddr(k));
}

SEXP mkKernel(cl_kernel k) {
    SEXP ptr;
    ptr = PROTECT(R_MakeExternalPtr(k, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, clFreeKernel, TRUE);
    Rf_setAttrib(ptr, R_ClassSymbol, mkString("clKernel"));
    UNPROTECT(1);
    return ptr;
}

cl_kernel getKernel(SEXP k) {
    if (!Rf_inherits(k, "clKernel") ||
	TYPEOF(k) != EXTPTRSXP)
	Rf_error("invalid OpenCL kernel");
    return (cl_kernel)R_ExternalPtrAddr(k);
}
