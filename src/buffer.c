#include <string.h>
#include "ocl.h"

#define USE_RINTERNALS 1
#include <Rinternals.h>

static void clFreeBuffer(SEXP buffer_exp)
{
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    clReleaseMemObject(buffer);
}

/* Create a OpenCL floating-point buffer */
SEXP clCreateFloatBuffer(SEXP context_exp, SEXP length_exp)
{
    cl_context context = getContext(context_exp);
    int len = Rf_asInteger(length_exp);
    cl_mem buffer;
    SEXP buffer_exp;
    cl_int last_ocl_error;

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len,
                            NULL, &last_ocl_error);
    if (!buffer)
        ocl_err("clCreateBuffer", last_ocl_error);

    buffer_exp = PROTECT(R_MakeExternalPtr(buffer, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(buffer_exp, clFreeBuffer, TRUE);
    Rf_setAttrib(buffer_exp, R_ClassSymbol, mkString("clFloatBuffer"));
    Rf_setAttrib(buffer_exp, Rf_install("context"), context_exp);
    UNPROTECT(1);
    return buffer_exp;
}

/* Retrieve the length of an floating-point buffer */
SEXP clGetFloatBufferLength(SEXP buffer_exp)
{
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    size_t size;

    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    return Rf_ScalarInteger(size / sizeof(float));
}

/* Read data from an OpenCL floating-point buffer */
SEXP clReadFloatBuffer(SEXP buffer_exp, SEXP indices)
{
    SEXP context_exp = Rf_getAttrib(buffer_exp, Rf_install("context"));
    SEXP queue_exp = Rf_getAttrib(context_exp, Rf_install("queue"));
    cl_command_queue queue = getCommandQueue(queue_exp);
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    size_t size;
    SEXP res;
    cl_int last_ocl_error;

    // TODO: Use indices!
    if (TYPEOF(indices) != STRSXP || LENGTH(indices) != 1
        || strcmp(CHAR(STRING_ELT(indices, 0)), "all"))
        Rf_error("arbitrary assignments not implemented yet, use []<-");

    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    res = PROTECT(Rf_allocVector(RAWSXP, size));

    last_ocl_error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size,
                                         RAW(res), 0, NULL, NULL);
    if (!buffer)
        ocl_err("clEnqueueReadBuffer", last_ocl_error);

    Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));
    UNPROTECT(1);
    return res;
}

/* Write data to a OpenCL floating-point buffer */
SEXP clWriteFloatBuffer(SEXP buffer_exp, SEXP indices, SEXP values)
{
    SEXP context_exp = Rf_getAttrib(buffer_exp, Rf_install("context"));
    SEXP queue_exp = Rf_getAttrib(context_exp, Rf_install("queue"));
    cl_command_queue queue = getCommandQueue(queue_exp);
    cl_mem buffer = (cl_mem)R_ExternalPtrAddr(buffer_exp);
    size_t size;
    cl_int last_ocl_error;

    // TODO: Use indices!
    if (TYPEOF(indices) != STRSXP || LENGTH(indices) != 1 || strcmp(CHAR(STRING_ELT(indices, 0)), "all"))
        Rf_error("arbitrary assignments not implemented yet, use []<-");

    // Get buffer size
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

    // Check input data
    if (TYPEOF(values) != RAWSXP || !inherits(values, "clFloat"))
        Rf_error("invalid single precision vector");
    if (LENGTH(values) != size)
        Rf_error("invalid input length");

    // Note that we do not have to block here.
    last_ocl_error = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, size,
                                          RAW(values), 0, NULL, NULL);
    if (last_ocl_error != CL_SUCCESS)
        ocl_err("clEnqueueWriteBuffer", last_ocl_error);

    return buffer_exp;
}
