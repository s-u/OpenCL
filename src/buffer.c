#include <string.h>
#include "ocl.h"

#define USE_RINTERNALS 1
#include <Rinternals.h>

/* Translate description string to internal type */
static ClType get_type(SEXP mode_exp)
{
    if (TYPEOF(mode_exp) != STRSXP || LENGTH(mode_exp) != 1)
        Rf_error("invalid mode");

    const char* mode = CHAR(STRING_ELT(mode_exp, 0));
    if (!strcmp(mode, "integer"))
        return CLT_INT;
    if (!strcmp(mode, "clFloat"))
        return CLT_FLOAT;
    if (!strcmp(mode, "numeric"))  // TODO: decide based on device capabilities
        return CLT_DOUBLE;
    Rf_error("invalid mode");
}

/* Translate internal type to description string */
static SEXP get_type_description(ClType type)
{
    switch (type) {
    case CLT_INT: return Rf_mkString("integer");
    case CLT_FLOAT: return Rf_mkString("clFloat");
    case CLT_DOUBLE: return Rf_mkString("numeric");
    default: return R_NilValue;
    }
}

/* Get size of a single element for the given type */
static size_t get_element_size(ClType type)
{
    switch (type) {
    case CLT_INT: return sizeof(cl_int);
    case CLT_FLOAT: return sizeof(cl_float);
    case CLT_DOUBLE: return sizeof(cl_double);
    default: return 0;
    }
}

/* Get size of a single SEXP vector element corresponding to the type */
static size_t get_sexp_element_size(ClType type)
{
    switch (type) {
    case CLT_INT: return sizeof(int);
    case CLT_FLOAT: return sizeof(Rbyte);
    case CLT_DOUBLE: return sizeof(double);
    default: return 0;
    }
}

/* Translate type to corresponding SEXP type */
static SEXPTYPE get_sexptype(ClType type)
{
    switch (type) {
    case CLT_INT: return INTSXP;
    case CLT_FLOAT: return RAWSXP;
    case CLT_DOUBLE: return REALSXP;
    default: return ANYSXP;     // dummy return value
    }
}

/* Create an OpenCL buffer */
SEXP cl_create_buffer(SEXP context_exp, SEXP length_exp, SEXP mode_exp)
{
    cl_context context = getContext(context_exp);
    int len = Rf_asInteger(length_exp);
    ClType type = get_type(mode_exp);
    cl_mem buffer;
    SEXP buffer_exp;
    cl_int last_ocl_error;

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, get_element_size(type) * len,
                            NULL, &last_ocl_error);
    if (!buffer)
        ocl_err("clCreateBuffer", last_ocl_error);

    buffer_exp = PROTECT(mkBuffer(buffer, type));
    Rf_setAttrib(buffer_exp, oclContextSymbol, context_exp);
    Rf_setAttrib(buffer_exp, oclModeSymbol, get_type_description(type));
    UNPROTECT(1);
    return buffer_exp;
}

/* Retrieve the length of an OpenCL buffer */
SEXP cl_get_buffer_length(SEXP buffer_exp)
{
    cl_mem buffer = getBuffer(buffer_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    size_t size;

    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    return Rf_ScalarInteger(size / get_element_size(type));
}

/* Read data from an OpenCL buffer */
SEXP cl_read_buffer(SEXP buffer_exp, SEXP indices)
{
    cl_mem buffer = getBuffer(buffer_exp);
    SEXP context_exp = Rf_getAttrib(buffer_exp, oclContextSymbol);
    SEXP queue_exp = Rf_getAttrib(context_exp, oclQueueSymbol);
    cl_command_queue queue = getCommandQueue(queue_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    size_t size;
    SEXP res;
    cl_int last_ocl_error;

    // TODO: Use indices!
    if (TYPEOF(indices) != STRSXP || LENGTH(indices) != 1
        || strcmp(CHAR(STRING_ELT(indices, 0)), "all"))
        Rf_error("arbitrary assignments not implemented yet, use []<-");

    // Get buffer size
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

    // Allocate appropriately sized target buffer
    res = PROTECT(Rf_allocVector(get_sexptype(type), size / get_sexp_element_size(type)));
    if (type == CLT_FLOAT)
        Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));

    last_ocl_error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size,
                                         RAW(res), 0, NULL, NULL);
    if (!buffer)
        ocl_err("clEnqueueReadBuffer", last_ocl_error);

    UNPROTECT(1);
    return res;
}

/* Write data to an OpenCL buffer */
SEXP cl_write_buffer(SEXP buffer_exp, SEXP indices, SEXP values)
{
    cl_mem buffer = getBuffer(buffer_exp);
    SEXP context_exp = Rf_getAttrib(buffer_exp, oclContextSymbol);
    SEXP queue_exp = Rf_getAttrib(context_exp, oclQueueSymbol);
    cl_command_queue queue = getCommandQueue(queue_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    size_t size;
    cl_int last_ocl_error;

    // TODO: Use indices!
    if (TYPEOF(indices) != STRSXP || LENGTH(indices) != 1 || strcmp(CHAR(STRING_ELT(indices, 0)), "all"))
        Rf_error("arbitrary assignments not implemented yet, use []<-");

    // Get buffer size
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

    // Check input data
    if (TYPEOF(values) != get_sexptype(type) ||
        (type == CLT_FLOAT && !inherits(values, "clFloat")))
        Rf_error("invalid input vector type");
    if (LENGTH(values) * get_sexp_element_size(type) != size)
        Rf_error("invalid input length");

    // Note that we do not have to block here.
    last_ocl_error = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, size,
                                          RAW(values), 0, NULL, NULL);
    if (last_ocl_error != CL_SUCCESS)
        ocl_err("clEnqueueWriteBuffer", last_ocl_error);

    return buffer_exp;
}
