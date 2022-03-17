#include <string.h>
#include "ocl.h"

#define USE_RINTERNALS 1
#include <Rinternals.h>
#include <R_ext/Visibility.h>

/* Translate description string to internal type */
ClType get_type(SEXP mode_exp)
{
    if (TYPEOF(mode_exp) != STRSXP || LENGTH(mode_exp) != 1)
        Rf_error("invalid mode");

    const char* mode = CHAR(STRING_ELT(mode_exp, 0));
    if (!strcmp(mode, "integer"))
        return CLT_INT;
    if (!strcmp(mode, "single"))
        return CLT_FLOAT;
    if (!strcmp(mode, "double"))
        return CLT_DOUBLE;
    Rf_error("invalid mode");
}

/* Translate internal type to description string */
SEXP get_type_description(ClType type)
{
    switch (type) {
    case CLT_INT: return Rf_mkString("integer");
    case CLT_FLOAT: return Rf_mkString("single");
    case CLT_DOUBLE: return Rf_mkString("double");
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

/* Translate type to corresponding SEXP type */
static SEXPTYPE get_sexptype(ClType type)
{
    switch (type) {
    case CLT_INT: return INTSXP;
    case CLT_FLOAT: return REALSXP;
    case CLT_DOUBLE: return REALSXP;
    default: return ANYSXP;     // dummy return value
    }
}

/* FLOAT <-> DOUBLE CONVERSION with NAs */
static const uint32_t cl_NaFloat = 0x7ff007a2;   /* 0x7A2 = 1954, as in R_NaReal */

/* Convert float to double value */
static inline double to_double(float value)
{
    if (memcmp(&value, &cl_NaFloat, sizeof(float)))
        return (double)value;
    else
        return R_NaReal;
}

/* Convert double to float */
static inline float to_float(double value)
{
    if (memcmp(&value, &R_NaReal, sizeof(double)))
        return (float)value;
    else {
        float naFloat;
        memcpy(&naFloat, &cl_NaFloat, sizeof(float));
        return naFloat;
    }
}

/* Create an OpenCL buffer */
attribute_visible SEXP cl_create_buffer(SEXP context_exp, SEXP length_exp, SEXP mode_exp)
{
    cl_context context = getContext(context_exp);
    ClType type = get_type(mode_exp);
    size_t len = (size_t) (Rf_asReal(length_exp) + 0.001);
    cl_mem buffer;
    SEXP buffer_exp;
    cl_int last_ocl_error;

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, get_element_size(type) * len,
                            NULL, &last_ocl_error);
    if (!buffer)
        ocl_err("clCreateBuffer", last_ocl_error);

    buffer_exp = Rf_protect(mkBuffer(buffer, type));
    Rf_setAttrib(buffer_exp, oclContextSymbol, context_exp);
    Rf_setAttrib(buffer_exp, oclModeSymbol, get_type_description(type));
    Rf_unprotect(1);
    return buffer_exp;
}

/* Retrieve the length of an OpenCL buffer */
attribute_visible SEXP cl_get_buffer_length(SEXP buffer_exp)
{ 
    cl_mem buffer = getBuffer(buffer_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    size_t size;

    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    return Rf_ScalarInteger(size / get_element_size(type));
}

static int contig_index(SEXP indices) {
    size_t ixn = 0, i0 = 0;
    int *ix = 0;
    if (TYPEOF(indices) == INTSXP) {
	ix = INTEGER(indices);
	ixn = XLENGTH(indices);
    } else if (indices == R_NilValue) /* all index is always ok */
	return 1;
    else /* should be REALSXP */
	return 0;

    if (ix) {
	size_t i = 0;
	if (ix[0] == NA_INTEGER || !ix[0])
	    return 0;
	i0 = ix[0] - 1; /* the value of the index, i.e. first element to retrieve */
	i++;
	while (i < ixn && ix[i - 1] + 1 == ix[i]) i++;
	if (i < ixn)
	    return 0;
    }
    return 1;
}

attribute_visible SEXP cl_supported_index(SEXP indices) {
    return Rf_ScalarLogical(contig_index(indices));
}

/* Read data from an OpenCL buffer */
attribute_visible SEXP cl_read_buffer(SEXP buffer_exp, SEXP indices)
{
    cl_mem buffer = getBuffer(buffer_exp);
    SEXP context_exp = Rf_getAttrib(buffer_exp, oclContextSymbol);
    SEXP queue_exp = Rf_getAttrib(context_exp, oclQueueSymbol);
    cl_command_queue queue = getCommandQueue(queue_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    SEXP wait_exp = Rf_getAttrib(buffer_exp, oclEventSymbol);
    cl_event wait = (TYPEOF(wait_exp) == EXTPTRSXP) ? getEvent(wait_exp) : NULL;
    size_t size, length, ixn = 0, i0 = 0, read_size = 0, els = get_element_size(type);
    int *ix = 0;
    SEXP res;
    float *intermediate = NULL;
    cl_int last_ocl_error;

    if (TYPEOF(indices) == INTSXP) {
	ix = INTEGER(indices);
	ixn = XLENGTH(indices);
    } else if (indices != R_NilValue) /* should be REALSXP */
	Rf_error("Sorry, long vector indexing is not supported (yet).");

    // Get buffer size
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    length = size / els;

    if (ix) { /* only contiguous writes are supported at this point */
	size_t i = 0;
	if (ix[0] == NA_INTEGER || !ix[0])
	    Rf_error("indices cannot contain NAs or 0");
	i0 = ix[0] - 1; /* the value of the index, i.e. first element to retrieve */
	i++;
	while (i < ixn && ix[i - 1] + 1 == ix[i]) i++;
	if (i < ixn)
	    Rf_error("Sorry, subseting on the GPU is only supported for a contiguous region.");
	if (i0 + ixn > length)
	    Rf_error("Subsetting range (%lu .. %lu) out of bounds (length is %lu).",
		     (unsigned long) (i0 + 1), (unsigned long) (i0 + ixn), (unsigned long) length );
	read_size = ixn * els;
	i0 *= els;
    } else read_size = size;

    // Allocate appropriately sized target buffer
    res = Rf_allocVector(get_sexptype(type), read_size / els);
    if (type == CLT_FLOAT) {
        intermediate = (float*)calloc(length, sizeof(float));
        if (intermediate == NULL)
            Rf_error("Out of memory");
    }

    last_ocl_error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, i0, read_size,
        (type == CLT_FLOAT) ? (Rbyte*)intermediate : DATAPTR(res), wait ? 1 : 0, wait ? &wait : NULL, NULL);
    if (last_ocl_error != CL_SUCCESS) {
        if (type == CLT_FLOAT)
            free(intermediate);
        ocl_err("clEnqueueReadBuffer", last_ocl_error);
    }

    if (type == CLT_FLOAT) {
        /* Convert to double values */
        size_t i;
        double *result = REAL(res);
        for (i = 0; i < length; i++)
            result[i] = to_double(intermediate[i]);
        free(intermediate);
    }

    return res;
}

/* Write data to an OpenCL buffer */
attribute_visible SEXP cl_write_buffer(SEXP buffer_exp, SEXP indices, SEXP values)
{
    cl_mem buffer = getBuffer(buffer_exp);
    SEXP context_exp = Rf_getAttrib(buffer_exp, oclContextSymbol);
    SEXP queue_exp = Rf_getAttrib(context_exp, oclQueueSymbol);
    cl_command_queue queue = getCommandQueue(queue_exp);
    ClType type = (ClType)Rf_asInteger(R_ExternalPtrTag(buffer_exp));
    size_t size, length;
    float *intermediate;
    int *ix = 0;
    size_t ixn = 0, N = 0, i0 = 0, write_size = 0, els = get_element_size(type);
    cl_int last_ocl_error;

    // Get buffer size
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    length = size / els;

    if (TYPEOF(indices) == INTSXP) {
	ix = INTEGER(indices);
	ixn = XLENGTH(indices);
    } else if (indices != R_NilValue) /* should be REALSXP */
	Rf_error("Sorry, long vector indexing is not supported (yet).");

    /* Check input data */
    if (TYPEOF(values) != get_sexptype(type))
        Rf_error("invalid input vector type: %d", TYPEOF(values));
    N = XLENGTH(values);
    if (ix && ixn != N)
	Rf_error("invalid replacement length, %lu elements but %lu values",
		 (unsigned long) ixn, (unsigned long) N);
    if (!ix && N != length)
	Rf_error("invalid replacement, got %lu values, but expected %lu ",
		 (unsigned long) N, (unsigned long) length);

    if (!N) /* empty replacement */
	return buffer_exp;

    if (ix) { /* only contiguous writes are supported at this point */
	size_t i = 0;
	if (ix[0] == NA_INTEGER || !ix[0])
	    Rf_error("indices cannot contain NAs or 0");
	i0 = ix[0] - 1; /* the value of the index, i.e. first element to retrieve */
	i++;
	while (i < ixn && ix[i - 1] + 1 == ix[i]) i++;
	if (i < ixn)
	    Rf_error("Sorry, sub-assignment on the GPU is only supported for a contiguous region.");
	if (i0 + ixn > length)
	    Rf_error("Sub-assignment range (%lu .. %lu) out of bounds (length is %lu).",
		     (unsigned long) (i0 + 1), (unsigned long) (i0 + ixn), (unsigned long) length );
	write_size = ixn * els;
	i0 *= els;
    } else write_size = size;

    if (type == CLT_FLOAT) {
        /* Convert doubles to floats */
        intermediate = (float*)calloc(N, sizeof(float));
        if (intermediate == NULL)
            Rf_error("Out of memory");
        size_t i;
        double *input = REAL(values);
        for (i = 0; i < N; i++)
            intermediate[i] = to_float(input[i]);
    }

    /* Note that we do not have to block here (other than for mem mgmt reasons) */
    last_ocl_error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, i0, write_size,
        (type == CLT_FLOAT) ? (Rbyte*)intermediate : DATAPTR(values), 0, NULL, NULL);
    if (last_ocl_error != CL_SUCCESS) {
        if (type == CLT_FLOAT)
            free(intermediate);
        ocl_err("clEnqueueWriteBuffer", last_ocl_error);
    }

    if (type == CLT_FLOAT)
        free(intermediate);

    return buffer_exp;
}
