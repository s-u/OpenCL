#define USE_RINTERNALS 1
#include <Rinternals.h>

#include <stdlib.h>
#include <string.h>

/* convert single precision object to a numeric vector */
SEXP float2double(SEXP fObject) {
    const float *f;
    double *d;
    SEXP res;
    int i, n;
    
    if (TYPEOF(fObject) != RAWSXP || !inherits(fObject, "clFloat"))
	Rf_error("invalid single precision object");
    n = LENGTH(fObject) / sizeof(float);
    res = Rf_allocVector(REALSXP, n);
    d = REAL(res);
    f = (const float*) RAW(fObject);
    for (i = 0; i < n; i++)
	d[i] = f[i];
    return res;
}

/* convert a numeric vector to a single precision object */
SEXP double2float(SEXP dObject) {
    const double *d;
    float *f;
    SEXP res;
    int i, n;

    dObject = Rf_coerceVector(dObject, REALSXP);
    n = LENGTH(dObject);
    d = REAL(dObject);
    res = PROTECT(Rf_allocVector(RAWSXP, n * sizeof(float)));
    f = (float*) RAW(res);
    for (i = 0; i < n; i++)
	f[i] = d[i];
    Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));
    UNPROTECT(1);
    return res;
}

/* return the length of the vector as the number of elements.
   This is more efficient than length(unclass(x)). */
SEXP clFloat_length(SEXP fObject) {
    return Rf_ScalarInteger(LENGTH(fObject) / sizeof(float));
}

/* set the length ofthe clFloat object, effectively resizing it */
SEXP clFloat_length_set(SEXP fObject, SEXP value) {
    SEXP res;
    int newLen = Rf_asInteger(value), cpy;
    if (newLen == LENGTH(fObject)) return fObject;
    if (newLen < 0)
	Rf_error("invalid length");
    if (newLen > 536870912)
	Rf_error("clFloat length cannot exceed 512Mb due to R vector length limitations");
    newLen *= sizeof(float);
    res = PROTECT(Rf_allocVector(RAWSXP, newLen));
    cpy = (newLen > LENGTH(fObject)) ? LENGTH(fObject) : newLen;
    memcpy(RAW(res), RAW(fObject), cpy);
    if (newLen > cpy) /* FIXME: we initialize to 0.0 - maybe we need NAs ? */
	memset(RAW(res) + cpy, 0, newLen - cpy);
    Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));
    UNPROTECT(1);
    return res;
}
