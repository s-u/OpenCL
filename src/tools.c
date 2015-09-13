#define USE_RINTERNALS 1
#include <Rinternals.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

uint32_t clFloat_NaReal_ = 0x7ff007a2;   /* 0x7A2 = 1954, as in R_NaReal */

/* Implementation of as.double.clFloat */
/* convert single precision object to a numeric vector */
SEXP float2double(SEXP fObject) {
    const float *f;
    double *d;
    SEXP res;
    int i, n;
    
    if (TYPEOF(fObject) != RAWSXP || !inherits(fObject, "clFloat"))
	Rf_error("invalid single precision object");
    n = LENGTH(fObject) / sizeof(float);
    res = PROTECT(Rf_allocVector(REALSXP, n));
    d = REAL(res);
    f = (const float*) RAW(fObject);
    for (i = 0; i < n; i++) {
        if (memcmp(f + i, &clFloat_NaReal_, sizeof(float)))
            d[i] = f[i];
        else
            d[i] = R_NaReal;
    }
    UNPROTECT(1);
    return res;
}

/* Implementation of clFloat = as.clFloat */
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
    for (i = 0; i < n; i++) {
        if (memcmp(d + i, &R_NaReal, sizeof(double)))
            f[i] = d[i];
        else
            ((int*)RAW(res))[i] = clFloat_NaReal_;
    }
    Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));
    UNPROTECT(1);
    return res;
}

/* Implementation of length.clFloat */
/* return the length of the vector as the number of elements.
   This is more efficient than length(unclass(x)). */
SEXP clFloat_length(SEXP fObject) {
    return Rf_ScalarInteger(LENGTH(fObject) / sizeof(float));
}

/* Implementation of length<-.clFloat */
/* set the length ofthe clFloat object, effectively resizing it */
SEXP clFloat_length_set(SEXP fObject, SEXP value) {
    SEXP res;
    int newLen = Rf_asInteger(value), cpy;
    if (newLen == LENGTH(fObject)) return fObject;
    if (newLen < 0)
	Rf_error("invalid length");
    if (newLen > 536870912)
	Rf_error("clFloat length cannot exceed 512M due to R vector length limitations");
    newLen *= sizeof(float);
    res = PROTECT(Rf_allocVector(RAWSXP, newLen));
    cpy = (newLen > LENGTH(fObject)) ? LENGTH(fObject) : newLen;
    memcpy(RAW(res), RAW(fObject), cpy);
    for (; cpy < newLen; cpy += sizeof(float))
        *((int *)(RAW(res) + cpy)) = clFloat_NaReal_;
    Rf_setAttrib(res, R_ClassSymbol, Rf_mkString("clFloat"));
    UNPROTECT(1);
    return res;
}
