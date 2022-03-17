#include <stdlib.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>

#include "c.h"

static const R_CallMethodDef CAPI[] = {
    {"cl_create_buffer",  (DL_FUNC) &cl_create_buffer, 3},
    {"cl_get_buffer_length",  (DL_FUNC) &cl_get_buffer_length, 1},
    {"cl_read_buffer",  (DL_FUNC) &cl_read_buffer, 2},
    {"cl_write_buffer",  (DL_FUNC) &cl_write_buffer, 3},
    {"cl_supported_index", (DL_FUNC) &cl_supported_index, 1},
    {"ocl_context",  (DL_FUNC) &ocl_context, 1},
    {"ocl_devices",  (DL_FUNC) &ocl_devices, 2},
    {"ocl_ez_kernel",  (DL_FUNC) &ocl_ez_kernel, 4},
    {"ocl_get_device_info",  (DL_FUNC) &ocl_get_device_info, 1},
    {"ocl_get_platform_info",  (DL_FUNC) &ocl_get_platform_info, 1},
    {"ocl_platforms",  (DL_FUNC) &ocl_platforms, 0},
    {"ocl_mem_limits", (DL_FUNC) &ocl_mem_limits, 2 },
    {NULL, NULL, 0}
};

static const R_ExternalMethodDef EAPI[] = {
  {"ocl_call", (DL_FUNC) &ocl_call, -1 },
  {NULL, NULL, 0}
};

void R_register_OpenCL(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CAPI, NULL, EAPI);
    R_useDynamicSymbols(dll, FALSE);
}
