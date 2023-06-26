/* list of the .Call API functions in ocl.c and buffers.c */

#include <Rinternals.h>

/* .Call */
extern SEXP cl_create_buffer(SEXP context_exp, SEXP length_exp, SEXP mode_exp);
extern SEXP cl_get_buffer_length(SEXP buffer_exp);
extern SEXP cl_read_buffer(SEXP buffer_exp, SEXP indices);
extern SEXP cl_write_buffer(SEXP buffer_exp, SEXP indices, SEXP values);
extern SEXP cl_supported_index(SEXP indices);
extern SEXP ocl_context(SEXP device_exp);
extern SEXP ocl_devices(SEXP platform, SEXP sDevType);
extern SEXP ocl_ez_kernel(SEXP context, SEXP k_name, SEXP code, SEXP mode);
extern SEXP ocl_get_device_info(SEXP device);
extern SEXP ocl_get_device_info_entry(SEXP device, SEXP sDI);
extern SEXP ocl_get_platform_info(SEXP platform);
extern SEXP ocl_platforms(void);
extern SEXP ocl_mem_limits(SEXP sTrigger, SEXP sHigh);
/* .External */
extern SEXP ocl_call(SEXP args);
