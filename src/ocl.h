#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

/* Sometime the actual runtime is not the one present during compilation
 * Hence it is useful for "more recent runtime functions" to be able to
 * test if they are present before trying to use them ...
 *
 * Using some code from the clinfo.c which is under CC0 licence.
 */

/* We will want to check for symbols in the OpenCL library.
 * On Windows, we must get the module handle for it, on Unix-like
 * systems we can just use RTLD_DEFAULT
 */
// We will be using 'dlsym' function to test presence in the used symbol in the running library :
#ifdef _MSC_VER
# include <windows.h>
# define dlsym GetProcAddress
# define DL_MODULE GetModuleHandle("OpenCL")
#else
# include <dlfcn.h>
# define DL_MODULE ((void*)0) /* This would be RTLD_DEFAULT */
#endif

typedef struct SEXPREC* SEXP;

/* Symbols */
extern SEXP oclDeviceSymbol;
extern SEXP oclQueueSymbol;
extern SEXP oclContextSymbol;
extern SEXP oclNameSymbol;
extern SEXP oclModeSymbol;
extern SEXP oclEventSymbol;
extern SEXP oclMessageSymbol;

/* Supported buffer data types */
typedef enum {
    CLT_INT,
    CLT_FLOAT,
    CLT_DOUBLE
} ClType;

/* from oclerr.c (others are from wrap.c) */
const char* ocl_errstr(cl_int errorCode);

/* Rf_error/warning with extra code handling */
void ocl_err(const char *str, cl_int error_code);
void ocl_warn(const char *str, cl_int error_code);

/* Encapsulation of a cl_platform_id as SEXP */
SEXP mkPlatformID(cl_platform_id id);
cl_platform_id getPlatformID(SEXP platform);

/* Encapsulation of a cl_device_id as SEXP */
SEXP mkDeviceID(cl_device_id id);
cl_device_id getDeviceID(SEXP device);

/* Encapsulation of a cl_context as SEXP */
SEXP mkContext(cl_context ctx);
cl_context getContext(SEXP ctx);

/* Encapsulation of a cl_command_queue as SEXP */
SEXP mkCommandQueue(cl_command_queue queue);
cl_command_queue getCommandQueue(SEXP queue_exp);

/* Encapsulation of a cl_mem as SEXP */
SEXP mkBuffer(cl_mem buffer, ClType type);
cl_mem getBuffer(SEXP buffer_exp);

/* Encapsulation of a cl_kernel as SEXP */
SEXP mkKernel(cl_kernel k);
cl_kernel getKernel(SEXP k);

/* Encapsulation of a cl_event as SEXP */
SEXP mkEvent(cl_event event);
cl_event getEvent(SEXP event_exp);

/* BUFFER HANDLING */
/* Mode string <-> buffer type */
ClType get_type(SEXP mode_exp);
SEXP get_type_description(ClType type);

/* Create an OpenCL buffer */
SEXP cl_create_buffer(SEXP context_exp, SEXP length_exp, SEXP mode_exp);

/* Retrieve the length of an OpenCL buffer */
SEXP cl_get_buffer_length(SEXP buffer_exp);

/* Read data from an OpenCL buffer */
SEXP cl_read_buffer(SEXP buffer_exp, SEXP indices);

/* Write data to an OpenCL buffer */
SEXP cl_write_buffer(SEXP buffer_exp, SEXP indices, SEXP values);
