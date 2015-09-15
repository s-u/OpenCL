# Print information about OpenCL objects
print.clDeviceID <- function(x, ...) {
  i <- .Call("ocl_get_device_info", x)
  cat("OpenCL device '", i$name, "'\n", sep='')
  invisible(x)
}

print.clPlatformID <- function(x, ...) {
  i <- .Call("ocl_get_platform_info", x)
  cat("OpenCL platform '", i$name, "'\n", sep='')
  invisible(x)
}

print.clContext <- function(x, ...) {
  cat("OpenCL context ")
  attr <- attributes(x)
  attributes(x) <- NULL
  print.default(x, ...)
  cat("  Device: "); print(attr$device)
  cat("  Queue: ");  print(attr$queue)
  attributes(x) <- attr
  invisible(x)
}

print.clCommandQueue <- function(x, ...) {
  cat("OpenCL command queue ")
  attr <- attributes(x)
  attributes(x) <- NULL
  print.default(x, ...)
  attributes(x) <- attr
  invisible(x)
}

# Interface for clKernel objects.
# The user sees them as list of their attributes.
names.clKernel <- function(x) names(attributes(x))
`$.clKernel` <- function(x, name) attr(x, name)
`$<-.clKernel` <- function(x, name, value) stop("Kernel properties are read-only")
print.clKernel <- function(x, ...) {
  cat("OpenCL kernel '", attr(x, "name"),"'\n", sep='')
  a <- attributes(x)
  a$class <- NULL
  a$name <- NULL
  print(a)
  invisible(x)
}

# Query platforms and devices
oclPlatforms <- function() .Call("ocl_platforms")
oclDevices <- function(platform = oclPlatforms()[[1]], type="default") .Call("ocl_devices", platform, type)

# Create a context
oclContext <- function(device) .Call("ocl_context", device)

# Compile a "simple kernel"
oclSimpleKernel <- function(context, name, code, precision = c("single", "double", "best")) {
  precision <- match.arg(precision)
  if (precision == "best") { # detect supported precision from the device
    precision <- if (any(grepl("cl_khr_fp64", oclInfo(device)$exts))) {
      code <- c("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n", gsub("\\bfloat\\b", "double", code))
      "double"
    } else "single"
  }
  .Call("ocl_ez_kernel", context, name, code, precision)
}

# Run a simple kernel and retrieve the result
oclRun <- function(kernel, size, ..., native.result=FALSE, wait=TRUE, dim=size) .External("ocl_call", kernel, size, native.result, wait, dim, ...)
oclResult <- function(context, wait = TRUE) .Call("ocl_collect_call", context, wait)

# Get extended information about OpenCL objects
oclInfo <- function(item) UseMethod("oclInfo")
oclInfo.clDeviceID <- function(item) .Call("ocl_get_device_info", item)
oclInfo.clPlatformID <- function(item) .Call("ocl_get_platform_info", item)
oclInfo.list <- function(item) lapply(item, oclInfo)
