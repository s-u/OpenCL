# Print information about OpenCL objects
print.clDeviceID <- function(x, ...) {
  i <- .Call(ocl_get_device_info, x)
  cat("OpenCL device '", i$name, "'\n", sep='')
  invisible(x)
}

print.clPlatformID <- function(x, ...) {
  i <- .Call(ocl_get_platform_info, x)
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
  cat("  Default precision: ", attr$precision, "\n", sep='')
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
oclPlatforms <- function() .Call(ocl_platforms)
oclDevices <- function(platform = oclPlatforms(),
                       type=c("all", "cpu", "gpu", "accelerator", "default")) {
    type <- match.arg(type)
    if ( inherits(platform, "clPlatformID") ) {
        return(.Call(ocl_devices, platform, type))
    }
    if ( is.list(platform) ) {
        ret <- unlist(lapply(X=platform, FUN=function(x) {stopifnot(inherits(x, "clPlatformID")); return(.Call(ocl_devices, x, type))} ))
        return(ret)
    }
    stop("Platform should be either a clPlatform object, or a list thereof")
}

# Create a context
oclContext <- function(device = "default", precision = c("best", "single", "double")) {
<<<<<<< HEAD
    precision <- match.arg(precision)

    # Choose device, if user was too lazy
    if (! inherits(class(device, "clDeviceID") ) {
        plat.can <- oclPlatforms()
        if ( ! is.null(getOption("ocl.default.platform")) ) {
            message(sprintf("Option 'ocl.default.platform' is set to '%s', we will only consider this platform to choose devices from.",
                            getOption("ocl.default.platform")))
            plat.names <- as.character(lapply(oclInfo(plat.can), function(info) info$name))
            plat.can <- plat.can[ which(plat.names == getOption("ocl.default.platform"))]
        }
        candidates <- oclDevices(platform=plat.can, type=device)
        if ( ! is.null(getOption("ocl.default.device")) ) {
            message(sprintf("Option 'ocl.default.device' is set to '%s', we will only consider this device to create a default context.",
                            getOption("ocl.default.device")))
            dev.names <- as.character(lapply(oclInfo(candidates), function(info) info$name))
            candidates <- candidates[ which(dev.names == getOption("ocl.default.device"))]
        }
        if (length(candidates) < 1)
            stop("No devices found")

        # Choose the "fastest" candidate in case of multiple GPUs.
        # (We might use a better mechanism in the future)
        # Anyway, alert the user that our choice was ambigous.
        if (length(candidates) > 1)
            warning("Found more than one device, choosing the fastest (freq * compute units)")
        speed <- as.numeric(lapply(oclInfo(candidates), function(info) info$max.frequency * info$compute.unit))
        device <- candidates[[which.max(speed)]]
    }

    # Create context
    context <- .Call(ocl_context, device)

    # Find precision
    if (precision == "best") {
        precision <- ifelse(
            any(oclInfo(device)$exts == "cl_khr_fp64"),
            "double", "single")
    }
    attributes(context)[["precision"]] = precision

    context
}

# Compile a "simple kernel"
oclSimpleKernel <- function(context, name, code, output.mode = c("numeric", "single", "double", "integer")) {
    if (!inherits(context,"clContext"))
        stop("invalid context")
    output.mode <- match.arg(output.mode)
    name <- as.character(name)
    code <- as.character(code)
    # Handle "numeric" type
    if (output.mode == "numeric")
        output.mode <- attributes(context)$precision

    # Add typedef for numeric and enable double precision extension, if required.
    if (output.mode == "double")
        code <- c(
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
            "typedef double numeric;\n",
            code)
    else    # output.mode == "single"
        code <- c(
            "typedef float numeric;\n",
            code)
  .Call(ocl_ez_kernel, context, name, code, output.mode)
}

# Run a simple kernel and retrieve the result
oclRun <- function(kernel, size, ..., dim=size) {
    if (!inherits(kernel, "clKernel"))
        stop("invalid kernel object (e.g., use oclSimpleKernel first)")
    .External(ocl_call, kernel, size, dim, ...)
}

# Get extended information about OpenCL objects
oclInfo <- function(item) UseMethod("oclInfo")
splitExts <- function(info) {
    info$exts <- strsplit(info$exts, " ", fixed=TRUE)[[1]]
    info
}
oclInfo.clDeviceID <- function(item) splitExts(.Call(ocl_get_device_info, item))
oclInfo.clPlatformID <- function(item) splitExts(.Call(ocl_get_platform_info, item))
oclInfo.list <- function(item) lapply(item, oclInfo)

oclMemLimits <- function(trigger=NULL, high=NULL)
    .Call(ocl_mem_limits, trigger, high)

## Low-level function mostly for diagnostics (not exported) allows the retrieval
## of arbitrary device info entries based on the integer ID (cl_device_info).
## The caller is expected to know how to interpret the bytes into the actual information.
## The int parameter can be set to 2, 4 or 8 which will convert the raw bytes into
## unsigned integers of the corresponding with (in native endianness). Otherwise
## a raw vector with the bytes is returned.
## NOTE: only entries up to 2k in size can be retrieved (there are no known
## entries in the standard that would be larger).
## Some useful uses:
## OpenCL:::.oclDeviceInfoEntry(d, 0x101FL, 8L)  ## global memory size
## OpenCL:::.oclDeviceInfoEntry(d, 0x1002L, 4L)  ## number of compute units
.oclDeviceInfoEntry <- function(device, entry.id, int=0L)
    .Call(ocl_get_device_info_entry, device, entry.id, int)
