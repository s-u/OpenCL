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
oclPlatforms <- function() .Call("ocl_platforms")
oclDevices <- function(platform = oclPlatforms()[[1]], type="gpu") .Call("ocl_devices", platform, type)

# Create a context
oclContext <- function(device = "gpu", precision = c("best", "single", "double")) {
    # Choose device, if user was too lazy
    if (class(device) != "clDeviceID") {
        candidates <- oclDevices(type=device)
        if (length(candidates) < 1)
            stop("No devices found")

        # Choose the "fastest" candidate in case of multiple GPUs.
        # (We might use a better mechanism in the future)
        # Anyway, alert the user that our choice was ambigous.
        if (length(candidates) > 1)
            warning("Found more than one device, choosing the fastest")
        freqs <- as.numeric(lapply(oclInfo(candidates), function(info) info$max.frequency))
        device <- candidates[[which.max(freqs)]]
    }

    # Create context
    context <- .Call("ocl_context", device)

    # Find precision
    precision <- match.arg(precision)
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
    output.mode <- match.arg(output.mode)
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
  .Call("ocl_ez_kernel", context, name, code, output.mode)
}

# Run a simple kernel and retrieve the result
oclRun <- function(kernel, size, ..., dim=size) .External("ocl_call", kernel, size, dim, ...)

# Get extended information about OpenCL objects
oclInfo <- function(item) UseMethod("oclInfo")
oclInfo.clDeviceID <- function(item) {
    info <- .Call("ocl_get_device_info", item)
    info$exts <- unlist(strsplit(info$exts, " "))
    info
}
oclInfo.clPlatformID <- function(item) .Call("ocl_get_platform_info", item)
oclInfo.list <- function(item) lapply(item, oclInfo)
