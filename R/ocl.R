print.clDeviceID <- function(x, ...) {
  i <- .Call("ocl_get_device_info", x)
  cat(" OpenCL device '", i$name, "'\n", sep='')
  x
}

print.clPlatformID <- function(x, ...) {
  i <- .Call("ocl_get_platform_info", x)
  cat(" OpenCl platform '", i$name, "'\n", sep='')
  x
}

oclPlatforms <- function() .Call("ocl_platforms")
oclDevices <- function(platform = oclPlatforms()[[1]], type="default") .Call("ocl_devices", platform, type)
oclSimpleKernel <- function(device, name, code, precision=c("single","double")) .Call("ocl_ez_kernel", device, name, code, match.arg(precision))
oclRun <- function(kernel, size, ..., native.result=FALSE, wait=TRUE) .External("ocl_call", kernel, size, native.result, wait, ...)
oclResult <- function(context) .Call("ocl_collect_call", context)

oclInfo <- function(item) UseMethod("oclInfo")
oclInfo.clDeviceID <- function(item) .Call("ocl_get_device_info", item)
oclInfo.clPlatformID <- function(item) .Call("ocl_get_platform_info", item)
oclInfo.list <- function(item) lapply(item, oclInfo)
