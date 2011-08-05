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
oclMakeKernel <- function(device, name, code) .Call("ocl_ez_kernel", device, name, code)
oclRun <- function(kernel, size, ...) .External("ocl_call_double", kernel, size, ...)
