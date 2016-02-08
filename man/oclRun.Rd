\name{oclRun}
\alias{oclRun}
\alias{oclResult}
\title{
Run a kernel using OpenCL.
}
\description{
  \code{oclRun} is used to execute code that has been compiled for
  OpenCL.

  \code{oclResult} collects results from an asynchronous \code{oclRun}
  call.
}
\usage{
oclRun(kernel, size, ..., native.result = FALSE, wait = TRUE, dim = size)
oclResult(context, wait = TRUE)
}
\arguments{
  \item{kernel}{kernel object as obtained from \code{\link{oclSimpleKernel}}}
  \item{size}{length of the output vector}
  \item{\dots}{additional arguments passed to the kernel}
  \item{native.result}{logical scalar, if \code{TRUE} then the
    result from a single-precision kernel is not converted to
    double-precision but returned as a \code{\link{clFloat}} object.}
  \item{wait}{logical scalar, if \code{TRUE} then \code{oclRun} waits
    for the operation to finish and returs the result. Otherwise the
    kernel is  only enqueued,  so it will be run in parallel to R and have 
    to be collected later with \code{oclResult}.}
  \item{dim}{numeric vector describing the global work
    dimensions, i.e., the index range that the kernel will be run
    on. The kernel can use \code{get_global_id(n)} to obtain the
    \code{(n + 1)}-th dimension index and \code{get_global_size(n)} to
    get the dimension. OpenCL standard supports only up to three
    dimensions, you can use use index vectors as arguments if more
    dimensions are required. Note that \code{dim} is not necessarily the
    dimension of the result although it can be.}
  \item{context}{context object that was returned by
    \code{oclRun(..., wait = FALSE)} call.}
}
\details{
  \code{oclRun} pushes kernel arguments, executes the kernel and
  retrieves the result. The kernel is expected to have either
  \code{__global double *} or \code{__global float *}
  type (write-only) as the first argument which will be used for the
  result and \code{const int} second argument denoting the result
  length. All other arguments are assumed to be read-only and will be
  filled accoding to the \code{\dots} values. Scalar values (vectors of
  length one) are passed as constants, vectors are passed as global
  objects. Only numeric (\code{int*}, \code{double*}),
  \code{\link{clFloat}} (\code{float*}) and logical (\code{int*})
  vectors are supported as kernel arguments. Numeric (double-precision)
  vectors are converted to single-precision automatically when using
  single-precision kernel. The caller is responsible for matching the
  argument types according to the kernel in a way similar to
  \code{\link{.C}} and \code{\link{.Call}}.

  \code{oclResult} retrieves the result of a previous operation that was
  enqueued using \code{oclRun(..., wait = FALSE)}. If
  \code{oclResult(..., wait = FALSE)} is used then \code{NULL} is
  returned in case the result is not ready yet. Note that results can
  be collected only once and the context object becomes invalid after a
  successful call to \code{oclResult} since all associated OpenCL
  objects are released.
}
\value{
  \code{oclRun}: for \code{wait = TRUE} is the result of the operation,
  a numeric vector of the length \code{size}.
  Otherwise \code{oclRun} returns a call context object that can be used
  by \code{oclResult} to retrieve the result.

  \code{oclResult}: Result of the previously started operation or
  \code{NULL} if \code{wait=FALSE} and the operation has not completed
  yet.
}
\author{
  Simon Urbanek
}
\seealso{
  \code{\link{oclSimpleKernel}}, \code{\link{clFloat}}
}
\examples{
library(OpenCL)
ctx = oclContext()

code = c("
__kernel void dnorm(
  __global float* output,
 const unsigned int count,
  __global float* input,
 const float mu, const float sigma)
{
  int i = get_global_id(0);
  if(i < count)
      output[i] = exp(-0.5f * ((input[i] - mu) / sigma) * ((input[i] - mu) / sigma))
      / (sigma * sqrt( 2 * 3.14159265358979323846264338327950288 ) );
};")
k.dnorm <- oclSimpleKernel(ctx, "dnorm", code, "single")
f <- function(x, mu=0, sigma=1, ...)
  oclRun(k.dnorm, length(x), x, mu, sigma, ...)

## expect differences since the above uses single-precision but
## it should be close enough
f(1:10/2) - dnorm(1:10/2)

## this is optional - use floats instead of regular numeric vectors
x <- clFloat(1:10/2)
f(x, native.result=TRUE)

## does the device support double-precision?
if (any(grepl("cl_khr_fp64", oclInfo(attributes(ctx)$device)$exts))) {
code = c("#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dnorm(
  __global double* output,
 const unsigned int count,
  __global double* input,
 const double mu, const double sigma)
{
  int i = get_global_id(0);
  if(i < count)
      output[i] = exp(-0.5f * ((input[i] - mu) / sigma) * ((input[i] - mu) / sigma))
      / (sigma * sqrt( 2 * 3.14159265358979323846264338327950288 ) );
};")
k.dnorm <- oclSimpleKernel(ctx, "dnorm", code, "double")
f <- function(x, mu=0, sigma=1)
  oclRun(k.dnorm, length(x), x, mu, sigma)

## probably not identical, but close...
f(1:10/2) - dnorm(1:10/2)
} else cat("\nSorry, your device doesn't support double-precision\n")

## Note that in practice you can use precision="best" in the first
## example which will pick "double" on devices that support it and
## "single" elsewhere
}
\keyword{interface}
