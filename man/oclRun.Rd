\name{oclRun}
\alias{oclRun}
\title{
Run a kernel using OpenCL.
}
\description{
  \code{oclRun} is used to execute code that has been compiled for OpenCL.
}
\usage{
oclRun(kernel, size, ...)
}
\arguments{
  \item{kernel}{kernel object as obtained from \code{\link{oclMakeKernel}}}
  \item{size}{length of the output vector}
  \item{\dots}{additional arguments passed to the kernel}
}
\details{
  \code{oclRun} pushes kernel arguments, executes the kernel and
  retrieves the result. The kernel is expected to have \code{__global double *}
  type (write-only) as the first argument which will be used for the
  result. All other arguments are assumed to be read-only and will be
  filled accoding to the \code{\dots} values. Scalar values (vectors of
  length one) are passed as constants, vectors are passed as global
  objects. Only numeric (\code{int*}, \code{double*}) and logical
  (\code{int*}) vectors are supported as kernel arguments. The caller is
  responsible for matching the argument types according to the kernel in
  a way similar to \code{\link{.C}} and \code{\link{.Call}}.
}
\value{
  Result of the operation, a numeric vector of the length \code{size}.
}
\author{
Simon Urbanek
}
\seealso{
\code{\link{oclMakeKernel}}
}
\examples{
library(OpenCL)
p = oclPlatforms()
d = oclDevices(p[[1]])

code = c(
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
"__kernel void dnorm(\n",
"  __global double* output,\n",
"  __global double* input,\n",
" const unsigned int count,\n",
" const double mu, const double sigma)\n",
"{\n",
"  int i = get_global_id(0);\n",
"  if(i < count)\n",
"      output[i] = exp(-0.5f * ((input[i] - mu)/sigma) * ((input[i] - mu)/sigma)) / sigma;\n",
"};")
k.dnorm <- oclMakeKernel(d[[1]], "dnorm", code)
f <- function(x, mu, sigma)
  oclRun(k.dnorm, length(x), x, length(x), 0, 1)
f(1:10/2)
}
\keyword{interface}
