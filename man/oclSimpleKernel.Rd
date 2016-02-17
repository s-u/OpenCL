\name{oclSimpleKernel}
\alias{oclSimpleKernel}
\title{
  Create and compile OpenCL kernel code.
}
\description{
  Creates a kernel object by compiling the supplied code.
  The kernel can then be used in \code{\link{oclRun}}.
}
\usage{
oclSimpleKernel(context, name, code, precision = c("single", "double", "best"))
}
\arguments{
  \item{context}{Context (as created by \code{\link{oclContext}})
    to compile the kernel in.
  }
  \item{name}{Name of the kernel function - must match the name used in
    the supplied code.}
  \item{code}{Character vector containing the code. The code will be
    concatenated (as-is, no newlines are added!) by the engine.}
  \item{precision}{Precision of the result floating-point argument in the
    kernel. If "best" is used then the kernel code is expected to
    use single-precision but it will be automatically augmented to
    double-precision (by replacing words "float" with "double" in the
    code and adding the \code{cl_khr_fp64} pragma) if supported by the
    device.}
}
\details{
  \code{oclSimpleKernel} builds the program specified by \code{code} and
  creates a kernel from the program.

  The kernel built by this function is simple in that it can have
  exactly one vector output and arbitrarily many inputs. The first
  argument of the kernel must be \code{__global double*} or
  \code{__global float*} for the output and the second argument must be
  \code{const unsigned int} for the length of the output vector.
  All additional arguments are optional. See \code{\link{oclRun}} for an
  example of a simple kernel.

  Note that building a kernel can take substantial amount of time
  (depending on the OpenCL implementation) so it is generally a good
  idea to compile a kernel once and re-use it many times.
}
\value{
  Kernel object that can be used by \code{\link{oclRun}}.
}
\author{
  Simon Urbanek, Aaron Puchert
}
\seealso{
  \code{\link{oclContext}}, \code{\link{oclRun}}
}
%\examples{
%}
\keyword{interface}
