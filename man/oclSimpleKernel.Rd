\name{oclSimpleKernel}
\alias{oclSimpleKernel}
\title{
  Create and compile OpenCL kernel code.
}
\description{
  \code{oclSimpleKernel} creates a kernel object by compiling the supplied
  code. The kernel can them be used in \code{\link{oclRun}}.

  
}
\usage{
oclSimpleKernel(device, name, code, precision = c("single", "double"))
}
\arguments{
  \item{device}{Device (element of the list returned by
    \code{\link{oclDevices}()}) to compile the kernel on.
  }
  \item{name}{Name of the kernel function - must match the name used in
    the supplied code.}
  \item{code}{character vector containg the code. The code will be
    concatenated (as-is, no newlines are added!) by the engine.}
  \item{precision}{precision of all floating-point arguments. Note that
    R uses only double-precision floating point representation, so
    single-precision computation requires temporary conversion of all
    input and output values and thus has significant overhead. However,
    not all devices support double-precision computation.}
}
\details{
  \code{oclSimpleKernel} creates a new OpenCL context, then creates and
  builds the program specified by code and finally creates a kernel
  from the program.

  The kernel built by this function is simple in that it can have
  exactly one vector output and arbitrarily many inputs. The first
  argument of the kernel must be \code{__global double*} for the output
  and the second argument must be \code{const int} for the length of the
  output vector. All additional arguments are optional. See
  \code{\link{oclRun}} for an example of a simple kernel.

  Note that building a kernel can take substantial amount of time
  (depending on the OpenCL implementation) so it is generally a good
  idea to compile a kernel once and re-use it many times.
}
\value{
  Kernel object that can be used by \code{\link{oclRun}}.
}
\author{
Simon Urbanek
}
\seealso{
  \code{\link{oclDevices}}, \code{\link{oclRun}}
}
%\examples{
%}
\keyword{interface}
