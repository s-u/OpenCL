\name{oclMakeKernel}
\alias{oclMakeKernel}
\title{
  Create and compile OpenCL kernel code.
}
\description{
  \code{oclMakeKernel} creates a kernel object by compiling the supplied
  code. The kernel can them be used in \code{\link{oclRun}}.
}
\usage{
oclMakeKernel(device, name, code)
}
\arguments{
  \item{device}{Device (element of the list returned by
    \code{\link{oclDevices}()}) to compile the kernel on.
  }
  \item{name}{Name of the kernel function - must match the name used in
    the supplied code.}
  \item{code}{character vector containg the code. The code will be
    concatenated (as-is, no newlines are added!) by the engine.}
}
\details{
  \code{oclMakeKernel} creates a new OpenCL context, then creates and
  builds the program specified by code and finally creates a kernel
  from the program.
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
