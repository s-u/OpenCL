\name{oclContext}
\alias{oclContext}
\title{
Create an OpenCL context for a given device.
}
\description{
  OpenCL contexts host kernels and buffers for the device they are hosted on.
  They also have an attached command queue, which allows out-of-order execution
  of all operations. Once you have a context, you can create a kernel in the
  context with \code{\link{oclSimpleKernel}}.
}
\usage{
oclContext(device = "default", precision = c("best", "single", "double"))
}
\arguments{
  \item{device}{Device object as obtained from \code{\link{oclDevices}} or a
  type as in \code{\link{oclDevices}}. In this case, a suitable device of the
  given type will be selected automatically.}
  \item{precision}{Default precision of the context. This is the precision that
  will be chosen by default for \code{numeric} buffers and kernels with
  \code{numeric} output mode.}
}
%\details{
%}
\value{
  An OpenCL context.
}
\author{
  Aaron Puchert
}
\seealso{
  \code{\link{oclDevices}}, \code{\link{oclSimpleKernel}}
}
\examples{
library(OpenCL)
cat("== Platforms:\n")
(platforms <- oclPlatforms())
if (length(platforms)) {
    cat("== Devices:\n")
    ## pick the first platform
    print(devices <- oclDevices(platforms[[1]]))
    if (length(devices)) {
        cat("== Context:\n")
        ## pick the first device
        print(ctx <- oclContext(devices[[1]]))
    }
    cat("== Default context:\n")
    ## Note that context can find device on its own
    ## (may be different from above if you have multiple devices)
    print(c2 <- oclContext())
}
}
\keyword{interface}
