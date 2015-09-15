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
oclContext(device)
}
\arguments{
  \item{device}{device object as obtained from \code{\link{oclDevices}}}
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
  \code{\link{clDevices}}, \code{\link{oclSimpleKernel}}
}
\examples{
library(OpenCL)
platform <- oclPlatforms()[[1]]
device <- oclDevices(platform)[[1]]
ctx <- oclContext(device)
print(ctx)
}
\keyword{interface}
