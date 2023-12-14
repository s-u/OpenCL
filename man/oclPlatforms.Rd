\name{oclPlatforms}
\alias{oclPlatforms}
\title{
Retrieve available OpenCL platforms.
}
\description{
\code{oclPlatforms} retrieves all available OpenCL platforms.
}
\usage{
oclPlatforms()
}
%\details{
%}
\value{
  List of available OpenCL platforms. If using OpenCL with Installable
  Client Driver (ICD) support, the result can be an empty list if no
  vendor ICD can be found. A warning is also issued in that case.
}
%\references{
%}
\author{
Simon Urbanek
}
\seealso{
\code{\link{oclDevices}}
}
\examples{
print(oclPlatforms())
}
\keyword{interface}
