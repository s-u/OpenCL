\name{oclDevices}
\alias{oclDevices}
\title{
Get a list of OpenCL devices.
}
\description{
\code{oclDevices} retrieves a list of OpenCL devices for the given platform.
}
\usage{
oclDevices(platform = oclPlatforms()[[1]], type = "default")
}
\arguments{
  \item{platform}{OpenCL platform (see \code{\link{oclPlatforms}})}
  \item{type}{Desired device type, character vector of length one. Valid
    values are \code{"cpu"}, \code{"gpu"}, \code{"accelerator"},
    \code{"all"}, \code{"default"}. Partial matches are allowed.}
}
%\details{
%}
\value{
  List of devices. May be empty.
}
\author{
Simon Urbanek
}
\seealso{
\code{\link{oclPlatforms}}
}
\examples{
p <- oclPlatforms()
if (length(p)) print(oclDevices(p[[1]],"all"))
}
\keyword{interface}
