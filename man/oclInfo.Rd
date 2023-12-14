\name{oclInfo}
\alias{oclInfo}
\alias{oclInfo.clDeviceID}
\alias{oclInfo.clPlatformID}
\alias{oclInfo.list}
\title{
  Retrieve information about an OpenCL object.
}
\description{
  Some OpenCL obejcts have information tokens associated with them. For
  example the device obejct has a name, vendor, list of extensions etc.
  \code{oclInfo} returns a list of such properties for the given object.
}
\usage{
oclInfo(item)
\method{oclInfo}{clDeviceID}(item)
\method{oclInfo}{clPlatformID}(item)
\method{oclInfo}{list}(item)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{item}{object to retrieve information properties from}
}
%\details{
%}
\value{
  List of properties. The properties vary by object type. Some common
  properties are \code{"name"}, \code{"vendor"}, \code{"version"},
  \code{"profile"} and \code{"exts"}.  
}
\author{
Simon Urbanek
}
\examples{
p <- oclPlatforms()
if (length(p)) {
    cat("== Platform information:\n")
    print(oclInfo(p[[1]]))
    d <- oclDevices(p[[1]])
    if (length(d)) {
        cat("== Device information:\n")
        print(oclInfo(d))
    }
}
}
\keyword{interface}
