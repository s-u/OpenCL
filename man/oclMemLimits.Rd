\name{oclMemLimits}
\alias{oclMemLimits}
\title{
  OpenCL Memory Management and Limits
}
\description{
  \code{oclMemLimits} manages the memory limits used internally to
  aid with \R{} garbage collection and reports used buffer memory.
}
\usage{
oclMemLimits(trigger = NULL, high = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{trigger}{size specification for trigger limit or \code{NULL} to
    not change}
  \item{high}{size specification for high mark limit or \code{NULL} to
    not change}  
}
\details{
  In principle the memory management is simple: as long as a reference
  to a GPU object exists in \R, that object is retained. As soon as \R{}
  removes the reference object, the corresponding GPU object is
  released. This is sounds easy execpt for one important detail: \R{}
  only releases unused objects when a garbage collection is run (see
  \code{\link{gc}}), but \R{} does not know about the GPU memory so it
  may not decide that it is necessary if little \R{} memory is used.

  As a user, you can explicitly call \code{gc()} to force all unused
  objects to be collected, but garbage collection is expensive so it may
  impact your computation. Therefore OpenCL tracks allocated memory
  sizes used by \code{\link{clBuffer}} buffers and will trigger R garbage
  collection automatically if certain limits are reached.

  There are two limits: \code{trigger} limit and \code{high} limit.
  The \code{trigger} limit is the threshold at which OpenCL will attempt
  to run garbage collection. This limit is checked before any buffer
  allocation. Once this limit is exceeded, OpenCL will run \code{gc()}
  to attempt to free memory. Hoever, if the current operation actually
  does require a lot of memory, no GPU memory may be freed. In that case
  running garbage collection would be wasteful, therefore OpenCL will
  disable further GC until the \code{high} limit is reached. Beyond that
  limit GC is always run.

  The limit size specifications can be one of the following: a positive
  integer numeric (in bytes) or a scalar string consisting of the
  integer numeric and an optional unit suffix. The following suffixes
  are supported: \code{"k"}, \code{"m"} and \code{"g"} - corresponding
  powers of 1024. Note that the reported sizes are always in bytes
  represented as numerics.
}
\value{
  List with followng components:
  \item{trigger}{active trigger limit (in bytes) or 0 if not active}
  \item{high}{active trigger limit (in bytes) or 0 if not active}
  \item{used}{number of bytes currently allocated in clBuffers on the
    GPU}
  \item{in.zone}{logical, \code{TRUE} if garbage collection is disabled
    due to the inability to reduce usage under \code{trigger}, i.e.,
    the usage is between \code{trigger} and \code{high}}
}
\author{
  Simon Urbanek
}
\note{
  Currently the default is to not enable the automatic garbage
  collection, becasue it is experimental and best settings will cary by
  the hardware used,but that is likey to change. It can always be disabled
  with \code{oclMemLimits(0,0)}.

  IMPORTANT: The current tracking is global to OpenCL, so it is based on
  all the memory used across all devices.
}
\seealso{
  \code{\link{gc}}, \code{\link{clBuffer}}
}
\examples{
oclMemLimits()
}
\keyword{interface}
