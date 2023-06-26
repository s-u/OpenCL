\name{clLocal}
\alias{clLocal}
\alias{is.clLocal}
\alias{print.clLocal}
\alias{length.clLocal}
\title{
  Create and handle local memory arguments for OpenCL kernels
}
\description{
  OpenCL kernels allow the use of local memory which is shared by all
  work-items of a work-group. In most cases, such memory is allocated inside
  the kernel at compile time such as \code{__local numeric temp[GROUP_SIZE]}.
  However, in some rare circumstances it may be desirable to allocate
  the buffer dynamically as an argument to the kernel. In that case the
  corresponding argument of the kernel is defined with the
  \code{__local} keyword and the caller has to specify the size of the
  local memory buffer at run-time when calling the kernel.

  The \code{clLocal()} function creates a specification of the local
  memory buffer. It is the only object that may be passed to a kernel
  argument declared with \code{__local}. The object is merely a
  specification that \code{\link{oclRun}} knows how to interpret,
  \code{clLocal} doesn't actually allocate any memory.

  By default, \code{size} is interpreted as bytes, but for convenience
  it can also specify the number of elements of a particular type. In
  the special case of \code{"numeric"} the actual size of one element
  (and thus the total buffer size) will depend on the context in which
  this specification is used (single or double precision).

  With \code{is.clLocal} one can check if an object is a local buffer
  specification.

  The methods \code{length.clLocal} and \code{print.clLocal} retrieve
  the length (number of elements) and print the contents, respectively.
}
\usage{
clLocal(length, mode = c("byte", "numeric", "single", "double", "integer"))
is.clLocal(x)
\method{print}{clLocal}(x, \dots)
\method{length}{clLocal}(x)
}
\arguments{
  \item{length}{numeric, length (number of elements) of the required
  buffer. The actual size will depend on \code{mode}.}
  \item{mode}{string, mode of the buffer (only used to compute the total
    size in bytes). The default is to treat \code{length} as the size in
    bytes (i.e., \code{"byte"} is aways allowed irrespective of the type
    of the kernel argument).}
  \item{x}{object}
  \item{\dots}{Ignored}
}
\value{
  \code{clLocal} returns an object of the class \code{"cLocal"}

  \code{is.clLocal} return \code{TRUE} for \code{"clLocal"} objects and
  \code{FALSE} otherwise.

  \code{print} method returns \code{x} invisibly.

  \code{length} returns a numeric scalar with the length (number of
  elements) in the buffer specification.
}
\note{
  The internal structure of the \code{clLocal} object should be
  considered private, may change and no user code should access its
  components. Similarly, \code{clLocal} objects are only legal when
  returned from the \code{clLocal()} function, they may not be created
  by other means or mutated.
}
\author{
Simon Urbanek
}
\seealso{
\code{\link{oclRun}}
}
\keyword{interface}
