\name{clFloat}
\alias{clFloat}
\alias{as.clFloat}
\alias{is.clFloat}
\alias{as.double.clFloat}
\alias{as.integer.clFloat}
\alias{as.character.clFloat}
\alias{print.clFloat}
\alias{length.clFloat}
\alias{length<-.clFloat}
\title{
  clFloat represents a single-precision vector that can be used with OpenCL.
}
\description{
  \code{clFloat} is a constructor of a single-precision vector - an
  obejct of the class \code{"clFloat"}. The current internal
  representation is a  raw vector of the machine-specific representation
  of the C \code{float} type but this may change in the future. The
  purpose of this type is to serve as a pass-though for OpenCL calls
  that involve single precision vectors such that they do not need to be
  converted to/from double precision and thus incurring the corresponding
  penalty. This type is not designed to be a full numeric vector
  replacement and thus opeartion such as arithmetics are intentionally
  not implemented.

  \code{as.clFloat} coerces objects into single-precision vectors
  essentially by calling \code{clFloat(as.numeric(x))}.

  \code{is.clFloat} returns \code{TRUE} the \code{x} is anobejct of the
  class \code{clFloat}

  Some most basic methods such as \code{length} or \code{print} are
  implemented as well as basic coersion methods.
}
\usage{
clFloat(x)
as.clFloat(x)
is.clFloat(x)
\method{as.double}{clFloat}(x, \dots)
\method{as.integer}{clFloat}(x, \dots)
\method{as.character}{clFloat}(x, \dots)
\method{print}{clFloat}(x, \dots)
\method{length}{clFloat}(x)
\method{length}{clFloat}(x) <- value
}
\arguments{
  \item{x}{object}
  \item{\dots}{arguments passed to subsequent methods}
  \item{value}{new length}
}
%\value{
%}
\author{
Simon Urbanek
}
\seealso{
\code{\link{oclRun}}
}
\examples{
x <- clFloat(1:10/2)
print(x)
length(x)
as.double(x)
as.character(x)
as.integer(x)
is.clFloat(x)
identical(x, as.clFloat(as.numeric(x)))

## clFloat is the return type of oclRun(..., native.result=TRUE)
## for single-precision kernels. It can also be used instead of
## numeric vectors in such kernels to avoid conversions.
## See oclRun() examples.
}
\keyword{interface}
