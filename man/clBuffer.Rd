\name{clBuffer}
\alias{clBuffer}
\alias{as.clBuffer}
\alias{as.double.clBuffer}
\alias{as.integer.clBuffer}
\alias{is.clBuffer}
\alias{print.clBuffer}
\alias{length.clBuffer}
\alias{[.clBuffer}
\alias{[<-.clBuffer}
\title{
  Create and handle OpenCL buffers
}
\description{
  OpenCL buffers are just like numeric or integer vectors that reside on the
  GPU and can directly be accessed by kernels. Both non-scalar arguments to
  \code{\link{oclRun}} and its return type are OpenCL buffers.

  Just like vectors in R, OpenCL buffers have a mode, which is (as of now) one
  of "numeric" (corresponds to \code{double} in OpenCL C), "clFloat"
  (\code{float}) or "integer" (\code{int}).

  The constructor \code{clBuffer} takes a context as created by
  \code{oclContext}, a length and a mode argument.

  The conversion function \code{as.clBuffer} creates an OpenCL buffer of the
  same length and mode as the argument and copies the data. Conversely,
  \code{as.double} (= \code{as.numeric}) and \code{as.integer} read a buffer and
  coerce the result as vector the appropriate mode.

  With \code{is.clBuffer} one can check if an object is an OpenCL buffer.

  The methods \code{length.clBuffer} and \code{print.clBuffer} retrieve the
  length and print the contents, respectively.

  Basic access to the data is available via \code{[\dots]}. As of now, only an
  empty selection is supported (which selects all elements), i.e. you can only
  select \code{buf[]}.
}
\usage{
clBuffer(context, length, mode = "numeric")
as.clBuffer(x, context)
is.clBuffer(x)
\method{as.double}{clBuffer}(buffer, \dots)
\method{as.integer}{clBuffer}(buffer, \dots)
\method{print}{clBuffer}(buffer, \dots)
\method{length}{clBuffer}(buffer)
\method{[}{clBuffer}(buffer, indices)
\method{[}{clBuffer}(buffer, indices) <- values
}
\arguments{
  \item{x}{Arbitrary object}
  \item{context}{OpenCL context as created by \code{\link{oclContext}}}
  \item{length}{Length of the required buffer}
  \item{mode}{Mode of the buffer, can be one of "numeric", "clFloat", "integer"}
  \item{buffer}{OpenCL buffer object (\code{clBuffer})}
  \item{indices}{Indices to access the buffer, must be omitted (as of now)}
  \item{values}{New values}
  \item{\dots}{Arguments passed to subsequent methods}
}
%\value{
%}
\author{
Aaron Puchert
}
\seealso{
\code{\link{oclContext}}, \code{\link{oclRun}}
}
\examples{
library(OpenCL)
ctx<-oclContext()

buf<-clBuffer(ctx, 16, "numeric")
# Do not write buf<-..., as this replaces buf with a vector.
buf[]<-sqrt(1:16)
buf

intbuf<-as.clBuffer(1:16, ctx)
print(intbuf)

length(buf)
as.numeric(buf)
buf[]

## clBuffer is the required argument and return type of oclRun.
## See oclRun() examples.
}
\keyword{interface}
