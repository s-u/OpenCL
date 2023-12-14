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
  of "double" or "numeric" (corresponds to \code{double} in OpenCL C), "single"
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

  Basic access to the data is available via \code{[\dots]}. Note that
  only contiguous memory operations are supported on GPU buffers, so if
  the index does not reference a contiguous region then the
  subsetting/assignment will be performed by retrieving the entire
  buffer and perfroming the operation in R which is very expensive.

  Note that unlike regular R objects GPU buffers are by design mutable,
  i.e. the object is only a reference to the GPU memory and thus any
  modification will affect all refernces. The contents can be emerged into R
  via \code{x[]} at which point the result is a regular R vector and no
  longer tied to the source buffer. Analogously, \code{x[] <- value} is
  the canonical way to replace the entire contents of the buffer where
  \code{value} must have the same length as the buffer (no recycling).
}
\usage{
clBuffer(context, length, mode = c("numeric", "single", "double", "integer"))
as.clBuffer(vector, context, mode = class(vector))
is.clBuffer(any)
\method{as.double}{clBuffer}(x, \dots)
\method{as.integer}{clBuffer}(x, \dots)
\method{print}{clBuffer}(x, \dots)
\method{length}{clBuffer}(x)
\method{[}{clBuffer}(x, i)
\method{[}{clBuffer}(x, i) <- value
}
\arguments{
  \item{context}{OpenCL context as created by \code{\link{oclContext}}}
  \item{length}{Length of the required buffer}
  \item{mode}{Mode of the buffer, can be one of "numeric", "single",
  "double" or "integer"}
  \item{vector}{Numeric or integer vector or \code{clFloat} object}
  \item{any}{Arbitrary object}
  \item{x}{OpenCL buffer object (\code{clBuffer})}
  \item{i}{index specifying elements to extract or replace}
  \item{value}{New values}
  \item{\dots}{Arguments passed to subsequent methods}
}
%\value{
%}
\author{
Aaron Puchert and Simon Urbanek
}
\seealso{
\code{\link{oclContext}}, \code{\link{oclRun}}
}
\examples{
library(OpenCL)

## Only proceed if this machine has at least one OpenCL platform
if (length(oclPlatforms())) {

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

buf[3:5]
buf[1:2] = 0
buf

## clBuffer is the required argument and return type of oclRun.
## See oclRun() examples.
}
}
\keyword{interface}
