# Creating a buffer in a context
clFloatBuffer <- function(context, length)
    .Call("clCreateFloatBuffer", context, length)

as.clFloatBuffer <- function(x, context) {
    buffer <- clFloatBuffer(context, length(x))
    buffer[] <- x
    buffer
}
as.double.clFloatBuffer <- function(buffer, ...) {
    as.double(.Call("clReadFloatBuffer", buffer, "all"))
}
is.clFloatBuffer <- function(x) inherits(x, "clFloatBuffer")

# Printing information about the buffer
print.clFloatBuffer <- function(buffer, ...) {
    cat("OpenCL floating-point buffer, size", length(buffer), "\n");
    print(.Call("clReadFloatBuffer", buffer, "all"), ...)
    invisible(buffer)
}
#as.character.clFloat <- function(x, ...) as.character(.Call("float2double", x), ...)

# Get and modify length
length.clFloatBuffer<- function(buffer) {
    .Call("clGetFloatBufferLength", buffer)
}
# For now, we don't allow to modify the length.
#"length<-.clFloatBuffer" <- function(x, value) {}

# Retrieve and overwrite data
`[.clFloatBuffer` <- function(buffer, indices) {
    if (missing(indices)) { indices = "all" }
    .Call("clReadFloatBuffer", buffer, indices)
}
`[<-.clFloatBuffer` <- function(buffer, indices, values) {
    if (missing(indices)) { indices = "all" }
    .Call("clWriteFloatBuffer", buffer, indices, as.clFloat(values))
}
