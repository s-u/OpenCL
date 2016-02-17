# Creating a buffer in a context
clBuffer <- function(context, length, mode = "numeric")
    .Call("cl_create_buffer", context, length, mode)

as.clBuffer <- function(vector, context) {
    buffer <- clBuffer(context, length(vector), class(vector))
    buffer[] <- vector
    buffer
}
as.double.clBuffer <- function(x, ...) {
    as.double(.Call("cl_read_buffer", x, "all"))
}
as.integer.clBuffer <- function(x, ...) {
    as.integer(.Call("cl_read_buffer", x, "all"))
}
is.clBuffer <- function(any) inherits(any, "clBuffer")

# Printing information about the buffer
print.clBuffer <- function(x, ...) {
    cat("OpenCL buffer,", length(x),
        "elements of type", attributes(x)$mode, "\n");
    print(.Call("cl_read_buffer", x, "all"), ...)
    invisible(x)
}

# Get and modify length
length.clBuffer<- function(x) {
    .Call("cl_get_buffer_length", x)
}
# For now, we don't allow to modify the length.
#"length<-.clFloatBuffer" <- function(x, value) {}

# Retrieve and overwrite data
`[.clBuffer` <- function(x, indices) {
    if (missing(indices)) { indices = "all" }
    .Call("cl_read_buffer", x, indices)
}
`[<-.clBuffer` <- function(x, indices, value) {
    if (missing(indices)) { indices = "all" }
    value <- do.call(paste("as", attributes(x)$mode, sep="."), list(value))
    .Call("cl_write_buffer", x, indices, value)
}
