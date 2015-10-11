# Creating a buffer in a context
clBuffer <- function(context, length, mode = "numeric")
    .Call("cl_create_buffer", context, length, mode)

as.clBuffer <- function(x, context) {
    buffer <- clBuffer(context, length(x), class(x))
    buffer[] <- x
    buffer
}
as.double.clBuffer <- function(buffer, ...) {
    as.double(.Call("cl_read_buffer", buffer, "all"))
}
is.clBuffer <- function(x) inherits(x, "clBuffer")

# Printing information about the buffer
print.clBuffer <- function(buffer, ...) {
    cat("OpenCL buffer,", length(buffer),
        "elements of type", attributes(buffer)$mode, "\n");
    print(.Call("cl_read_buffer", buffer, "all"), ...)
    invisible(buffer)
}

# Get and modify length
length.clBuffer<- function(buffer) {
    .Call("cl_get_buffer_length", buffer)
}
# For now, we don't allow to modify the length.
#"length<-.clFloatBuffer" <- function(x, value) {}

# Retrieve and overwrite data
`[.clBuffer` <- function(buffer, indices) {
    if (missing(indices)) { indices = "all" }
    .Call("cl_read_buffer", buffer, indices)
}
`[<-.clBuffer` <- function(buffer, indices, values) {
    if (missing(indices)) { indices = "all" }
    values <- do.call(paste("as", attributes(buffer)$mode, sep="."), list(values))
    .Call("cl_write_buffer", buffer, indices, values)
}
