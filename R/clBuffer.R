# Creating a buffer in a context
clBuffer <- function(context, length, mode = c("numeric", "single", "double", "integer"))
{
    if (!inherits(context, "clContext"))
        stop("Invalid context")
    mode <- match.arg(mode)
    if (mode == "numeric")
        mode <- attributes(context)$precision
    .Call(cl_create_buffer, context, length, mode)
}

as.clBuffer <- function(vector, context, mode = class(vector)) {
    buffer <- clBuffer(context, length(vector), mode)
    buffer[] <- vector
    buffer
}
as.double.clBuffer <- function(x, ...) {
    as.double(.Call(cl_read_buffer, x, NULL))
}
as.integer.clBuffer <- function(x, ...) {
    as.integer(.Call(cl_read_buffer, x, NULL))
}
is.clBuffer <- function(any) inherits(any, "clBuffer")

# Printing information about the buffer
print.clBuffer <- function(x, ...) {
    stopifnot(is.clBuffer(x))
    cat("OpenCL buffer,", length(x),
        "elements of type", attributes(x)$mode, "\n");
    print(.Call(cl_read_buffer, x, NULL), ...)
    invisible(x)
}

# Get and modify length
length.clBuffer<- function(x) {
    stopifnot(is.clBuffer(x))
    .Call(cl_get_buffer_length, x)
}
# For now, we don't allow to modify the length.
#"length<-.clFloatBuffer" <- function(x, value) {}

# Retrieve and overwrite data
`[.clBuffer` <- function(x, i) {
    # convert i to either NULL (all) or integer index
    ix <- if (missing(i)) NULL else seq_along(x)[i]
    # check if we can retrieve this en-block
    if (.Call(cl_supported_index, ix))
        .Call(cl_read_buffer, x, ix)
    else ## emerge all and use R for subsetting - bad, but all we can do
        .Call(cl_read_buffer, x, NULL)[i]
}

`[<-.clBuffer` <- function(x, i, value) {
    # convert i to either NULL (all) or integer index
    ix <- if (missing(i)) NULL else seq_along(x)[i]

    # do we have to emerge and replace in R?
    if (!.Call(cl_supported_index, ix)) {
        if (length(x) > 1e4)
            warning("Non-contiguous sub-assignment on clBuffer, this has to be done in CPU memory by copying the entire buffer, so is very inefficient")
        y <- x[]
        y[i] <- value
        x[] <- y
        return(x)
    }

    # Determine expected class for value.
    targetClass <- switch(attributes(x)$mode,
                          single=, double="numeric",
                          integer="integer",
                          stop("Invalid buffer class ", attributes(x)$mode))

    # Convert if necessary - target is either numeric or integer
    if (!inherits(value, targetClass))
        value <- if (targetClass == "numeric") as.numeric(value) else as.integer(value)

    # recycling
    if (length(value) < length(ix)) value <- rep(value, length.out=length(ix))

    .Call(cl_write_buffer, x, ix, value)
}
