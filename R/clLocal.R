clLocal <- function(length, mode = c("byte", "numeric", "single", "double", "integer")) {
    if (!is.numeric(length) || length(length) != 1 || !is.finite(length))
        stop("invalid length")
    mode <- match.arg(mode)
    ## -1 represend numberics which can be 4 or 8 depending on capabilities
    sizes <- as.list(c(byte=1L, numeric=-1L, single=4L, double=8L, integer=4L))
    m <- sizes[[mode]]
    structure(list(length=length, elt=m, mode=mode), class="clLocal")
}

is.clLocal <- function(x) inherits(x, "clLocal")

print.clLocal <- function(x, ...) {
    cat("OpenCL local buffer placeholder,", length(x),
	"elements of type", x$mode, "\n");
    invisible(x)
}

length.clLocal <- function(x) x$length
