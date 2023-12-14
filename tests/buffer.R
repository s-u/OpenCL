# 0. Create context
library(OpenCL)

if (!length(oclPlatforms())) {
    cat("== Cannot run tests as there is no platform")
    q("no")
}

ctx<-oclContext()

# 1. Create single-precision buffer and fill with values
buf <- clBuffer(ctx, 16, "single")
buf[] <- 1:16

#    Inspect the resulting buffer
class(buf)
print(attributes(buf)$mode)
print(buf)
length(buf)

#    subsetting
buf[2:5]       # contiguous
buf[c(1,6)]    # non-contiguous
buf[-1]        # negative
buf[buf[] > 6] # logical
buf[c(NA, 4)]  # NA

#    subassignment
buf[2:3] = 0 # contiguous
buf[1:5]
sum(buf[1:4])
buf[5:4] = c(1,2) # non-contiguous (reversed)
buf[1:5]

#    Check if memory accounting works.
oclMemLimits()$used
rm(buf)
invisible(gc())
oclMemLimits()$used

# 2. The same for an integer buffer
ints <- clBuffer(ctx, 32, "integer")
ints[] <- 16:47

#    Inspect the resulting buffer
class(ints)
print(attributes(ints)$mode)
print(ints)
length(ints)

# 3. Let's see if we can guess the type correctly
numeric.buf <- as.clBuffer(sqrt(3:18), ctx)
print(numeric.buf)
integer.buf <- as.clBuffer(-1:14, ctx)
print(integer.buf)

# 4. Other functions
c(is.clBuffer(ints), is.clBuffer(1:16), is.clBuffer(ctx))
