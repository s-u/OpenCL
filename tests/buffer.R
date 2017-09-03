# 0. Create context
library(OpenCL)
ctx<-oclContext()

# 1. Create numeric buffer and fill with values
buf <- clBuffer(ctx, 16, "numeric")
buf[] <- 1:16

#    Inspect the resulting buffer
class(buf)
print(attributes(buf)$mode)
print(buf)
length(buf)

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
c(is.clBuffer(buf), is.clBuffer(1:16), is.clBuffer(ctx))
