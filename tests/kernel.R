# 0. Create context and read kernel file
library(OpenCL)

if (!length(oclPlatforms())) {
    cat("== Cannot run tests as there is no platform")
    q("no")
}

ctx <- oclContext()
code <- readChar("kernel.cl", nchars=file.info("kernel.cl")$size)

# 1. Create kernel without inputs and run it
linear <- oclSimpleKernel(ctx, "linear", code, "integer")
oclRun(linear, 4)
oclRun(linear, 32)

# 2. Run kernel with a numeric input buffer
square <- oclSimpleKernel(ctx, "square", code)
input <- as.clBuffer(sqrt(1:16), ctx)
output <- oclRun(square, 16, input)
output
oclRun(square, 16, output)

# 3. Run kernel with a buffer argument and a scalar argument
multiply <- oclSimpleKernel(ctx, "multiply", code)
input <- as.clBuffer((1:16)^2, ctx)
output <- oclRun(multiply, 16, input, 2.5)
output
