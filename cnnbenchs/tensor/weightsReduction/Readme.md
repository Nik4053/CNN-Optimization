Optimisations:
 - registers: use registers to store all filter results first and reduce at the end
 - shfl: use shfl operation for reduction
 - reordered: reorder the for loops to make better use of compile time unrolling
 - work: make every thread do more work
 - padded:
 - reuseIndex
 - reduction: rausgezogene reduction
