# Exact feasibility probe for fifth-order compact SBP closures.
#
#   julia --project=. -t 1 bench/closureenergy.jl rows=4,5,6,7,8,9,10,11,12
#   julia --project=. -t 1 bench/closureenergy.jl bandwidth=12
#   julia --project=. -t 1 bench/closureenergy.jl rows=7 bandwidth=7 interior=e6
#
# D=A\B, symmetric tridiagonal A, and B+B'=(5/3)*diag(-1,0,...,1)
# imply summation by parts in H=(3/5)A whenever A is positive definite.
# A[1,2]=0 makes strong endpoint injection an H-orthogonal projection.
# Interior A and B are exactly Lele C6. This probes the additional boundary
# support needed for fifth-order polynomial exactness; it does not assume
# that a consistent linear system admits a positive norm.
# bandwidth>1 permits a full boundary block on the implicit side. The e6
# control uses the standard explicit sixth-order interior and admits the
# known restricted-full-norm fifth-order moment family from seven rows.

module ClosureEnergy

using CompactLES, LinearAlgebra, Printf
const Q = Rational{BigInt}

function constraints(r; bandwidth=1, degree=5, implicit=Q[1//3],
                     explicit=Q[7//9,1//36], restricted=true)
    r >= 3 || error("at least three closure rows required")
    variables = [(:diagonal, j, j) for j in 1:r]
    append!(variables, [(:offdiagonal, j, k) for j in (restricted ? 2 : 1):r-1
                        for k in j+1:min(r,j+bandwidth)])
    append!(variables, [(:skew, j, k) for j in 1:r for k in j+1:r])
    matrix = zeros(Q, (degree+1)*r, length(variables))
    target = zeros(Q, (degree+1)*r)
    value(k, p) = Q(k-1)^p
    derivative(k, p) = p == 0 ? zero(Q) : p * Q(k-1)^(p-1)
    for j in 1:r, p in 0:degree
        row = (degree+1)*(j-1)+p+1
        boundary_flux = sum(k*explicit[k] for k in eachindex(explicit))
        known = j == 1 ? -boundary_flux*value(1,p) : zero(Q)
        for k in eachindex(explicit)
            j+k > r && (known += explicit[k]*value(j+k,p))
        end
        for k in eachindex(implicit)
            j+k > r && (known -= implicit[k]*derivative(j+k,p))
        end
        target[row] = -known
        for (column, (kind, a, b)) in enumerate(variables)
            if kind == :diagonal
                j == a && (matrix[row,column] = -derivative(a,p))
            elseif kind == :offdiagonal
                j == a && (matrix[row,column] = -derivative(b,p))
                j == b && (matrix[row,column] = -derivative(a,p))
            else
                j == a && (matrix[row,column] = value(b,p))
                j == b && (matrix[row,column] = -value(a,p))
            end
        end
    end
    return matrix, target, variables
end

function affine_solution(matrix, target)
    augmented = hcat(matrix, target)
    m, n = size(matrix)
    operations = Matrix{Q}(I,m,m)
    pivots = Int[]
    row = 1
    for column in 1:n
        pivot = findfirst(i -> !iszero(augmented[i,column]), row:m)
        pivot === nothing && continue
        chosen = row + pivot - 1
        if chosen != row
            augmented[row,:], augmented[chosen,:] =
                copy(augmented[chosen,:]), copy(augmented[row,:])
            operations[row,:], operations[chosen,:] =
                copy(operations[chosen,:]), copy(operations[row,:])
        end
        divisor = augmented[row,column]
        augmented[row,:] ./= divisor
        operations[row,:] ./= divisor
        for i in 1:m
            i == row && continue
            factor = augmented[i,column]
            iszero(factor) && continue
            augmented[i,:] .-= factor .* augmented[row,:]
            operations[i,:] .-= factor .* operations[row,:]
        end
        push!(pivots, column)
        row += 1
        row > m && break
    end
    contradiction = findfirst(i -> all(iszero, augmented[i,1:n]) &&
                                  !iszero(augmented[i,n+1]), 1:m)
    consistent = contradiction === nothing
    if !consistent
        witness = operations[contradiction,:] ./ augmented[contradiction,n+1]
        # Exact certificate of inconsistency: y'M=0, y'b=1.
        all(iszero, transpose(matrix)*witness) || error("invalid left-null witness")
        dot(witness,target) == 1 || error("invalid contradiction witness")
        return (; consistent, rank=length(pivots), particular=Q[],
                basis=zeros(Q,n,0), witness)
    end
    free = setdiff(1:n, pivots)
    particular = zeros(Q,n)
    basis = zeros(Q,n,length(free))
    for (i,pivot) in enumerate(pivots)
        particular[pivot] = augmented[i,n+1]
        for (j,column) in enumerate(free)
            basis[pivot,j] = -augmented[i,column]
        end
    end
    for (j,column) in enumerate(free)
        basis[column,j] = 1
    end
    matrix*particular == target || error("invalid particular solution")
    all(iszero,matrix*basis) || error("invalid homogeneous solution")
    return (; consistent, rank=length(pivots), particular, basis, witness=Q[])
end

function matrices(r, variables, parameters, n=2r+9)
    n >= 2r+2 || error("need at least two interior rows")
    A = Matrix(SymTridiagonal(ones(n), fill(1/3,n-1)))
    B = zeros(n,n)
    for j in r+1:n-r
        B[j,j-1] = -7/9; B[j,j+1] = 7/9
        B[j,j-2] = -1/36; B[j,j+2] = 1/36
    end
    for j in 1:r, k in r+1:n-r
        B[j,k] = -B[k,j]
    end
    A[1,2] = A[2,1] = 0
    B[1,1] = -5/6
    for ((kind,j,k), v) in zip(variables, parameters)
        if kind == :skew
            B[j,k] = v; B[k,j] = -v
        else
            A[j,k] = A[k,j] = v
        end
    end
    for j in 1:r, k in 1:n
        A[n+1-j,n+1-k] = A[j,k]
        A[n+1-k,n+1-j] = A[j,k]
        B[n+1-j,n+1-k] = -B[j,k]
        B[n+1-k,n+1-j] = -B[k,j]
    end
    return A, B
end

function main(args=ARGS)
    opts = CompactLES.script_args(args, (rows="4,5,6,7,8,9,10,11,12",
        bandwidth=1, degree=5, interior="c6", restricted=true))
    BLAS.set_num_threads(1)
    for r in parse.(Int,split(opts.rows,','))
        implicit, explicit = opts.interior == "c6" ? (Q[1//3],Q[7//9,1//36]) :
                             opts.interior == "c4" ? (Q[1//4],Q[3//4]) :
                             opts.interior == "e6" ? (Q[],Q[3//4,-3//20,1//60]) :
                             error("unknown interior")
        matrix, target, variables = constraints(r; bandwidth=opts.bandwidth,
            degree=opts.degree, implicit=implicit, explicit=explicit,
            restricted=opts.restricted)
        solved = affine_solution(matrix,target)
        @printf("rows=%d variables=%d rank=%d consistent=%s free=%d\n",
                r, size(matrix,2), solved.rank, solved.consistent, size(solved.basis,2))
        if solved.consistent && opts.interior == "c6"
            A,B = matrices(r,variables,Float64.(solved.particular))
            Printf.format(stdout, Printf.Format(
                    "  particular min eig(A) %+.6e max eig(A) %.6e " *
                    "norm parameter dimension %d\n"),
                    eigmin(Symmetric(A)),eigmax(Symmetric(A)),
                    rank(Float64.(solved.basis[
                        findall(v -> v[1] != :skew, variables),:])))
        end
        flush(stdout)
    end
end

end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ClosureEnergy.main()
