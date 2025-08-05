module PyTyphoonJL

export OpticalFlowCore, Typhoon, RMSE, solve, solve_fast

import Base: size
using Wavelets

struct OpticalFlowCore
    shape::Tuple{Int,Int}
    dtype::DataType
    interpolation_order::Int
    sigma_blur::Float64
end

function OpticalFlowCore(shape::Tuple{Int,Int}; dtype=Float32, interpolation_order=3, sigma_blur=0.5)
    OpticalFlowCore(shape, dtype, interpolation_order, sigma_blur)
end

"""Compute displaced frame difference."""
function DFD(core::OpticalFlowCore, im0::AbstractMatrix, im1::AbstractMatrix, U::Tuple{AbstractMatrix,AbstractMatrix})
    # simple forward warp with bilinear interpolation
    u1, u2 = U
    m, n = core.shape
    out = similar(im0, core.dtype)
    for i in 1:m, j in 1:n
        x = i + u1[i,j]
        y = j + u2[i,j]
        x1 = clamp(floor(Int, x), 1, m)
        y1 = clamp(floor(Int, y), 1, n)
        x2 = clamp(x1+1, 1, m)
        y2 = clamp(y1+1, 1, n)
        dx = x - x1
        dy = y - y1
        v11 = im1[x1,y1]
        v12 = im1[x1,y2]
        v21 = im1[x2,y1]
        v22 = im1[x2,y2]
        out[i,j] = (1-dx)*(1-dy)*v11 + (1-dx)*dy*v12 + dx*(1-dy)*v21 + dx*dy*v22
    end
    return out - im0
end

"""Compute value and gradient of DFD."""
function DFD_gradient(core::OpticalFlowCore, im0::AbstractMatrix, im1::AbstractMatrix, U::Tuple{AbstractMatrix,AbstractMatrix})
    dfd = DFD(core, im0, im1, U)
    # simple gradient using forward differences
    m, n = core.shape
    gx = zeros(core.dtype, m, n)
    gy = zeros(core.dtype, m, n)
    for i in 1:m-1, j in 1:n
        gx[i,j] = dfd[i+1,j] - dfd[i,j]
    end
    for i in 1:m, j in 1:n-1
        gy[i,j] = dfd[i,j+1] - dfd[i,j]
    end
    return sum(dfd.^2)/2, (gx, gy)
end

mutable struct Typhoon
    core::Union{Nothing,OpticalFlowCore}
    wav_filter::OrthoFilter
    levels_decomp::Int
    im0::Union{Nothing,AbstractMatrix}
    im1::Union{Nothing,AbstractMatrix}
end

Typhoon(;shape=nothing) = Typhoon(shape === nothing ? nothing : OpticalFlowCore(shape), wavelet(WT.haar), 0, nothing, nothing)

function solve(ty::Typhoon, im0::AbstractMatrix, im1::AbstractMatrix; wav="haar", mode="zero", levels_decomp::Int=3, levels_estim::Union{Nothing,Int}=nothing, U0=nothing, interpolation_order::Int=3, sigma_blur::Float64=0.5)
    # select wavelet
    w = hasproperty(WT, Symbol(wav)) ? wavelet(getfield(WT, Symbol(wav))) : wavelet(WT.haar)
    ty.wav_filter = w
    # adjust levels
    max_lvl = Wavelets.maxtransformlevels([size(im0)...])
    levels_decomp = min(levels_decomp, max_lvl)
    ty.levels_decomp = levels_decomp
    # pad images to match wavelet requirements
    block = 2^levels_decomp
    m, n = size(im0)
    pad_m = (block - mod(m, block)) % block
    pad_n = (block - mod(n, block)) % block
    M, N = m + pad_m, n + pad_n
    im0p = zeros(eltype(im0), M, N); im0p[1:m,1:n] = im0
    im1p = zeros(eltype(im1), M, N); im1p[1:m,1:n] = im1
    ty.core = OpticalFlowCore((M, N); interpolation_order=interpolation_order, sigma_blur=sigma_blur)
    ty.im0, ty.im1 = im0p, im1p
    # initialize motion field
    if U0 === nothing
        U0 = (zeros(ty.core.dtype, M, N), zeros(ty.core.dtype, M, N))
    else
        u10 = zeros(ty.core.dtype, M, N); u20 = zeros(ty.core.dtype, M, N)
        u10[1:size(U0[1],1),1:size(U0[1],2)] .= U0[1]
        u20[1:size(U0[2],1),1:size(U0[2],2)] .= U0[2]
        U0 = (u10, u20)
    end
    c1 = dwt(U0[1], w, levels_decomp)
    c2 = dwt(U0[2], w, levels_decomp)
    x0 = vcat(vec(c1), vec(c2))
    fg = create_cost_function(ty, size(c1))
    x = x0
    for _ in 1:10
        _, g = fg(x)
        x -= 1e-3 * g
    end
    ncoef = length(c1)
    c1 = reshape(x[1:ncoef], size(c1))
    c2 = reshape(x[ncoef+1:end], size(c2))
    u1 = idwt(c1, w, levels_decomp)
    u2 = idwt(c2, w, levels_decomp)
    return (u1[1:m,1:n], u2[1:m,1:n])
end

solve_fast(ty::Typhoon, im0::AbstractMatrix, im1::AbstractMatrix; wav="haar", levels_decomp::Int=3, levels_estim::Union{Nothing,Int}=nothing, U0=nothing) = solve(ty, im0, im1; wav=wav, mode="periodization", levels_decomp=levels_decomp, levels_estim=levels_estim, interpolation_order=1, sigma_blur=0.0)

function create_cost_function(ty::Typhoon, shape)
    function fg(x)
        n = prod(shape)
        c1 = reshape(x[1:n], shape)
        c2 = reshape(x[n+1:2n], shape)
        u1 = idwt(c1, ty.wav_filter, ty.levels_decomp)
        u2 = idwt(c2, ty.wav_filter, ty.levels_decomp)
        f, (g1, g2) = DFD_gradient(ty.core, ty.im0, ty.im1, (u1, u2))
        g1c = dwt(g1, ty.wav_filter, ty.levels_decomp)
        g2c = dwt(g2, ty.wav_filter, ty.levels_decomp)
        return f, vcat(vec(g1c), vec(g2c))
    end
    return fg
end

function solve_pyramid(im0::AbstractMatrix, im1::AbstractMatrix; levels_pyr::Int=1, solve_fast::Bool=false, wav="haar", levels_decomp::Int=3, levels_estim::Union{Nothing,Int}=nothing, kwargs...)
    typhoon = Typhoon()
    U = solve(typhoon, im0, im1; wav=wav, levels_decomp=levels_decomp, levels_estim=levels_estim, kwargs...)
    return U, typhoon
end

function RMSE(Ua::Tuple, Ub::Tuple)
    diff = [ua - ub for (ua, ub) in zip(Ua, Ub)]
    return sqrt(mean(sum(d.^2 for d in diff)))
end

end
