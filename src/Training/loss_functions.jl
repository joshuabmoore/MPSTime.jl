contract_tensor = ITensors._contract

abstract type LossFunction <: Function end

abstract type KLDLoss <: LossFunction end
abstract type MSELoss <: LossFunction end

struct Loss_Grad_MSE <: MSELoss end
struct Loss_Grad_KLD <: KLDLoss end
struct Loss_Grad_KLD_slow <: KLDLoss end

struct Loss_Grad_mixed <: LossFunction end
struct Loss_Grad_default <: LossFunction end


loss_grad_MSE = Loss_Grad_MSE()
loss_grad_KLD = Loss_Grad_KLD()
loss_grad_KLD_slow = Loss_Grad_KLD_slow()

loss_grad_mixed = Loss_Grad_mixed()
loss_grad_default = Loss_Grad_default()

#######################################################
function kron_add!(k::AbstractVector, x1::AbstractVector, x2::AbstractVector)
    l1,l2 = length(x1),length(x2)
    @turbo for i = eachindex(x1), j =eachindex(x2)
        k[j + l2*(i-1)] += x1[i] * x2[j]
    end
end

function kron_add!(k::AbstractVector, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    x2a = kron(x2,x3)
    l1,l2 = length(x1),length(x2a)
    @turbo for i = eachindex(x1), j =eachindex(x2a)
        k[j + l2*(i-1)] += x1[i] * x2a[j]
    end
end

function kron_add!(k::AbstractVector, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector, x4::AbstractVector)
    xa = kron(x1,x2)
    xb = kron(x3,x4)
    la,lb = length(xa),length(xb)
    @turbo for i = eachindex(xa), j =eachindex(xb)
        k[j + lb*(i-1)] += xa[i] * xb[j]
    end
end

function kron_scale(scale::Float64,x1::AbstractVector, x2::AbstractVector)
    l1,l2 = length(x1),length(x2)
    out = Vector{eltype(x1)}(undef, l1*l2)
    @turbo for i = eachindex(x1), j=eachindex(x2)
        out[j + l2*(i-1)] = x1[i] * x2[j] / scale
    end
    return out
end


function kron_conj1(x1::AbstractVector, x2::AbstractVector)
    l1,l2 = length(x1),length(x2)
    out = Vector{eltype(x1)}(undef, l1*l2)
    @turbo for i = eachindex(x1), j=eachindex(x2)
        out[j + l2*(i-1)] = x1[i] * conj(x2[j])
    end
    return out
end

function kron_conj2(x1::AbstractVector, x2::AbstractVector)
    l1,l2 = length(x1),length(x2)
    out = Vector{eltype(x1)}(undef, l1*l2)
    @turbo for i = eachindex(x1), j=eachindex(x2)
        out[j + l2*(i-1)] = conj(x1[i] * x2[j])
    end
    return out
end


function kron_scaleadd!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yp::Base.RefValue{Float64}, x1::AbstractVector, x2::AbstractVector)
    l2 = length(x2)
    yhat = 0.
    scale = yp[]
    @turbo for i = eachindex(x1), j =eachindex(x2)
        idx = j + l2*(i-1)
        phi = conj(x1[i] * x2[j])
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yp[] = yhat
    return -log(abs2(yhat))
end

function kron_scaleadd_firstsite!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yp::Base.RefValue{Float64}, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yp[]
    @turbo for i = eachindex(x1), j =eachindex(x2a)
        idx = j + l2*(i-1)
        phi = x1[i] * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yp[] = yhat
    return -log(abs2(yhat))
end

function kron_scaleadd!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yp::Base.RefValue{Float64}, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yp[]
    @turbo for i = eachindex(x1), j =eachindex(x2a)
        idx = j + l2*(i-1)
        phi = conj(x1[i]) * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yp[] = yhat
    return -log(abs2(yhat))
end


function kron_scaleadd!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yp::Base.RefValue{Float64}, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector, x4::AbstractVector)
    xa = kron_conj2(x1,x2)
    xb = kron_conj2(x3,x4)
    lb = length(xb)
    yhat = 0.
    scale = yp[]
    @turbo for i = eachindex(xa), j = eachindex(xb)
        idx = j + lb*(i-1)
        phi = xa[i] * xb[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yp[] = yhat
    return -log(abs2(yhat))
end

# function (::Loss_Grad_KLD)(::TrainSeparate{true}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     cnums = ETSs.class_distribution
#     TSs = ETSs.timeseries
#     label_idx = inds(BT)[1]

#     losses = zero(real(eltype(BT)))
#     grads = Tensor(zeros(eltype(BT), size(BT)), inds(BT))
#     no_label = inds(BT)[2:end]
#     phit_scaled = Tensor(eltype(BT), no_label)
#     # phi_tilde = Tensor(eltype(BT), no_label)


#     i_prev = 0
#     for (ci, cn) in enumerate(cnums)
#         y = onehot(label_idx => ci)
#         bt = tensor(BT * y)
#         phit_scaled .= zero(eltype(bt))

#         c_inds = (i_prev+1):(cn+i_prev)
#         @inbounds @fastmath loss = mapreduce((LEP,REP, prod_state) -> KLD_iter!(phit_scaled,bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
#         @inbounds @fastmath losses += loss/cn # maybe doing this with a combiner instead will be more efficient
#         @inbounds @fastmath @. $selectdim(grads,1, ci) -= conj(phit_scaled) / cn

#         i_prev += cn

#     end


#     return losses, itensor(eltype(BT), grads, inds(BT))

# end

function (::Loss_Grad_KLD)(
        ::TrainSeparate{false}, 
        bts::BondTensor, 
        LE::PCache, 
        RE::PCache,
        ETSs::EncodedTimeSeriesSet, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    nts, nsites = size(TSs)[2:3]
    num_type = eltype(eltype(bts))

    losses = zero(real(num_type))
    phit_scaled = zeros(num_type, size(bts)) 
    yhat = Ref{Float64}(1.)
    phit_prev = Vector{num_type}(undef, size(bts,1)) 


    i_prev=0
    for (ci, cn) in enumerate(cnums)
        bt = view(bts, :, ci)
        yhat[] = 1.
        phit_prev .= 0
        c_inds = (i_prev+1):(cn+i_prev)

        if lid == 1
            # first site
            if rid == nsites
                # two site MPS
                @inbounds @fastmath losses += mapreduce(
                    (rs,ls)->kron_scaleadd!(view(phit_scaled,:,ci), view(phit_prev, :), bt, yhat, rs, ls),
                    +,
                    eachcol(view(TSs, :, c_inds, rid)),
                    eachcol(view(TSs, :, c_inds, lid)),
                )
            else
                @inbounds @fastmath losses += mapreduce(
                    (rc,rs,ls)->kron_scaleadd_firstsite!(view(phit_scaled,:,ci), view(phit_prev, :), bt, yhat, rc, rs, ls),
                    +,
                    view(RE, c_inds, rid+1),
                    eachcol(view(TSs, :, c_inds, rid)),
                    eachcol(view(TSs, :, c_inds, lid)),
                )
            end

        elseif rid == nsites
            # last site; not a two site MPS
            @inbounds @fastmath losses += mapreduce(
                (rs,lc,ls)->kron_scaleadd!(view(phit_scaled,:,ci), view(phit_prev, :), bt, yhat, rs, lc, ls), 
                +,
                eachcol(view(TSs, :, c_inds, rid)),
                view(LE, c_inds, lid-1),
                eachcol(view(TSs, :, c_inds, lid)),
            )

        else
            # in the bulk
            @inbounds @fastmath losses += mapreduce(
                (rc,rs,lc,ls)->kron_scaleadd!(view(phit_scaled,:,ci), view(phit_prev, :), bt, yhat, rc, rs, lc, ls), 
                +,
                view(RE, c_inds, rid+1),
                eachcol(view(TSs, :, c_inds, rid)),
                view(LE, c_inds, lid-1),
                eachcol(view(TSs, :, c_inds, lid)),
            )

        end
        @inbounds @fastmath @. phit_scaled[:, ci] = -conj(phit_scaled[:, ci] + phit_prev / yhat[] / nts )
        i_prev += cn
    end
    losses /= nts

    return losses, phit_scaled

end
#####################################################################################################  MSE LOSS

# function MSE_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
#     product_state::PState, lid::Int, rid::Int) 
#     """Computes the Mean squared error loss function derived from KL divergence and its gradient"""


#     yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

#     # convert the label to ITensor
#     label_idx = inds(yhat)[1]
#     y = onehot(label_idx => (product_state.label_index))

#     diff_sq = abs2.(yhat - y)
#     sum_of_sq_diff = sum(diff_sq)
#     loss = 0.5 * real(sum_of_sq_diff)

#     # construct the gradient - return dC/dB
#     gradient = (yhat - y) * conj(phi_tilde)

#     return [loss, gradient]

# end


# function (::Loss_Grad_MSE)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     TSs = ETSs.timeseries
#     loss,grad = mapreduce((LEP,REP, prod_state) -> MSE_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
#     loss /= length(TSs)
#     grad ./= length(TSs)

#     return loss, grad

# end

# ###################################################################################################  Mixed loss


# function mixed_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
#     product_state::PState, lid::Int, rid::Int; alpha=5) 
#     """Returns the loss and gradient that results from mixing the logarithmic loss and mean squared error loss with mixing parameter alpha"""

#     yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

#     # convert the label to ITensor
#     label_idx = inds(yhat)[1]
#     y = onehot(label_idx => (product_state.label_index))
#     f_ln = (yhat *y)[1]
#     log_loss = -log(abs2(f_ln))

#     # construct the gradient - return dC/dB
#     log_gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

#     # MSE
#     diff_sq = abs2.(yhat - y)
#     sum_of_sq_diff = sum(diff_sq)
#     MSE_loss = 0.5 * real(sum_of_sq_diff)

#     # construct the gradient - return dC/dB
#     MSE_gradient = (yhat - y) * conj(phi_tilde)


#     return [log_loss + alpha*MSE_loss, log_gradient + alpha*MSE_gradient]

# end


# function (::Loss_Grad_mixed)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int; alpha=5)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     TSs = ETSs.timeseries
#     loss,grad = mapreduce((LEP,REP, prod_state) -> mixed_iter(BT,LEP,REP,prod_state,lid,rid; alpha=alpha),+, eachcol(LE), eachcol(RE),TSs)
    
#     loss /= length(TSs)
#     grad ./= length(TSs)

#     return loss, grad

# end


# ######################### old  generic Loss_Grad function
# function (::Loss_Grad_default)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     TSs = ETSs.timeseries
#     loss,grad = mapreduce((LEP,REP, prod_state) -> lg_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
#     loss /= length(TSs)
#     grad ./= length(TSs)

#     return loss, grad

# end

# function (::Loss_Grad_default)(::TrainSeparate{true}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     cnums = ETSs.class_distribution
#     TSs = ETSs.timeseries
#     label_idx = find_index(BT, "f(x)")

#     losses = ITensor(real(eltype(BT)), label_idx)
#     grads = ITensor(eltype(BT), inds(BT))

#     i_prev=0
#     for (ci, cn) in enumerate(cnums)
#         y = onehot(label_idx => ci)

#         c_inds = (i_prev+1):cn
#         loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE)[c_inds], eachcol(RE)[c_inds],TSs[c_inds])

#         losses += loss  / cn # maybe doing this with a combiner instead will be more efficient
#         grads += grad / cn
#         i_prev = cn
#     end


#     return losses, grads

# end
