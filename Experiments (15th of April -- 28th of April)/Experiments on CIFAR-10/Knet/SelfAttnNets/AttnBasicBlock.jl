using Pkg
for p in ("Knet","Images","ImageMagick", "MAT", "IterTools","Statistics", "Plots", "LinearAlgebra")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end
using Knet, MAT, Images, Random
using Base.Iterators: flatten, cycle, take
using IterTools
using LinearAlgebra
using Statistics: mean
using Plots; default(fmt=:png,ls=:auto)
import Base: length, size, iterate
include(Knet.dir("data", "cifar.jl"))
import Knet: Data

#Pkg.update("Knet")
Knet.gpu(0)

let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})) : at
end

kaiming(h, w, i, o) = Float32(sqrt(2 / (w * h * o))) .* randn(Float32, h, w, i, o)

struct Conv1x1
    w
    stride
    padding
end

function Conv1x1(cx::Int, cy::Int; stride = 1, padding = 0)
    w = param(1, 1, cx, cy; init = kaiming, atype = atype())
    return Conv1x1(w, stride, padding)
end

function (c::Conv1x1)(x)
    return conv4(c.w, x ; padding = c.padding, stride = c.stride)
end

struct Conv3x3
    w
    stride
    padding
end

function Conv3x3(cx::Int, cy::Int; stride = 1, padding = 1)
    w = param(3, 3, cx, cy; init = kaiming, atype = atype())
    return Conv3x3(w, stride, padding)
end

function (c::Conv3x3)(x)
    return conv4(c.w, x ; padding = c.padding, stride = c.stride)
end

struct Conv
    w
    stride
    padding
end

function Conv(w1::Int, w2::Int, cx::Int, cy::Int; stride = 1, padding = 0)
    w = param(w1, w2, cx, cy; init = kaiming, atype = atype())
    return Conv(w, stride, padding)
end

function (c::Conv)(x)
    return conv4(c.w, x ; padding = c.padding, stride = c.stride)
end

struct Deconv
    w
    stride
    padding
end

function Deconv(w1::Int, w2::Int, cy::Int, cx::Int; stride = 1, padding = 0)
    w = param(w1, w2, cy, cx; init = xavier_normal, atype = atype())
    return Deconv(w, stride, padding)
end

function (c::Deconv)(x)
    return deconv4(c.w, x ; padding = c.padding, stride = c.stride)
end

struct Batchnorm
    w
    m
end 

function Batchnorm(w1::Int)
    w = KnetArray(bnparams(Float32, w1))
    m = bnmoments()
    return Batchnorm(w,m)
end

function (b::Batchnorm)(x)
    return batchnorm(x, b.m, b.w)
end

struct Dense
    w
    b
end

function Dense(i::Int,o::Int; bias = true)     
    w = param(o,i; init = xavier, atype = atype())
    if bias
        b = param0(o)
    else
        b = nothing
    end
    return Dense(w,b)
end

function (d::Dense)(x)
    if d.b != nothing
        return (d.w * mat(x)) .+ d.b
    else
        return (d.w * mat(x))
    end
end

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

####--------------------------------------------------------------------------------####
####--------------------------------------------------------------------------------####

function rel_logits_2d(flat_q, rel, kernel_size)
    h2, d, Nh, b = size(flat_q)
    flat_q = reshape(flat_q, (1, h2, d, Nh, b))
    rel = flatten_rel(rel, kernel_size)
    rel_logits = rel .* flat_q
    rel_logits = reshape(sum(rel_logits, dims = 3),(h2, h2, Nh, b))
    return rel_logits
end

function flatten_hw(input)
    if ndims(input) == 5
        h, w, d, Nh, b = size(input)
        new_size = (h*w, d, Nh, b)
    else
        h, w, d, b = size(input)
        new_size = (h*w, d, b)
    end
    return reshape(input, new_size)
end

function flatten_rel(rel, kernel_size)
    h2, w2, c, Nh, b = size(rel)
    temp_size = (kernel_size, kernel_size, kernel_size, kernel_size, c, Nh, b)
    rel = permutedims(reshape(rel, temp_size),[1,3,2,4,5,6,7])
    new_size = (h2, w2, c, Nh, b)
    rel = reshape(rel, new_size)
    return rel
end

function split_heads_2d(inputs, Nh)
    h, w, d, b = size(inputs)
    ret_shape = (h, w, floor(Int,d/Nh), Nh, b)
    out = reshape(inputs, ret_shape)
    return out
end

function combine_heads_2d(inputs)
    h, w, dh, Nh, b  = size(inputs)
    ret_shape = (h,w, dh*Nh, b)
    return reshape(inputs, ret_shape)
end

function odims(input, kernel_size, stride, padding, dv)
    inh,inw,inc,b = size(input)
    out_dims_h = floor(Int,(((inh-kernel_size) + 2*padding)/stride + 1))
    out_dims = (out_dims_h^2, dv, b)
    return out_dims, out_dims_h
end

function pad_zeros(img, pad)
    h,w,c,b = size(img)
    c_pad = KnetArray{Float32}(zeros((h,pad,c,b))); 
    r_pad = KnetArray{Float32}(zeros((pad,w+2*pad,c,b)));
    img_temp_p = cat(c_pad,img, dims = 2); 
    img_temp_p = cat(img_temp_p,c_pad, dims = 2); 
    img_temp_p = cat(r_pad,img_temp_p, dims = 1); 
    img_temp_p = cat(img_temp_p,r_pad, dims = 1); 
    return img_temp_p
end

struct self_attention_2d
    conv_q
    conv_k
    conv_v
    conv_rel
    deconv_rel
    conv_attn
    kernel_size
    stride
    padding
    dk
    dv
    Nh
    dkh
    dvh
    drop_p
end

function self_attention_2d(input_dims, kernel_size, stride, padding, Nh, dk, dv; drop_p = 0.2)
    conv_q = Conv(1,1, input_dims, dk)
    conv_k = Conv(1,1, input_dims, dk)
    conv_v = Conv(1,1, input_dims, dv)
    conv_rel = Conv(3, 3, input_dims, dk; padding = 1)
    deconv_rel = Deconv(kernel_size, kernel_size, dk, dk; stride = kernel_size)
    conv_attn = Conv(1,1, dv, dv)
    
    dkh = floor(Int, dk/Nh)
    dvh = floor(Int, dv/Nh)
    
    stride = stride
    padding = padding
    
    return self_attention_2d(conv_q, conv_k, conv_v, conv_rel, deconv_rel,
                                conv_attn, kernel_size, stride, padding, dk,
                                dv, Nh, dkh, dvh, drop_p)
end

function (s::self_attention_2d)(x)
    
    out_dims, out_h = odims(x, s.kernel_size, s.stride, s.padding, s.dv)
    out = nothing
    x = pad_zeros(x, s.padding)
    imh, imw, imc, b = size(x)
    
    for i = 1:s.stride:imh-s.kernel_size+1
        for j = 1:s.stride:imw-s.kernel_size+1

            x_patch = x[i:i+s.kernel_size-1, j:j+s.kernel_size-1, :, :]
            
            #x_patch = dropout(x_patch, s.drop_p)
            
            _, _, _, b = size(x_patch)
            q = s.conv_q(x_patch)
            k = s.conv_k(x_patch)
            v = s.conv_v(x_patch)

            rel = s.conv_rel(x_patch)
            rel = s.deconv_rel(rel)

            q = q .* (s.dkh ^ -0.5)

            q = split_heads_2d(q,s.Nh)
            k = split_heads_2d(k,s.Nh)
            v = split_heads_2d(v,s.Nh)
            rel = split_heads_2d(rel,s.Nh)

            flat_q = flatten_hw(q)
            flat_k = flatten_hw(k)
            flat_v = flatten_hw(v)

            logits = bmm(flat_q, flat_k, transB = true)
            logits = permutedims(logits, [2,1,3,4])

            rel_logits = rel_logits_2d(flat_q, rel, s.kernel_size)

            logits += rel_logits

            weights = softmax(logits; dims = 1)

            attn = bmm(weights, flat_v, transA = true)

            attn = reshape(attn, (s.kernel_size, s.kernel_size, s.dvh, s.Nh, :))
            attn = combine_heads_2d(attn) 

            attn = s.conv_attn(attn)
            
            flat_attn = flatten_hw(attn)
            attn_out = sum(flat_attn, dims = 1)
            
            if out == nothing
                out = attn_out
            else
                out = cat(out, attn_out; dims = 1)
            end
            
        end
    end
    
    out = permutedims(reshape(out, (out_h, out_h, s.dv, :)), [2, 1, 3, 4])
    
    return out
end

####--------------------------------------------------------------------------------####
####--------------------------------------------------------------------------------####


struct BasicBlock
    version
    expansion
    conv1
    bn1
    conv2
    bn2
    activation
    downsample
    stride
end

function BasicBlock(version, inplanes, planes; expansion = 1, stride = 1, is_downsample = false)
    
    if version == "Conv"
        conv1 = Conv3x3(inplanes, planes*expansion, stride = stride, padding = 1)
    elseif version == "Attn"
        conv1 = self_attention_2d(inplanes, 3, 1, 1,
                                    8, planes*expansion, planes*expansion)
    end
    
    bn1 = Batchnorm(planes*expansion)
    
    if version == "Conv"
        conv2 = Conv3x3(planes*expansion, planes*expansion, stride = 1, padding = 1)
    elseif version == "Attn"
        conv2 = self_attention_2d(planes*expansion, 3, 1, 1, 
                                    8, planes*expansion, planes*expansion)
    end
    
    bn2 = Batchnorm(planes*expansion)
    
    activation = relu
    
    if is_downsample
        downsample = Chain(Conv1x1(inplanes, planes*expansion, stride = stride),
                            Batchnorm(planes*expansion))
    else
        downsample = nothing
    end
    
    return BasicBlock(version, expansion, conv1, bn1, conv2, bn2, activation, downsample, stride)
end

function (b::BasicBlock)(x)
    identity = x
    
    out = b.conv1(x)
    if b.stride != 1 && b.version == "Attn"
        out = pool(out; mode = 2)
    end
    out = b.bn1(out)
    if b.version == "Conv"
        out = b.activation.(out)
    end
    
    out = b.conv2(out)
    out = b.bn2(out)
    
    if b.downsample != nothing
        identity = b.downsample(x)
    end
    
    out += identity
    if b.version == "Conv"
        out = b.activation.(out)
    end
    
    return out
    
end

function _make_stage(block, block_version, block_expansion, inplanes, planes, num_blocks; stride = 1) 
    is_downsample = false
    if stride != 1 || inplanes != planes * block_expansion
        is_downsample = true
    end
    layers = []
    push!(layers, block(block_version, inplanes, planes; expansion = block_expansion, stride = stride, is_downsample = is_downsample))
    inplanes = planes * block_expansion
    for _ in range(1, stop = num_blocks)
        push!(layers, block(block_version, inplanes, planes; expansion =  block_expansion))
    end
    return Chain(layers...), inplanes
end


mutable struct ResNetAttn
    conv_i
    bn_i
    activation
    stage_1
    stage_2
    stage_3
    stage_4
    avg_pool
    #fc_1
    #bn_1
    fc_out
end

function ResNetAttn(layers, num_classes = 10)
    
    inplanes = 16
    conv_i = Conv(3,3,3,16; stride = 1, padding = 1)
    bn_i = Batchnorm(16)
    activation = relu
    
    stage_1, inplanes = _make_stage(BasicBlock, "Conv", 1, inplanes, 16, layers[1]; stride = 2)
    stage_2, inplanes = _make_stage(BasicBlock, "Conv", 1, inplanes, 32, layers[2]; stride = 2)
    stage_3, inplanes = _make_stage(BasicBlock, "Attn", 1, inplanes, 64, layers[3]; stride = 2)
    stage_4, inplanes = _make_stage(BasicBlock, "Attn", 1, inplanes, 64, layers[4]; stride = 2)
    
    avg_pool = pool
    #fc_1 = Dense(64, 128)
    #bn_1 = Batchnorm(128)
    fc_out = Dense(64, 10)
    
    return ResNetAttn(conv_i, bn_i, activation, stage_1, stage_2, stage_3, stage_4, avg_pool, fc_out) 
    
end

function (r::ResNetAttn)(x)
    x = r.conv_i(x)
    x = r.bn_i(x)
    x = r.activation.(x)
    x = r.stage_1(x)
    x = r.stage_2(x)
    x = r.stage_3(x)
    x = r.stage_4(x)
    x = r.avg_pool(x; window = size(x,1), mode = 2)
    x = mat(x)
    #x = r.fc_1(x)
    #x = r.bn_1(x)
    #x = relu.(x)
    out = r.fc_out(x)
    return out
end

function (r::ResNetAttn)(x,y)
    scores = r(x)
    loss = nll(scores, y)
    return loss
end

function (r::ResNetAttn)(d::Data)
    mean_loss = mean(r(x,y) for (x,y) in d)
    return mean_loss
end


function loaddata()
    @info("Loading CIFAR 10...")
    xtrn, ytrn, xtst, ytst, = cifar10()
    #= Subtract mean of each feature
    where each channel is considered as
    a single feature following the CNN
    convention=#
    mn = mean(xtrn, dims=(1,2,4))
    xtrn = xtrn .- mn
    xtst = xtst .- mn
    @info("Loaded CIFAR 10")
    return (xtrn, ytrn), (xtst, ytst)
end

(xtrn, ytrn), (xtst, ytst) = loaddata();

summary.([xtrn,ytrn,xtst,ytst])

dtst = minibatch(xtst, ytst, 200;
                     partial=true,
                     xtype=atype());


const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, ".", "checkpoints"))

function train(name, loadname, xtrn, ytrn, dtst, epochs; weight_decay = 1e-6, optimizer = Adam, momentum = 0.99, lr = 0.1)
    
    loadfile = abspath(joinpath(SAVEDIR, loadname))
    
    bestmodel_path = abspath(joinpath(SAVEDIR, name*"-best.jld2"))

    len_dtst = length(dtst)
   
    history = Dict()
    bestacc = 0.0
    
    if isfile(loadfile)
        println("Resuming from the model stored in %file....")
        Knet.@load loadfile net history
        overall_iter = history["overall_iter"]
        sumloss = history["sumloss"]
        avgloss = history["avgloss"]
        currloss = history["currloss"]
        if haskey(history, bestacc)
            bestacc = history["bestacc"]
        end
    else
        println("Creating a new model from scratch....")
        net = ResNetAttn([1, 2, 3, 1])
        avgloss = []
        sumloss = 0
        currloss = []
        overall_iter = 0
    end
    
    iteration = 0
    
    println("Model is successfully built and loaded....")
    
    for p in params(net)
        #p.opt = optimizer(; gamma = momentum, lr = lr)
        p.opt = optimizer(; lr = lr)
    end
    
    for e in 1:epochs
        
        dtrn = minibatch(xtrn, ytrn, 50;
                   shuffle=true,
                   xtype=atype());
        len_dtrn = length(dtrn)
        
        acc_epoch_sum = 0
        acc_avg = 0
        for (i,(x,y)) in enumerate(dtrn)
            #x = augment(x)
            J = @diff (net(x,y) + sum([(weight_decay * sum(i.value .* i.value)) for i in params(net)]))
            #J = @diff net(x,y)
            loss = value(J)
            iteration += 1
            sumloss += loss
            avg_temp = sumloss / (overall_iter + iteration)
            acc_t = accuracy(net,[(x,y)])
            acc_epoch_sum += acc_t
            acc_avg = acc_epoch_sum / len_dtrn
            for p in params(net)
                g = grad(J, p)
                update!(p, g)
            end
            push!(currloss, loss)
            push!(avgloss, avg_temp)
            println("epoch: $e iter: $iteration / $(len_dtrn*epochs)    loss: $loss   avg_loss: $avg_temp    accuracy: $acc_t")
            flush(stdout)
            
            """
            lrr = Any
            if iteration < 19500
                lr_step = 6e-5
            elseif iteration >= 19500 && iteration < 30000
                lr_step = -1e-4
            else
                lr_step = -1e-7
            end
            for p in params(net)
                p.opt.lr =  p.opt.lr + lr_step
                lrr = p.opt.lr
            end
            """
        end
        println("Average Training Accuracy during this epoch: $acc_avg")
        println("Calculating this version's performance on the test set...")
        acc_tst_epoch_sum = 0
        acc_tst_avg = 0 
        for (i,(x,y)) in enumerate(dtst)
            acc_tst_t = accuracy(net,[(x,y)])
            println("iter: $i / $(len_dtst)  accuracy: $acc_tst_t")
            acc_tst_epoch_sum += acc_tst_t
            acc_tst_avg = acc_tst_epoch_sum / len_dtst
        end
        println("------  Epoch finished : epoch: $e    Test Accuracy: $acc_tst_avg  -----------")
        flush(stdout)
        if acc_tst_avg > bestacc
            bestacc = acc_tst_avg
            history["bestacc"] = bestacc
            history["overall_iter"] = overall_iter + iteration
            history["sumloss"] = sumloss
            history["avgloss"] = avgloss
            history["currloss"] = currloss
            println("Best version of the model successfully saved....")
            Knet.@save bestmodel_path net history
        end
        Knet.gc()
        lrr = Any
        if e % 5 == 0
            for p in params(net)
                p.opt.lr =  p.opt.lr/2
                lrr = p.opt.lr
            end
            println("Learning rate reduced to: $lrr")
        end
    end
end

endswith(string(PROGRAM_FILE), "AttnBasicBlock.jl") && train("attnnet_20_BasicBlock_1231", "", xtrn, ytrn, dtst, 30; lr = 0.002)

