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


struct BasicBlock
    expansion
    conv1
    bn1
    conv2
    bn2
    activation
    downsample
    stride
end

function BasicBlock(inplanes, planes; expansion = 1, stride = 1, is_downsample = false)
    
    conv1 = Conv3x3(inplanes, planes*expansion, stride = stride, padding = 1)
    bn1 = Batchnorm(planes*expansion)
    
    conv2 = Conv3x3(planes*expansion, planes*expansion, stride = 1, padding = 1)
    bn2 = Batchnorm(planes*expansion)
    
    activation = relu
    
    if is_downsample
        downsample = Chain(Conv1x1(inplanes, planes*expansion, stride = stride),
                            Batchnorm(planes*expansion))
    else
        downsample = nothing
    end
    
    return BasicBlock(expansion, conv1, bn1, conv2, bn2, activation, downsample, stride)
end

function (b::BasicBlock)(x)
    identity = x
    
    out = b.conv1(x)
    out = b.bn1(out)
    out = b.activation.(out)
    
    out = b.conv2(out)
    out = b.bn2(out)
    
    if b.downsample != nothing
        identity = b.downsample(x)
    end
    
    out += identity
    out = b.activation.(out)
    
    return out
    
end

function _make_stage(block, block_expansion, inplanes, planes, num_blocks; stride = 1) 
    is_downsample = false
    if stride != 1 || inplanes != planes * block_expansion
        is_downsample = true
    end
    layers = []
    push!(layers, block(inplanes, planes; expansion = block_expansion, stride = stride, is_downsample = is_downsample))
    inplanes = planes * block_expansion
    for _ in range(1, stop = num_blocks)
        push!(layers, block(inplanes, planes; expansion =  block_expansion))
    end
    return Chain(layers...), inplanes
end


mutable struct ResNet
    conv_i
    bn_i
    activation
    stage_1
    stage_2
    stage_3
    avg_pool
    fc_out
end

function ResNet(layers, num_classes = 10)
    
    inplanes = 16
    conv_i = Conv(3,3,3,16; stride = 1, padding = 1)
    bn_i = Batchnorm(16)
    activation = relu
    
    stage_1, inplanes = _make_stage(BasicBlock, 1, inplanes, 16, layers[1])
    stage_2, inplanes = _make_stage(BasicBlock, 1, inplanes, 32, layers[2]; stride = 2)
    stage_3, inplanes = _make_stage(BasicBlock, 1, inplanes, 64, layers[3]; stride = 2)
    
    avg_pool = pool
    fc_out = Dense(64, 10)
    
    return ResNet(conv_i, bn_i, activation, stage_1, stage_2, stage_3,  avg_pool, fc_out) 
    
end

function (r::ResNet)(x)
    x = r.conv_i(x)
    x = r.bn_i(x)
    x = r.activation.(x)
    x = r.stage_1(x)
    x = r.stage_2(x)
    x = r.stage_3(x)
    x = r.avg_pool(x; window = size(x,1), mode = 2)
    x = mat(x)
    out = r.fc_out(x)
    return out
end

function (r::ResNet)(x,y)
    scores = r(x)
    loss = nll(scores, y)
    return loss
end

function (r::ResNet)(d::Data)
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

function train(name, loadname, xtrn, ytrn, dtst, epochs; weight_decay = 1e-4, optimizer = Momentum, momentum = 0.99, lr = 0.1)
    
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
        net = ResNet([2, 2, 2])
        avgloss = []
        sumloss = 0
        currloss = []
        overall_iter = 0
    end
    
    iteration = 0
    
    println("Model is successfully built and loaded....")
    
    for p in params(net)
        p.opt = optimizer(; gamma = momentum, lr = lr)
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
        
        
    
    end
end

endswith(string(PROGRAM_FILE), "ResBasicBlock.jl") && train("resnet_20_BasicBlock_LRsched", "", xtrn, ytrn, dtst, 40; lr = 0.008)

