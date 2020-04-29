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
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray : Array)) : at
end

struct Conv1x1
    w
    stride
    padding
end

function Conv1x1(cx::Int, cy::Int; stride = 1, padding = 0)
    w = param(1, 1, cx, cy)
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
    w = param(3, 3, cx, cy)
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
    w = param(w1, w2, cx, cy)
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
    w = param(w1, w2, cy, cx)
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
function Dense(i::Int,o::Int)     
    w = param(o,i)
    b = param0(o)
    return Dense(w,b)
end

function (d::Dense)(x)
    return (d.w * mat(x)) .+ d.b
end

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


struct Bottleneck
    expansion
    conv1
    bn1
    conv2
    bn2
    conv3
    bn3
    activation
    downsample
    stride
end

function Bottleneck(inplanes, planes, expansion; stride = 1,  padding = 1, is_downsample = false, 
                        groups = 1, base_width = 64)
    
    width = Int(planes * (base_width / 64.)) * groups
    
    conv1 = Conv1x1(inplanes,width)
    bn1 = Batchnorm(width)
    
    conv2 = Conv3x3(width,width, stride = stride, padding = padding)
    bn2 = Batchnorm(width)
    
    conv3 = Conv1x1(width, planes*expansion)
    bn3 = Batchnorm(planes*expansion)
    activation = relu
    
    if is_downsample
        downsample = Chain(Conv1x1(inplanes, planes*expansion, stride = stride),
                            Batchnorm(planes*expansion))
    else
        downsample = nothing
    end
    
    return Bottleneck(expansion, conv1, bn1, conv2, bn2, conv3, bn3, activation, downsample, stride)
end

function (b::Bottleneck)(x)
    identity = x
    
    out = b.conv1(x)
    out = b.bn1(out)
    out = b.activation.(out)
    
    out = b.conv2(out)
    out = b.bn2(out)
    out = b.activation.(out)
    
    out = b.conv3(out)
    out = b.bn3(out)
    
    if b.downsample != nothing
        identity = b.downsample(x)
    end
    
    out += identity
    out = b.activation.(out)
    
    return out
    
end

function _make_stage(inplanes, block, block_expansion, planes, blocks; stride = 1) 
    is_downsample = false
    if stride != 1 || inplanes != planes * block_expansion
        is_downsample = true
    end
    layers = []
    push!(layers, block(inplanes, planes, block_expansion; stride = stride, is_downsample = is_downsample))
    inplanes = planes * block_expansion
    for _ in range(1, stop = blocks)
        push!(layers, block(inplanes, planes, block_expansion))
    end
    return Chain(layers...), inplanes
end

mutable struct ResNet
    base_width
    conv_i
    bn_i
    activation
    max_pool
    bottleneck_conv_1
    bottleneck_conv_2
    bottleneck_conv_3
    bottleneck_conv_4
    avg_pool
    fc
end

function ResNet(layers, num_classes = 10)
    
    global inplanes = 64
    base_width = 64
    conv_i = Conv(7,7,3,64; stride = 1, padding = 3)
    bn_i = Batchnorm(64)
    activation = relu
    max_pool = pool
    
    layer_1, inplanes = _make_stage(inplanes, Bottleneck, 4, 64, layers[1])
    layer_2, inplanes = _make_stage(inplanes, Bottleneck, 4, 128, layers[2]; stride = 2)
    layer_3, inplanes = _make_stage(inplanes, Bottleneck, 4, 128, layers[3]; stride = 2)
    layer_4, inplanes = _make_stage(inplanes, Bottleneck, 4, 256, layers[4]; stride = 2)
    
    avg_pool = pool
    fc = Dense(256 * 4, num_classes)
    
    return ResNet(base_width, conv_i, bn_i, activation, max_pool, layer_1, layer_2, layer_3, layer_4, avg_pool, fc) 
    
end

function (r::ResNet)(x)
    x = r.conv_i(x)
    x = r.bn_i(x)
    x = r.activation.(x)
    x = r.max_pool(x; window = 3, stride = 2, padding = 1)
    x = r.bottleneck_conv_1(x)
    x = r.bottleneck_conv_2(x)
    x = r.bottleneck_conv_3(x)
    x = r.bottleneck_conv_4(x)
    x = r.avg_pool(x; window = size(x,1), mode = 2)
    x = mat(x)
    x = r.fc(x)
    return x
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

dtrn = minibatch(xtrn, ytrn, 150;
                   shuffle=true,
                   xtype=atype());

dtst = minibatch(xtst, ytst, 150;
                     partial=true,
                     xtype=atype());



const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, ".", "checkpoints"))

function train(name, loadname, dtrn, dtst, epochs; weight_decay = 1e-4, optimizer = Momentum, momentum = 0.9, lr = 0.1)
    
    loadfile = abspath(joinpath(SAVEDIR, loadname))
    
    bestmodel_path = abspath(joinpath(SAVEDIR, name*"-best.jld2"))
    
    len = length(dtrn)
   
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
        net = ResNet([1, 2, 4, 1])
        avgloss = []
        sumloss = 0
        currloss = []
        overall_iter = 0
    end
    
    iteration = 0
    
    println("Model is successfully built and loaded....")
    
    for p in params(net)
        p.opt = optimizer(; gclip = 0.1, gamma = momentum, lr = lr)
    end
    
    for e in 1:epochs
        acc_epoch_sum = 0
        acc_avg = 0
        for (i,(x,y)) in enumerate(dtrn)
            J = @diff (net(x,y) + sum([(weight_decay * sum(i.value .* i.value)) for i in params(net)]))
            loss = value(J)
            iteration += 1
            sumloss += loss
            avg_temp = sumloss / (overall_iter + iteration)
            acc_t = accuracy(net,[(x,y)])
            acc_epoch_sum += acc_t
            acc_avg = acc_epoch_sum / len
            for p in params(net)
                g = grad(J, p)
                update!(p, g)
            end
            push!(currloss, loss)
            push!(avgloss, avg_temp)
            println("epoch: $e iter: $iteration / $(len*epochs)    loss: $loss   avg_loss: $avg_temp    accuracy: $acc_t")
            flush(stdout)
        end
        println("Average Training Accuracy during this epoch: $acc_avg")
        println("Calculating this version's performance on the test set...")
        acc = accuracy(net, dtst)
        println("------  Epoch finished : epoch: $e    Test Accuracy: $acc  -----------")
        flush(stdout)
        if acc > bestacc
            bestacc = acc
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
        if e % 8 == 0
            for p in params(net)
                p.opt.lr =  p.opt.lr/2
                lrr = p.opt.lr
            end
            println("Learning rate reduced to: $lrr")
        end
    end
end

endswith(string(PROGRAM_FILE), "ResNet_CIFAR.jl") && train("resnet_20_SGD", "", dtrn, dtst, 60; lr = 0.1)

