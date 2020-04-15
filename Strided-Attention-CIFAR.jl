using Pkg
for p in ("Knet","Images","ImageMagick", "MAT", "LinearAlgebra")
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
    return d.w * mat(x) .+ d.b
end

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

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
    out_dims_h = Int(((inh-kernel_size) + 2*padding)/stride + 1)
    out_dims = (out_dims_h^2, dv, b)
    return out_dims, out_dims_h
end

function pad_zeros(img, pad)
    h,w,c,b = size(img)
    c_pad = KnetArray{Float32}(zeros((h,pad,c,b))); #display(size(c_pad))
    r_pad = KnetArray{Float32}(zeros((pad,w+2*pad,c,b))); #display(size(r_pad))
    img_temp_p = cat(c_pad,img, dims = 2); #display(size(img_temp_p))
    img_temp_p = cat(img_temp_p,c_pad, dims = 2); #display(size(img_temp_p))
    img_temp_p = cat(r_pad,img_temp_p, dims = 1); #display(size(img_temp_p))
    img_temp_p = cat(img_temp_p,r_pad, dims = 1); #display(size(img_temp_p))
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
end

function self_attention_2d(input_dims, kernel_size, stride, padding, Nh, dk, dv)
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
                                conv_attn, kernel_size, stride, padding, dk, dv, Nh, dkh, dvh)
end

function (s::self_attention_2d)(x)
    
    out_dims, out_h = odims(x, s.kernel_size, s.stride, s.padding, s.dv)
    out = nothing
    x = pad_zeros(x, s.padding)
    imh, imw, imc, b = size(x)
    
    for i = 1:s.stride:imh-s.kernel_size+1
        for j = 1:s.stride:imw-s.kernel_size+1

            x_patch = x[i:i+s.kernel_size-1, j:j+s.kernel_size-1, :, :]
            
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
            
            #attn_out = pool(attn_out; window = s.kernel_size, mode = 2)
            flat_attn = flatten_hw(attn)
            attn_out = sum(flat_attn, dims = 1)
            
            #attn_out = reshape(attn_out, (1, s.dv, :))
            
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

struct Bottleneck
    version
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

function Bottleneck(version, inplanes, planes, expansion; stride = 1,  padding = 1, is_downsample = false, 
                        groups = 1, base_width = 64)
    
    width = Int(planes * (base_width / 64.)) * groups
    
    conv1 = Conv1x1(inplanes,width)
    bn1 = Batchnorm(width)
    if version == "Conv"
        conv2 = Conv3x3(width,width, stride = stride, padding = padding)
    elseif version == "Attn"
        conv2 = self_attention_2d(width, 5, 1, 2, 4, width, width)
    end    
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
    
    return Bottleneck(version, expansion, conv1, bn1, conv2, bn2, conv3, bn3, activation, downsample, stride)
end

function (b::Bottleneck)(x)
    identity = x
    
    out = b.conv1(x)
    out = b.bn1(out)
    out = b.activation.(out)
    
    out = b.conv2(out)
    if b.stride != 1 && b.version == "Attn"
        out = pool(out; mode = 2)
    end
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

function _make_stage(inplanes, block, block_version, block_expansion, planes, blocks; stride = 1) #dilate = False
    is_downsample = false
    if stride != 1 || inplanes != planes * block_expansion
        is_downsample = true
    end
    layers = []
    push!(layers, block(block_version, inplanes, planes, block_expansion; stride = stride, is_downsample = is_downsample))
    inplanes = planes * block_expansion
    for _ in range(1, stop = blocks)
        push!(layers, block(block_version, inplanes, planes, block_expansion))
    end
    return Chain(layers...), inplanes
end

mutable struct ResNetAttn
    base_width
    conv_i
    bn_i
    activation
    max_pool
    bottleneck_conv_1
    bottleneck_conv_2
    bottleneck_attn_1
    #bottleneck_attn_2
    avg_pool
    fc
end

function ResNetAttn(layers, num_classes = 10)
    
    global inplanes = 64
    base_width = 64
    conv_i = Conv(7,7,3,64; stride = 1, padding = 3)
    bn_i = Batchnorm(64)
    activation = relu
    max_pool = pool
    
    layer_1, inplanes = _make_stage(inplanes, Bottleneck, "Conv", 4, 64, layers[1]; stride = 2)
    layer_2, inplanes = _make_stage(inplanes, Bottleneck, "Conv", 4, 128, layers[2]; stride = 2)
    layer_3, inplanes = _make_stage(inplanes, Bottleneck, "Attn", 4, 256, layers[3]; stride = 2)
    #layer_4, inplanes = _make_stage(inplanes, Bottleneck, "Attn", 4, 256, layers[4]; stride = 2)
    
    avg_pool = pool
    fc = Dense(256 * 4, num_classes)
    
    return ResNetAttn(base_width, conv_i, bn_i, activation, max_pool, layer_1, layer_2, layer_3,
                        avg_pool, fc) #layer4
    
end

function (r::ResNetAttn)(x)
    x = r.conv_i(x)
    x = r.bn_i(x)
    x = r.activation.(x)
    x = r.max_pool(x; window = 3, stride = 2, padding = 1)
    x = r.bottleneck_conv_1(x)
    x = r.bottleneck_conv_2(x)
    x = r.bottleneck_attn_1(x)
    #x = r.bottleneck_attn_2(x)
    x = r.avg_pool(x; window = 4, mode = 2)
    x = mat(x)
    x = r.fc(x)
    return x
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

dtrn = minibatch(xtrn, ytrn, 25;
                   shuffle=true,
                   xtype=atype());

dtst = minibatch(xtst, ytst, 20;
                     partial=true,
                     xtype=atype());

#=
function train(file, dtrn, dtst, epochs; lr = 0.001)
    net = ResNetAttn([3, 4, 4])
    avgloss = []
    sumloss = 0
    currloss = []
    len = length(dtrn)
    iteration = 0
    ind = []
    for e = 1:epochs
        for (i,v) in enumerate(adam(net, dtrn; lr = lr))
            iteration += 1
            push!(ind, iteration)
            push!(currloss,v)
            sumloss += v
            avg_temp = sumloss / iteration
            push!(avgloss, avg_temp)
            println("iteration: $i / $len   loss: $v")
        end
        acc = accuracy(net, dtst)
        println("epoch: $e    test_acc: $acc")
        Knet.gc()
    end
    Knet.save(file,"net", net)
    return ind, currloss, avgloss
end=#

function train(file, dtrn, dtst, epochs; lr = 0.001, resume = false)
    if resume
        @info("Resuming from the model stored in %file....")
        net = Knet.load(file, "net")
    else
        @info("Creating a new model from scratch....")
        net = ResNetAttn([3, 4, 4])
    end
    
    @info("Model is successfully built and loaded....")
    
    for p in params(net)
        p.opt = Adam(; lr = lr)
    end
    
    avgloss = []
    sumloss = 0
    currloss = []
    acc = []
    len = length(dtrn)
    ind = []
    iteration = 0
    for e = 1:epochs
        for (x,y) in dtrn
            J = @diff net(x,y)
            loss = value(J)
            iteration += 1
            sumloss += loss
            avg_temp = sumloss / iteration
            for p in params(net)
                g = grad(J, p)
                update!(p, g)
            end
            #acc_t = accuracy(net,[(x,y)])
            #push!(currloss,loss)
            #push!(ind, iteration)
            #push!(avgloss, avg_temp)
            #push!(acc, acc_t)
            println("epoch: $e      iteration: $iteration / $(len*epochs)     loss: $loss      avg_loss: $avg_temp")
            flush(stdout)
        end
        acc = accuracy(net, dtst)
        println("------  Epoch finished : epoch: $e    Test Accuracy: $acc  -----------")
        flush(stdout)
        Knet.gc()
    end
    Knet.save(file,"net", net)
    @info("Model successfully saved....")
end

endswith(string(PROGRAM_FILE), "Strided-Attention-CIFAR.jl") && train("cifar.jld2", dtrn, dtst, 10; resume = true)