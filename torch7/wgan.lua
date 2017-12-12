require 'image'
require 'math'
require 'gnuplot'
require 'cunn'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'dp'
require 'optim'

local LSTM = require 'libs/LSTM' 
local model_utils = require 'libs/model_utils'
local data = require 'utils/data'

-- commond line input options
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing command line input')
cmd:text()
cmd:text('Options')
cmd:option('-plot',true,'plot graphs')
cmd:option('-batchsize',100,'batch size')
cmd:option('-optimization','rmsprop','optimization')
cmd:option('-epoch',800000,'number of epoch to run')
cmd:option('-learningrate',5e-4,'learning rate')
cmd:option('-datapath','images/','images path')
cmd:option('-saveresult',true,'save generated images')
cmd:text()

-- parse input params
opt = cmd:parse(arg or {})

discriminatorState = { learningRate = opt.learningrate }
generatorState = { learningRate = opt.learningrate }

data.dataLoader(opt.datapath, opt.batchsize)

-- model : images dimension set as 64 * 64
local discriminator = nn.Sequential()
discriminator:add(nn.Reshape(3,64,64))
discriminator:add(nn.SpatialConvolution(3,32,4,4,2,2,1,1):noBias()):add(nn.LeakyReLU(0.2, true))
discriminator:add(nn.SpatialConvolution(32,64,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(64)):add(nn.LeakyReLU(0.2, true))
discriminator:add(nn.SpatialConvolution(64,128,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(128)):add(nn.LeakyReLU(0.2, true))
discriminator:add(nn.SpatialConvolution(128,256,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(256)):add(nn.LeakyReLU(0.2, true))
discriminator:add(nn.SpatialConvolution(256,1,4,4,1,1):noBias())
discriminator:add(nn.Mean(1))

local generator = nn.Sequential()
generator:add(nn.Linear(100, 16384)):add(nn.ReLU())
generator:add(nn.Reshape(1024,4,4))
generator:add(nn.SpatialFullConvolution(1024,512,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(512)):add(nn.ReLU())
generator:add(nn.SpatialFullConvolution(512,256,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(256)):add(nn.ReLU())
generator:add(nn.SpatialFullConvolution(256,128,4,4,2,2,1,1):noBias()):add(nn.SpatialBatchNormalization(128)):add(nn.ReLU())
generator:add(nn.SpatialFullConvolution(128,3,4,4,2,2,1,1):noBias()):add(nn.Sigmoid())

discriminator:cuda()
generator:cuda()

local discriminator_params, discriminator_grads = discriminator:getParameters()
discriminator_params:uniform(-0.01, 0.01)

local generator_params, generator_grads = generator:getParameters()
generator_params:uniform(-0.05, 0.05)

local real = torch.CudaTensor(1,1):fill(-1)
local fake = real:clone():mul(-1)

function discriminator_eval(params)
    if params ~= discriminator_params then
        discriminator_params:copy(params)
    end
    discriminator_grads:zero()

    ------------------ against generative network -------------------
    local inputs = torch.CudaTensor(opt.batchsize,100):uniform(-1,1)

    local fake_inputs = generator:forward(inputs)
    local fpred = discriminator:forward(fake_inputs):clone()

    discriminator:backward(fake_inputs, fake)

    ------------------- real data -------------------
    batch_inputs = batch_inputs:cuda()

    local rpred = discriminator:forward(batch_inputs):clone()

    discriminator:backward(batch_inputs, real)

    return rpred:mean() - fpred:mean(), discriminator_grads
end

function generator_eval(params)
    if params ~= generator_params then
        generator_params:copy(params)
    end
    generator_grads:zero()

    local inputs = torch.CudaTensor(opt.batchsize,100):uniform(-1,1)

    local fake_inputs = generator:forward(inputs)

    local fpred = discriminator:forward(fake_inputs)

    local dloss = discriminator:backward(fake_inputs, real)
    generator:backward(inputs, dloss)

    if opt.saveresult then image.save('fake_images.jpg', fake_inputs[1]) end

    return fpred:mean(), generator_grads
end

for i = 0, opt.epoch do
    batch_inputs = data.batch(64)

    local _, discriminator_loss, generator_loss
    _, discriminator_loss = optim[opt.optimization](discriminator_eval, discriminator_params, discriminatorState)
    discriminator_params:clamp(-0.01,0.01)

    if i % 5 == 0 then
      _, generator_loss = optim[opt.optimization](generator_eval, generator_params, generatorState)

      output = string.format('ite: %-10d discriminator loss: %-10.3f generator_loss: %-10.3f', i, discriminator_loss[1], generator_loss[1])
      print(output)
    end
end
