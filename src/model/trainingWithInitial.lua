require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'
require 'math'
----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputHessian', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 1, 'threads')

-- for all models:
cmd:option('-loss', 'bce', 'type of loss function to minimize:bce | margin | softmargin')
--cmd:option('-model', 'conv-psd', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsizeX', 38, 'sizeX of each input patch')
cmd:option('-inputsizeY', 78, 'sizeY of each input patch')
cmd:option('-nfiltersin', 3, 'number of input convolutional filters')
cmd:option('-nfiltersout', 32, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0.9, 'gradient momentum')
cmd:option('-maxiter', 11933*2, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', true, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 1000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 7, 'size of convolutional kernels')

-- logging:
cmd:option('-statinterval', 1000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:addTime('psd')
cmd:log(params.rundir .. '/log.txt', params)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(1) -- by default, use GPU 1
torch.manualSeed(params.seed)
local statinterval = torch.floor(params.statinterval / params.batchsize)*params.batchsize
local hessianinterval = torch.floor(params.hessianinterval / params.batchsize)*params.batchsize
print (statinterval)
print (hessianinterval)


torch.setnumthreads(params.threads)

----------------------------------------------------------------------
-- load data
dataset = torch.load('/home/deguoxi/torch/extra/unsupgpu/src/train.t7')
print ("data set size")
print (#dataset.data)
----------------------------------------------------------------------
-- create model
   local conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   local kw, kh = params.kernelsize, params.kernelsize
   local W,H = params.inputsizeX, params.inputsizeY
   local padw, padh = torch.floor(params.kernelsize/2.0), torch.floor(params.kernelsize/2.0)
   --local padw, padh = 0, 0
   local batchSize = params.batchsize or 1
   local noutLabel = 2
   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }] 
   local outputFeatures = conntable[{ {},2 }]:max()
   local inputFeatures = conntable[{ {},1 }]:max()
--model with conv -> tanh -> maxpooling -> conv -> tanh -> maxplooing -> linear logistic regression
--todo an parallel with subsampling from the previous layer
   module  = nn.Sequential()
   module:add(nn.SpatialConvolution(inputFeatures,outputFeatures, kw, kh, 1, 1, padw, padh))
   module:add(nn.Tanh())
   module:add(nn.Diag(outputFeatures))
   module:add(nn.SpatialMaxPooling(2, 2 [, 2, 2, 0, 0]))
   module:add(nn.SpatialConvolution(32, 64, kw, kh, 1,1, padw, padh))
   module:add(nn.Tanh())
   module:add(nn.Diag(64))
   module:add(nn.SpatialMaxPooling(2, 2 [, 2, 2, 0, 0])) -- output of size 3*5*15, 64 feature maps, 
   module.add(nn.Linear(64*3*5*15, noutLabel, bias = true))
   module:cuda()

   -- verbose
   print('==> constructed convolutional predictive sparse decomposition (PSD) unpervised classification neural network')

----------------------------------------------------------------------
-- loss function 
print '==> define loss'

if cmd.loss == 'margin'
    criterion = nn.MarginCriterion()
elseif cmd.losss = 'softmargin'
    criterion = nn.SoftMarginCriterion()
else if cmd.loss = 'bce'
--Binary Cross Entropy between the target and the output:
--loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
    module.add(nn.Sigmoid())
    criterion = nn.BCECriterion()
end
print '==> here is the loss function:'
print(criterion)
------------------------------------------------------------------------
-- model copy initialization 
--todo 






-------------------------------------------------------------------------
-- train model



