require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'
--parameter  setting
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 1, 'threads')

-- for all models:
cmd:option('-model', 'conv-psd', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsizeX', 38, 'sizeX of each input patch')
cmd:option('-inputsizeY', 78, 'sizeY of each input patch')
cmd:option('-nfiltersin', 3, 'number of input convolutional filters')
cmd:option('-nfiltersout', 16, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 5, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0.9, 'gradient momentum')
cmd:option('-maxiter', 1500, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', true, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 1000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 7, 'size of convolutional kernels')

-- logging:
cmd:option('-statinterval', 200, 'interval for saving stats and models')
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


--torch.manualSeed(params.seed)

torch.setnumthreads(params.threads)