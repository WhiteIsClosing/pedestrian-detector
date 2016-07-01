require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'
--train model
function trainModel(module, dataset)
	if params.hessian then
   nn.hessian.enable()
   module:initDiagHessianParameters()
end

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()

----------------------------------------------------------------------
-- train model
--
print(module)
print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do

   --------------------------------------------------------------------
   -- update diagonal hessian parameters
   --
   if params.hessian and math.fmod(t , hessianinterval) == 1 then
      -- some extra vars:
      local batchsize = params.batchsize
      local hessiansamples = params.hessiansamples
      local minhessian = params.minhessian
      local maxhessian = params.maxhessian
      local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
      etas = etas or ddl_ddx:clone()

      print('==> estimating diagonal hessian elements')
      
      for ih = 1,hessiansamples,batchsize do
        local inputs  = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsizeX,params.inputsizeY)
        local targets = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsizeX,params.inputsizeY)
        for i = ih,ih+batchsize-1 do
          -- next
          local input  = dataset.data[i]
          --input:resize(torch.CudaTensor(1,3*96,160))
          local target = dataset.data[i]
          --target:resize(torch.CudaTensor(1,3*96,160))
          inputs[{i-ih+1,{},{},{}}] = input
          targets[{i-ih+1,{},{},{}}] = target
        end
   
        local inputs_ = inputs:cuda()
        local targets_ = targets:cuda()
        module:updateOutput(inputs_,targets_)
        --print ('Done one fw pass')
        -- gradient
        dl_dx:zero()
        module:updateGradInput(inputs_, targets_)
        module:accGradParameters(inputs_, targets_) 

        -- hessian
        ddl_ddx:zero()
        module:updateDiagHessianInput(inputs_, targets_)
        module:accDiagHessianParameters(inputs_, targets_)

        -- accumulate
        ddl_ddx_avg:add(batchsize/hessiansamples, ddl_ddx)
      end

      -- cap hessian params
      print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
      ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
      ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
      print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

      -- generate learning rates
      etas:fill(1):cdiv(ddl_ddx_avg)
      collectgarbage()
   end

   --------------------------------------------------------------------
   -- progress
   --
   iter = iter+1
   xlua.progress(iter*params.batchsize, params.statinterval)

   --------------------------------------------------------------------
   -- create mini-batch
   --
   local example = dataset.data[t]
   local inputs = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsizeX,params.inputsizeY)
   local targets = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsizeX,params.inputsizeY)
   for i = t,math.min(t+params.batchsize-1, 1799) do
      -- load new sample
      print(i)
      local sample = dataset.data[i]
      local input = sample:clone()
      local target = sample:clone()
      inputs[{i-t+1,{},{},{}}] = input
      targets[{i-t+1,{},{},{}}] = target
   end
   
   inputs_ = inputs:cuda()
   targets_ = targets:cuda()
   --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      f = f + module:updateOutput(inputs_, targets_)

      -- gradients
      module:updateGradInput(inputs_, targets_)
      module:accGradParameters(inputs_, targets_)
      
      -- normalize
      --dl_dx:div(#inputs)
      --f = f/#inputs
      dl_dx:div(inputs:size(1))
      f = f/(inputs:size(1))

      -- return f and df/dx
      return f,dl_dx
   end

   --------------------------------------------------------------------
   -- one SGD step
   --
   sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]

   -- normalize
   if params.model:find('psd') then
      module:normalize()
   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --print (t .. ' ' .. statinterval .. ' ' .. math.fmod(t , statinterval) )
   if math.fmod(t , statinterval) == 1 then

      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)
      -- get temperate convolutional layer result tensor for next layer unsupervised training input
      --todo
end