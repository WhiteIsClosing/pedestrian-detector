--load data from disk, which is the output for the previous layer. 
-- and the output of the function is input for next layer.
require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'
require 'math'
------------------------------------------------------------------------
--parameter settings
cmd = torch.CmdLine()
cmd:text()
cmd:text('input data for next unsupervised layer')
cmd:option('-dir', 'firstlayer', 'subdirectory to save experiments in')
cmd:option('-inputsizeX', 38, 'sizeX of each input patch')
cmd:option('-inputsizeY', 78, 'sizeY of each input patch')
cmd:option('-nfiltersin', 3, 'number of input convolutional filters')
cmd:option('-nfiltersout', 32, 'number of output convolutional filters')

------------------------------------
function maxpooling(path)
	local kW = 2
	local kH = 2 
	local dW = 2
	local dH = 2 
	local padW = 0
	local padH = 0
	local vch = 3 -- color channel
	trainOut = torch.load(path) --input data for maxpooling
	
	model = nn.Sequential()
	--Applies 2D max-pooling operation in kWxkH regions by step size dWxdH step
	model:add(nn.SpatialMaxPooling(kW, kH [, dW, dH, padW, padH]))
	for t = 1, #trainOut, 1 do
		local pooled = model.forward(trainOut[t])
		poolOut{t,{},{},{}} = pooled
	end
    torch.save('maxpooled', poolOut)
end

----------------------------------------------------------------------
-- load data
dataset = torch.load('/home/deguoxi/torch/extra/unsupgpu/src/train.t7')
print ('data set size')
print (#dataset.data)
----------------------------------------------------------------------
-- create model
   local conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   local kw, kh = params.kernelsize, params.kernelsize
   local W,H = params.inputsizeX, params.inputsizeY
   local padw, padh = torch.floor(params.kernelsize/2.0), torch.floor(params.kernelsize/2.0)
   --local padw, padh = 0, 0
   local batchSize = params.batchsize or 1
   -- connection table:
   local outputFeatures = conntable[{ {},2 }]:max()
   local inputFeatures = conntable[{ {},1 }]:max()
--model with conv -> tanh -> maxpooling
   module  = nn.Sequential()
   module:add(nn.SpatialConvolution(inputFeatures,outputFeatures, kw, kh, 1, 1, padw, padh))
   module:add(nn.Tanh())
   module:add(nn.Diag(outputFeatures))
   module:add(nn.SpatialMaxPooling(2, 2 [, 2, 2, 0, 0]))
   module:cuda()
   -- verbose
   print('==> constructed convolutional predictive sparse decomposition (PSD) unpervised classification neural network')
--copy weight from disk for the convulitional layer (filters weight 32*3*7*7 , bias 32*3*7*1)
trainedW = torch.load(' ')
module.modules[1].weight = trainedW
--forward image data through the network, save the output data on disk
poolOut = torch.Tensor(#trainOut, 3, 19, 38)
imgData = torch.load(path) 
for t 1, #imgData, 1 do
	local pooled  = module.forward(trainOut[t])
	poolOut{t,{},{},{}} = pooled
end
torch.save('maxpooled', poolOut)
