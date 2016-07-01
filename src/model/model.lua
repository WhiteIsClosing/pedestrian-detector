require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'
--autoencoder and decoder model build
function buildmodel()
   local conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   local kw, kh = params.kernelsize, params.kernelsize
   local W,H = params.inputsizeX, params.inputsizeY
   local padw, padh = 0, 0
   local batchSize = params.batchsize or 1
   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }] 
   local outputFeatures = conntable[{ {},2 }]:max()
   local inputFeatures = conntable[{ {},1 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(inputFeatures,outputFeatures, kw, kh, 1, 1, padw, padh))
   encoder:add(nn.Tanh())
   
   encoder:add(nn.Diag(outputFeatures))
   print(encoder)
   print(encoder.modules[1].weight)
   -- decoder is L1 solution:
   print(kw, kh, W, H, padw, padh, params.lambda, batchSize) 
   print(decodertable)
   decoder = unsupgpu.SpatialConvFistaL1(decodertable, kw, kh, W-kw+1, H-kh+1, padw, padh, params.lambda, batchSize) -- here w, h should be the input size or output?
   print(decoder)
   -- PSD autoencoder
   module = unsupgpu.PSD(encoder, decoder, params.beta)

   module:cuda()
   
   print('==> constructed convolutional predictive sparse decomposition (PSD) auto-enocdder')
end
