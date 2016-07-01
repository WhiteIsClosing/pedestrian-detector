require 'sys'
require 'image'
require 'image'
require 'optim'
require 'cunn'
require 'torch'
require 'cutorch'
print(sys.COLORS.red ..  '==> preprocessing data')

function getdata(filename)
   local data = torch.load(filename)
   local dataset ={}

   local std = std or 0.2
   local nsamples = data:size(1)
   local nrows = data:size(2)
   local ncols = data:size(3)

   function dataset:size()
      return nsamples
   end

   function dataset:selectPatch(nr,nc)
      local imageok = false
      if simdata_verbose then
         print('selectPatch')
      end
      while not imageok do
         --image index
         local i = math.ceil(torch.uniform(1e-12,nsamples))
         local im = data:select(1,i)
         -- select some patch for original that contains original + pos
         local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
         local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
         local patch = im:narrow(1,ri,nr)
         patch = patch:narrow(2,ci,nc)
         local patchstd = patch:std()
         if data_verbose then
            print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
         end
         if patchstd > std then
            if data_verbose then
               print(patch:min(),patch:max())
            end
            return patch,i,im
         end
      end
   end

   local dsample = torch.Tensor(inputsize*inputsize)

   function dataset:conv()
      dsample = torch.Tensor(1,inputsize,inputsize)
   end

   setmetatable(dataset, {__index = function(self, index)
                                       local sample,i,im = self:selectPatch(inputsize, inputsize)
                                       dsample:copy(sample)
                                       return {dsample,dsample,im}
                                    end})
   return dataset
end


local channels = {'g'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
trainData = torch.load('/home/deguoxi/torch/extra/unsupgpu/src/train.t7')
testData = torch.load('/home/deguoxi/torch/extra/unsupgpu/src/test.t7')
print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics:')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data:')
--save created dataset:
torch.save('train.t7',trainData)
torch.save('test.t7',testData)
-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

--[[if opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end]]
