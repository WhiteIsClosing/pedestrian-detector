require 'sys'
require 'image'
require 'optim'
require 'cunn'
require 'torch'
require 'cutorch'


function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

-- convert rgb to grayscale by averaging channel intensities
function rgb2gray(im)
  -- Image.rgb2y uses a different weight mixture

  local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
  if dim ~= 3 then
     print('<error> expected 3 channels')
     return im
  end

  -- a cool application of tensor:select
  local r = im:select(1, 1)
  local g = im:select(1, 2)
  local b = im:select(1, 3)

  local z = torch.Tensor(w, h):zero()

  -- z = z + 0.21r
  z = z:add(0.21, r)
  z = z:add(0.72, g)
  z = z:add(0.07, b)
  return z
end

--main function 
local ivch = 3  --color channels
local labelPerson = 1 -- label for person and background:
local labelBg = 2
local trainPosDir = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/posALL/'
local trainPosImaNumber = #ls(trainPosDir)
local trainNegDir = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/negALL/'
local trainNegImaNumber = #ls(trainNegDir)
local width =38
local height =78
local trSize = trainPosImaNumber + trainNegImaNumber

trainData = {
    data = torch.Tensor(trSize, ivch, width, height),
    labels = torch.Tensor(trSize),
    size = function() return trSize end
}

   -- shuffle dataset: get shuffled indices in this variable:
   local trShuffle = torch.randperm(trSize) -- train shuffle
   
   -- load person train data: 2416 images
   for i = 1, trSize, 2 do
      img = image.loadPNG(trainPosDir..ls(trainPosDir)[(i-1)/2+1],ivch) -- we pick all of the images in train!
      imgScale = image.scale(img, width, height,  bilinear)
      --grayimage = rgb2gray(imgScale)
      trainData.data[trShuffle[i]] = imgScale
      trainData.labels[trShuffle[i]] = labelPerson

      -- load background data:
      imgNeg = image.loadPNG(trainNegDir..ls(trainNegDir)[(i-1)/2+1],ivch)
      cropScale = image.scale(imgNeg, width, height,  bilinear)
      trainData.data[trShuffle[i+1]] = cropScale
      trainData.labels[trShuffle[i+1]] = labelBg 
   end

--trainData.data = trainData.data:cuda()
--trainData.labels = trainData.labels:cuda()

 --save created dataset:
 torch.save('train.t7',trainData)
 torch.save('test.t7',testData)

-- Displaying the dataset architecture ---------------------------------------
print(sys.COLORS.red ..  'Training Data:')
print(trainData)
print()

print(sys.COLORS.red ..  'Test Data:')
print(testData)
print()