require 'image'
require 'sys'
require 'math'
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

function augumention()
	local trainDir = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/pos/'
	local savePath = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/posAU/'
	local ivch = 3  
	local setSize = 2396
	for i = 1, setSize, 1 do 
    	img = image.loadPNG(trainDir..ls(trainDir)[i+1],ivch) -- we pick all of the images in train!
    	imgScale = image.scale(img, 38, 78, bilinear)
    	imgR =image.hflip(imgR, imgScale)
    	--print(img)
    	--print(imgR)
        s = savePath..ls(trainDir)[i+1]
        --print (s)
    	image.save(s, imgR)
 	end
end
augumention()

