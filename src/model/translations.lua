require 'image'
require 'sys'
require 'math'
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

function translations()
	local trainDir = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/posAU/'
	local savePath = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/postrans4/'
	local ivch = 3  
	local setSize = 2396
    local width = 38
    local height = 78
	for i = 1, setSize, 1 do 
    	img = image.loadPNG(trainDir..ls(trainDir)[i+1],ivch) -- we pick all of the images in train!
    	imgScale = image.scale(img, width, height, bilinear)
        --imgR = image.translate(imgR IM x, y)
        local xtrans = math.random(-2,2)
        local ytrans = math.random(-2,2)
    	imgR =image.translate(imgR, imgScale, xtrans, ytrans)
        crop = image.crop(imgR, math.abs(xtrans),0,width,height-math.abs(ytrans)):clone()
        imgS = image.scale(crop, width, height, bilinear):clone()
    	--print(img)
    	--print(imgR)
        s = savePath..ls(trainDir)[i+1]
        --print (s)
    	image.save(s, imgS)
 	end
end
translations()

