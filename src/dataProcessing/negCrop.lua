require 'image'
require 'sys'
require 'math'
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

function augumention()
	local trainDir = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/neg/'
	local savePath = '/home/deguoxi/torch/extra/unsupgpu/INRIAPerson/trainUSE/negCROP/'
	local ivch = 3  
	local setSize = 662
    local width = 38
    local height = 78
	for i = 1, setSize, 1 do 
    	img = image.loadPNG(trainDir..ls(trainDir)[i+1],ivch) -- we pick all of the images in train!
        if height <480 then
            xtrans = math.random(0,160)
            ytrans = math.random(0,160)
        else  
            xtrans = math.random(0,400)
            ytrans = math.random(0,400)
        end
        imgCrop = image.crop(img, xtrans, ytrans, xtrans+width, ytrans+height):clone()
        --imgR =image.hflip(imgR, imgCrop)
        s = savePath..tostring(setSize*19 + i)..(".png")
        --print (s)
        --print (s)
    	image.save(s, imgCrop)
 	end
end
augumention()
