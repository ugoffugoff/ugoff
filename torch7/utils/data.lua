require 'torch'

local function dir(path)
    local list = paths.dir(path)

    if list ~= nil then
      for i = 1, #list do
        if list[i] == '.' then table.remove(list,i) end
      end
      for i = 1, #list do
        if list[i] == '..' then table.remove(list,i) end
      end
    end
    return list
end

local function filePath(path, files)
    local list = {}

    for k in ipairs(files) do
      list[k] = path .. files[k]
    end

    return list
end

local data = {}
data.__index = data

data.nSamples = 0
data.counter = 0

function data.dataLoader(path, batchsize)
    local files = dir(path)
    data.nSamples = #files
    data.images = filePath(path, files)
    data.batchSize = batchsize or 32
end

function data.batch(dimension)
    local start_index = data.counter * data.batchSize + 1
    local end_index = math.min(data.nSamples, (data.counter + 1) * data.batchSize)
  
    local filelist = {}

    for i = start_index, end_index do
      filelist[#filelist+1] =  data.images[i]
    end
  
    local batch_inputs = torch.Tensor(#filelist,3,dimension,dimension)
  
    for i = 1, #filelist do
      local img = image.load(filelist[i])
      batch_inputs[i]:copy(img)
    end

    if end_index == data.nSamples then
      data.counter = 0
    else
      data.counter = data.counter + 1
    end

    return batch_inputs
end

return data
