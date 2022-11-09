function [picdata,numTag] = txtImgFldRead(filepath, DatName)
% eg: DatInfo = txtImgFldRead('G:\tools for dicom files\test\','testDat.mat')
% coded by Mengzhou Li, Feb 2019

di0 = dir([filepath,'*.txt']);
di = deldotslashfiles(di0);
numframe = length(di);
disp(numframe)
img = load([filepath,di(1).name]);
picdata = zeros([numframe,size(img)],'uint16');

o = zeros(1,numframe);
for k = 1:numframe
    c = split(di(k).name,'_');
    c = split(c{end},'.');
    o(k) = str2double(c{end-1});
end
[numTag,oi] = sort(o);

h = waitbar(0,'Start reading data, please wait');
for k = 1:numframe
    img = load([filepath,di(oi(k)).name]);
    picdata(k,:) = uint16(img(:));
    waitbar(k/numframe,h,['Reading file ', num2str(k),' out of ',num2str(numframe)])
end
close(h)

save(DatName,'picdata','numTag','-v7.3');
disp('Finished!');
end

function oi = deldotslashfiles(di)
numfig = length(di);
k = 1;
for itag = 1:numfig
    if startsWith(di(itag).name,'._')
    else
        oi(k) = di(itag);
        k = k + 1;
    end
end
end 
