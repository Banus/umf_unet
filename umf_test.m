out_folder='./';

plot_figures=true;
model_path = [out_folder, 'trainedModels/'];
load([model_path,'unet_v1_published.mat'],'net')

% testing
tsize=256; gsize=256;
images=[struct('folder', out_folder,'name', 'sub-131_sam-8_Image_em'),];  %#ok

outDir=[out_folder, 'predictions/'];
warning('off','MATLAB:MKDIR:DirectoryExists');
mkdir(outDir);

for i=1:numel(images)
  % prediction on overlapped tiles and majority voting
  fname=[images(i).folder,images(i).name,'.tiff'];
  [I,~,CI]=image_segment(net,fname,2*tsize,2*gsize); mask=CI>0.5;
  imwrite(mask,[outDir,images(i).name,'_pred.png'])
  out=labeloverlay(I,mask,'Colormap','autumn');
  imwrite(out,[outDir,images(i).name,'_overlay.tiff'])

  % evaluation with respect to the manual annotations
  gt=imread([images(i).folder,images(i).name,'.png']);
  mask=postprocess_predictions(mask,50,1);
  [pq,sq,rq]=panoptic_quality(gt,mask);
  tp=sum(mask&gt,'all'); fp=sum(mask&~gt,'all'); fn=sum(~mask&gt,'all');
  dice=2*tp/(2*tp+fp+fn); jaccard=tp/(tp+fp+fn);

  % report results
  fprintf('\nResult for %s:\n Panoptic Quality = %.3f\n', images(i).name, pq);
  fprintf(' Segmentation Quality = %.3f\n Recognition Quality = %.3f\n',sq,rq);
  fprintf(' DICE score = %.3f\n Jaccard coefficient = %.3f\n',dice,jaccard);

  % plot
  if plot_figures
    overlay=imoverlay(I,mask&gt,[0 1 0]);
    overlay=imoverlay(overlay,mask&~gt,[0 0 1]);
    overlay=imoverlay(overlay,~mask&gt,[1 0 0]);
    figure, imshow(overlay);
  end
end


%
% utility function
%

function [I,BI,CI]=image_segment(net,fname,tsize,gsize)
  %IMAGE_SEGMENT Run U-Net model on overlapped tiles and combine predictions
  %with majority voting.
  %   [I,BI,CI]=IMAGE_SEGMENT(net,fname,tsize,gsize) returns the original image
  %   I, the border mask BI and the fiber mask CI given the trained U-Net
  %   model (net), the name of the image (fname), the input size (tsize)
  %   and the outputs size (gsize). If tsize != gsize, e.g., when using
  %   valid convolutions, the step between tiles is determined by gsize;
  %   otherwise, it is tsize/8.
  if tsize==gsize, n_steps=8; else, n_steps=1; end

  I=imread(fname);
  I=I(:,:,1);
  [numRows,numCols]=size(I);

  strd=gsize/n_steps; n_perc=0;
  BI=zeros(numRows,numCols);
  CI=zeros(numRows,numCols);
  cnt=zeros(numRows,numCols);

  for i=1:strd:(numRows-tsize+1)
    for j=1:strd:(numCols-tsize+1)
      ie=i+tsize-1; je=j+tsize-1; dsize=int32((tsize-gsize)/2);
      gi=i+dsize; gj=j+dsize; gie=gi+gsize-1; gje=gj+gsize-1;
      subI=histeq(I(i:ie,j:je));
      C=uint8(semanticseg(subI,net));
      BI(gi:gie,gj:gje)=BI(gi:gie,gj:gje)+(C==2);
      CI(gi:gie,gj:gje)=CI(gi:gie,gj:gje)+(C==1);
      cnt(gi:gie,gj:gje)=cnt(gi:gie,gj:gje)+1;

      % update progress
      progress=fix((i*numCols+j)/(numRows*numCols)*100);
      if progress>n_perc, n_perc=progress; fprintf('.'); end
    end
  end
  CI=CI./cnt; BI=BI./cnt;
end


function img = postprocess_predictions(img, size_thr, n_steps)
  %POSTOPROCESS_PREDICTIONS Removes instances smaller than 'size_thr' and
  %dilates the remaining instances up to n_step times, stopping when they touch
  %other instances.
  
  [nr,nc]=size(img); IN=zeros(nr,nc);
  % label connected components and removes small ones (<size_thr pixels)
  CC=bwconncomp(img);
  for i=1:CC.NumObjects
    IN(CC.PixelIdxList{i})=i;
    if numel(CC.PixelIdxList{i})<size_thr,img(CC.PixelIdxList{i})=0; end
  end

  % dilate up to n_step pixels, if after dilation a connected component merges
  % to  another component revert back the dilation for that specific component
  for j=1:n_steps
    img_old=img; img=imdilate(img, strel('disk',1));
    IN_old=IN; IN=zeros(nr,nc);
    CC=bwconncomp(img);
    for i=1:CC.NumObjects
      IN(CC.PixelIdxList{i})=i;
      % if an instance cover two or more labels, dilation caused an overlap
      % between instances: roll-back dilation
      unique_labels=unique(IN_old(CC.PixelIdxList{i}));
      unique_labels=unique_labels(unique_labels~=0);
      if length(unique_labels)>=2
        img(CC.PixelIdxList{i})=img_old(CC.PixelIdxList{i});
        IN(CC.PixelIdxList{i})=IN_old(CC.PixelIdxList{i});
      end
    end
  end

end
