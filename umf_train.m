% standard paths
datapath='./dataset/';
out_folder='./';
data_folder=[out_folder,'tiles/'];
model_path=[out_folder,'trainedModels'];

rng(777);  % reproducibility
tsize=256; gsize=256;

warning('off','MATLAB:MKDIR:DirectoryExists');
mkdir(fullfile(data_folder,'im_train'));
mkdir(fullfile(data_folder,'gt_train'));
mkdir(fullfile(data_folder,'im_test'));
mkdir(fullfile(data_folder,'gt_test'));

% get list of training/test images with annotations and generate tiles
train_im_dir=fullfile(datapath,'full_im_train/');
train_gt_dir=fullfile(datapath,'full_gt_train/');
test_im_dir=fullfile(datapath,'full_im_test/');
test_gt_dir=fullfile(datapath,'full_gt_test/');

im_list_train=fullfile(train_im_dir,{dir([train_im_dir,'*.tiff']).name});
gt_list_train=fullfile(train_gt_dir,{dir([train_gt_dir,'*.tiff']).name});
im_list_test=fullfile(test_im_dir,{dir([test_im_dir,'*.tiff']).name});
gt_list_test=fullfile(test_gt_dir,{dir([test_gt_dir,'*.tiff']).name});
assert(length(im_list_train)==length(gt_list_train));
assert(length(im_list_test)==length(gt_list_test));

im_list=[im_list_train,im_list_test]; gt_list=[gt_list_train,gt_list_test];
j_test=length(im_list_train);
sample_area(im_list, gt_list, tsize, gsize, j_test, data_folder);

% train the network
classes =["UF", "UFB", "BG"];
labelIDs=[255   120    0];

labelDir=fullfile(data_folder,'gt_train');
pxds=pixelLabelDatastore(labelDir,classes,labelIDs);
imgDir=fullfile(data_folder,'im_train');
imds=imageDatastore(imgDir);
pximds=pixelLabelImageDatastore(imds,pxds);

tbl=countEachLabel(pxds);
imageFreq=tbl.PixelCount./tbl.ImagePixelCount;
classWeights=median(imageFreq)./imageFreq;

imgDir=fullfile(test_folder,'im_test');
imds=imageDatastore(imgDir);
labelDir=fullfile(test_folder,'gt_test');
pxds=pixelLabelDatastore(labelDir,classes,labelIDs);
pximdsVal=pixelLabelImageDatastore(imds,pxds);

inputSize=[2*tsize 2*tsize];
numClasses=numel(classes);

layers=unetLayers(inputSize,numClasses);

layers=add_batchnorm(layers,'Encoder-Stage-1-ReLU-1','BN1');
layers=add_batchnorm(layers,'Encoder-Stage-1-ReLU-2','BN2');
layers=add_batchnorm(layers,'Encoder-Stage-2-ReLU-1','BN3');
layers=add_batchnorm(layers,'Encoder-Stage-2-ReLU-2','BN4');
layers=add_batchnorm(layers,'Encoder-Stage-3-ReLU-1','BN5');
layers=add_batchnorm(layers,'Encoder-Stage-3-ReLU-2','BN6');
layers=add_batchnorm(layers,'Encoder-Stage-4-ReLU-1','BN7');
layers=add_batchnorm(layers,'Encoder-Stage-4-ReLU-2','BN8');

layers=add_batchnorm(layers,'Bridge-ReLU-1','BNB1');
layers=add_batchnorm(layers,'Bridge-ReLU-2','BNB2');
layers=add_batchnorm(layers,'Decoder-Stage-1-UpReLU','BNB11');
layers=add_batchnorm(layers,'Decoder-Stage-1-ReLU-1','BN12');
layers=add_batchnorm(layers,'Decoder-Stage-1-ReLU-2','BN13');
layers=add_batchnorm(layers,'Decoder-Stage-2-UpReLU','BN14');
layers=add_batchnorm(layers,'Decoder-Stage-2-ReLU-1','BN15');
layers=add_batchnorm(layers,'Decoder-Stage-2-ReLU-2','BN16');
layers=add_batchnorm(layers,'Decoder-Stage-3-UpReLU','BN17');
layers=add_batchnorm(layers,'Decoder-Stage-3-ReLU-1','BN18');
layers=add_batchnorm(layers,'Decoder-Stage-3-ReLU-2','BN19');
layers=add_batchnorm(layers,'Decoder-Stage-4-UpReLU','BN20');
layers=add_batchnorm(layers,'Decoder-Stage-4-ReLU-1','BN21');
layers=add_batchnorm(layers,'Decoder-Stage-4-ReLU-2','BN22'); 

pxLayer=pixelClassificationLayer('Name','labels','Classes',tbl.Name,...
                                 'ClassWeights',classWeights);
layers=replaceLayer(layers,"Segmentation-Layer",pxLayer);

options=trainingOptions('sgdm',... 
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',1,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9,... 
    'InitialLearnRate',0.01,...
    'L2Regularization',0.005,...
    'MaxEpochs',4,...  
    'MiniBatchSize',2,...
    'ValidationData',pximdsVal,...
    'ValidationFrequency',250,...
    'ValidationPatience',500,...
    'Shuffle','every-epoch',...
    'Plots','training-progress');

[net,~]=trainNetwork(pximds,layers,options); 
save([model_path,'\unet_train.mat'],'net')


%
% Helper functions
%

function indices = sample_indices(stats, n_samples)
  %SAMPLE_INDICES Given the shape statistics, computes the sampling scores and
  %sample n_samples indices.
  %   indices = SAMPLE_INDICES(stats, n_samples) returns n_sample fiber
  %   indices sampled using a multinomial distribution on normalized
  %   scores for the fibers, where the scores are the product of area and
  %   circularity of the fibers. 'stats' are shape statistics as returned by
  %   REGIONPROPS and they must include 'Area' and 'Perimeter'.
  circ=4*pi*[stats.Area]./[stats.Perimeter].^2; prob=1./circ;
  Aeff=[stats.Area].*prob; p=Aeff./sum(Aeff); % sampling effective area
  [~,indices]=max(mnrnd(1,p,n_samples),[],2); indices=indices'; 
end


function layers = add_batchnorm(layers, name, bn_name)
  %ADD_BATCHNORM Adds a batch normalization layer with name 'bn_name' to the
  %network layers, in the position after the layer 'name'.
  larray=[batchNormalizationLayer('Name',bn_name) reluLayer('Name',name)];
  layers=replaceLayer(layers,name,larray);
end


function [I, G, numRows, numCols] = load_image(imname, gtname)
  %LOAD_IMAGE Loads an image and its corresponding annotations and converts
  %the annotations in a class+border format, regardless of the original format.
  %   [I, G, numRows, numCols] = LOAD_IMAGE(imname, gtname) loads the image
  %   from the imname path and the annotations from the gtname file and
  %   returns them with the number of rows and columns of the image. The
  %   annotation may be images of the borders of the region (Neurolucida
  %   format) or Matlab arrays with multiple annotated regions and without
  %   borders; the code converts all the formats in unmyelinated fibers
  %   (UMF) regions with a 1-pixel border.
  disp(['Loading: ',imname]);
  I=imread(imname); I=I(:,:,1); numRows=size(I,1); numCols=size(I,2);
  BW=uint8(zeros(numRows,numCols));
  if endsWith(gtname, '.mat')
    % Matlab array with multiple annotations, UMF's id is 2, 
    G=(load(gtname).gt==2);
    BW(G)=1; G=imerode(G,strel('disk',1)); BW(G)=2;
  else
    % Image with annotation borders, as exported from e.g. Neurolucida
    G=logical(imread(gtname)); BW(G)=2; BW=imfill(BW); BW(G)=1; % fill+border
  end
  G=uint8(120*(BW==1)+255*(BW==2));
end


function sample_area(im_list, gt_list, tsize, gsize, j_test, data_folder)
  %SAMPLE_AREA Sample image and annotation tiles centered on fibers using area
  %and shape statistics to define a probability score for each fiber.
  %   SAMPLE_AREA(ds1, ds2, tsize, gsize, j_test, data_folder) takes a list
  %   of large images (im_list) and their annotations (gt_list) and extracts
  %   image tiles of size 2*tsize and annotation tiles of size 2*gsize (usually
  %   tsize==gsize). Any image with index larger than j_test is assigned to
  %   the validation set. The tiles are stored in a set of directories in
  %   the 'data_folder' path.
  for j=1:numel(im_list)
    [I,G,numRows,numCols]=load_image(im_list{j},gt_list{j});
    
    CC=bwconncomp(G==255,4);
    stats=regionprops(CC,'Area','Perimeter');
    idx=find([stats.Area]>100); stats=stats(idx);  % remove small elements
    CC.PixelIdxList=CC.PixelIdxList(idx); CC.NumObjects=numel(idx);

    is_test=(j>=j_test);
    if is_test, nmin=1000;
    else, nmin=min(2000,3*numel(stats));  % max: all tiles + 2 augmentations
    end

    if is_test
      im_path="im_test/"; lbl_path="gt_test/";
      step=30*ceil(CC.NumObjects/nmin);  % select only a small validation set
      indices=1:step:CC.NumObjects;      % and sample fibers uniformly
    else
      im_path="im_train/"; lbl_path="gt_train/";
      indices=sample_indices(stats, nmin);
    end
  
    nn=1;
    for i=indices
      pts=CC.PixelIdxList{i}; [ii,jj] = ind2sub([numRows,numCols],pts);

      % center jitter augmentation (only for training)
      if is_test==1, cntr=floor(mean([ii jj],1));
      else, idx=randsample(length(ii),1); cntr=[ii(idx) jj(idx)]; end

      is=cntr(1)-(tsize-1); ie=cntr(1)+tsize;
      js=cntr(2)-(tsize-1); je=cntr(2)+tsize;
      gis=cntr(1)-(gsize-1); gie=cntr(1)+gsize;
      gjs=cntr(2)-(gsize-1); gje=cntr(2)+gsize;
      
      % extract tiles
      if ie<=numRows && is>=1 && je<=numCols && js>=1  % inside the image
        I1=histeq(I(is:ie,js:je)); G1=G(gis:gie,gjs:gje);
        if size(I1,1)~=2*tsize, disp(j); end  % sanity check

        if ~is_test  % random flips augmentation (only for training)
          if rand>0.5, I1=fliplr(I1); G1=fliplr(G1); end
          if rand>0.5, I1=flipud(I1); G1=flipud(G1); end
        end

        tname=[num2str(j),'_',num2str(i),'_',num2str(nn)]; nn=nn+1;
        imwrite(I1,strcat(data_folder,im_path,tname,'.tiff'));
        imwrite(G1,strcat(data_folder,lbl_path,tname,'.png'));
      end
    end
  end
end
