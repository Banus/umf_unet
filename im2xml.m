function im2xml(fname_EM,size_thr,mode)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fname_EM: input segmentation file name
% size_thr: minimum element size
% mode: either convhull (findx the convex hull) or fill (only hole fill)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[folder, base_fn, ~] = fileparts(fname_EM);
fname_xml = fullfile(folder, sprintf('%s.xml', base_fn));

if nargin < 3, mode = 'convhull'; end
if nargin < 2, size_thr = 250; end

%-- define template file name and scaling factor
template_xml = 'template.xml';
scale_x = 0.0137;
scale_y = 0.0137;

%-- read xml template
fid = fopen(template_xml,'r');
if fid < 0
  error("XML template not found. Ensure that %s is in the working directory.", ...
        template_xml);
end
tline = fgetl(fid);
A = cell(0,1);
while ischar(tline)
  A{end+1,1} = tline; %#ok<AGROW>
  tline = fgetl(fid);
end
fclose(fid);

%-- log image file name and scaling factors
A{10} = sprintf('%s',['    <filename>',fname_EM,'</filename>']);
A{16} = sprintf('%s',['    <scale x="',num2str(scale_x),'" y="',num2str(scale_y),'"/>']);

%-- read image file
img = imread(fname_EM);

%-- clean image and fill holes
CC = bwconncomp(img);
for i=1:CC.NumObjects
  if numel(CC.PixelIdxList{i}) < size_thr
    img(CC.PixelIdxList{i})=0;
  end
end
img = imdilate(img, strel('disk', 2));
if strcmp(mode,'fill')
  img = imfill(img,'hole');
  % img = bwconvhull(img,'objects');  % optional
end
% imwrite(img, 'preprocessed_hull.tiff');  % for debug

%-- trace the boundary of individual connected components
if strcmp(mode,'fill')
  bound_pts = bwboundaries(img);
  % points are returned as y,x coordinates but we need x,y
  bound_pts = arrayfun(@(v) fliplr(v), bound_pts);
else
  bound_pts = {regionprops(img,'ConvexHull').ConvexHull}';
end

%-- log the boundary coordinates into a xml file
fid = fopen(fname_xml,'w');
for i = 1:numel(A)
  fprintf(fid,'%s\n', A{i});
end

for iobj = 1 : length(bound_pts)
  [x, y] = reducem(bound_pts{iobj}(:,1), bound_pts{iobj}(:,2));
  fprintf(fid,'%s\n','<contour name="Unmyelinated Axon" color="#FF8000" closed="true" shape="Contour">');
  fprintf(fid,'%s\n','  <property name="GUID"><s></s></property>');
  fprintf(fid,'%s\n','  <property name="FillDensity"><n>0</n></property>');
  for idx=1:numel(x)
    fprintf(fid,'%s\n',['  <point x="',num2str(round(x(idx)*scale_x,2)),...
                        '" y="',num2str(-round(y(idx)*scale_y,2)),...
                        '" z="0.00" d="0.03"/>']);
  end
  fprintf(fid,'%s\n','</contour>');
  if mod(iobj,200) == 0, fprintf('.'); end
end
fprintf('\n');
fprintf(fid,'%s','</mbf>');
fclose(fid);

end
