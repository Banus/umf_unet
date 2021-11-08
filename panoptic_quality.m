function [pq, sq, rq] = panoptic_quality(GT, PM)
%PANOPTIC_QUALITY computes the panoptic quality, segmentation quality and
%recognition quality of a segmentation [1].
%   [PQ, SQ, RQ] = PANOPTIC_QUALITY(GT, PM) return the Panoptic Quality
%   (PQ), Segmentation Quality (SQ) and Recognition Quality (RQ) while
%   comparing manually annotated instances in GT and predicted isntances in
%   PM. GT and PM are binary masks where the background class (0) is
%   ignored, and each connected component is considered a separate
%   instance. If no instances are matched, PQ=SQ=RQ=0.
%
% Adapted from https://github.com/tensorflow/models/blob/master/research/deeplab/evaluation/panoptic_quality.py
%
% References
% -----------
% [1] Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019).
%     Panoptic segmentation. In Proceedings of the IEEE/CVF Conference on
%     Computer Vision and Pattern Recognition (pp. 9404-9413).

gt=bwlabel(GT); pm=bwlabel(PM); gt=gt(:); pm=pm(:);
[gt_areas, ~]=hist(gt,unique(gt)); [pm_areas, ~]=hist(pm,unique(pm));
max_instances=max(max(gt,[],"all"),max(pm,[],"all"))+1;
% define an unique code for each PM/GT pair, unique values are intersections
intrs=gt.*max_instances+pm; [intr_areas, intr_ids]=hist(intrs,unique(intrs));

TP=0; iou=0;
for i=1:length(intr_ids)
  intr_id=intr_ids(i);
  if intr_id<max_instances, continue; end  % skip background
  gt_id=fix(intr_id/max_instances)+1;
  pm_id=rem(intr_id, max_instances)+1;
  intr=intr_areas(i);
  % remove intersection with background (first max_instance IDs)
  unio=gt_areas(gt_id)+pm_areas(pm_id)-intr;
  if unio>0, iou_i=intr/unio; else, iou_i=0; end
  if iou_i>0.5, iou=iou+iou_i; TP=TP+1; end  % match only for IoU>0.5
end

FN=max(gt)-TP; FP=max(pm)-TP;
if TP>0, sq=iou/TP; else, sq=0; end
rq=TP/(TP+0.5*FP+0.5*FN);
pq=sq*rq;

end

