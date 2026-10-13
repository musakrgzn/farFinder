"""Actual optical flow with synthetic semantic dropouts, not mocked tracking."""

import cv2
import numpy as np
import pytest

from farfinder.head.capture import HeadMaskTracker, HeadSegmenter, CaptureSelector


def scene(dx=0, dy=0, category=1, foreground=True):
    rng=np.random.default_rng(7)
    gray=rng.integers(35,220,(240,360),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(3,3),0.7)
    mask=np.zeros(gray.shape,np.uint8)
    cv2.ellipse(mask,(180,120),(80,90),0,0,360,255,-1)
    image=np.full((240,360,3),30,np.uint8)
    image[mask>0]=np.repeat(gray[:,:,None],3,axis=2)[mask>0]
    transform=np.float32([[1,0,dx],[0,1,dy]])
    image=cv2.warpAffine(image,transform,(360,240),borderValue=(30,30,30))
    mask=cv2.warpAffine(mask,transform,(360,240),flags=cv2.INTER_NEAREST)
    categories=np.zeros(mask.shape,np.uint8)
    probabilities=np.zeros((6,*mask.shape),np.float32)
    probabilities[0]=1
    if foreground:
        categories[mask>0]=category
        probabilities[0,mask>0]=0.1
        probabilities[category,mask>0]=0.9
    return image,mask,categories,probabilities


def resolve(tracker,values,now):
    image,mask,categories,probabilities=values
    semantic=HeadSegmenter.semantic_mask(categories,probabilities)
    return tracker.resolve(image,semantic,probabilities,now)


def test_skin_only_side_view_is_tracked_from_detected_head():
    tracker=HeadMaskTracker()
    first=resolve(tracker,scene(),0)
    assert first.reason=="OK"
    second_scene=scene(5,2,category=2)
    tracked=resolve(tracker,second_scene,0.1)
    assert tracked.reason=="OK"
    assert tracked.source=="temporal_tracking"
    assert tracked.tracking_inliers>=16
    assert tracked.tracking_error_px<1
    overlap=np.count_nonzero((tracked.mask>0)&(second_scene[1]>0))/np.count_nonzero(tracked.mask)
    assert overlap>0.98
    assert np.median(np.where(tracked.mask>0)[1])-np.median(np.where(first.mask>0)[1])==pytest.approx(5,abs=1)


def test_tracker_cannot_bootstrap_from_undifferentiated_body_skin():
    tracker=HeadMaskTracker()
    result=resolve(tracker,scene(category=2),0)
    assert not np.any(result.mask)
    assert "No reliable" in result.reason


def test_tracked_pixels_need_current_foreground_evidence():
    tracker=HeadMaskTracker()
    resolve(tracker,scene(),0)
    lost=resolve(tracker,scene(5,2,foreground=False),0.1)
    assert not np.any(lost.mask)
    assert "foreground support" in lost.reason
    assert tracker.mask is None


def test_unrelated_image_cannot_inherit_the_previous_head_mask():
    tracker=HeadMaskTracker()
    resolve(tracker,scene(),0)
    image,mask,categories,probabilities=scene(category=2)
    image=np.random.default_rng(40).integers(0,256,image.shape,dtype=np.uint8)
    result=resolve(tracker,(image,mask,categories,probabilities),0.1)
    assert not np.any(result.mask)


def test_tracking_expires_without_a_fresh_semantic_anchor():
    tracker=HeadMaskTracker(max_frames=2,max_seconds=3)
    resolve(tracker,scene(),0)
    assert resolve(tracker,scene(3,1,category=2),0.1).reason=="OK"
    assert resolve(tracker,scene(6,2,category=2),0.2).reason=="OK"
    expired=resolve(tracker,scene(9,3,category=2),0.3)
    assert not np.any(expired.mask)
    assert "expired" in expired.reason
    assert resolve(tracker,scene(9,3,category=1),0.4).reason=="OK"


def test_tracking_time_limit_and_session_reset():
    tracker=HeadMaskTracker(max_frames=100,max_seconds=3)
    resolve(tracker,scene(),0)
    assert "expired" in resolve(tracker,scene(3,1,category=2),3.1).reason
    resolve(tracker,scene(),4)
    tracker.reset()
    assert not np.any(resolve(tracker,scene(category=2),4.1).mask)


def test_capture_metadata_reports_tracking_as_estimated():
    tracker=HeadMaskTracker()
    resolve(tracker,scene(),0)
    values=scene(4,1,category=2)
    tracked=resolve(tracker,values,0.1)
    accepted,quality=CaptureSelector().evaluate(values[0],tracked,0.1)
    assert quality["head_mask_source"]=="temporal_tracking"
    assert quality["head_mask_status"]=="ESTIMATED"
    assert quality["tracking_inliers"]>=16
